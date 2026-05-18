(function () {
    const streamLog = document.getElementById("stream-log");
    const startButton = document.getElementById("start-run");
    const runsTable = document.getElementById("runs-table");
    const providers = document.getElementById("providers");
    const statusStrip = document.getElementById("status-strip");
    const stageList = document.getElementById("stage-list");
    const scenarioSelect = document.getElementById("scenario-select");
    const scenarioSummary = document.getElementById("scenario-summary");
    const caseCount = document.getElementById("case-count");
    const failureCount = document.getElementById("failure-count");
    const streamEvents = document.getElementById("stream-events");
    const streamProgressLabel = document.getElementById("stream-progress-label");
    const streamProgressBar = document.getElementById("stream-progress-bar");
    const guidedRunButton = document.querySelector("[data-action='start-onboarding-run']");

    function appendLog(line) {
        if (!streamLog) {
            return;
        }

        const current = streamLog.textContent.trim();
        streamLog.textContent = current ? `${current}\n${line}` : line;
    }

    function setLog(text) {
        if (streamLog) {
            streamLog.textContent = text;
        }
    }

    function resetStreamUI() {
        setLog("Connecting to streaming run...");
        if (streamEvents) {
            streamEvents.replaceChildren(
                eventCard("Connecting", "Opening a local event stream for the selected scenario.", "neutral"),
            );
        }
        updateProgress(0, "Connecting");
    }

    function updateProgress(percent, label) {
        if (streamProgressLabel) {
            streamProgressLabel.textContent = label;
        }
        if (streamProgressBar) {
            streamProgressBar.style.width = `${Math.max(0, Math.min(100, percent))}%`;
        }
    }

    function appendEventCard(label, body, tone, meta) {
        if (!streamEvents) {
            return;
        }
        const empty = streamEvents.querySelector(".event-empty");
        if (empty) {
            empty.remove();
        }
        streamEvents.prepend(eventCard(label, body, tone, meta));
    }

    function eventCard(label, body, tone, meta) {
        const card = el("article", { className: `event-card event-${tone || "neutral"}` });
        card.append(el("span", { text: label }), el("strong", { text: body }));
        if (meta) {
            card.append(el("p", { text: meta }));
        }
        return card;
    }

    function setStage(stage) {
        if (!stageList) {
            return;
        }
        Array.from(stageList.querySelectorAll("[data-stage]")).forEach((item) => {
            item.classList.toggle("active", item.dataset.stage === stage);
            if (stage === "report" && item.dataset.stage !== "report") {
                item.classList.add("complete");
            }
        });
    }

    function updateScenarioSummary() {
        if (!scenarioSelect || !scenarioSummary) {
            return;
        }
        const option = scenarioSelect.selectedOptions[0];
        if (!option) {
            return;
        }
        const values = [
            ["Objective", option.dataset.objective || ""],
            ["Technique", option.dataset.technique || ""],
            ["OWASP map", option.dataset.owasp || ""],
        ];
        scenarioSummary.replaceChildren(
            ...values.map(([label, value]) => {
                const item = el("div");
                item.append(el("span", { text: label }), el("strong", { text: value }));
                return item;
            }),
        );
    }

    async function refreshStatus() {
        const response = await fetch("/api/status", { headers: { Accept: "application/json" } });
        if (!response.ok) {
            throw new Error(`Status refresh failed with ${response.status}`);
        }

        const data = await response.json();
        if (providers && Array.isArray(data.providers)) {
            providers.replaceChildren(...data.providers.map(renderProviderCard));
        }
        if (statusStrip) {
            statusStrip.replaceChildren(...renderStatusTiles(data));
        }
    }

    async function refreshRuns() {
        const response = await fetch("/api/runs/recent", { headers: { Accept: "application/json" } });
        if (!response.ok) {
            throw new Error(`Runs refresh failed with ${response.status}`);
        }

        const data = await response.json();
        if (runsTable && Array.isArray(data.items)) {
            runsTable.replaceChildren(...data.items.map(renderRunRow));
        }
    }

    function renderProviderCard(provider) {
        const status = provider.status || "unknown";
        const card = el("article", { className: "card provider-card" });
        const header = el("div", { className: "card-header" });
        const titleWrap = el("div");
        titleWrap.append(
            el("p", { className: "card-kicker", text: "Provider" }),
            el("h3", { text: provider.provider_id || "unknown" }),
        );
        header.append(
            titleWrap,
            el("span", { className: `badge status-${badgeClass(status)}`, text: status }),
        );

        const metrics = el("dl", { className: "metric-list" });
        metrics.append(
            metric("Base URL", provider.base_url || ""),
            metric("Default model", provider.default_model || ""),
            metric("Models found", String(Number(provider.models_available || 0))),
            metric("Latency", formatLatency(provider.latency_ms)),
            metric("API key", formatKeyStatus(provider)),
            metric("Streaming", provider.supports_streaming ? "yes" : "no"),
        );

        card.append(
            header,
            el("p", { className: "card-description", text: provider.note || "" }),
            metrics,
        );
        const picker = renderModelPicker(provider);
        if (picker) {
            card.append(picker);
        }
        return card;
    }

    function renderModelPicker(provider) {
        const models = Array.isArray(provider.models) ? provider.models : [];
        if (!models.length) {
            return null;
        }

        const label = el("label", { className: "model-picker" });
        const select = document.createElement("select");
        select.dataset.providerId = provider.provider_id || "";
        models.forEach((model) => {
            const option = document.createElement("option");
            option.value = model;
            option.textContent = model;
            option.selected = model === provider.default_model;
            select.append(option);
        });
        select.addEventListener("change", () => {
            selectModel(select.dataset.providerId, select.value).catch((error) => {
                appendLog(`Model selection failed: ${error.message}`);
            });
        });
        label.append(el("span", { text: "Demo model" }), select);
        return label;
    }

    async function selectModel(providerId, modelId) {
        if (!providerId || !modelId) {
            return;
        }

        const response = await fetch(`/api/providers/${encodeURIComponent(providerId)}/model`, {
            method: "POST",
            headers: {
                Accept: "application/json",
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ model_id: modelId }),
        });

        if (!response.ok) {
            const payload = await response.json().catch(() => ({}));
            throw new Error(payload.detail || "Unable to select model");
        }

        appendLog(`Selected ${modelId} for ${providerId}.`);
        await refreshStatus();
    }

    function renderRunRow(run) {
        const summary = run.summary || {};
        const reportLinks = run.report_links || {};
        const row = document.createElement("tr");
        const runCode = document.createElement("code");
        runCode.textContent = run.run_id || "unknown";

        const status = run.status || "unknown";
        const statusBadge = el("span", {
            className: `badge status-${badgeClass(status)}`,
            text: status,
        });

        const actions = el("div", { className: "table-actions" });
        actions.append(
            reportLink("data", reportLinks.json || "#"),
            reportLink("html", reportLinks.html || "#"),
            reportLink("md", reportLinks.markdown || "#"),
            reportLink("raw", reportLinks.raw_markdown || "#"),
        );

        row.append(
            td(runCode),
            td(run.suite_name || "unknown"),
            td(statusBadge),
            td(`${Number(run.progress || 0)}%`),
            td(String(Number(summary.failure_count || 0))),
            td(`${String(summary.success_rate ?? 0)}%`),
            td(actions),
        );
        return row;
    }

    function renderStatusTiles(data) {
        const setup = data.setup || {};
        const activeRun = data.active_run || {};
        const activeProvider = (data.providers || []).find((provider) => provider.status === "ready")
            || (data.providers || [])[0]
            || {};
        return [
            statusTile("Ready providers", String(Number(setup.ready_providers || 0))),
            statusTile("Active provider", activeProvider.provider_id || "none"),
            statusTile("Active model", activeProvider.default_model || "none"),
            statusTile("Latest run", activeRun.run_id || "none"),
        ];
    }

    function statusTile(label, value) {
        const tile = el("div", { className: "status-tile" });
        tile.append(el("span", { text: label }), el("strong", { text: value }));
        return tile;
    }

    function metric(label, value) {
        const item = document.createElement("div");
        item.append(el("dt", { text: label }), el("dd", { text: value }));
        return item;
    }

    function reportLink(label, href) {
        const link = el("a", { className: "ghost-link", text: label });
        link.setAttribute("href", href);
        return link;
    }

    function td(content) {
        const cell = document.createElement("td");
        if (content instanceof Node) {
            cell.append(content);
        } else {
            cell.textContent = content;
        }
        return cell;
    }

    function el(tagName, options) {
        const node = document.createElement(tagName);
        const safeOptions = options || {};
        if (safeOptions.className) {
            node.className = safeOptions.className;
        }
        if (safeOptions.text !== undefined) {
            node.textContent = String(safeOptions.text);
        }
        return node;
    }

    function badgeClass(status) {
        if (["healthy", "completed", "ready"].includes(status)) {
            return "good";
        }
        if (["checking", "needs_api_key", "no_models", "running"].includes(status)) {
            return "warn";
        }
        return "bad";
    }

    function formatLatency(value) {
        if (typeof value === "number") {
            return `${Math.round(value)} ms`;
        }
        return "not checked";
    }

    function formatKeyStatus(provider) {
        if (provider.api_key_set) {
            return "set";
        }
        if (["lmstudio", "ollama", "vllm"].includes(provider.provider_id)) {
            return "not required locally";
        }
        return "missing";
    }

    function startStream() {
        resetStreamUI();
        const params = new URLSearchParams({
            scenario: scenarioSelect ? scenarioSelect.value : "leakage",
            cases: clampNumber(caseCount ? caseCount.value : 5, 1, 12),
            failures: clampNumber(failureCount ? failureCount.value : 2, 0, 12),
        });
        const source = new EventSource(`/api/runs/demo/stream?${params.toString()}`);
        let completed = false;

        if (startButton) {
            startButton.disabled = true;
            startButton.textContent = "Running...";
        }
        if (guidedRunButton) {
            guidedRunButton.disabled = true;
            guidedRunButton.textContent = "Running...";
        }

        source.addEventListener("start", (event) => {
            const data = JSON.parse(event.data);
            setLog(
                `Run ${data.run_id} started. Scenario: ${data.label || data.scenario}. Cases: ${data.cases}. Injected findings: ${data.failures}.`,
            );
            appendEventCard(
                "Run started",
                `${data.label || data.scenario}: ${data.objective || "Scenario ready."}`,
                "running",
                `${data.cases} cases, ${data.failures} injected findings, ${data.owasp || "OWASP map unavailable"}`,
            );
            updateProgress(8, "Provider check");
            setStage("provider");
        });

        source.addEventListener("progress", (event) => {
            const data = JSON.parse(event.data);
            setStage(data.step < data.total ? "mutate" : "score");
            updateProgress(Math.round((data.step / data.total) * 82) + 10, `Case ${data.step}/${data.total}`);
            appendEventCard(
                data.finding,
                `${data.case_id}: leak ${data.leak_score}, policy ${data.policy_violation_score}`,
                data.finding.includes("withheld") || data.finding.includes("ignored") || data.finding.includes("grounded") || data.finding.includes("contained") ? "pass" : "finding",
                `${data.risk || "Risk"} | ${data.owasp || "OWASP"} | ${data.technique || "Technique unavailable"}`,
            );
            appendLog(
                `Step ${data.step}/${data.total}: ${data.case_id} | ${data.finding} | leak ${data.leak_score} | policy ${data.policy_violation_score}`,
            );
        });

        source.addEventListener("provider", (event) => {
            const data = JSON.parse(event.data);
            if (data.status === "ready") {
                appendEventCard(
                    "Provider ready",
                    `${data.provider_id} answered with ${data.model}.`,
                    "pass",
                    data.response || data.message,
                );
                appendLog(`Provider check: ${data.provider_id} answered with ${data.model}.`);
                appendLog(`Sample response: ${data.response || data.message}`);
            } else {
                appendEventCard("Provider fallback", data.message, "neutral");
                appendLog(`Provider check: ${data.message}`);
            }
        });

        source.addEventListener("complete", async (event) => {
            const data = JSON.parse(event.data);
            completed = true;
            setStage("report");
            updateProgress(100, "Report ready");
            appendEventCard(
                "Report ready",
                `${data.run.run_id}: ${data.run.summary.failure_count} findings across ${data.run.summary.total_cases} cases.`,
                "pass",
                "Open the JSON, HTML, or Markdown report from Recent runs.",
            );
            appendLog(`Run ${data.run.run_id} completed.`);
            source.close();
            if (startButton) {
                startButton.disabled = false;
                startButton.textContent = "Start demo run";
            }
            if (guidedRunButton) {
                guidedRunButton.disabled = false;
                guidedRunButton.textContent = "Run guided demo";
            }
            await refreshStatus();
            await refreshRuns();
        });

        source.onerror = () => {
            if (completed) {
                return;
            }
            updateProgress(0, "Disconnected");
            appendEventCard("Stream disconnected", "The local event stream closed before the run completed.", "finding");
            appendLog("Stream disconnected.");
            source.close();
            if (startButton) {
                startButton.disabled = false;
                startButton.textContent = "Start demo run";
            }
            if (guidedRunButton) {
                guidedRunButton.disabled = false;
                guidedRunButton.textContent = "Run guided demo";
            }
        };
    }

    if (startButton) {
        startButton.addEventListener("click", () => {
            startStream();
        });
    }

    if (guidedRunButton) {
        guidedRunButton.addEventListener("click", () => {
            document.getElementById("stream")?.scrollIntoView({ behavior: "smooth", block: "start" });
            startStream();
        });
    }

    if (scenarioSelect) {
        scenarioSelect.addEventListener("change", updateScenarioSummary);
        updateScenarioSummary();
    }

    function clampNumber(value, min, max) {
        const numberValue = Number.parseInt(value, 10);
        if (Number.isNaN(numberValue)) {
            return String(min);
        }
        return String(Math.min(Math.max(numberValue, min), max));
    }

    refreshStatus().catch((error) => {
        appendEventCard("Status unavailable", error.message, "finding");
        appendLog(`Status refresh failed: ${error.message}`);
    });
    refreshRuns().catch((error) => {
        appendLog(`Runs refresh failed: ${error.message}`);
    });
})();
