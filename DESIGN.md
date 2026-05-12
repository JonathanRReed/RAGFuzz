# RAGFuzz Design Context

## Design Register

Product UI. Design serves a technical workflow and should feel like a focused security cockpit, not a marketing page.

## Visual System

- Theme: dark, because users are running local technical reviews in an IDE or terminal-adjacent setting.
- Color strategy: restrained tinted neutrals with one warm risk accent for primary actions, active states, and findings.
- Radius: 8px or smaller for panels, controls, and cards.
- Typography: system sans for UI, system monospace for ids, stream events, and code-like report references.
- Layout: dense but readable. Prefer clear bands, split panels, tables, and inline status blocks over decorative card grids.

## Interaction Rules

- Every async action needs visible progress and recoverable failure copy.
- Streaming output should be structured into scannable event cards, with the raw log retained for accessibility and debugging.
- Onboarding must be optional, replayable, and attached to real controls.
- Demo mode must clearly explain its in-memory behavior without undermining real CLI artifact persistence.
- Reports and provider checks should always communicate whether data is real local state or sample demo evidence.

## Accessibility

- Keep touch targets at least 44px high.
- Preserve visible focus states.
- Do not rely on color alone for findings or status.
- Respect reduced motion.
- Avoid horizontal scroll on mobile except inside data tables.
