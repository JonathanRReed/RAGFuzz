"""Grammar-based mutator for obfuscation attacks."""

from __future__ import annotations

from typing import Any

from ragfuzz.mutators.base import Mutator


class GrammarMutator(Mutator):
    """Mutator that applies grammatical obfuscations."""

    def __init__(
        self, name: str = "grammar", config: dict[str, Any] | None = None, rule: str | None = None
    ):
        """Initialize the grammar mutator.

        Args:
            name: Unique identifier for this mutator.
            config: Configuration including rule type.
            rule: Rule type (convenience alternative to config).
        """
        super().__init__(name, config)
        if rule is not None:
            self.rule = rule
        elif config:
            self.rule = config.get("rule", "homoglyph_and_whitespace")
        else:
            self.rule = "homoglyph_and_whitespace"

    async def mutate(self, input_text: str, _context: dict[str, Any] | None = None) -> str:
        """Apply a grammatical mutation to the input text.

        Args:
            input_text: The input text to mutate.

        Returns:
            The mutated text.
        """
        if self.rule == "homoglyph_and_whitespace":
            return self._apply_homoglyph_and_whitespace(input_text)
        elif self.rule == "markdown_nesting":
            return self._apply_markdown_nesting(input_text)
        elif self.rule == "json_fragmentation":
            return self._apply_json_fragmentation(input_text)
        else:
            return input_text

    def _apply_homoglyph_and_whitespace(self, text: str) -> str:
        """Apply homoglyph substitution and whitespace obfuscation."""
        homoglyphs = {
            "a": "а",
            "e": "е",
            "i": "і",
            "o": "о",
            "c": "с",
        }

        result = []
        for char in text:
            if char in homoglyphs:
                import random

                if random.random() < 0.3:
                    result.append(homoglyphs[char])
                else:
                    result.append(char)
            elif char == " ":
                import random

                if random.random() < 0.2:
                    result.append("  ")
                else:
                    result.append(char)
            else:
                result.append(char)

        return "".join(result)

    def _apply_markdown_nesting(self, text: str) -> str:
        """Apply nested markdown formatting."""
        prefixes = ["**", "__", "``"]
        import random

        prefix = random.choice(prefixes)
        return f"{prefix}{text}{prefix}"

    def _apply_json_fragmentation(self, text: str) -> str:
        """Apply JSON fragmentation obfuscation."""
        import json

        if text.startswith("{"):
            try:
                data = json.loads(text)
                return json.dumps(data, indent=2)
            except json.JSONDecodeError:
                pass

        return text
