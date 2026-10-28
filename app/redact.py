"""Privacy layer: strip personal data before text reaches the language model.

This is deliberately simple and rule-based. It catches the most common
identifiers in German business documents (emails, phone numbers, IBANs)
and replaces them with typed placeholders, so the model still understands
that *something* was there.
"""

import re
from dataclasses import dataclass, field

# Order matters: IBANs contain long digit runs that the phone pattern would
# otherwise grab first.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){3,7}(?:[ ]?[A-Z0-9]{1,3})?\b")),
    # Phone numbers start with +, 00 or 0 (German style). Dots are not allowed,
    # so dates like 14.10.2026 and amounts like 1.234.567 are left alone.
    ("PHONE", re.compile(r"(?<![\w.])(?:\+\d{1,3}|00\d{1,3}|0)[\d /()-]{6,}\d\b")),
]


@dataclass
class RedactionResult:
    text: str
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def redact(text: str) -> RedactionResult:
    counts: dict[str, int] = {}
    for label, pattern in _PATTERNS:
        text, n = pattern.subn(f"[{label}]", text)
        if n:
            counts[label] = counts.get(label, 0) + n
    return RedactionResult(text=text, counts=counts)
