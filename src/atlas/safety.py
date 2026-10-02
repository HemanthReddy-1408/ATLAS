"""Defence against indirect prompt injection. Crawled pages are *untrusted data*; a page that tells the model what to
do must never reach it. Three layers: hidden-text removal at extraction, a pattern scanner that quarantines
instruction-like chunks at ingestion, and output sanitising (no exfiltration images / foreign links)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

QUARANTINE_AT = 0.75  # risk score at/above which a chunk is withheld from every index

# (regex, weight, label). weight 3 alone quarantines; two weight-2 signals quarantine.
_PATTERNS: list[tuple[re.Pattern[str], int, str]] = [
    (re.compile(r"\b(?:ignore|disregard|forget|override)\b[^.\n]{0,40}\b(?:previous|prior|above|earlier|all|any|your)\b[^.\n]{0,40}"
                r"\b(?:instructions?|prompts?|rules|guidelines|context)\b", re.I), 3, "instruction override"),
    (re.compile(r"\b(?:reveal|print|show|repeat|leak|output|disclose)\b[^.\n]{0,30}\b(?:system prompt|your instructions|"
                r"hidden instructions|api[- ]?keys?|secrets?|credentials?|passwords?)\b", re.I), 3, "prompt/secret exfiltration"),
    (re.compile(r"\b(?:note|message|instruction)s? (?:to|for) (?:the )?(?:ai|assistants?|language models?|llms?|chatbots?|agents?)\b", re.I),
     3, "addresses the AI directly"),
    (re.compile(r"\byou are now\b|\bact as (?:an? )?(?:unrestricted|jailbroken|different|evil)\b|\bnew (?:persona|role)\b", re.I), 2, "role hijack"),
    (re.compile(r"<\|?(?:im_start|im_end|system|assistant)\|?>|\[/?INST\]|^\s*(?:system|assistant)\s*:", re.I | re.M), 2, "chat-template markers"),
    (re.compile(r"\bdo not (?:tell|mention|reveal|inform|disclose)\b[^.\n]{0,40}\b(?:user|human|anyone)\b", re.I), 2, "concealment instruction"),
    (re.compile(r"\b(?:assistant|ai|model) (?:must|should|will) (?:now )?(?:say|answer|respond|reply|state|claim|output)\b|"
                r"\btell the user (?:that)?\b", re.I), 2, "directive to the model"),
    (re.compile(r"!\[[^\]]*\]\(https?://[^)]*\)", re.I), 2, "markdown image (exfiltration channel)"),
    (re.compile(r"\b(?:send|post|upload|forward|exfiltrate)\b[^.\n]{0,50}https?://", re.I), 2, "exfiltration URL"),
    (re.compile("[​-‏⁠‪-‮﻿]{3,}"), 2, "invisible characters"),
]


@dataclass
class Finding:
    label: str
    weight: int
    snippet: str


@dataclass
class ScanResult:
    score: float = 0.0
    findings: list[Finding] = field(default_factory=list)

    @property
    def quarantine(self) -> bool:
        return self.score >= QUARANTINE_AT

    @property
    def suspicious(self) -> bool:
        return self.score >= 0.5

    def notes(self) -> str:
        return "; ".join(f"{f.label} (“{f.snippet}”)" for f in self.findings)


def scan(text: str) -> ScanResult:
    findings, total = [], 0
    for rx, weight, label in _PATTERNS:
        m = rx.search(text)
        if m:
            findings.append(Finding(label, weight, re.sub(r"\s+", " ", m.group(0))[:70]))
            total += weight
    return ScanResult(min(1.0, total / 4), findings)


_IMG = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")


def sanitize_output(text: str, allowed_urls: set[str] | None = None) -> str:
    """Remove markdown images (zero-click exfiltration) and links to URLs that are not cited evidence."""
    text = _IMG.sub("", text)
    allowed = allowed_urls or set()
    return _LINK.sub(lambda m: m.group(0) if m.group(2) in allowed else m.group(1), text)
