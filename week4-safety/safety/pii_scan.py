"""Redact PII before a retrieved chunk is placed in a prompt.

Presidio is used when it is installed. The built-in patterns cover the identifiers
this corpus actually contains: emails, phone numbers, and street addresses.
"""

import re

EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE = re.compile(r"(?:\+\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?)?\d{3,4}[\s-]?\d{3,4}")
STREET = re.compile(r"\b\d{1,5}\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\s(?:Street|St|Road|Rd|Avenue|Ave|Dock)\b")


def _builtin(text: str) -> tuple[str, list[dict]]:
    findings = []
    redacted = text
    for name, pattern in (("EMAIL_ADDRESS", EMAIL), ("PHONE_NUMBER", PHONE), ("STREET_ADDRESS", STREET)):
        for match in pattern.finditer(text):
            value = match.group(0)
            if name == "PHONE_NUMBER" and len(re.sub(r"\D", "", value)) < 8:
                continue
            findings.append({
                "entity_type": name,
                "start": match.start(),
                "end": match.end(),
                "score": 0.85,
                "text": value,
            })
        redacted = pattern.sub(f"<{name}>", redacted)
    return redacted, findings


def scan_and_redact(text: str, entities: list[str] | None = None) -> tuple[str, list[dict]]:
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
    except ImportError:
        redacted, findings = _builtin(text)
    else:
        analyzer = AnalyzerEngine()
        anonymizer = AnonymizerEngine()
        results = analyzer.analyze(text=text, language="en", entities=entities)
        anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
        findings = [
            {"entity_type": item.entity_type, "start": item.start, "end": item.end, "score": item.score}
            for item in results
        ]
        redacted = anonymized.text
    if entities:
        findings = [item for item in findings if item["entity_type"] in entities]
    return redacted, findings
