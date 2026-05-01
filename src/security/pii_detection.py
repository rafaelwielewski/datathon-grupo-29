from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class PiiMatch:
    label: str
    value: str


_EMAIL_RE = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
_PHONE_RE = re.compile(r'\b(?:\+?55\s?)?(?:\(?\d{2}\)?\s?)?\d{4,5}-?\d{4}\b')
_CPF_RE = re.compile(r'\b\d{3}\.\d{3}\.\d{3}-\d{2}\b|\b\d{11}\b')
_CNPJ_RE = re.compile(r'\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b|\b\d{14}\b')
_CEP_RE = re.compile(r'\b\d{5}-\d{3}\b')
_CREDIT_CARD_RE = re.compile(r'\b(?:\d{4}[- ]?){3}\d{4}\b')


def find_pii(text: str) -> list[PiiMatch]:
    matches: list[PiiMatch] = []
    matches.extend([PiiMatch('EMAIL', m.group(0)) for m in _EMAIL_RE.finditer(text)])
    matches.extend([PiiMatch('PHONE', m.group(0)) for m in _PHONE_RE.finditer(text)])
    matches.extend([PiiMatch('CPF', m.group(0)) for m in _CPF_RE.finditer(text)])
    matches.extend([PiiMatch('CNPJ', m.group(0)) for m in _CNPJ_RE.finditer(text)])
    matches.extend([PiiMatch('CEP', m.group(0)) for m in _CEP_RE.finditer(text)])
    matches.extend([PiiMatch('CREDIT_CARD', m.group(0)) for m in _CREDIT_CARD_RE.finditer(text)])
    return matches


def redact_pii(text: str) -> str:
    redacted = _EMAIL_RE.sub('[REDACTED_EMAIL]', text)
    redacted = _PHONE_RE.sub('[REDACTED_PHONE]', redacted)
    redacted = _CPF_RE.sub('[REDACTED_CPF]', redacted)
    redacted = _CNPJ_RE.sub('[REDACTED_CNPJ]', redacted)
    redacted = _CEP_RE.sub('[REDACTED_CEP]', redacted)
    redacted = _CREDIT_CARD_RE.sub('[REDACTED_CC]', redacted)
    return redacted
