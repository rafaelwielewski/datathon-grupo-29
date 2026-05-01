from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from src.security.pii_detection import redact_pii

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GuardrailResult:
    is_valid: bool
    reason: str


class InputGuardrail:
    """Validate and sanitize user input before LLM use."""

    INJECTION_PATTERNS = [
        r'ignore\s+(all\s+)?previous\s+instructions',
        r'you\s+are\s+now\s+a',
        r'system:\s*',
        r'<\|im_start\|>',
        r'\[INST\]',
        r'forget\s+(everything|all|your\s+instructions)',
        r'developer\s+message:\s*',
        r'tool\s+call:\s*',
    ]

    def __init__(self, max_length: int = 1000):
        self.max_length = max_length
        self._compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS]

    def validate(self, user_input: str) -> GuardrailResult:
        if not user_input.strip():
            return GuardrailResult(False, 'Input rejected: empty question.')

        if len(user_input) > self.max_length:
            return GuardrailResult(False, f'Input rejected: exceeds max length ({self.max_length}).')

        for pattern in self._compiled_patterns:
            if pattern.search(user_input):
                logger.warning(
                    'Prompt injection detected; matched_pattern=%s input_length=%d',
                    pattern.pattern,
                    len(user_input),
                )
                return GuardrailResult(False, 'Input blocked: suspicious pattern detected.')

        return GuardrailResult(True, 'OK')


class OutputGuardrail:
    """Redact PII from model outputs before returning to users."""

    def sanitize(self, llm_output: str) -> str:
        sanitized = redact_pii(llm_output)
        if sanitized != llm_output:
            logger.warning('PII detected in model output; redaction applied.')
        return sanitized
