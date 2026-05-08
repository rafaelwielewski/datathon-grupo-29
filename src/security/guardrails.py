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
        # English — role/instruction override
        r'ignore\s+(all\s+)?previous\s+instructions',
        r'disregard\s+(all\s+)?previous\s+instructions',
        r'you\s+are\s+now\s+a',
        r'act\s+as\s+(?:a\s+|an\s+)(?!aviation)',
        r'pretend\s+(you\s+are|to\s+be)',
        r'forget\s+(everything|all|your\s+instructions)',
        r'override\s+(your\s+)?(instructions|rules|guidelines)',
        r'new\s+instructions?\s*:',
        r'your\s+(real|true|actual)\s+(purpose|goal|task)\s+is',
        # English — prompt delimiters / jailbreak tokens
        r'<\|im_start\|>',
        r'<\|im_end\|>',
        r'\[INST\]',
        r'<</SYS>>',
        r'###\s*(instruction|system|human|assistant)\s*:',
        r'system\s*:',
        r'developer\s+message:\s*',
        r'tool\s+call:\s*',
        r'<tool_call>',
        r'<function_calls>',
        # English — data exfiltration / scope escape
        r'reveal\s+(your\s+)?(system\s+prompt|instructions|training)',
        r'print\s+(your\s+)?(system\s+prompt|instructions)',
        r'what\s+(are|were)\s+your\s+(original\s+)?instructions',
        r'repeat\s+(everything|all)\s+(above|before)',
        # Portuguese — role/instruction override
        r'ignore\s+(todas\s+as\s+)?instru[çc][õo]es\s+anteriores',
        r'esque[çc]a\s+(tudo|todas\s+as\s+instru[çc][õo]es)',
        r'voc[êe]\s+(agora\s+[ée]|[ée]\s+agora)',
        r'finja\s+(ser|que\s+[ée])',
        r'aja\s+como\s+(um|uma)\s+(?!assistente\s+de\s+avia)',
        r'novas?\s+instru[çc][õo]es?\s*:',
        r'seu\s+(verdadeiro|real)\s+(objetivo|prop[oó]sito|papel)\s+[ée]',
        r'desconsidere\s+(as\s+)?(instru[çc][õo]es|regras)',
        r'substitua\s+(as\s+)?(instru[çc][õo]es|regras)',
        # Portuguese — data exfiltration
        r'revele?\s+(as\s+)?(instru[çc][õo]es|prompt\s+do\s+sistema)',
        r'mostre?\s+(as\s+)?(instru[çc][õo]es|prompt\s+do\s+sistema)',
        r'repita\s+(tudo|todas)\s+(acima|anterior)',
        r'qual\s+[ée]\s+o\s+seu\s+prompt\s+de\s+sistema',
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
