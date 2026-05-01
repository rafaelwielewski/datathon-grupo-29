# OWASP Mapping - LLM Top 10 (2025)

## Resumo
Mapeamento de ameaças relevantes ao agente LLM do Datathon e mitigacoes aplicadas.

| OWASP LLM Top 10 | Risco no projeto | Mitigacao |
|---|---|---|
| LLM01: Prompt Injection | Usuario tenta alterar instrucao do agente | Input guardrail com padroes de injecao e limite de tamanho |
| LLM02: Insecure Output Handling | Resposta do LLM pode conter PII | Output guardrail com redacao de PII (email/telefone/CPF/CNPJ/CEP/cartao) |
| LLM04: Model Denial of Service | Entradas enormes sobrecarregam o LLM | Limite de tamanho no input + FastAPI schema |
| LLM05: Supply Chain | Dependencias e KB com conteudo inseguro | Dependencias declaradas no pyproject com versoes minimas, KB revisada |
| LLM08: Excessive Agency | Agente executa acoes nao autorizadas | Tools limitadas a leitura de dados e calculos locais |

## Evidencias
- Guardrails implementados em src/security/guardrails.py.
- Redacao de PII em src/security/pii_detection.py (email, telefone, CPF, CNPJ, CEP, cartao).
- API valida tamanho do input em src/serving/app.py.
