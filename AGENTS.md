# AGENTS.md

Guidance for AI coding agents (GitHub Copilot, Claude, Codex and similar) working in this repository. Read this before making changes.

## Project overview

A two-stage credit card fraud pipeline:

1. **Machine learning** scores each transaction and writes `fraud_risk_scores.csv` (fraud probability, 0-100 risk score, risk band, recommended action).
2. **Generative AI** (`fraud_llm.py`) reads that CSV and uses Llama 3.3 70B on Groq to add a plain-English `narration` and an `action_plan`, writing `fraud_risk_scores_llm.csv`.

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows PowerShell: .\venv\Scripts\Activate.ps1
pip install -r Requirements.txt
echo "GROQ_API_KEY=your_key_here" > .env
```

Python 3.10+ is recommended.

## Commands

| Task | Command |
|---|---|
| Add GenAI narratives to the scored CSV | `python fraud_llm.py` |
| ML environment and library setup | `python Fraud_Detection.py` |

`Fraud_Detection.py` currently contains the environment and imports for Stage 1. New modelling code (feature engineering, SMOTE, model benchmarking, SHAP, risk scoring) belongs in this file or in a new module next to it, and must write `fraud_risk_scores.csv` with the schema below.

## Data contract between the stages

`fraud_llm.py` stops if any of these columns are missing from `fraud_risk_scores.csv`:

`transaction_id, amount, is_international, velocity_24h, fraud_probability, risk_score, risk_band, recommended_action`

`is_fraud` is also present in the sample data. Keep column names stable; if you rename one, update the `required` set in `fraud_llm.py` in the same change.

## Conventions for the LLM stage

- Keep `response_format={"type": "json_object"}` and the two output keys `narration` (string, 2-3 sentences) and `action_plan` (list of 3-5 short steps).
- Keep `TEMPERATURE` low (currently 0.2) so outputs stay consistent and auditable.
- The system prompt tells the model to use only the values provided. Do not add instructions that invite the model to guess missing data.
- On failure the script retries with increasing delay and then writes a clear `[LLM error: ...]` value instead of crashing. Preserve this so one bad call does not lose a whole batch.
- Model name, temperature, token limit and retry count are constants at the top of `fraud_llm.py`; change them there, not inline.

## Security and data

- Never hard-code `GROQ_API_KEY` or commit `.env` (it is in `.gitignore`; use `.env.example` for placeholders).
- Do not add real cardholder or transaction data to this repository.

## Before you open a pull request

1. Run `python fraud_llm.py` on the sample CSV and check one High and one Low example in the printed output.
2. Confirm the output CSV still has the original columns plus `narration` and `action_plan`.
3. Keep changes small and describe the effect on the output in the PR description.
