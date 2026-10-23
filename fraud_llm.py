"""
fraud_llm_narration.py
----------------------
Reads `fraud_risk_scores.csv` (output of the ML model) and uses Groq's
llama-3.3-70b-versatile to add two columns:

    - narration   : 2-3 sentence plain-English explanation
    - action_plan : numbered next steps for the fraud ops team

Output: fraud_risk_scores_llm.csv

Setup
-----
1. Activate your venv:
       .\venv\Scripts\Activate.ps1
2. Install dependencies:
       pip install pandas groq python-dotenv tqdm
3. Create a .env file in the same folder as this script with:
       GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
4. Run:
       python fraud_llm_narration.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_CSV  = SCRIPT_DIR / "fraud_risk_scores.csv"
OUTPUT_CSV = SCRIPT_DIR / "fraud_risk_scores_llm.csv"

LLM_MODEL   = "llama-3.3-70b-versatile"
TEMPERATURE = 0.2
MAX_TOKENS  = 400
MAX_RETRIES = 2

SYSTEM_PROMPT = (
    "You are a senior fraud-risk analyst at a card-issuing bank. "
    "For each transaction, you receive the ML model output (risk score, band, "
    "recommended action) and the underlying features. "
    "You must produce a concise, professional explanation and a clear, "
    "numbered plan of action for the operations team. "
    "Be specific - reference the actual values (amount, velocity, "
    "international flag). Do not invent data that was not provided. "
    "Always respond with valid JSON only, no markdown fences, with exactly "
    "two keys: \"narration\" (string, 2-3 sentences) and \"action_plan\" "
    "(array of 3-5 short imperative steps)."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_api_key() -> None:
    """Load GROQ_API_KEY from a .env file next to this script."""
    load_dotenv(SCRIPT_DIR / ".env")
    if not os.getenv("GROQ_API_KEY"):
        sys.exit(
            "GROQ_API_KEY not set. Create a .env file next to this script "
            "with a line like:\n    GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx"
        )


def build_user_prompt(row: dict) -> str:
    """Format one transaction row for the LLM."""
    return (
        f"Transaction ID: {row['transaction_id']}\n"
        f"Amount: ${row['amount']:.2f}\n"
        f"International transaction: {'Yes' if row['is_international'] else 'No'}\n"
        f"Velocity (txns in last 24h): {row['velocity_24h']}\n"
        f"ML fraud probability: {row['fraud_probability']:.3f}\n"
        f"Risk score (0-100): {row['risk_score']}\n"
        f"Risk band: {row['risk_band']}\n"
        f"Recommended action (from model): {row['recommended_action']}\n\n"
        'Return JSON: {"narration": "...", "action_plan": ["...", "..."]}'
    )


def llm_explain(client: Groq, row: dict) -> dict:
    """Call Groq, parse JSON, return {narration, action_plan}."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": build_user_prompt(row)},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                response_format={"type": "json_object"},
            )
            parsed = json.loads(resp.choices[0].message.content)
            return {
                "narration":   str(parsed.get("narration", "")).strip(),
                "action_plan": parsed.get("action_plan", []),
            }
        except Exception as e:
            if attempt == MAX_RETRIES:
                return {"narration": f"[LLM error: {e}]", "action_plan": []}
            time.sleep(1.5 * (attempt + 1))


def format_plan(steps: list) -> str:
    """Turn a list of steps into '1. ... | 2. ... | 3. ...' for CSV storage."""
    return " | ".join(f"{i}. {s}" for i, s in enumerate(steps, 1))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    load_api_key()

    if not INPUT_CSV.exists():
        sys.exit(f"Input file not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows from {INPUT_CSV.name}")

    required = {
        "transaction_id", "amount", "is_international", "velocity_24h",
        "fraud_probability", "risk_score", "risk_band", "recommended_action",
    }
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"Input CSV is missing required columns: {sorted(missing)}")

    client = Groq()  # picks up GROQ_API_KEY from env

    narrations, action_plans = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="LLM enriching"):
        out = llm_explain(client, row.to_dict())
        narrations.append(out["narration"])
        action_plans.append(format_plan(out["action_plan"]))

    df["narration"]   = narrations
    df["action_plan"] = action_plans

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {OUTPUT_CSV.name}  ({df.shape[0]} rows, {df.shape[1]} cols)")

    # Quick sanity preview: one High and one Low
    print("\n--- Sample outputs ---")
    for band in ("High", "Low"):
        sample = df[df["risk_band"] == band].head(1)
        if sample.empty:
            continue
        r = sample.iloc[0]
        print(f"\n[{band}] {r['transaction_id']}  ${r['amount']:.2f}  "
              f"score={r['risk_score']}")
        print(f"Narration : {r['narration']}")
        print("Action plan:")
        for step in str(r["action_plan"]).split(" | "):
            print(f"  {step}")


if __name__ == "__main__":
    main()