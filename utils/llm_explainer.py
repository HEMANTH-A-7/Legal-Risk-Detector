import os
from typing import Optional

from openai import OpenAI


def _get_openai_client() -> Optional[OpenAI]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def explain_clause(sentence: str, risk_type: str, severity: str) -> str:
    """
    LLM-based plain-English explanation of a detected risk clause.

    Uses GPT-4o-mini via the /explain route. Falls back to
    template-based generate_explanation() in nlp_utils.py if unavailable.

    Args:
        sentence: The original contract clause text.
        risk_type: Detected risk category (Liability, Penalty, etc.)
        severity: HSE severity label (High, Medium, Low)

    Returns:
        Plain-English explanation string (1-2 sentences for non-lawyers).

    Raises:
        RuntimeError: If OPENAI_API_KEY is not set.
    """
    client = _get_openai_client()
    if client is None:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    system = (
        "You explain legal contract clauses to non-lawyers in plain English. "
        "Be concise and specific, avoid legal jargon, and do not add extra sections."
    )
    user = (
        "Explain the following sentence in 1-2 short sentences for a non-lawyer. "
        "Also briefly say why it is categorized as the given risk type and severity.\n\n"
        f"Risk type: {risk_type}\n"
        f"Severity: {severity}\n"
        f"Sentence: {sentence}"
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )

    content = (resp.choices[0].message.content or "").strip()
    return content or "Unable to generate an explanation for this clause."
