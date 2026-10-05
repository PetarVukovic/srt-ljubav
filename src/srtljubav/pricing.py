"""Gemini pricing table and cost estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# USD per 1M tokens. For free quota runs, pricing_mode="free" reports 0.
# NOTE: the 3.x Flash rates below are the promotional prices Google lists
# through 2026-12-31; on 2027-01-01 they double to 1.50/7.50. Re-check
# https://ai.google.dev/gemini-api/docs/pricing before pricing a 2027 run.
GEMINI_PRICING_USD_PER_1M: dict[str, dict[str, float]] = {
    "gemini-3-flash-preview": {"input": 0.50, "output": 3.00},
    "gemini-3.5-flash": {"input": 0.75, "output": 3.75},
    "gemini-3.6-flash": {"input": 0.75, "output": 3.75},
    "gemini-3.7-flash": {"input": 0.75, "output": 3.75},
    "gemini-3.8-flash": {"input": 0.75, "output": 3.75},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-flash-latest": {"input": 0.30, "output": 2.50},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
}

# What an unrecognised model in the 3.x Flash generation costs. Guessing the
# legacy preview's cheaper 0.50/3.00 hid a third of the bill on every
# gemini-3.5/3.6/3.7/3.8 call, so an unknown sibling is billed at its
# generation's rate instead -- erring high keeps the report honest.
GEMINI_3_FLASH_FALLBACK: dict[str, float] = {"input": 0.75, "output": 3.75}

# Google bills flex and batch at half the standard rate and priority at 1.8x.
# The ratio is identical for every Flash model, so one multiplier keeps the
# price table above single-sourced instead of four near-copies of it.
SERVICE_TIER_MULTIPLIER: dict[str, float] = {
    "standard": 1.0,
    "flex": 0.5,
    "batch": 0.5,
    "priority": 1.8,
}


def service_tier_multiplier(service_tier: str | None) -> float:
    return SERVICE_TIER_MULTIPLIER.get((service_tier or "standard").lower(), 1.0)


@dataclass(frozen=True)
class UsageSummary:
    prompt_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0

    def __add__(self, other: UsageSummary) -> UsageSummary:
        return UsageSummary(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            estimated_cost_usd=self.estimated_cost_usd + other.estimated_cost_usd,
        )


def estimate_usage_cost_usd(
    model_name: str,
    prompt_tokens: int,
    output_tokens: int,
    pricing_mode: str = "standard",
    service_tier: str | None = None,
) -> float:
    if pricing_mode == "free":
        return 0.0

    model_key = model_name.replace("models/", "")
    pricing = GEMINI_PRICING_USD_PER_1M.get(model_key)
    if pricing is None and "flash-lite" in model_key:
        pricing = GEMINI_PRICING_USD_PER_1M["gemini-2.5-flash-lite"]
    elif pricing is None and "gemini-3" in model_key and "flash" in model_key:
        pricing = GEMINI_3_FLASH_FALLBACK
    elif pricing is None and "flash" in model_key:
        pricing = GEMINI_PRICING_USD_PER_1M["gemini-2.5-flash"]
    elif pricing is None:
        return 0.0

    multiplier = service_tier_multiplier(service_tier)
    return multiplier * (
        (prompt_tokens / 1_000_000 * pricing["input"])
        + (output_tokens / 1_000_000 * pricing["output"])
    )


def summarize_usage_records(
    records: list[dict[str, Any]],
    model_name: str,
    pricing_mode: str,
    service_tier: str | None = None,
) -> UsageSummary:
    prompt_tokens = sum(int(record.get("prompt_tokens") or 0) for record in records)
    output_tokens = sum(int(record.get("output_tokens") or 0) for record in records)
    total_tokens = sum(int(record.get("total_tokens") or 0) for record in records)
    estimated_cost = sum(
        estimate_usage_cost_usd(
            model_name=str(record.get("model") or model_name),
            prompt_tokens=int(record.get("prompt_tokens") or 0),
            output_tokens=int(record.get("output_tokens") or 0),
            pricing_mode=pricing_mode,
            # Historical runs predate the tier field and were billed as standard.
            service_tier=record.get("service_tier") or service_tier,
        )
        for record in records
    )
    return UsageSummary(
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        estimated_cost_usd=estimated_cost,
    )


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
