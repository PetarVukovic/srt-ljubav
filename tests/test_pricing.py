import pytest

from srtljubav.pricing import (
    UsageSummary,
    estimate_usage_cost_usd,
    summarize_usage_records,
)


def test_estimate_usage_cost_for_gemini_flash_standard():
    cost = estimate_usage_cost_usd(
        model_name="gemini-flash-latest",
        prompt_tokens=1_000_000,
        output_tokens=1_000_000,
        pricing_mode="standard",
    )

    assert abs(cost - 2.80) < 1e-9


def test_estimate_usage_cost_for_gemini_3_flash_preview_standard():
    cost = estimate_usage_cost_usd(
        model_name="gemini-3-flash-preview",
        prompt_tokens=1_000_000,
        output_tokens=1_000_000,
        pricing_mode="standard",
    )

    assert abs(cost - 3.50) < 1e-9


def test_free_pricing_mode_is_zero():
    assert (
        estimate_usage_cost_usd(
            model_name="gemini-3-flash-preview",
            prompt_tokens=1_000_000,
            output_tokens=1_000_000,
            pricing_mode="free",
        )
        == 0.0
    )


def test_usage_summary_addition():
    a = UsageSummary(prompt_tokens=1, output_tokens=2, total_tokens=3, estimated_cost_usd=0.5)
    b = UsageSummary(prompt_tokens=10, output_tokens=20, total_tokens=30, estimated_cost_usd=1.5)

    total = a + b

    assert total == UsageSummary(
        prompt_tokens=11, output_tokens=22, total_tokens=33, estimated_cost_usd=2.0
    )


def test_flex_service_tier_halves_the_standard_price():
    standard = estimate_usage_cost_usd(
        model_name="gemini-3-flash-preview",
        prompt_tokens=1_000_000,
        output_tokens=1_000_000,
        pricing_mode="standard",
        service_tier="standard",
    )
    flex = estimate_usage_cost_usd(
        model_name="gemini-3-flash-preview",
        prompt_tokens=1_000_000,
        output_tokens=1_000_000,
        pricing_mode="standard",
        service_tier="flex",
    )

    assert abs(standard - 3.50) < 1e-9
    assert abs(flex - 1.75) < 1e-9


def test_priority_service_tier_costs_more_than_standard():
    priority = estimate_usage_cost_usd(
        model_name="gemini-3-flash-preview",
        prompt_tokens=1_000_000,
        output_tokens=0,
        pricing_mode="standard",
        service_tier="priority",
    )

    assert abs(priority - 0.90) < 1e-9


def test_unset_service_tier_bills_as_standard():
    assert estimate_usage_cost_usd(
        model_name="gemini-3-flash-preview",
        prompt_tokens=1_000_000,
        output_tokens=1_000_000,
        pricing_mode="standard",
        service_tier=None,
    ) == estimate_usage_cost_usd(
        model_name="gemini-3-flash-preview",
        prompt_tokens=1_000_000,
        output_tokens=1_000_000,
        pricing_mode="standard",
    )


def test_free_pricing_mode_stays_zero_on_flex():
    assert (
        estimate_usage_cost_usd(
            model_name="gemini-3-flash-preview",
            prompt_tokens=1_000_000,
            output_tokens=1_000_000,
            pricing_mode="free",
            service_tier="flex",
        )
        == 0.0
    )


def test_summarize_prefers_the_tier_recorded_on_each_call():
    records = [
        {
            "model": "gemini-3-flash-preview",
            "prompt_tokens": 1_000_000,
            "output_tokens": 0,
            "total_tokens": 1_000_000,
            "service_tier": "flex",
        },
        {
            "model": "gemini-3-flash-preview",
            "prompt_tokens": 1_000_000,
            "output_tokens": 0,
            "total_tokens": 1_000_000,
        },
    ]

    summary = summarize_usage_records(
        records, model_name="gemini-3-flash-preview", pricing_mode="standard"
    )

    assert abs(summary.estimated_cost_usd - (0.25 + 0.50)) < 1e-9


def test_gemini_3_8_flash_is_not_billed_at_the_legacy_preview_rate():
    """The 3.x Flash generation costs 0.75/3.75, not the preview's 0.50/3.00.

    Guards a silent 33 % undercount: ``gemini-3.8-flash`` used to fall through
    the ``"gemini-3" in model_key`` branch onto ``gemini-3-flash-preview``, so
    every CSV and every ``srt-ljubav report`` row understated the real bill.
    """
    cost = estimate_usage_cost_usd(
        "gemini-3.8-flash", prompt_tokens=1_000_000, output_tokens=1_000_000
    )
    assert cost == pytest.approx(0.75 + 3.75)


def test_every_named_gemini_3_flash_generation_shares_one_rate():
    for model in ("gemini-3.5-flash", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash"):
        cost = estimate_usage_cost_usd(
            model, prompt_tokens=1_000_000, output_tokens=1_000_000
        )
        assert cost == pytest.approx(0.75 + 3.75), model


def test_an_unknown_gemini_3_flash_never_falls_back_to_the_cheaper_preview():
    """An unrecognised 3.x Flash must not be priced below the generation it is in.

    Guessing low is the dangerous direction: it hides spend. A future
    ``gemini-3.9-flash`` should bill at the current generation's rate.
    """
    cost = estimate_usage_cost_usd(
        "gemini-3.9-flash", prompt_tokens=1_000_000, output_tokens=1_000_000
    )
    assert cost == pytest.approx(0.75 + 3.75)
