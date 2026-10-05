from scripts.scan_untranslated import _actionable_indices


def test_isolated_suspicious_entries_do_not_block_upload():
    assert _actionable_indices([4, 18]) == []


def test_three_consecutive_suspicious_entries_block_upload():
    assert _actionable_indices([4, 10, 11, 12, 20]) == [10, 11, 12]
