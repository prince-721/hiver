import pandas as pd

from src.data.cleaner import clean_text, filter_pairs


def test_clean_text_strips_urls_and_handles():
    raw = "Hey @BrandSupport check https://example.com/abc please"
    cleaned = clean_text(raw)
    assert "http" not in cleaned
    assert "@BrandSupport" not in cleaned
    assert "check" in cleaned and "please" in cleaned


def test_filter_pairs_drops_too_short():
    df = pd.DataFrame({
        "customer_message": ["hi", "This is a normal length customer complaint about billing issues."],
        "support_response": ["ok this is a fine length response here", "short"],
    })
    out = filter_pairs(df, min_chars=8, max_chars=500)
    assert len(out) == 0  # both rows fail on one side or the other
    assert out.attrs["n_before"] == 2
