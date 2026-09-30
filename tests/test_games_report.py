"""Report 03 text helpers: the Kyle bridge paragraph must handle module D present or absent."""

from markout.games import kyle
from markout.games import report


def test_bridge_text_without_module_d():
    text = "\n".join(report.text_bridge(kyle.kyle_bridge({})))
    assert "needs module D" in text
    assert "β/√R²" in text          # says what the table will contain once D has run


def test_bridge_text_with_module_d():
    micro = {"kyle_bridge": {
        "INTC": {"beta_ticks_per_1000_shares": 0.02, "r2": 0.74, "mean_depth_best": 13600.0},
        "AAPL": {"beta_ticks_per_1000_shares": 4.1, "r2": 0.43, "mean_depth_best": 152.0}}}
    text = "\n".join(report.text_bridge(kyle.kyle_bridge(micro)))
    assert "| INTC |" in text and "| AAPL |" in text
    assert "0.02 (INTC) to 4.10 (AAPL)" in text
    assert "trade-based Kyle λ is not computed" in text


def test_tables_escape_pipes():
    out = report.table([{"E[V | buy]": "a | b"}])
    assert "E[V \\| buy]" in out and "a \\| b" in out
