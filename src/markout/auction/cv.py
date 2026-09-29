"""Walk-forward splits by day, and the sealed holdout.

Design choices, and why:
- Split by day, never by row: every stock on a day shares the market's moves, so a
  random row split puts near-duplicate information on both sides.
- Expanding-window walk-forward: every test day is later than every training day,
  as in live trading. A first "calibration" block produces out-of-sample
  predictions that are used only to calibrate the forecasts of the next block.
- Purging is unnecessary here: each label spans 60 s inside a single day, so no
  label straddles a train/test boundary.
- An embargo exists to stop *training* rows that come after a test block from
  seeing it through lookback features, which happens in k-fold CV. Walk-forward
  has no such rows, so the 1-day embargo kept here is only a cheap guard.
- Days 421-480 are the holdout: scored once, by `evaluation.holdout.HoldoutGuard`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Fold:
    k: int
    train: tuple[int, int]  # inclusive day range
    test: tuple[int, int]
    role: str  # "calibration" or "evaluation"


@dataclass(frozen=True)
class Design:
    n_days: int = 481
    research_end: int = 420
    first_eval: int = 181
    n_eval: int = 4
    calib_len: int = 60
    embargo: int = 1
    weight_days: tuple[int, int] = (0, 120)  # index weights estimated before any test block

    @classmethod
    def scaled(cls, n_days: int, embargo: int = 1) -> "Design":
        """Same proportions as the real 481-day design, for smaller (synthetic) data."""
        f = (n_days - 1) / 480
        research_end = round(420 * f)
        first_eval = round(181 * f)
        n_eval = 4
        block = (research_end - first_eval + 1) // n_eval
        research_end = first_eval + n_eval * block - 1
        calib_start = first_eval - block
        return cls(n_days=n_days, research_end=research_end, first_eval=first_eval, n_eval=n_eval,
                   calib_len=block, embargo=embargo, weight_days=(0, calib_start - 1))

    @property
    def block(self) -> int:
        return (self.research_end - self.first_eval + 1) // self.n_eval

    @property
    def eval_days(self) -> tuple[int, int]:
        return (self.first_eval, self.research_end)

    @property
    def holdout(self) -> tuple[int, int]:
        return (self.research_end + 1, self.n_days - 1)

    def folds(self) -> list[Fold]:
        starts = [self.first_eval - self.calib_len] + [self.first_eval + i * self.block for i in range(self.n_eval)]
        lengths = [self.calib_len] + [self.block] * self.n_eval
        out = []
        for k, (s, n) in enumerate(zip(starts, lengths)):
            out.append(Fold(k=k, train=(0, s - 1 - self.embargo), test=(s, s + n - 1),
                            role="calibration" if k == 0 else "evaluation"))
        return out

    def final_fold(self) -> Fold:
        """Train on the whole research period, test on the holdout."""
        lo, hi = self.holdout
        return Fold(k=len(self.folds()), train=(0, lo - 1 - self.embargo), test=(lo, hi), role="holdout")
