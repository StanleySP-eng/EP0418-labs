"""
ep0418 — verification checks for EP0418 lab stubs.

    from ep0418 import check

    check.no_sentinels(df, cols=["humidity_pct"], sentinel=-999)
    check.rows_accounted_for(raw, clean, grid, explain="...")
    check.report()

Design notes for tutors
-----------------------
* Nothing here raises. A failed check is recorded and explained; the notebook
  keeps running. Beginners who hit a traceback stop working, so failures are
  reported as prose with a suggested next step instead.
* Three outcomes, not two. PASS and FAIL behave as expected. NOTE marks a
  legitimate-but-noteworthy result — most importantly a model that loses to its
  baseline, which the module treats as a valid finding to report honestly rather
  than as a failure to hide.
* Re-running a cell replaces that check's earlier result rather than appending a
  duplicate, so the report stays truthful in a notebook worked out of order.
* pandas and numpy only. No install step in Colab.

Every check maps to a syllabus outcome; see the SLO tag in each docstring.
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd

__all__ = ["check", "backtest", "show", "league"]

PASS, FAIL, NOTE = "PASS", "FAIL", "NOTE"

_PLACEHOLDERS = {
    "", "-", "n/a", "na", "none", "nil", "todo", "tbc", "asdf", "test",
    "explain", "because", "reason", "idk", "no reason", "nothing",
}


def _times(obj, col=None):
    """Pull a DatetimeIndex out of a DataFrame/Series however it is stored."""
    if col is not None:
        if not isinstance(obj, pd.DataFrame):
            raise TypeError("col= was given but the object is not a DataFrame")
        if col not in obj.columns:
            return None
        return pd.DatetimeIndex(pd.to_datetime(obj[col], errors="coerce"))
    if isinstance(obj.index, pd.DatetimeIndex):
        return obj.index
    if isinstance(obj, pd.DataFrame):
        for c in obj.columns:
            if pd.api.types.is_datetime64_any_dtype(obj[c]):
                return pd.DatetimeIndex(obj[c])
    return None


def _n(x):
    """Row count for a DataFrame, Series or plain sequence."""
    try:
        return len(x)
    except TypeError:
        return 0


def _unfilled(x):
    """True when a stub placeholder (`...`) has not been replaced yet."""
    return x is Ellipsis or x is None


def _safe(fn):
    """
    Two guards wrapped around every check.

    1. If the student has not replaced the stub's `...` yet, say so plainly
       instead of raising AttributeError on an ellipsis object.
    2. If anything else goes wrong inside a check, record it as a failure rather
       than raising. A beginner who hits a traceback stops working; a beginner
       who reads "this check could not run because X" keeps going.
    """
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        name = fn.__name__
        if any(_unfilled(a) for a in args):
            return self._record(
                f"{name}::unfilled", FAIL, name,
                "Not there yet: nothing has made this value.",
                "Run Gemini's code first: click Accept and run on its green cell (or paste its "
                "code into a new cell and run it), then run this cell again. If you did, Gemini "
                "used a different name: type in the chat 'call the result' and the name this cell uses.")
        try:
            return fn(self, *args, **kwargs)
        except Exception as exc:                      # noqa: BLE001 — deliberate net
            return self._record(
                f"{name}::error", FAIL, name,
                f"This check could not run: {type(exc).__name__}: {exc}",
                "That usually means the value passed in is not the shape the check "
                "expected — print it and look at its type and columns before "
                "continuing.")
    return wrapper


class _Check:
    def __init__(self):
        self._results = {}
        self._reported = False

    # ---------------------------------------------------------------- internals
    def _record(self, key, status, label, detail, fix=None):
        if self._reported:                      # a new run started after report()
            self._results.clear()
            self._reported = False
        self._results[key] = (status, label, detail, fix)
        return status == PASS

    def reset(self):
        """Clear all recorded results. Rarely needed — report() resets for you."""
        self._results.clear()
        self._reported = False

    # ---------------------------------------------------------------- structure
    @_safe
    def columns_present(self, df, cols):
        """The columns you expect actually exist. (SLO 2.1)"""
        cols = list(cols)
        missing = [c for c in cols if c not in df.columns]
        key = f"columns_present::{','.join(cols)}"
        if missing:
            return self._record(
                key, FAIL, f"columns_present({', '.join(cols)})",
                f"Missing from the DataFrame: {missing}. "
                f"Present columns are: {list(df.columns)}.",
                "Check spelling and capitalisation — 'Close' and 'close' are different "
                "columns. Print df.columns to see exactly what you have.")
        return self._record(key, PASS, f"columns_present({len(cols)} cols)",
                            "All expected columns present.")

    @_safe
    def rows_at_least(self, df, minimum):
        """Enough data to be worth analysing. (SLO 1.1)"""
        n = _n(df)
        key = "rows_at_least"
        if n < minimum:
            return self._record(
                key, FAIL, f"rows_at_least({minimum})",
                f"Only {n:,} rows — fewer than the {minimum:,} this analysis needs.",
                "Did a filter or dropna() remove more than you intended? Check the row "
                "count immediately after each step to find where they went.")
        return self._record(key, PASS, f"rows_at_least({minimum})", f"{n:,} rows.")

    # ---------------------------------------------------------------- data quality
    @_safe
    def no_sentinels(self, df, cols, sentinel=-999):
        """No 'sensor did not respond' codes left masquerading as real numbers. (SLO 4.1)"""
        cols = [cols] if isinstance(cols, str) else list(cols)
        hits = {}
        for c in cols:
            if c in df.columns:
                n = int((df[c] == sentinel).sum())
                if n:
                    hits[c] = n
        key = f"no_sentinels::{','.join(cols)}::{sentinel}"
        if hits:
            worst = ", ".join(f"{c}: {n}" for c, n in hits.items())
            return self._record(
                key, FAIL, f"no_sentinels({sentinel})",
                f"Found values equal to {sentinel} — {worst}.",
                f"{sentinel} is the logger's way of writing 'no reading'. pandas counts "
                f"it as a real number, so it drags your averages down and isna() will "
                f"never find it. Replace it first:\n"
                f"        df.loc[df[col] == {sentinel}, col] = np.nan")
        return self._record(key, PASS, f"no_sentinels({sentinel})",
                            f"No {sentinel} values in {len(cols)} column(s).")

    @_safe
    def no_missing(self, df, cols=None):
        """No blanks left in the columns you are about to analyse. (SLO 4.2)"""
        sub = df if cols is None else df[[c for c in
                                          ([cols] if isinstance(cols, str) else cols)
                                          if c in df.columns]]
        counts = sub.isna().sum()
        bad = counts[counts > 0]
        key = f"no_missing::{'all' if cols is None else str(cols)}"
        if len(bad):
            return self._record(
                key, FAIL, "no_missing",
                "Blank values remain — " + ", ".join(f"{c}: {int(v)}" for c, v in bad.items()) + ".",
                "Decide and justify: fill them (ffill, interpolate), drop those rows, or "
                "keep them as NaN and use methods that skip blanks. Any of the three can "
                "be right — say which you chose and why.")
        return self._record(key, PASS, "no_missing", "No blank values.")

    @_safe
    def values_in_range(self, df, col, low, high):
        """Values are physically possible for this quantity. (SLO 4.1)"""
        key = f"values_in_range::{col}"
        if col not in df.columns:
            return self._record(key, FAIL, f"values_in_range({col})",
                                f"Column '{col}' not found.", "Check the column name.")
        s = pd.to_numeric(df[col], errors="coerce")
        bad = int(((s < low) | (s > high)).sum())
        if bad:
            return self._record(
                key, FAIL, f"values_in_range({col})",
                f"{bad} value(s) outside the plausible range {low} to {high} "
                f"(actual range {s.min():.4g} to {s.max():.4g}).",
                "Either these are data errors you should handle, or your range is wrong "
                "for this quantity. Look at the offending rows before deciding — an "
                "impossible value is often the most interesting row in the file.")
        return self._record(key, PASS, f"values_in_range({col})",
                            f"All values within {low} to {high}.")

    @_safe
    def zeros_are_real(self, df, col, max_zeros=0):
        """Zeros mean 'zero', not 'device was off'. (SLO 4.2)"""
        key = f"zeros_are_real::{col}"
        if col not in df.columns:
            return self._record(key, FAIL, f"zeros_are_real({col})",
                                f"Column '{col}' not found.", "Check the column name.")
        n = int((pd.to_numeric(df[col], errors="coerce") == 0).sum())
        if n > max_zeros:
            return self._record(
                key, NOTE, f"zeros_are_real({col})",
                f"{n} row(s) have {col} exactly 0 (you allowed {max_zeros}).",
                "A zero from a device that was switched off is not the same fact as a "
                "genuine zero, but pandas averages them identically. If these are "
                "'not recorded', convert them to NaN before you compute anything:\n"
                f"        df.loc[df['{col}'] == 0, '{col}'] = np.nan")
        return self._record(key, PASS, f"zeros_are_real({col})",
                            f"{n} zero value(s), within the {max_zeros} you allowed.")

    # ---------------------------------------------------------------- time series
    @_safe
    def no_duplicate_timestamps(self, df, col=None):
        """Each moment in time appears once. (SLO 4.1)"""
        key = f"no_duplicate_timestamps::{col}"
        t = _times(df, col)
        if t is None:
            return self._record(key, FAIL, "no_duplicate_timestamps",
                                "Could not find a datetime column or index.",
                                "Parse your time column first: "
                                "pd.to_datetime(df['timestamp'])")
        n = int(t.duplicated().sum())
        if n:
            return self._record(
                key, FAIL, "no_duplicate_timestamps",
                f"{n} duplicated timestamp(s).",
                "Loggers repeat rows when they retry. Decide whether to keep the first, "
                "the last, or average them:\n"
                "        df = df.drop_duplicates(subset='timestamp', keep='first')")
        return self._record(key, PASS, "no_duplicate_timestamps",
                            f"All {len(t):,} timestamps unique.")

    @_safe
    def sorted_in_time(self, df, col=None):
        """Rows run forwards in time. (SLO 3.3)"""
        key = f"sorted_in_time::{col}"
        t = _times(df, col)
        if t is None:
            return self._record(key, FAIL, "sorted_in_time",
                                "Could not find a datetime column or index.",
                                "Parse your time column first.")
        if not t.is_monotonic_increasing:
            return self._record(
                key, FAIL, "sorted_in_time", "Timestamps are not in increasing order.",
                "Anything using .shift(), .diff() or .rolling() is meaningless on "
                "unsorted rows. Sort before you analyse:\n"
                "        df = df.sort_values('timestamp')")
        return self._record(key, PASS, "sorted_in_time", "Timestamps increase throughout.")

    @_safe
    def regular_cadence(self, obj, freq, col=None):
        """Rows sit on an evenly spaced time grid. (SLO 3.3)"""
        key = f"regular_cadence::{freq}"
        t = _times(obj, col)
        if t is None:
            return self._record(key, FAIL, f"regular_cadence({freq})",
                                "Could not find a datetime column or index.",
                                "Set the time column as the index before resampling: "
                                "df.set_index('timestamp')")
        if len(t) < 3:
            return self._record(key, FAIL, f"regular_cadence({freq})",
                                f"Only {len(t)} rows — too few to have a cadence.")
        # accept both "1h" and the bare pandas aliases students meet in
        # resample(): "h", "D", "5min", "W"
        want = pd.Timedelta(pd.tseries.frequencies.to_offset(freq))
        gaps = t.to_series().diff().dropna()
        odd = gaps[gaps != want]
        if len(odd):
            common = odd.value_counts().head(3)
            shown = ", ".join(f"{str(k)} x{v}" for k, v in common.items())
            # every odd gap a whole number of steps: the grid was made and then
            # the empty slots were dropped (usually a .dropna() the student did
            # not ask for), which is a different fix from never gridding at all
            multiples = all((g % want) == pd.Timedelta(0) for g in odd)
            if multiples:
                return self._record(
                    key, FAIL, f"regular_cadence({freq})",
                    f"{len(odd)} gap(s) are a whole number of {freq} steps, e.g. {shown}. "
                    f"The grid was regular and then the empty slots were removed.",
                    "Look for a .dropna() (or a filter) after the resample and take it out. "
                    "Empty slots must stay in the data as NaN so that gaps are visible:\n"
                    f"        s = df.set_index('timestamp')['value'].resample('{freq}').mean()"
                    "   # no .dropna()")
            return self._record(
                key, FAIL, f"regular_cadence({freq})",
                f"{len(odd)} gap(s) are not exactly {freq}. Most common: {shown}.",
                "Real feeds arrive at irregular moments, so .shift(168) means '168 rows "
                "ago', not 'one week ago'. Put the data on a fixed grid first:\n"
                f"        s = df.set_index('timestamp')['value'].resample('{freq}').mean()")
        return self._record(key, PASS, f"regular_cadence({freq})",
                            f"All {len(gaps):,} gaps are exactly {freq}.")

    @_safe
    def rows_accounted_for(self, *stages, explain=None, names=None):
        """
        Reconcile row counts across your cleaning steps. (SLO 4.2, 9.2)

        This is the keystone check. Losing rows is normal; losing rows you cannot
        account for is how a silent bug reaches your results. If any stage changes
        the row count you must say why, in your own words.
        """
        key = "rows_accounted_for"
        if len(stages) < 2:
            return self._record(key, FAIL, "rows_accounted_for",
                                "Pass at least two stages, e.g. (raw, clean).")
        counts = [_n(s) for s in stages]
        labels = list(names) if names else [f"stage {i}" for i in range(len(stages))]
        if len(labels) != len(counts):
            labels = [f"stage {i}" for i in range(len(counts))]

        steps, changed = [], False
        for i in range(1, len(counts)):
            d = counts[i] - counts[i - 1]
            if d:
                changed = True
            steps.append(f"{labels[i-1]} {counts[i-1]:,} -> {labels[i]} {counts[i]:,} "
                         f"({d:+,})")
        trail = "; ".join(steps)

        if not changed:
            return self._record(key, PASS, "rows_accounted_for",
                                f"Row count unchanged at {counts[0]:,}.")

        clean_expl = (explain or "").strip()
        low = clean_expl.lower()
        if (len(clean_expl) < 25 or low in _PLACEHOLDERS
                or low.startswith("todo") or "todo(you)" in low):
            return self._record(
                key, FAIL, "rows_accounted_for",
                f"Row count changed and you have not explained it. {trail}.",
                "Every row that disappeared, disappeared for a reason you chose. Write "
                "it down — that sentence is assessed evidence of verification:\n"
                "        check.rows_accounted_for(raw, clean, grid,\n"
                "            explain='dropped 140 duplicate timestamps; resampling "
                "created 66 empty slots across the 11 June outage')")
        return self._record(key, PASS, "rows_accounted_for",
                            f"{trail}. Explained: \"{clean_expl}\"")

    # ---------------------------------------------------------------- modelling
    @_safe
    def compare_to_baseline(self, model_score, baseline_score, metric="MAE",
                            lower_is_better=True):
        """
        Score your model against the baseline honestly. (SLO 6.2, 6.4)

        Losing to the baseline is NOT a failed check — it is a legitimate result
        that the module asks you to report. It is recorded as a NOTE so it stays
        visible in your report rather than quietly disappearing.
        """
        key = "compare_to_baseline"
        try:
            m, b = float(model_score), float(baseline_score)
        except (TypeError, ValueError):
            return self._record(key, FAIL, "compare_to_baseline",
                                "Scores are not numbers.",
                                "Compute both scores before comparing them.")
        if not (np.isfinite(m) and np.isfinite(b)):
            return self._record(
                key, FAIL, "compare_to_baseline",
                f"A score is NaN or infinite (model {m}, baseline {b}).",
                "This almost always means the two series did not overlap. Compare only "
                "rows where BOTH the actual value and the prediction exist:\n"
                "        m = actual.notna() & pred.notna()")
        better = m < b if lower_is_better else m > b
        gap = abs(m - b) / abs(b) * 100 if b else float("nan")
        if better:
            return self._record(key, PASS, "compare_to_baseline",
                                f"Model {metric} {m:,.4g} beats baseline {b:,.4g} "
                                f"({gap:.1f}% better).")
        return self._record(
            key, NOTE, "compare_to_baseline",
            f"Model {metric} {m:,.4g} does NOT beat baseline {b:,.4g} "
            f"({gap:.1f}% worse).",
            "This is a real result, not a mistake, and you are not penalised for it. "
            "Report it plainly and say what you think the baseline is capturing that "
            "your model is not. A sophisticated model that loses to a naive one is "
            "evidence about the problem, and hiding it is the only wrong move here.")

    @_safe
    def same_horizon(self, model_horizon, baseline_horizon, unit=""):
        """Model and baseline forecast the same distance ahead. (SLO 6.2)"""
        key = "same_horizon"
        if model_horizon != baseline_horizon:
            return self._record(
                key, FAIL, "same_horizon",
                f"Model forecasts {model_horizon}{unit} ahead but the baseline was "
                f"scored at {baseline_horizon}{unit}.",
                "This comparison is not valid — baselines get easier or harder with "
                "horizon, so you can appear to win purely by scoring them differently. "
                "Score both at the horizon your system actually uses.")
        return self._record(key, PASS, "same_horizon",
                            f"Both scored at {model_horizon}{unit} ahead.")

    @_safe
    def signal_is_shifted(self, raw_signal, used_signal, by=1):
        """Your trading/decision signal does not use information from the future."""
        key = "signal_is_shifted"
        try:
            raw = pd.Series(raw_signal).reset_index(drop=True)
            used = pd.Series(used_signal).reset_index(drop=True)
        except Exception:
            return self._record(key, FAIL, "signal_is_shifted",
                                "Could not read the two signals as Series.")
        expected = raw.shift(by)
        both = expected.notna() & used.notna()
        if not both.any():
            return self._record(key, FAIL, "signal_is_shifted",
                                "No overlapping rows to compare.")
        mismatch = int((expected[both] != used[both]).sum())
        if mismatch:
            return self._record(
                key, FAIL, "signal_is_shifted",
                f"{mismatch} row(s) differ from the signal shifted by {by}.",
                "If you compute a signal from today's close and act on today's close, "
                "you used information you would not have had — lookahead bias, and the "
                "commonest way a student backtest produces a fake return. Shift it:\n"
                f"        used = raw.shift({by})")
        return self._record(key, PASS, "signal_is_shifted",
                            f"Signal is shifted by {by}; no future information used.")

    @_safe
    def flagged_at_least(self, mask, minimum, what="readings"):
        """A detector flagged at least `minimum` rows. (SLO 5.1)"""
        key = f"flagged_at_least::{what}"
        try:
            n = int(pd.Series(mask).fillna(False).astype(bool).sum())
        except Exception:
            return self._record(key, FAIL, f"flagged_at_least({what})",
                                "Could not read the detector output as True/False values.",
                                "The detector should produce one True/False per row.")
        if n < minimum:
            return self._record(
                key, FAIL, f"flagged_at_least({what})",
                f"Flagged {n} {what}; at least {minimum} expected.",
                "Check which column the detector looks at and how long its window is. "
                "A stuck sensor shows up as a rolling standard deviation of exactly "
                "zero on the channel that is stuck, not as an outlier.")
        return self._record(key, PASS, f"flagged_at_least({what})",
                            f"Flagged {n:,} {what}.")

    # ---------------------------------------------------------------- response
    @_safe
    def response_fired(self, evidence, minimum=1):
        """An automated response actually happened. (SLO 7.3)"""
        key = "response_fired"
        if isinstance(evidence, bool):
            n = 1 if evidence else 0
        elif isinstance(evidence, (int, np.integer)):
            n = int(evidence)
        elif isinstance(evidence, str):
            n = 1 if evidence.strip() else 0
        else:
            n = _n(evidence)
        if n < minimum:
            return self._record(
                key, FAIL, "response_fired",
                f"Found {n} response(s); at least {minimum} expected.",
                "A prediction nobody receives is not a decision system. Trigger "
                "something real — a Telegram message, an email, a dashboard row — and "
                "pass the evidence in, e.g. the list of messages you sent.")
        return self._record(key, PASS, "response_fired", f"{n} response(s) fired.")

    # ---------------------------------------------------------------- report

    # ---------------------------------------------------------------- backtest league
    @_safe
    def position_is_valid(self, position, prices):
        """Your strategy's position is -1, 0 or 1 on every trading day of the price file."""
        key = "position_is_valid"
        pos = pd.Series(position)
        if len(pos) != len(prices):
            return self._record(
                key, FAIL, "position_is_valid",
                f"The position has {len(pos):,} rows; the price file has {len(prices):,}.",
                "Return one position per row of the price table, in the same order. "
                "If you dropped the warm-up rows of a rolling window, fill them with 0 "
                "instead:  position = position.fillna(0)")
        bad = pos[~pos.isin([-1, 0, 1])]
        if len(bad):
            sample = ", ".join(str(v) for v in bad.unique()[:5])
            return self._record(
                key, FAIL, "position_is_valid",
                f"{len(bad):,} row(s) are not -1, 0 or 1 (for example {sample}).",
                "A position is a decision, not a forecast: 1 means hold the ETF, 0 means "
                "hold cash, -1 means short. NaN usually means a rolling window has not "
                "filled yet; fill those with 0.")
        return self._record(key, PASS, "position_is_valid",
                            f"{len(pos):,} positions, all -1, 0 or 1.")

    @_safe
    def no_lookahead(self, strategy, prices, cuts=None):
        """
        Your strategy does not use future prices. (SLO 6.2, 9.1)

        The test is a truncation test: run the strategy on the first k rows only and
        compare with the first k rows of its answer on the full file. If today's
        decision changes when tomorrow's price is removed, it was using tomorrow's
        price. This catches shift(-1), centred rolling windows and the like, which
        the scorer's one-day delay alone cannot. Twelve cut points are used, so a
        rule that leaks one day (shift(-1)) is caught at almost every cut.
        """
        key = "no_lookahead"
        if cuts is None:                      # 12 cuts: a one-row leak at a cut is
            cuts = np.linspace(0.3, 0.97, 12)  # missed with probability about 1 in 4000
        full = pd.Series(strategy(prices.copy())).reset_index(drop=True)
        for frac in cuts:
            k = int(len(prices) * frac)
            part = pd.Series(strategy(prices.iloc[:k].copy())).reset_index(drop=True)
            if len(part) != k:
                return self._record(
                    key, FAIL, "no_lookahead",
                    f"On the first {k:,} rows the strategy returned {len(part):,} positions.",
                    "The strategy must return one position per row it is given.")
            diff = int((part.fillna(0) != full.iloc[:k].fillna(0)).sum())
            if diff:
                first = int(np.argmax((part.fillna(0) != full.iloc[:k].fillna(0)).values))
                return self._record(
                    key, FAIL, "no_lookahead",
                    f"Removing the rows after day {k:,} changed {diff:,} earlier decision(s), "
                    f"the first on day {first:,}.",
                    "The strategy is reading prices from after the day it is deciding for. "
                    "Look for shift() with a negative number, rolling(..., center=True), "
                    "or a statistic computed over the whole file (such as the overall "
                    "mean) and used on every day. A decision for day t may use rows up "
                    "to and including t only.")
        return self._record(key, PASS, "no_lookahead",
                            f"Decisions unchanged when the future is removed at "
                            f"{len(cuts)} cut points.")

    @_safe
    def trades_at_least(self, position, minimum=4):
        """The strategy actually trades; buy-and-hold under a new name is not an entry."""
        key = "trades_at_least"
        pos = pd.Series(position).fillna(0)
        n = int((pos.diff().fillna(0) != 0).sum())
        if n < minimum:
            return self._record(
                key, FAIL, "trades_at_least",
                f"The position changes {n} time(s); the league asks for at least {minimum}.",
                "A strategy that never changes its mind is buy-and-hold, which is already "
                "on the leaderboard as the baseline. Your entry has to decide something.")
        return self._record(key, PASS, "trades_at_least",
                            f"The position changes {n:,} times.")

    def report(self, verbose=True):
        """Print the summary. Returns True if nothing failed."""
        rows = list(self._results.values())
        self._reported = True
        if not rows:
            print("No checks were run.")
            return False

        n_pass = sum(1 for r in rows if r[0] == PASS)
        n_fail = sum(1 for r in rows if r[0] == FAIL)
        n_note = sum(1 for r in rows if r[0] == NOTE)
        width = 66

        print("=" * width)
        print("EP0418 VERIFICATION REPORT")
        print("=" * width)
        for status, label, detail, fix in rows:
            print(f"[{status}] {label}")
            if status == PASS and not verbose:
                continue
            if detail:
                print(f"       {detail}")
            if fix and status != PASS:
                for line in fix.split("\n"):
                    print(f"       {line}")
            print()
        print("-" * width)
        bits = [f"{n_pass} passed"]
        if n_fail:
            bits.append(f"{n_fail} failed")
        if n_note:
            bits.append(f"{n_note} to report")
        print(", ".join(bits))
        if n_fail:
            print("\nFix the failures above, then run this cell again.")
        elif n_note:
            print("\nNothing is broken. The NOTE items are real results — write them up.")
        else:
            print("\nAll checks passed. Carry on with the next step; at the end of the notebook,"
                  " write the mistake you caught in the Mistake caught box.")
        print("=" * width)
        return n_fail == 0


check = _Check()


# =============================================================================
# Backtest league scorer. Locked, like the checks: students read it, never edit it.
# =============================================================================

class _Backtest:
    """
    Turn a daily position into a scored equity curve.

        from ep0418 import backtest
        result = backtest.run(prices, position)      # prices has a Close column
        print(result)

    Rules of the scorer, in words (the code below is the only authority):

    * You decide at the close. The position you hold at the close of day t earns
      the move from the close of day t to the close of day t+1. The scorer shifts
      your position by one day for you, so a decision cannot earn the move that
      produced it.
    * Every change of position costs COST of the money moved. Switching from
      long to short moves two units and costs twice.
    * Scores are computed over the rows you pass in, so the same function scores
      the public file and the hidden period.
    """
    COST = 0.001          # 0.1% of the amount traded, each time the position changes
    DAYS = 252            # trading days in a year, for annualising

    def run(self, prices, position, cost=None):
        cost = self.COST if cost is None else cost
        close = pd.Series(prices["Close"]).astype(float).reset_index(drop=True)
        pos = pd.Series(position).astype(float).fillna(0).reset_index(drop=True)
        if len(pos) != len(close):
            raise ValueError(f"position has {len(pos)} rows, prices has {len(close)}")
        ret = close.pct_change().fillna(0)
        held = pos.shift(1).fillna(0)                       # yesterday's decision earns today
        traded = held.diff().abs().fillna(held.abs().iloc[0] if len(held) else 0)
        daily = held * ret - traded * cost
        equity = (1 + daily).cumprod()
        return self._metrics(daily, equity, int((traded > 0).sum()))

    def buy_and_hold(self, prices, cost=None):
        return self.run(prices, np.ones(len(prices)), cost)

    def _metrics(self, daily, equity, trades):
        n = len(daily)
        years = n / self.DAYS if n else float("nan")
        total = float(equity.iloc[-1] - 1) if n else float("nan")
        ann = float((1 + total) ** (1 / years) - 1) if n and total > -1 else float("nan")
        vol = float(daily.std() * np.sqrt(self.DAYS)) if n > 1 else float("nan")
        sharpe = float(daily.mean() / daily.std() * np.sqrt(self.DAYS)) if n > 1 and daily.std() > 0 else 0.0
        mdd = float((equity / equity.cummax() - 1).min()) if n else float("nan")
        return _Result(total=total, annualised=ann, volatility=vol, sharpe=sharpe,
                       max_drawdown=mdd, trades=trades, days=n, equity=equity)


class _Result(dict):
    """A dict of metrics that prints as a small table."""
    def __init__(self, **kw):
        super().__init__(**kw)
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)
    def __str__(self):
        return (f"  total return   {self['total']:+.1%}\n"
                f"  annualised     {self['annualised']:+.1%}\n"
                f"  volatility     {self['volatility']:.1%}\n"
                f"  Sharpe ratio   {self['sharpe']:.2f}\n"
                f"  max drawdown   {self['max_drawdown']:.1%}\n"
                f"  trades         {self['trades']}   over {self['days']} trading days")
    __repr__ = __str__


backtest = _Backtest()


# =============================================================================
# show: plain, labelled plots for the labs. One call per picture, so every lab
# draws the same way: a thin dark line, grey for comparisons, one accent for
# anything flagged. Nothing here raises; a bad input prints a sentence instead.
# =============================================================================

class _Show:
    INK, GREY, ACCENT, LIGHT = "#222222", "#888888", "#b03030", "#dddddd"
    W = 10

    def _ax(self, h=3.0, n=1):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(n, 1, figsize=(self.W, h * n), sharex=(n > 1))
        for ax in (axes if n > 1 else [axes]):
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.grid(False)
        return fig, axes

    def _series(self, obj, col=None):
        """A Series with a DatetimeIndex, from a Series or a DataFrame + column."""
        if isinstance(obj, pd.DataFrame):
            if col is None:
                raise ValueError("give the column to plot, e.g. col='co2_ppm'")
            t = _times(obj)
            if t is None:
                raise ValueError("no datetime column or index found")
            return pd.Series(obj[col].values, index=t, name=col)
        s = pd.Series(obj)
        if not isinstance(s.index, pd.DatetimeIndex):
            s.index = pd.to_datetime(s.index, errors="coerce")
        return s

    def _done(self, ax, title, ylabel):
        import matplotlib.pyplot as plt
        if title:
            ax.set_title(title, loc="left", fontsize=10)
        if ylabel:
            ax.set_ylabel(ylabel)
        plt.tight_layout(); plt.show()

    def series(self, obj, col=None, title=None, ylabel=None, span=None, hline=None):
        """One series over time. span=(start, end) shades a period; hline draws a level."""
        try:
            s = self._series(obj, col)
            fig, ax = self._ax()
            ax.plot(s.index, s.values, color=self.INK, lw=0.8)
            if span is not None:
                ax.axvspan(pd.Timestamp(span[0]), pd.Timestamp(span[1]), color=self.LIGHT, zorder=0)
            if hline is not None:
                ax.axhline(hline, color=self.GREY, lw=0.8, ls="--")
            self._done(ax, title, ylabel or getattr(s, "name", None))
        except Exception as e:                                        # noqa: BLE001
            print(f"show.series could not draw this: {e}")

    def flags(self, obj, mask, col=None, title=None, ylabel=None, label="flagged"):
        """A series with the rows where mask is True marked in the accent colour."""
        try:
            s = self._series(obj, col)
            m = pd.Series(mask).fillna(False).astype(bool).values
            fig, ax = self._ax()
            ax.plot(s.index, s.values, color=self.INK, lw=0.8)
            ax.scatter(s.index[m], s.values[m], color=self.ACCENT, s=12, zorder=3,
                       label=f"{label}: {int(m.sum()):,}")
            ax.legend(loc="upper right", frameon=False, fontsize=9)
            self._done(ax, title, ylabel or getattr(s, "name", None))
        except Exception as e:                                        # noqa: BLE001
            print(f"show.flags could not draw this: {e}")

    def before_after(self, before, after, col, title=None):
        """Two panels on the same time axis: the column before and after a step."""
        try:
            a, b = self._series(before, col), self._series(after, col)
            fig, axes = self._ax(2.4, 2)
            axes[0].plot(a.index, a.values, color=self.INK, lw=0.8)
            axes[0].set_title(f"before: {len(a):,} rows", loc="left", fontsize=10)
            axes[1].plot(b.index, b.values, color=self.INK, lw=0.8)
            axes[1].set_title(f"after: {len(b):,} rows", loc="left", fontsize=10)
            for ax in axes:
                ax.set_ylabel(col)
            if title:
                fig.suptitle(title, x=0.01, ha="left", fontsize=10)
            import matplotlib.pyplot as plt
            plt.tight_layout(); plt.show()
        except Exception as e:                                        # noqa: BLE001
            print(f"show.before_after could not draw this: {e}")

    def compare(self, actual, title=None, ylabel=None, **others):
        """Actual in dark; each named keyword series in grey, dashed. show.compare(s, baseline=b, model=m)"""
        try:
            s = self._series(actual)
            fig, ax = self._ax()
            ax.plot(s.index, s.values, color=self.INK, lw=0.9, label="actual")
            styles = ["--", ":", "-."]
            for i, (name, o) in enumerate(others.items()):
                o = self._series(o)
                ax.plot(o.index, o.values, color=self.GREY if i else self.ACCENT, lw=0.9,
                        ls=styles[i % 3], label=name)
            ax.legend(loc="upper right", frameon=False, fontsize=9, ncol=len(others) + 1)
            self._done(ax, title, ylabel or getattr(s, "name", None))
        except Exception as e:                                        # noqa: BLE001
            print(f"show.compare could not draw this: {e}")

    def profile(self, obj, col=None, by="hour", title=None):
        """Mean by hour of day (by='hour') or day of week (by='dow')."""
        try:
            s = self._series(obj, col)
            if by == "hour":
                g = s.groupby(s.index.hour).mean(); xlabel = "hour of day"
            else:
                g = s.groupby(s.index.dayofweek).mean(); xlabel = "day of week (0 = Monday)"
            fig, ax = self._ax(2.6)
            ax.plot(g.index, g.values, color=self.INK, lw=0.9, marker="o", ms=3)
            ax.set_xlabel(xlabel); ax.set_xticks(list(g.index))
            self._done(ax, title, f"mean {getattr(s, 'name', '')}".strip())
        except Exception as e:                                        # noqa: BLE001
            print(f"show.profile could not draw this: {e}")

    def alerts(self, obj, threshold, col=None, title=None, **fired):
        """A series, its threshold, and one row of markers per named alert mask.
        show.alerts(df, 1000, col='co2_ppm', naive=mask1, debounced=mask2)"""
        try:
            s = self._series(obj, col)
            fig, ax = self._ax(3.2)
            ax.plot(s.index, s.values, color=self.INK, lw=0.7)
            ax.axhline(threshold, color=self.GREY, lw=0.8, ls="--", label=f"threshold {threshold}")
            top = float(np.nanmax(s.values)); step = (top - float(np.nanmin(s.values))) * 0.06
            for i, (name, m) in enumerate(fired.items()):
                m = pd.Series(m).fillna(False).astype(bool).values
                y = top + step * (i + 1)
                ax.scatter(s.index[m], [y] * int(m.sum()), s=8, marker="|",
                           color=self.ACCENT if i == 0 else self.GREY, label=f"{name}: {int(m.sum()):,} alerts")
            ax.legend(loc="upper left", frameon=False, fontsize=9, ncol=len(fired) + 1)
            self._done(ax, title, getattr(s, "name", None))
        except Exception as e:                                        # noqa: BLE001
            print(f"show.alerts could not draw this: {e}")

    def stages(self, counts, title="Rows at each stage"):
        """Row counts through a pipeline: show.stages({'ingest': 17354, 'clean': 17214, ...})."""
        try:
            names, vals = list(counts.keys()), [int(v) for v in counts.values()]
            fig, ax = self._ax(0.5 * len(names) + 0.8)
            ax.barh(names[::-1], vals[::-1], color=self.LIGHT, edgecolor=self.INK, lw=0.6)
            for i, v in enumerate(vals[::-1]):
                ax.text(v, i, f" {v:,}", va="center", fontsize=9)
            ax.set_xlim(0, max(vals) * 1.18)
            self._done(ax, title, None)
        except Exception as e:                                        # noqa: BLE001
            print(f"show.stages could not draw this: {e}")

    def pipeline_code(self, names=("ingest", "clean", "analyse", "decide", "respond", "pipeline")):
        """Put the code of the live pipeline stages in one box, to copy into an AI that
        cannot see the notebook (ChatGPT, Claude, Copilot). Only the versions in use now
        are shown, under the names pipeline() calls them by."""
        import inspect, html, __main__
        g, parts = vars(__main__), []
        for name in names:
            f = g.get(name)
            if f is None:
                parts.append(f"# ---- {name}: not defined in this notebook yet")
                continue
            note = f" (this is {f.__name__})" if getattr(f, "__name__", name) != name else ""
            if getattr(f, "fake", False):
                note += " - still the fake skeleton from Task 1"
            try:
                src = inspect.getsource(f).rstrip()
            except (TypeError, OSError):
                src = "# (its code could not be read; paste its cell by hand)"
            parts.append(f"# ---- {name}{note}\n{src}")
        text = "\n\n".join(parts)
        try:
            from IPython.display import HTML, display
            rows = min(30, text.count("\n") + 2)
            display(HTML(f'<textarea readonly rows="{rows}" style="width:100%;font-family:monospace;font-size:12px">'
                         f'{html.escape(text)}</textarea>'))
            print("Click inside the box, press Ctrl+A (Cmd+A on a Mac), then Ctrl+C. "
                  "In the other AI's chat, paste it under the prompt.")
        except Exception:                                             # noqa: BLE001
            print(text)


show = _Show()


# =============================================================================
# league: locked scorers for the Beat-the-Baseline League, one per track.
# Students read these; they never edit them. Every scorer takes a strategy
# function and the data, scores it on the rows after `start` (the hidden period
# when the tutor runs it), and checks that the strategy used no future rows.
# =============================================================================

class _League:
    """
    Track 2  strategy(df) -> boolean Series, True on rows to alert.  Lower cost wins:
             cost = alert rows + uncovered rows (occupied, CO2 > 1000, no alert in the
             last 30 minutes). Baseline: the fixed threshold.
    Track 3  strategy(history) -> True to water today. Called once per day with the
             rows before that day. cost = 2 x missed waterings + wasted waterings.
             Baseline: the best of three dumb rules (yesterday, always water, never water).
    Track 4  strategy(history) -> the next 24 hourly values. Called once per day with
             the hourly series up to midnight. MAE. Baseline: the same 24 hours last week.
    Track 5  strategy(history) -> tomorrow's steps. Called once per day. MAE.
             Baseline: mean of the last 7 recorded days.
    Track 1  use ep0418.backtest (the backtest league).
    """
    COVER = 6                      # rows (30 minutes) an alert covers, Track 2
    HORIZON = 24                   # hours, Track 4

    # ------------------------------------------------------------ helpers
    def _result(self, track, metric, value, baseline, n, lower_is_better=True, detail=""):
        return _LeagueResult(track=track, metric=metric, value=float(value), baseline=float(baseline),
                       n=int(n), lower_is_better=lower_is_better,
                       beats_baseline=(value < baseline) if lower_is_better else (value > baseline),
                       detail=detail)

    @staticmethod
    def _split(df, start, col):
        t = pd.to_datetime(df[col])
        if start is None:
            start = t.iloc[int(len(t) * 0.8)]
        start = pd.Timestamp(start)
        if t.dt.tz is not None and start.tzinfo is None:
            start = start.tz_localize(t.dt.tz)
        return start

    # ------------------------------------------------------------ Track 2
    def building_cost(self, df, alert):
        a = pd.Series(alert).fillna(False).astype(bool).reset_index(drop=True)
        d = df.reset_index(drop=True)
        need = (d.occupancy > 0) & (d.co2_ppm > 1000)
        covered = a.rolling(self.COVER, min_periods=1).max().astype(bool)
        uncovered = need & ~covered
        return int(a.sum()), int(uncovered.sum())

    def score_building(self, strategy, df, start=None):
        d = df.sort_values("timestamp").reset_index(drop=True)
        start = self._split(d, start, "timestamp")
        full = pd.Series(strategy(d.copy())).fillna(False).astype(bool).reset_index(drop=True)
        if len(full) != len(d):
            raise ValueError(f"strategy returned {len(full)} values for {len(d)} rows")
        k = int((d.timestamp < start).sum())
        trues = np.flatnonzero(full.values[: max(len(d) - 1, 1)]) + 1   # cut just after an alert
        cuts = list(trues[np.linspace(0, len(trues) - 1, 8).astype(int)]) if len(trues) else []
        cuts += [int(len(d) * f) for f in (0.3, 0.6, 0.9)]
        for c in sorted(set(int(x) for x in cuts)):       # truncation test at several cuts
            part = pd.Series(strategy(d.iloc[:c].copy())).fillna(False).astype(bool).reset_index(drop=True)
            if len(part) == c and int((part != full.iloc[:c]).sum()):
                raise ValueError("decisions before a cut changed when later rows were removed: "
                                 "the strategy is reading the future")
        test = d.iloc[k:].reset_index(drop=True)
        alerts, missed = self.building_cost(test, full.iloc[k:])
        b_alerts, b_missed = self.building_cost(test, (test.co2_ppm > 1000))
        return self._result(2, "cost (alerts + uncovered)", alerts + missed, b_alerts + b_missed, len(test),
                            detail=f"{alerts} alerts, {missed} uncovered; fixed threshold {b_alerts} + {b_missed}")

    # ------------------------------------------------------------ Track 3
    def score_farm(self, strategy, df, start=None):
        d = df.sort_values("date").reset_index(drop=True)
        d["date"] = pd.to_datetime(d["date"])
        start = self._split(d, start, "date")
        need = d.rain_mm_mean < 1.0
        idx = d.index[d.date >= start]
        missed = wasted = 0
        rules = {"yesterday": [0, 0], "always water": [0, 0], "never water": [0, 0]}
        for i in idx:
            hist = d.iloc[:i].copy()
            water = bool(strategy(hist))
            dumb = {"yesterday": bool(need.iloc[i - 1]) if i > 0 else False,
                    "always water": True, "never water": False}
            if need.iloc[i]:
                missed += (not water)
                for r, w in dumb.items(): rules[r][0] += (not w)
            else:
                wasted += water
                for r, w in dumb.items(): rules[r][1] += w
        cost = lambda mw: 2 * mw[0] + mw[1]
        best = min(rules, key=lambda r: cost(rules[r]))
        return self._result(3, "cost (2 x missed + wasted)", 2 * missed + wasted, cost(rules[best]), len(idx),
                            detail=f"{missed} missed, {wasted} wasted; best dumb rule '{best}' "
                                   f"{rules[best][0]} missed + {rules[best][1]} wasted")

    # ------------------------------------------------------------ Track 4
    def score_city(self, strategy, s, start=None):
        s = pd.Series(s).sort_index()
        if not isinstance(s.index, pd.DatetimeIndex):
            raise ValueError("give an hourly Series with a DatetimeIndex (resample first)")
        if start is None:
            start = s.index[int(len(s) * 0.8)].normalize()
        start = pd.Timestamp(start)
        if s.index.tz is not None and start.tzinfo is None:
            start = start.tz_localize(s.index.tz)
        days = pd.date_range(start.normalize(), s.index[-1].normalize(), freq="D")
        err, b_err, n = [], [], 0
        for day in days:
            hist = s[s.index < day]
            if len(hist) < 24 * 8:
                continue
            pred = np.asarray(strategy(hist.copy()), dtype=float).ravel()[: self.HORIZON]
            actual = s[(s.index >= day) & (s.index < day + pd.Timedelta(hours=self.HORIZON))]
            base = hist[(hist.index >= day - pd.Timedelta(days=7)) &
                        (hist.index < day - pd.Timedelta(days=7) + pd.Timedelta(hours=self.HORIZON))]
            m = min(len(pred), len(actual), len(base))
            for p, a, b in zip(pred[:m], actual.values[:m], base.values[:m]):
                if np.isfinite(a) and np.isfinite(p) and np.isfinite(b):
                    err.append(abs(a - p)); b_err.append(abs(a - b)); n += 1
        if not n:
            raise ValueError("no scorable hours after the start date")
        return self._result(4, f"MAE, {self.HORIZON} h ahead", float(np.mean(err)), float(np.mean(b_err)), n,
                            detail=f"{len(days)} forecast days")

    # ------------------------------------------------------------ Track 5
    def score_self(self, strategy, df, start=None):
        d = df.sort_values("date").reset_index(drop=True)
        d["date"] = pd.to_datetime(d["date"])
        start = self._split(d, start, "date")
        idx = d.index[d.date >= start]
        err, b_err = [], []
        for i in idx:
            hist = d.iloc[:i].copy()
            actual = d.steps.iloc[i]
            if not np.isfinite(actual):
                continue
            pred = float(strategy(hist))
            recent = hist.steps.replace(0, np.nan).dropna().tail(7)
            base = float(recent.mean()) if len(recent) else np.nan
            if np.isfinite(pred) and np.isfinite(base):
                err.append(abs(actual - pred)); b_err.append(abs(actual - base))
        return self._result(5, "MAE, steps, 1 day ahead", float(np.mean(err)), float(np.mean(b_err)), len(err))

    # ------------------------------------------------------------ dispatch
    def score(self, track, strategy, data, start=None):
        """league.score(track, strategy, data, start): score a strategy on the rows after start."""
        fn = {2: self.score_building, 3: self.score_farm, 4: self.score_city, 5: self.score_self}.get(int(track))
        if fn is None:
            raise ValueError("track must be 2, 3, 4 or 5 (Track 1 uses ep0418.backtest)")
        return fn(strategy, data, start)


class _LeagueResult(dict):
    """A dict of league results that prints as a short report."""
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)
    def __str__(self):
        verdict = "beats the baseline" if self["beats_baseline"] else "does not beat the baseline"
        return (f"  track {self['track']}   {self['metric']}\n"
                f"  yours      {self['value']:,.2f}\n"
                f"  baseline   {self['baseline']:,.2f}\n"
                f"  scored on  {self['n']:,} rows   -> {verdict}"
                + (f"\n  {self['detail']}" if self.get("detail") else ""))
    __repr__ = __str__


league = _League()
