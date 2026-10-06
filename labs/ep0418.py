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

__all__ = ["check"]

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
                "This step has not been written yet.",
                "The line above still reads `...`. Ask your AI assistant for the "
                "implementation, paste it in, read it to check it does what you asked, "
                "then run this cell again.")
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
            print("\nAll checks passed. Record what you changed in verification_log.md.")
        print("=" * width)
        return n_fail == 0


check = _Check()
