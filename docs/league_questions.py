"""
EP0418 Prediction League: every lab's question in one place.

The entry page runs this file in the browser to build the test cases; the nightly job
runs the same file to freeze entries and to settle them. For each lab:
  data      the file in docs/data/ that predict(history) receives (cut at the close)
  kind      "week": the question is about a Monday-to-Sunday week
            "coe":  the question is about the next COE bidding exercise after the close,
                    whenever LTA holds it (the target is ("after", <last exercise seen>))
  answer    what really happened (None until the data has it)
  baseline  the simple rule the board compares everyone with
"""
import pandas as pd

HOURLY = ("taxi_hourly.csv", "psi_hourly.csv")      # one row per hour, timestamped in Singapore time

TAXI_COLS = "timestamp (one reading per hour, Singapore time) and taxi_count"
RAIN_COLS = "date, rain_mm_mean (island average, mm), rain_mm_max (wettest gauge, mm) and stations_reporting"
PSI_COLS = ("timestamp (one reading per hour, Singapore time), psi_north, psi_south, psi_east, psi_west, "
            "psi_central (the 24-hour PSI in each region) and psi_max (the highest of the five)")
COE_COLS = "month, bidding_no, vehicle_class, quota, bids_success, bids_received and premium (S$)"

LABS = {
    "1": dict(title="How bad will the haze get next week?", unit="PSI", data="psi_hourly.csv", kind="week",
              week=("2026-10-26", "2026-11-01"), cols=PSI_COLS,
              ask="The highest 24-hour PSI reading in any region of Singapore during the target week.",
              baseline_says="the highest reading in the last seven days"),
    "2": dict(title="How much rain next week?", unit="mm", data="rain_daily.csv", kind="week",
              week=("2026-11-02", "2026-11-08"), cols=RAIN_COLS,
              ask="Singapore's island-average rainfall in mm, summed over the target week.",
              baseline_says="next week will be like last week (the last seven days, summed)"),
    "3": dict(title="How many taxis at 8 am on Monday?", unit="taxis", data="taxi_hourly.csv", kind="week",
              week=("2026-11-09", "2026-11-15"), cols=TAXI_COLS,
              ask="Taxis available across Singapore in the 8 o'clock reading on the Monday of the target week.",
              baseline_says="the average of every Monday 8 o'clock reading so far"),
    "4": dict(title="How many dry days next week?", unit="days", data="rain_daily.csv", kind="week",
              week=("2026-11-16", "2026-11-22"), cols=RAIN_COLS,
              ask="On how many of the seven days of the target week the island-average rain is zero (under 0.05 mm).",
              baseline_says="the share of dry days in the last sixty days, times seven, rounded"),
    "5": dict(title="What will the busiest hour's taxi count be?", unit="taxis", data="taxi_hourly.csv", kind="week",
              week=("2026-11-30", "2026-12-06"), cols=TAXI_COLS,
              ask="The largest hourly taxi-availability count during the target week.",
              baseline_says="the largest hourly count in the last seven days"),
    "6": dict(title="How many taxis at 6 pm on Friday?", unit="taxis", data="taxi_hourly.csv", kind="week",
              week=("2026-12-07", "2026-12-13"), cols=TAXI_COLS,
              ask="Taxis available in the 18 o'clock reading on the Friday of the target week.",
              baseline_says="the 18 o'clock reading on the last Friday so far"),
    "7": dict(title="What will the next Cat A COE premium be?", unit="S$", data="coe.csv", kind="coe",
              close="2027-01-11", cols=COE_COLS,
              ask="The Category A premium at the next COE bidding exercise after entries close.",
              baseline_says="the Category A premium at the most recent exercise"),
    "8": dict(title="Which day will be wettest?", unit="(1 = Mon … 7 = Sun)", data="rain_daily.csv", kind="week",
              week=("2027-01-18", "2027-01-24"), cols=RAIN_COLS,
              ask="Which day of the target week has the most island-average rain: 1 for Monday through 7 for Sunday.",
              baseline_says="the weekday with the most rain on average over the last nine weeks"),
}
STARTERS = {
    "1": '''import pandas as pd

def predict(history):
    # the baseline: the highest reading in the last seven days
    last_week = history[history["timestamp"] > history["timestamp"].max() - pd.Timedelta(days=7)]
    print("highest PSI each day:", last_week.groupby(last_week["timestamp"].dt.date)["psi_max"].max().tolist())
    return float(last_week["psi_max"].max())
''',
    "2": '''def predict(history):
    # the baseline: next week will be like last week
    last_week = history["rain_mm_mean"].tail(7)
    print("last 7 days:", last_week.round(1).tolist())
    return float(last_week.sum())
''',
    "3": '''def predict(history):
    # the baseline: the average of every Monday 8 o'clock reading so far
    ts = history["timestamp"]
    mondays = history[(ts.dt.weekday == 0) & (ts.dt.hour == 8)]
    print("Mondays at 8:", mondays["taxi_count"].tolist())
    return float(mondays["taxi_count"].mean())
''',
    "4": '''def predict(history):
    # the baseline: the share of dry days in the last sixty days, times seven
    dry = history["rain_mm_mean"].tail(60) < 0.05
    print("dry days in the last sixty:", int(dry.sum()))
    return float(round(dry.mean() * 7))
''',
    "5": '''import pandas as pd

def predict(history):
    # the baseline: the busiest hour of the last seven days
    last_week = history[history["timestamp"] > history["timestamp"].max() - pd.Timedelta(days=7)]
    print("hours in the last week:", len(last_week))
    return float(last_week["taxi_count"].max())
''',
    "6": '''def predict(history):
    # the baseline: the 18 o'clock reading on the last Friday
    ts = history["timestamp"]
    fridays = history[(ts.dt.weekday == 4) & (ts.dt.hour == 18)]
    print("last three Fridays at 6 pm:", fridays["taxi_count"].tail(3).tolist())
    return float(fridays["taxi_count"].iloc[-1])
''',
    "7": '''def predict(history):
    # the baseline: the Category A premium at the most recent exercise
    cat_a = history[history["vehicle_class"] == "Category A"]
    print("last three premiums:", cat_a["premium"].tail(3).tolist())
    return float(cat_a["premium"].iloc[-1])
''',
    "8": '''def predict(history):
    # the baseline: the weekday with the most rain on average over the last nine weeks
    recent = history.tail(63)
    by_day = recent.groupby(recent["date"].dt.dayofweek)["rain_mm_mean"].mean()
    print("average mm, Monday to Sunday:", by_day.round(1).tolist())
    return float(by_day.idxmax() + 1)
''',
}

for _q in LABS.values():
    _q.setdefault("close", _q["week"][0] if "week" in _q else None)   # entries close when the week starts


def any_ai_prompt(lab, idea="[write your idea here in a few words, or delete this line]"):
    """A prompt for any AI outside Colab (ChatGPT, Claude, Copilot): it cannot see the notebook,
    so every rule the league checks is spelled out. Shown on the league page and in the labs."""
    q = LABS[lab]
    names = {"psi_hourly.csv": "timestamp, psi_north, psi_south, psi_east, psi_west, psi_central, psi_max",
             "taxi_hourly.csv": "timestamp, taxi_count",
             "rain_daily.csv": "date, rain_mm_mean, rain_mm_max, stations_reporting",
             "coe.csv": "month, bidding_no, vehicle_class, quota, bids_success, bids_received, premium"}[q["data"]]
    when = ("the first COE bidding exercise after entries close" if q["kind"] == "coe"
            else f"the week of Monday {q['week'][0]} to Sunday {q['week'][1]}")
    if q["data"] in HOURLY:
        data = (" The timestamp column is timezone-aware (Asia/Singapore): compare it only with "
                "timezone-aware times, for example pd.Timestamp(\"2026-10-26\", tz=\"Asia/Singapore\").")
    elif q["data"] == "rain_daily.csv":
        data = " The date column is a pandas datetime, one row per day, with no time zone."
    else:
        data = (" month is text such as \"2026-07\", there are two exercises a month (bidding_no 1 and 2), "
                "and vehicle_class holds text such as \"Category A\" (every category is in the table).")
    return f"""I am entering a prediction contest. Write one Python function for it.

The question: {q['ask']} The target is {when}.
The simple rule to beat: {q['baseline_says']}.
My idea: {idea}

The contest runs the function automatically, so follow these rules exactly:
1. Name it predict, with exactly one input called history:  def predict(history):
2. history is a pandas DataFrame that the contest passes in, oldest row first. Its column names are exactly: {names}. What they hold: {q['cols']}.{data}
3. Do not read any file and do not ask for input. All the data is already in history.
4. Return one number with return float(...). Not a list, a table, text, None or NaN.
5. Use only pandas, numpy and the Python standard library, and finish in under 30 seconds.
6. Give only the import lines and the function: no example data, no test code, nothing that calls predict.

After the code, explain the idea in one sentence."""


def load(lab, csv_file):
    q = LABS[lab]
    if q["data"] in HOURLY:
        # an explicit format: the browser page parses this ten times faster than parse_dates
        df = pd.read_csv(csv_file)
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S%z")
    elif q["data"] == "rain_daily.csv":
        df = pd.read_csv(csv_file, parse_dates=["date"])
    else:
        df = pd.read_csv(csv_file, dtype={"month": str})
    return df


def _t(day):                       # midnight at the start of a Singapore day, comparable with the taxi timestamps
    return pd.Timestamp(day).tz_localize("Asia/Singapore")


def _key(df):
    return df["month"] + "-" + df["bidding_no"].astype(str)


def coe_after(df):
    """At the close: the next exercise after the latest Category A result in the data."""
    return ("after", _key(df[df["vehicle_class"] == "Category A"]).iloc[-1])


def history(lab, df, target):
    """The rows predict() may see when predicting this target (a week start, or a COE exercise)."""
    q = LABS[lab]
    if q["kind"] == "coe":
        if target[0] == "after":
            return df[_key(df) <= target[1]].reset_index(drop=True)
        month, no = target
        key = df["month"] + "-" + df["bidding_no"].astype(str)
        return df[key < f"{month}-{no}"].reset_index(drop=True)
    if q["data"] in HOURLY:
        return df[df["timestamp"] < _t(target)].reset_index(drop=True)
    return df[df["date"] < pd.Timestamp(target)].reset_index(drop=True)


def _week_rows(lab, df, start):
    q = LABS[lab]
    if q["data"] in HOURLY:
        return df[(df["timestamp"] >= _t(start)) & (df["timestamp"] < _t(start) + pd.Timedelta(days=7))]
    return df[(df["date"] >= pd.Timestamp(start)) & (df["date"] < pd.Timestamp(start) + pd.Timedelta(days=7))]


def _reading(df, day, hour):
    ts = df["timestamp"]
    hit = df[(ts.dt.date == pd.Timestamp(day).date()) & (ts.dt.hour == hour)]
    return float(hit["taxi_count"].iloc[0]) if len(hit) else None


def answer(lab, df, target):
    """What really happened, or None if the data does not have it yet."""
    if lab == "7":
        a = df[df["vehicle_class"] == "Category A"]
        if target[0] == "after":
            hit = a[_key(a) > target[1]]
        else:
            month, no = target
            hit = a[(a["month"] == month) & (a["bidding_no"].astype(str) == str(no))]
        return float(hit["premium"].iloc[0]) if len(hit) else None
    w = _week_rows(lab, df, target)
    if lab in ("2", "4", "8"):
        if len(w) < 7:
            return None
        if lab == "2":
            return float(w["rain_mm_mean"].sum())
        if lab == "4":
            return float((w["rain_mm_mean"] < 0.05).sum())
        return float(w.loc[w["rain_mm_mean"].idxmax(), "date"].isoweekday())
    if lab == "3":
        return _reading(df, target, 8)
    if lab == "6":
        return _reading(df, pd.Timestamp(target) + pd.Timedelta(days=4), 18)
    if lab == "5":
        return float(w["taxi_count"].max()) if len(w) >= 150 else None
    if lab == "1":
        return float(w["psi_max"].max()) if len(w) >= 150 else None


def baseline(lab, hist):
    """The simple rule's number, from the history alone."""
    if lab == "1":
        return float(hist[hist["timestamp"] > hist["timestamp"].max() - pd.Timedelta(days=7)]["psi_max"].max())
    if lab == "2":
        return float(hist["rain_mm_mean"].tail(7).sum())
    if lab == "3":
        ts = hist["timestamp"]
        return float(hist[(ts.dt.weekday == 0) & (ts.dt.hour == 8)]["taxi_count"].mean())
    if lab == "4":
        return float(round((hist["rain_mm_mean"].tail(60) < 0.05).mean() * 7))
    if lab == "5":
        return float(hist[hist["timestamp"] > hist["timestamp"].max() - pd.Timedelta(days=7)]["taxi_count"].max())
    if lab == "6":
        ts = hist["timestamp"]
        return float(hist[(ts.dt.weekday == 4) & (ts.dt.hour == 18)]["taxi_count"].iloc[-1])
    if lab == "7":
        return float(hist[hist["vehicle_class"] == "Category A"]["premium"].iloc[-1])
    if lab == "8":
        d = hist.tail(63)
        return float(d.groupby(d["date"].dt.dayofweek)["rain_mm_mean"].mean().idxmax() + 1)


def past_targets(lab, df, n=3):
    """The n most recent targets whose answer is already in the data: the page's test cases."""
    q = LABS[lab]
    if q["kind"] == "coe":
        ex = df[df["vehicle_class"] == "Category A"][["month", "bidding_no"]].astype(str).values.tolist()
        return [tuple(e) for e in ex[-n:]]
    last = (df["date"] if "date" in df else df["timestamp"].dt.tz_localize(None)).max().normalize()
    last_sunday = last - pd.Timedelta(days=(last.weekday() + 1) % 7)
    out = []
    k = 0
    while len(out) < n and k < 20:
        start = str((last_sunday - pd.Timedelta(days=6 + 7 * k)).date())
        if answer(lab, df, start) is not None:
            out.insert(0, start)
        k += 1
    return out


def target(lab, df=None):
    """What this lab is predicting. For the COE question it depends on the data at the close."""
    q = LABS[lab]
    return coe_after(df) if q["kind"] == "coe" else q["week"][0]
