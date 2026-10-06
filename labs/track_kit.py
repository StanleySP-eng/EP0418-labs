"""
EP0418 track kit: a ready-made `ingest()` and `clean()` for each project track,
for Laboratory 7 and CA3. Copy the pair for your track into your notebook and
adapt it; read every line first.

Every ingest returns a DataFrame with a `timestamp` column (renamed from `Date`
or `date` where needed) and the track's value columns, sorted in time. Every
clean handles that file's documented faults and returns the same kind of table,
on a regular grid where the track has one, with empty slots kept as NaN.

    from track_kit import ingest_smart_self as ingest, clean_smart_self as clean
"""
import numpy as np
import pandas as pd


def _load(csv, time_col):
    df = pd.read_csv(csv, parse_dates=[time_col])
    if time_col != "timestamp":
        df = df.rename(columns={time_col: "timestamp"})
    return df.sort_values("timestamp").reset_index(drop=True)


# ------------------------------------------------------------ Track 1
def ingest_smart_money(csv="smart_money_ohlcv.csv"):
    """Daily bars of the STI ETF: Open, High, Low, Close, Volume."""
    return _load(csv, "Date")


def clean_smart_money(df):
    """No sentinels in this file. Drop duplicate days, drop rows with a missing
    Close, keep trading days only (no reindexing to a calendar: markets close)."""
    df = df.drop_duplicates("timestamp", keep="last")
    df = df[df.Close.notna()]
    return df.sort_values("timestamp").reset_index(drop=True)


# ------------------------------------------------------------ Track 2
def ingest_smart_building(csv="smart_building_sensors.csv"):
    """Classroom sensors every 5 minutes: temp_c, humidity_pct, co2_ppm,
    occupancy, energy_kwh."""
    return _load(csv, "timestamp")


def clean_smart_building(df):
    """The Laboratory 3 cleaning: -999 humidity is a dropout, duplicate logger
    rows are kept last, and the 5-minute grid makes the 11 June outage visible."""
    df = df.copy()
    df.loc[df.humidity_pct == -999, "humidity_pct"] = np.nan
    df = df.drop_duplicates("timestamp", keep="last").sort_values("timestamp")
    grid = df.set_index("timestamp").resample("5min").mean()      # NaN kept
    return grid.reset_index()


# ------------------------------------------------------------ Track 3
def ingest_smart_farm(csv="sg_rainfall_daily.csv"):
    """Island-wide daily rainfall from NEA: rain_mm_mean, rain_mm_max,
    stations_reporting."""
    return _load(csv, "date")


def clean_smart_farm(df):
    """Rainfall cannot be negative. Put the series on a daily calendar so that
    any missing day is an empty slot, not a silently shorter file."""
    df = df.copy()
    for c in ("rain_mm_mean", "rain_mm_max"):
        df.loc[df[c] < 0, c] = np.nan
    df = df.drop_duplicates("timestamp", keep="last")
    daily = df.set_index("timestamp").asfreq("D")                  # NaN kept
    return daily.reset_index()


# ------------------------------------------------------------ Track 4
def ingest_smart_city(csv="sg_taxi_hourly.csv"):
    """Taxis available island-wide, one reading roughly every hour: taxi_count."""
    return _load(csv, "timestamp")


def clean_smart_city(df):
    """Readings arrive at irregular moments; resample onto an hourly grid, mean
    within the hour, empty hours kept as NaN. Implausibly low counts (a feed
    glitch, e.g. 24 taxis island-wide) are left for you to decide about."""
    df = df.drop_duplicates("timestamp", keep="last")
    s = df.set_index("timestamp").taxi_count.resample("h").mean()  # NaN kept
    return s.reset_index()


# ------------------------------------------------------------ Track 5
def ingest_smart_self(csv="smart_self_daily.csv"):
    """One row per day from a wearable: steps, resting_hr, sleep_hours,
    active_minutes."""
    return _load(csv, "date")


def clean_smart_self(df):
    """steps == 0 means the watch was not worn, not that the person did not
    move: make it NaN. Step counts above 30,000 are implausible for this
    wearer: NaN. Put the rows on a daily calendar so absent dates show."""
    df = df.copy()
    df.loc[df.steps == 0, "steps"] = np.nan
    df.loc[df.steps > 30000, "steps"] = np.nan
    df = df.drop_duplicates("timestamp", keep="last")
    daily = df.set_index("timestamp").asfreq("D")                  # NaN kept
    return daily.reset_index()


# ------------------------------------------------------------ Track 6
def ingest_free_form(csv="example_data.csv", time_col="timestamp"):
    """Your own file. Change `csv` and `time_col` to match it."""
    return _load(csv, time_col)


def clean_free_form(df, value_col="value", low=None, high=None):
    """Generic: drop duplicate timestamps, mark values outside [low, high] as
    NaN. Add your own file's faults here once you have profiled it."""
    df = df.drop_duplicates("timestamp", keep="last").copy()
    if low is not None:
        df.loc[df[value_col] < low, value_col] = np.nan
    if high is not None:
        df.loc[df[value_col] > high, value_col] = np.nan
    return df.sort_values("timestamp").reset_index(drop=True)


KIT = {
    1: (ingest_smart_money, clean_smart_money),
    2: (ingest_smart_building, clean_smart_building),
    3: (ingest_smart_farm, clean_smart_farm),
    4: (ingest_smart_city, clean_smart_city),
    5: (ingest_smart_self, clean_smart_self),
    6: (ingest_free_form, clean_free_form),
}
