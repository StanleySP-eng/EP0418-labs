"""Append every missing day up to yesterday (Singapore time) to docs/data/rain_daily.csv."""
import csv, datetime as dt, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from rain import rain_daily_row

PATH = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "data", "rain_daily.csv")


def main(limit=400):
    rows = list(csv.DictReader(open(PATH)))
    last = dt.date.fromisoformat(rows[-1]["date"])
    yesterday = (dt.datetime.utcnow() + dt.timedelta(hours=8)).date() - dt.timedelta(days=1)
    day, added = last + dt.timedelta(days=1), 0
    while day <= yesterday and added < limit:
        r = rain_daily_row(day.isoformat())
        with open(PATH, "a", newline="") as f:
            csv.writer(f).writerow([r["date"], r["rain_mm_mean"], r["rain_mm_max"], r["stations_reporting"]])
        print(r, flush=True)
        day += dt.timedelta(days=1); added += 1
    print(f"added {added} day(s); file ends {day - dt.timedelta(days=1)}")


if __name__ == "__main__":
    main()
