"""Append every missing hour up to the last full hour (Singapore time) to docs/data/taxi_hourly.csv.
One reading per hour, taken just before the hour ends (HH:59), like the Lab 3 file."""
import csv, datetime as dt, json, os, time, urllib.request

PATH = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "data", "taxi_hourly.csv")
SGT = dt.timezone(dt.timedelta(hours=8))
UA = {"User-Agent": "Mozilla/5.0 (EP0418 league)"}


def reading(hour_start):
    when = (hour_start + dt.timedelta(minutes=59)).strftime("%Y-%m-%dT%H:%M:%S")
    for i in range(5):
        try:
            d = json.load(urllib.request.urlopen(urllib.request.Request(
                f"https://api.data.gov.sg/v1/transport/taxi-availability?date_time={when}", headers=UA), timeout=60))
            p = d["features"][0]["properties"]
            ts = dt.datetime.fromisoformat(p["timestamp"])
            return ts.strftime("%Y-%m-%d %H:%M:%S") + "+08:00", int(p["taxi_count"])
        except Exception:                          # noqa: BLE001
            if i == 4:
                raise
            time.sleep(5 * (i + 1))


def main(limit=24 * 400):
    rows = list(csv.reader(open(PATH)))
    last = dt.datetime.fromisoformat(rows[-1][0]).astimezone(SGT).replace(minute=0, second=0, microsecond=0)
    end = dt.datetime.now(SGT).replace(minute=0, second=0, microsecond=0)   # the hour now running is not complete
    hour, added = last + dt.timedelta(hours=1), 0
    with open(PATH, "a", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        while hour < end and added < limit:
            ts, n = reading(hour)
            if dt.datetime.fromisoformat(ts).astimezone(SGT).replace(minute=0, second=0) == hour:
                w.writerow([ts, n]); added += 1          # a stale snapshot (feed outage) leaves the hour empty
            hour += dt.timedelta(hours=1)
            time.sleep(0.3)
    print(f"taxi: added {added} hour(s)")


if __name__ == "__main__":
    main()
