"""Append every missing day up to yesterday (Singapore time) to docs/data/psi_hourly.csv.
One row per hour: the 24-hour PSI in each of the five regions, and psi_max, the highest
of the five (the number the haze question is about). With no file yet it starts on
2025-01-01."""
import csv, datetime as dt, json, os, time, urllib.request

PATH = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "data", "psi_hourly.csv")
SGT = dt.timezone(dt.timedelta(hours=8))
UA = {"User-Agent": "Mozilla/5.0 (EP0418 league)"}
REGIONS = ["north", "south", "east", "west", "central"]
HEADER = ["timestamp"] + [f"psi_{r}" for r in REGIONS] + ["psi_max"]
FIRST = dt.date(2025, 1, 1)


def day_rows(day):
    for i in range(5):
        try:
            d = json.load(urllib.request.urlopen(urllib.request.Request(
                f"https://api-open.data.gov.sg/v2/real-time/api/psi?date={day}", headers=UA), timeout=60))
            break
        except Exception:                          # noqa: BLE001
            if i == 4:
                raise
            time.sleep(5 * (i + 1))
    out = []
    for it in d["data"]["items"]:
        psi = it["readings"]["psi_twenty_four_hourly"]
        vals = [psi.get(r) for r in REGIONS]
        if any(v is None for v in vals):
            continue
        ts = dt.datetime.fromisoformat(it["timestamp"]).astimezone(SGT)
        out.append([ts.strftime("%Y-%m-%d %H:%M:%S") + "+08:00"] + vals + [max(vals)])
    return sorted(out)


def main(limit=800):
    if os.path.exists(PATH):
        rows = list(csv.reader(open(PATH)))
        last = dt.datetime.fromisoformat(rows[-1][0]).date() if len(rows) > 1 else FIRST - dt.timedelta(days=1)
    else:
        with open(PATH, "w", newline="") as f:
            csv.writer(f, lineterminator="\n").writerow(HEADER)
        last = FIRST - dt.timedelta(days=1)
    yesterday = dt.datetime.now(SGT).date() - dt.timedelta(days=1)
    day, added = last + dt.timedelta(days=1), 0
    while day <= yesterday and added < limit:
        rows = day_rows(day.isoformat())
        with open(PATH, "a", newline="") as f:
            csv.writer(f, lineterminator="\n").writerows(rows)
        added += 1
        if added % 30 == 0:
            print(day, len(rows), "hours", flush=True)
        day += dt.timedelta(days=1)
        time.sleep(1.0)                            # the API answers 429 when asked too fast
    print(f"psi: added {added} day(s); file ends {day - dt.timedelta(days=1)}")


if __name__ == "__main__":
    main()
