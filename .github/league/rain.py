"""Daily rainfall rows in the Lab 2 file's format, from data.gov.sg's five-minute readings."""
import json, time, urllib.request

UA = {"User-Agent": "Mozilla/5.0 (EP0418 league)"}


def _get(url, tries=5):
    for i in range(tries):
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60))
        except Exception:                          # noqa: BLE001
            if i == tries - 1:
                raise
            time.sleep(6 * (i + 1))


def rain_daily_row(day):
    """One day: island-average rain summed over the day (rain_mm_mean), the wettest
    gauge's total (rain_mm_max) and how many gauges reported (stations_reporting)."""
    base = f"https://api-open.data.gov.sg/v2/real-time/api/rainfall?date={day}"
    tok, island, per_station, n = None, 0.0, {}, 0
    for _ in range(40):
        d = _get(base + (f"&paginationToken={tok}" if tok else ""))["data"]
        for r in d["readings"]:
            vals = r["data"]
            if vals:
                island += sum(x["value"] for x in vals) / len(vals); n += 1
                for x in vals:
                    per_station[x["stationId"]] = per_station.get(x["stationId"], 0.0) + x["value"]
        tok = d.get("paginationToken")
        if not tok:
            break
        time.sleep(1.5)                            # the API answers 429 when asked too fast
    if n == 0:
        raise RuntimeError(f"no rainfall readings for {day}")
    return {"date": day, "rain_mm_mean": round(island, 3),
            "rain_mm_max": round(max(per_station.values()), 1), "stations_reporting": len(per_station),
            "readings": n}
