"""Refresh docs/data/coe.csv from LTA's COE Bidding Results dataset (same columns as the Lab 7 file)."""
import csv, io, json, os, time, urllib.request

PATH = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "data", "coe.csv")
UA = {"User-Agent": "Mozilla/5.0 (EP0418 league)"}
POLL = "https://api-open.data.gov.sg/v1/public/api/datasets/d_69b3380ad7e51aff3a7dcc84eba52b8a/poll-download"



def _get(u, tries=6):
    for i in range(tries):                         # data.gov.sg answers 429 when asked too often
        try:
            return urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=120).read()
        except Exception:                          # noqa: BLE001
            if i == tries - 1:
                raise
            time.sleep(20 * (i + 1))


url = json.loads(_get(POLL))["data"]["url"]
body = _get(url).decode()
new = list(csv.reader(io.StringIO(body)))
old = list(csv.reader(open(PATH))) if os.path.exists(PATH) else []
assert new[0] == ["month", "bidding_no", "vehicle_class", "quota", "bids_success", "bids_received", "premium"], new[0]
if len(new) >= len(old):
    with open(PATH, "w", newline="") as f:
        csv.writer(f, lineterminator="\n").writerows(new)
print(f"coe: {len(new) - 1} rows (was {max(len(old) - 1, 0)})")
