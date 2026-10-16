"""
The Prediction League's nightly job (run by .github/workflows/league.yml).

    python run_league.py fetch  out.json     # read entries from the private sheet (needs ENTRIES_CSV_URL)
    python run_league.py freeze entries.json # at a lab's close: run each person's latest code once, freeze the number
    python run_league.py settle              # after the target week: score the frozen numbers against reality

Only names, numbers and short status lines are written to docs/data/. Code never is.
"""
import csv, datetime as dt, io, json, os, subprocess, sys, tempfile, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("LEAGUE_DATA") or os.path.join(HERE, "..", "..", "docs", "data")
sys.path.insert(0, os.path.join(HERE, "..", "..", "docs"))
import league_questions as Q                        # the questions, shared with the entry page
SGT = dt.timezone(dt.timedelta(hours=8))
TIME_LIMIT = int(os.environ.get("LEAGUE_TIME_LIMIT", "120"))


def now_sgt():
    return dt.datetime.now(SGT)


def fetch(out):
    url = os.environ["ENTRIES_CSV_URL"]
    text = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60).read().decode()
    rows = list(csv.DictReader(io.StringIO(text)))
    json.dump(rows, open(out, "w"))
    print(f"fetched {len(rows)} entries")


def _when(ts):          # the sheet writes M/D/YYYY H:MM:SS in Singapore time
    return dt.datetime.strptime(ts.strip(), "%m/%d/%Y %H:%M:%S").replace(tzinfo=SGT)


def freeze(entries_path):
    rows = json.load(open(entries_path))
    os.remove(entries_path)                     # nothing on disk once the student code starts running
    only = os.environ.get("LEAGUE_ONLY")        # tests: freeze one lab with a stand-in close
    for lab, q in Q.LABS.items():
        if only and lab != only:
            continue
        out = os.path.join(DATA, f"frozen_lab{lab}.json")
        close_day = os.environ.get("LEAGUE_CLOSE") or q["close"]
        close = dt.datetime.fromisoformat(close_day).replace(tzinfo=SGT)
        if os.path.exists(out) or now_sgt() < close:
            continue
        latest = {}
        for r in rows:
            name = " ".join(r.get("Name", "").split())
            if r.get("Lab", "").strip() != lab or not name or not r.get("Code", "").strip():
                continue
            when = _when(r["Timestamp"])
            if when < close and (name.lower() not in latest or when > latest[name.lower()][0]):
                latest[name.lower()] = (when, name, r["Code"])
        df = Q.load(lab, os.path.join(DATA, q["data"]))
        target = os.environ.get("LEAGUE_TARGET") or Q.target(lab)
        if isinstance(target, str) and "," in target:
            target = tuple(target.split(","))
        hist = Q.history(lab, df, target)
        results = []
        with tempfile.TemporaryDirectory() as tmp:
            hpath = os.path.join(tmp, "history.csv")
            hist.to_csv(hpath, index=False)
            os.chmod(hpath, 0o444)
            for key in sorted(latest):
                when, name, code = latest[key]
                work = tempfile.mkdtemp(dir=tmp)
                try:
                    p = subprocess.run([sys.executable, "-I", os.path.join(HERE, "child.py")],
                                       input=json.dumps({"code": code, "history": hpath, "lab": lab,
                                                         "questions": os.path.join(HERE, "..", "..", "docs")}),
                                       capture_output=True, text=True, timeout=TIME_LIMIT, cwd=work,
                                       env={"PATH": os.environ.get("PATH", ""), "HOME": work}, start_new_session=True)
                    tail = p.stdout.rsplit("@@RESULT@@", 1)
                    res = json.loads(tail[1]) if len(tail) == 2 else {"status": "error", "error": "the code stopped Python"}
                except subprocess.TimeoutExpired:
                    res = {"status": f"time limit ({TIME_LIMIT} s)"}
                results.append({"name": name, "submitted": when.strftime("%Y-%m-%d %H:%M"), **res})
        json.dump({"lab": lab, "title": q["title"], "close": close_day, "target": target,
                   "history_rows": len(hist), "baseline": Q.baseline(lab, hist), "entries": results}, open(out, "w"), indent=1)
        ok = sum(r["status"] == "ok" for r in results)
        print(f"lab {lab}: froze {len(results)} entries ({ok} returned a number)")


def settle():
    for lab, q in Q.LABS.items():
        fz, out = os.path.join(DATA, f"frozen_lab{lab}.json"), os.path.join(DATA, f"results_lab{lab}.json")
        if not os.path.exists(fz) or os.path.exists(out):
            continue
        f = json.load(open(fz))
        target = tuple(f["target"]) if isinstance(f["target"], list) else f["target"]
        actual = Q.answer(lab, Q.load(lab, os.path.join(DATA, q["data"])), target)
        if actual is None:
            continue                            # reality has not answered yet
        ranked = sorted((e for e in f["entries"] if e["status"] == "ok"), key=lambda e: abs(e["value"] - actual))
        for e in ranked:
            e["off_by"] = abs(e["value"] - actual)
        json.dump({**f, "actual": actual, "baseline_off_by": abs(f["baseline"] - actual),
                   "ranked": ranked, "no_number": [e for e in f["entries"] if e["status"] != "ok"]}, open(out, "w"), indent=1)
        print(f"lab {lab}: actual {actual}; {len(ranked)} ranked")


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"fetch": lambda: fetch(sys.argv[2]), "freeze": lambda: freeze(sys.argv[2]), "settle": settle}[cmd]()
