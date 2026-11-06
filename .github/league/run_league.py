"""
The Prediction League's nightly job (run by .github/workflows/league.yml).

    python run_league.py fetch  out.json     # read entries from the private sheet (needs ENTRIES_CSV_URL)
    python run_league.py freeze entries.json # at a lab's close: freeze each person's latest entry
                                             #   lock "close":  run the code once on the data up to the close
                                             #   lock "submit": run it on the data as it stood when it was
                                             #   submitted (from git history), so the number is the one the
                                             #   page showed them
    python run_league.py settle              # after the target week: score the frozen numbers against reality

Only names, numbers and short status lines are written to docs/data/. Code never is.
"""
import csv, datetime as dt, io, json, os, subprocess, sys, tempfile, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
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


def _versions(rel, when, n=2):
    """The last n commits of a data file at or before `when`, newest first (needs full git history)."""
    out = subprocess.run(["git", "log", f"-{n}", f"--before={when.isoformat()}", "--format=%H", "--", rel],
                         cwd=REPO, capture_output=True, text=True, check=True).stdout.split()
    return out


def _file_at(rel, commit, dest):
    data = subprocess.run(["git", "show", f"{commit}:{rel}"], cwd=REPO, capture_output=True, check=True).stdout
    open(dest, "wb").write(data)


def _run(code, hpath, lab, tmp):
    work = tempfile.mkdtemp(dir=tmp)
    try:
        p = subprocess.run([sys.executable, "-I", os.path.join(HERE, "child.py")],
                           input=json.dumps({"code": code, "history": hpath, "lab": lab,
                                             "cache": os.path.join(tmp, "downloads"),
                                             "questions": os.path.join(HERE, "..", "..", "docs")}),
                           capture_output=True, text=True, timeout=TIME_LIMIT, cwd=work,
                           env={"PATH": os.environ.get("PATH", ""), "HOME": work}, start_new_session=True)
        tail = p.stdout.rsplit("@@RESULT@@", 1)
        return json.loads(tail[1]) if len(tail) == 2 else {"status": "error", "error": "the code stopped Python"}
    except subprocess.TimeoutExpired:
        return {"status": f"time limit ({TIME_LIMIT} s)"}


def _history_file(lab, df, target, tmp, name):
    hist = Q.history(lab, df, target)
    hpath = os.path.join(tmp, name)
    hist.to_csv(hpath, index=False)
    os.chmod(hpath, 0o444)
    return hist, hpath


def _at_submit(lab, q, target, when, code, shown, tmp):
    """Run the code on the data the page had when it was submitted. The page loads the data when it
    opens, so a page opened just before a nightly update still had the older file: try the newest
    version at that time first, then the one before, and keep the one that gives the number shown."""
    rel = f"docs/data/{q['data']}"
    tmp, tried = tempfile.mkdtemp(dir=tmp), []           # one folder per entry: history files are read-only
    for i, commit in enumerate(_versions(rel, when)):
        raw = os.path.join(tmp, f"raw_{i}.csv")
        _file_at(rel, commit, raw)
        hist, hpath = _history_file(lab, Q.load(lab, raw), target, tmp, f"history_{i}.csv")
        res = {**_run(code, hpath, lab, tmp), "history_rows": len(hist)}
        tried.append(res)
        if shown is None or (res["status"] == "ok" and abs(res["value"] - shown) < 0.0015):
            return {**res, "checked": shown is not None}
    if not tried:
        return {"status": "no data at that time"}
    return {**tried[0], "checked": False}      # the shown number was not reproduced: use the re-run


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
                latest[name.lower()] = (when, name, r["Code"], r.get("Prediction", ""))
        df = Q.load(lab, os.path.join(DATA, q["data"]))
        target = os.environ.get("LEAGUE_TARGET") or Q.target(lab, df)
        if isinstance(target, str) and "," in target:
            target = tuple(target.split(","))
        results = []
        with tempfile.TemporaryDirectory() as tmp:
            hist, hpath = _history_file(lab, df, target, tmp, "history.csv")
            for key in sorted(latest):
                when, name, code, shown = latest[key]
                if q["lock"] == "submit":
                    try:
                        shown = float(shown)
                    except ValueError:
                        shown = None
                    res = _at_submit(lab, q, target, when, code, shown, tmp)
                else:
                    res = _run(code, hpath, lab, tmp)
                results.append({"name": name, "submitted": when.strftime("%Y-%m-%d %H:%M"), **res})
        json.dump({"lab": lab, "title": q["title"], "close": close_day, "target": target, "lock": q["lock"],
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
