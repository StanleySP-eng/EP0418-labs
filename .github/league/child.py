"""Runs ONE entry. Reads {"code": ..., "history": path} on stdin, prints one JSON line last."""
import contextlib, io, json, math, sys, traceback
import pandas as pd

job = json.loads(sys.stdin.read())
sys.path.insert(0, job["questions"])
import league_questions as Q                     # the same loader the page uses, so dtypes match
hist = Q.load(job["lab"], job["history"])
sys.path.pop(0)
buf, res = io.StringIO(), {}
try:
    ns = {"__name__": "__main__"}
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        exec(compile(job["code"], "<your code>", "exec"), ns)
        val = ns["predict"](hist.copy())
    if isinstance(val, bool) or val is None or isinstance(val, (str, list, tuple, dict, pd.Series, pd.DataFrame)):
        res = {"status": "wrong output type"}
    else:
        v = float(val)
        res = {"status": "ok", "value": round(v, 3)} if math.isfinite(v) else {"status": "wrong output type"}
except BaseException as e:                                    # noqa: BLE001
    res = {"status": "error", "error": "".join(traceback.format_exception_only(type(e), e)).strip()[:300]}
sys.__stdout__.write("\n@@RESULT@@" + json.dumps(res) + "\n")
