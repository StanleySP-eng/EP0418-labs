"""
Downloads inside predict(history): fetched once, then reused.

Public APIs such as data.gov.sg refuse a burst of requests (HTTP 429). The league page
runs an entry four times (three past cases and the entry), and at the close every
entry runs one after another from the same machine, so the same address is asked for
again and again. install() wraps requests and urllib.request so that each address is
downloaded once and every later call gets the saved copy; a 429 waits and retries.
At the close the saved copies sit in one folder shared by all entries, so everyone's
function sees the same data.
"""
import hashlib, io, json, os, time

_MEM = {}
_DIR = None
WAITS = (2, 4, 6, 8)                 # seconds between retries after a 429 (20 s at most)
UA = "Mozilla/5.0 (EP0418 Prediction League)"


def _key(url):
    return hashlib.sha256(url.encode()).hexdigest()[:32]


def _load(url):
    k = _key(url)
    if k in _MEM:
        return _MEM[k]
    if _DIR:
        path = os.path.join(_DIR, k + ".json")
        if os.path.exists(path):
            meta = json.load(open(path))
            meta["body"] = bytes.fromhex(meta["body"])
            _MEM[k] = meta
            return meta
    return None


def _save(url, status, headers, body):
    meta = {"url": url, "status": status, "headers": dict(headers), "body": body}
    _MEM[_key(url)] = meta
    if _DIR and status == 200:
        tmp = os.path.join(_DIR, _key(url) + ".tmp")
        json.dump({**meta, "body": body.hex()}, open(tmp, "w"))
        os.replace(tmp, os.path.join(_DIR, _key(url) + ".json"))
    return meta


class _Reply(io.BytesIO):
    """What urlopen returns: read(), status, getcode(), headers, and with-blocks."""
    def __init__(self, meta):
        super().__init__(meta["body"])
        self.status, self.headers, self.url = meta["status"], meta["headers"], meta["url"]

    def getcode(self):
        return self.status

    def geturl(self):
        return self.url


def install(cache_dir=None):
    global _DIR
    _DIR = cache_dir
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    import urllib.request, urllib.error
    real_urlopen = urllib.request.urlopen

    def urlopen(url, *args, **kwargs):
        if isinstance(url, str):
            url = urllib.request.Request(url)
        if not url.has_header("User-agent"):     # data.gov.sg answers urllib's default name with 403
            url.add_header("User-Agent", UA)
        full = url.full_url
        meta = _load(full)
        if meta is None:
            for i, wait in enumerate((0,) + WAITS):
                time.sleep(wait)
                try:
                    with real_urlopen(url, *args, **kwargs) as r:
                        meta = _save(full, r.status, r.headers.items(), r.read())
                    break
                except urllib.error.HTTPError as e:
                    if e.code != 429 or i == len(WAITS):
                        raise
        return _Reply(meta)

    urllib.request.urlopen = urlopen

    try:
        import requests
    except ImportError:
        return
    real_request = requests.sessions.Session.request

    def request(self, method, url, *args, **kwargs):
        if method.upper() != "GET":
            return real_request(self, method, url, *args, **kwargs)
        req = requests.Request("GET", url, params=kwargs.get("params")).prepare()
        meta = _load(req.url)
        if meta is None:
            for i, wait in enumerate((0,) + WAITS):
                time.sleep(wait)
                r = real_request(self, method, url, *args, **kwargs)
                if r.status_code != 429 or i == len(WAITS):
                    break
            meta = _save(req.url, r.status_code, r.headers.items(), r.content)
        out = requests.models.Response()
        out.status_code, out._content, out.url = meta["status"], meta["body"], meta["url"]
        out.headers = requests.structures.CaseInsensitiveDict(meta["headers"])
        out.encoding = "utf-8"
        return out

    requests.sessions.Session.request = request
