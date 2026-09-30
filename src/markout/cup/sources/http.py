"""A polite HTTP client: on-disk JSON cache with a TTL, at most one request per host per
`min_interval` seconds, a clear user agent, and no retries that hammer a failing host."""

from __future__ import annotations

import hashlib
import json
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from markout.paths import DATA

CACHE = DATA / "cup" / "cache"
USER_AGENT = "markout-research/0.1 (personal, read-only; github.com/markout)"


class Http:
    def __init__(self, cache_dir: Path = CACHE, min_interval: float = 1.0, ttl: float = 600.0, timeout: float = 20.0):
        self.cache_dir, self.min_interval, self.ttl, self.timeout = Path(cache_dir), min_interval, ttl, timeout
        self.last: dict[str, float] = {}
        self.lock = threading.Lock()

    def _key(self, url: str) -> Path:
        return self.cache_dir / f"{hashlib.sha256(url.encode()).hexdigest()[:24]}.json"

    def get_json(self, url: str, params: dict | None = None, ttl: float | None = None, cache: bool = True) -> Any:
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        ttl = self.ttl if ttl is None else ttl
        path = self._key(url)
        if cache and path.exists() and time.time() - path.stat().st_mtime < ttl:
            return json.loads(path.read_text())["body"]
        host = urllib.parse.urlparse(url).netloc
        with self.lock:
            wait = self.min_interval - (time.monotonic() - self.last.get(host, 0.0))
            if wait > 0:
                time.sleep(wait)
            self.last[host] = time.monotonic()
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:  # noqa: S310 (fixed https hosts)
            body = json.loads(r.read().decode())
        if cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"url": url, "fetched": time.time(), "body": body}))
        return body

    def prune(self, max_age: float = 86400.0) -> int:
        """Delete cache files older than `max_age` seconds (the disk is small)."""
        n = 0
        for f in self.cache_dir.glob("*.json"):
            if time.time() - f.stat().st_mtime > max_age:
                f.unlink()
                n += 1
        return n


DEFAULT = Http()
