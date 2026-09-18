"""Polite HTTP for every online sense the brain has.

Requests are written curl-style (Host first, then a short fixed header set)
because some content delivery edges, arXiv's included, answer 406 to the
shape Python's urllib writes. Each host gets a minimum interval between
requests so the brain never hammers a site.
"""

from __future__ import annotations

import gzip
import http.client
import json
import os
import ssl
import threading
import time
import urllib.parse
from typing import Any

USER_AGENT = "bigbrain/0.1 (trading knowledge brain; +https://github.com/muhammadhamkah/Big-Brain-Time)"

# Seconds between requests to the same host. Conservative on purpose.
HOST_INTERVALS: dict[str, float] = {
    "export.arxiv.org": 3.0,
    "arxiv.org": 3.0,
    "www.reddit.com": 2.0,
    "oauth.reddit.com": 2.0,
    "api.github.com": 1.0,
    "raw.githubusercontent.com": 1.0,
}
DEFAULT_INTERVAL = 1.0

_last_request: dict[str, float] = {}
_lock = threading.Lock()


class HTTPStatusError(Exception):
    def __init__(self, status: int, url: str, headers: dict[str, str], body: bytes) -> None:
        super().__init__(f"HTTP {status} from {url}")
        self.status, self.url, self.headers, self.body = status, url, headers, body


def _wait_for_host(host: str) -> None:
    """Reserve the next slot for ``host`` and sleep until it. Threads on different hosts never wait on each other."""
    with _lock:
        interval = HOST_INTERVALS.get(host, DEFAULT_INTERVAL)
        now = time.monotonic()
        slot = max(now, _last_request.get(host, 0.0) + interval)
        _last_request[host] = slot
    if slot > now:
        time.sleep(slot - now)


def _proxy_for(host: str) -> tuple[str, int] | None:
    """Honour HTTPS_PROXY / NO_PROXY the way urllib does, since http.client does not."""
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if not proxy:
        return None
    for skip in (os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or "").split(","):
        skip = skip.strip().lstrip("*").lstrip(".")
        if skip and (host == skip or host.endswith("." + skip)):
            return None
    parts = urllib.parse.urlsplit(proxy if "://" in proxy else f"http://{proxy}")
    return parts.hostname or "", parts.port or 3128


def _connect(host: str, port: int, timeout: float) -> http.client.HTTPSConnection:
    context = ssl.create_default_context()
    proxy = _proxy_for(host)
    if proxy:
        conn = http.client.HTTPSConnection(proxy[0], proxy[1], timeout=timeout, context=context)
        conn.set_tunnel(host, port)
        return conn
    return http.client.HTTPSConnection(host, port, timeout=timeout, context=context)


def http_get(
    url: str,
    timeout: float = 30.0,
    max_redirects: int = 3,
    headers: dict[str, str] | None = None,
    user_agent: str = USER_AGENT,
) -> bytes:
    """GET ``url`` and return the body. Raises HTTPStatusError on non-200."""
    for _ in range(max_redirects + 1):
        parts = urllib.parse.urlsplit(url)
        if parts.scheme != "https":
            raise ValueError(f"only https URLs are supported: {url}")
        host = parts.hostname or ""
        _wait_for_host(host)
        conn = _connect(host, parts.port or 443, timeout)
        try:
            path = parts.path or "/"
            if parts.query:
                path += f"?{parts.query}"
            conn.putrequest("GET", path, skip_host=True, skip_accept_encoding=True)
            conn.putheader("Host", host)
            conn.putheader("User-Agent", user_agent)
            conn.putheader("Accept", (headers or {}).get("Accept", "*/*"))
            conn.putheader("Accept-Encoding", "gzip")
            for name, value in (headers or {}).items():
                if name.lower() not in ("host", "user-agent", "accept", "accept-encoding"):
                    conn.putheader(name, value)
            conn.endheaders()
            resp = conn.getresponse()
            body = resp.read()
            resp_headers = {k: v for k, v in resp.getheaders()}
        finally:
            conn.close()
        if resp.status in (301, 302, 303, 307, 308) and resp.getheader("Location"):
            url = urllib.parse.urljoin(url, resp.getheader("Location"))
            if url.startswith("http://"):  # never drop to plain http; nearly every host serves https too
                url = "https://" + url[len("http://"):]
            continue
        if resp_headers.get("Content-Encoding", "").lower() == "gzip" or resp_headers.get("content-encoding", "").lower() == "gzip":
            body = gzip.decompress(body)
        if resp.status != 200:
            raise HTTPStatusError(resp.status, url, resp_headers, body)
        return body
    raise HTTPStatusError(310, url, {}, b"too many redirects")


def http_json(url: str, headers: dict[str, str] | None = None, timeout: float = 30.0) -> Any:
    merged = {"Accept": "application/json"}
    merged.update(headers or {})
    return json.loads(http_get(url, timeout=timeout, headers=merged).decode("utf-8"))


def http_text(url: str, headers: dict[str, str] | None = None, timeout: float = 30.0) -> str:
    return http_get(url, timeout=timeout, headers=headers).decode("utf-8", errors="replace")
