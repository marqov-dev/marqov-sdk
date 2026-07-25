#!/usr/bin/env python3
"""Pre-release link check: every URL in shipped source + docs must resolve.

A wrong or dead URL in a published package is caught by NO unit test — nothing
imports or fetches a prose URL. This is that gate. Run it before tagging a release.

    python tools/check_urls.py            # scan the default surfaces
    python tools/check_urls.py PATH ...   # scan specific files/dirs

Classification — the only distinction that matters is *resolves vs. doesn't*:

  FAIL  — the host does not resolve (DNS failure). The URL points at a domain that
          does not exist. This is the real bug: 0.3.0 shipped a default
          ``platform.marqov.com`` that did not resolve. Release-blocking (exit 1).

  WARN  — resolved but not a clean 2xx/3xx: a 4xx/5xx (a provider API returns 401/403
          to an unauthenticated GET *by design*; or a page path is wrong), OR a
          transient network failure (timeout / connection refused). WARN never blocks
          — a provider outage on release day must not stop your tag. Eyeball each.
          (No hardcoded host allowlist: it rots the moment a new provider is added.)

  OK    — 2xx/3xx (redirects followed).
"""
from __future__ import annotations

import re
import socket
import sys
import urllib.error
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_SURFACES = ["marqov", "README.md", "CHANGELOG.md", "CONTRIBUTING.md", "SECURITY.md", "docs"]
_SUFFIXES = {".py", ".md", ".rst", ".txt", ".toml", ".cfg"}
_URL_RE = re.compile(r"https?://[A-Za-z0-9._~:/?#@!$&*+,;=%()-]+")
_TRAILING = ")].,>\"'`"
_SKIP = re.compile(r"localhost|127\.0\.0\.1|example\.(com|org)|\{")
_TIMEOUT = 10


def _iter_files(surfaces: list[str]):
    for s in surfaces:
        p = _ROOT / s
        if p.is_file():
            yield p
        elif p.is_dir():
            for f in p.rglob("*"):
                if f.is_file() and f.suffix in _SUFFIXES:
                    yield f


def _urls(surfaces: list[str]):
    seen: set[str] = set()
    for f in _iter_files(surfaces):
        for m in _URL_RE.findall(f.read_text(encoding="utf-8", errors="replace")):
            u = m.rstrip(_TRAILING)
            if u not in seen and not _SKIP.search(u):
                seen.add(u)
                yield u


def _check(url: str) -> tuple[str, str]:
    """Return (kind, detail). kind is 'ok' | 'warn' | 'fail'.

    fail == the host does not resolve (DNS). Everything else that isn't a clean
    2xx/3xx is a warn — including transient network errors, which get one retry first.
    """
    req = urllib.request.Request(url, headers={"User-Agent": "marqov-link-check"})
    for attempt in (1, 2):
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
                code = r.status
                return ("ok", str(code)) if 200 <= code < 400 else ("warn", f"HTTP {code}")
        except urllib.error.HTTPError as e:
            # Resolved — the host is real, it just returned 4xx/5xx (auth-gated API,
            # or a wrong path). Never release-blocking; a human eyeballs it.
            return ("warn", f"HTTP {e.code}")
        except urllib.error.URLError as e:
            reason = e.reason
            if isinstance(reason, socket.gaierror):
                return ("fail", f"host does not resolve: {reason}")
            if attempt == 1:  # transient (timeout / refused) — retry once
                continue
            return ("warn", f"unreachable (transient?): {reason}")
        except (socket.timeout, TimeoutError) as e:
            if attempt == 1:
                continue
            return ("warn", f"timeout: {e}")
    return ("warn", "unreachable after retry")


def main(argv: list[str]) -> int:
    surfaces = argv[1:] or _DEFAULT_SURFACES
    fail, warn, ok = [], [], []
    for u in sorted(_urls(surfaces)):
        kind, detail = _check(u)
        {"ok": ok, "warn": warn, "fail": fail}[kind].append((u, detail))

    for u, d in ok:
        print(f"ok    {d:<26} {u}")
    for u, d in warn:
        print(f"WARN  {d:<26} {u}")
    for u, d in fail:
        print(f"FAIL  {d:<26} {u}")

    if fail:
        print(f"\nFAIL: {len(fail)} URL(s) point at a host that does not resolve — "
              "fix before release.", file=sys.stderr)
        return 1
    print(f"\nOK: {len(ok)} clean, {len(warn)} warn (eyeball), 0 non-resolving.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
