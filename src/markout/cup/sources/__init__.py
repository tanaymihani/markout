"""Free, read-only reference data. Every network call goes through `http.Http`
(cached, rate-limited per host, identifying user agent). No keys, no paid sources."""

from markout.cup.sources.base import ExternalQuote  # noqa: F401
