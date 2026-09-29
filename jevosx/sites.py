"""Web site identity for saved logins: host names, https pages, and which hosts a saved login may be used on."""

from __future__ import annotations

import re
from urllib.parse import urlsplit

from .errors import JevOSXError

# Shared hosting: anyone can own a subdomain, so a login saved for one of these matches that exact host only.
SHARED_SUFFIXES = frozenset(
    {
        "github.io",
        "gitlab.io",
        "herokuapp.com",
        "vercel.app",
        "netlify.app",
        "pages.dev",
        "workers.dev",
        "web.app",
        "firebaseapp.com",
        "appspot.com",
        "azurewebsites.net",
        "cloudfront.net",
        "blogspot.com",
        "wordpress.com",
        "glitch.me",
        "onrender.com",
        "fly.dev",
        "surge.sh",
        "ngrok.io",
        "ngrok-free.app",
        "s3.amazonaws.com",
    }
)
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_HOST = re.compile(r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)*[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")


class SiteError(JevOSXError):
    """Not a usable website host name."""


def normalize_site(site: str) -> str:
    """'https://www.GitHub.com/login' → 'github.com'. Raises SiteError for anything that is not a host name."""
    text = site.strip().lower()
    if not text:
        raise SiteError("empty site")
    parts = urlsplit(text if "://" in text else f"https://{text}")
    host = (parts.hostname or "").rstrip(".")
    if host.startswith("www."):
        host = host[4:]
    if host not in LOCAL_HOSTS and (not _HOST.match(host) or "." not in host):
        raise SiteError(f"{site!r} is not a website host name (example: github.com)")
    return host


def page_host(url: str | None) -> str | None:
    """Host of a page URL when a saved login may be used there: https, or http on this Mac only."""
    if not url:
        return None
    try:
        parts = urlsplit(url.strip())
        host = (parts.hostname or "").lower().rstrip(".")
    except ValueError:
        return None
    if not host or parts.scheme not in ("https", "http"):
        return None
    if parts.scheme == "http" and host not in LOCAL_HOSTS:
        return None  # never send a password over plain http
    return host


def host_matches(saved: str, host: str) -> bool:
    """The saved host itself or a subdomain of it (never a look-alike such as github.com.evil.io)."""
    if host == saved or (host.startswith("www.") and host[4:] == saved):
        return True
    if saved in SHARED_SUFFIXES or any(saved.endswith("." + s) for s in SHARED_SUFFIXES):
        return False
    return host.endswith("." + saved)


def mask(username: str) -> str:
    """'matthew@gmail.com' → 'm•••@gmail.com'; 'octocat' → 'o•••t'. Enough for Jev to tell accounts apart."""
    name, at, domain = username.partition("@")
    if at:
        return f"{name[:1]}•••@{domain}"
    return f"{username[:1]}•••{username[-1:]}" if len(username) > 2 else "•••"
