import re
from bs4 import BeautifulSoup
from hashlib import sha1
from datetime import datetime, timezone
from urllib.parse import (
    urlparse, urlunparse, urljoin, parse_qsl, urlencode
)

# ---- Heuristics ----
JS_REQUIRED_PATTERNS = [
    r"javascript is required",
    r"enable javascript",
    r"please enable javascript",
    r"requires javascript",
]

TRACKING_QUERY_KEYS = {
    "gclid", "fbclid", "yclid", "msclkid",
    "ref", "ref_src", "source", "spm",
}
TRACKING_QUERY_PREFIXES = ("utm_",)

DEFAULT_EXCLUDE_PATH_KEYWORDS = [
    "/login", "/logout", "/signup", "/register",
    "/account", "/profile", "/settings",
    "/search", "/tag/", "/tags/", "/category/", "/categories/",
    "/share", "/comment", "/comments",
]

DEFAULT_DISALLOWED_EXTS = {
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg",
    ".mp4", ".mov", ".avi", ".mkv", ".mp3", ".wav",
    ".pdf", ".zip", ".rar", ".7z", ".tar", ".gz",
    ".css", ".js", ".json", ".xml",
    ".woff", ".woff2", ".ttf", ".otf",
}


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_url(url: str) -> str:
    """Basic normalization: strip spaces, drop fragments."""
    url = (url or "").strip()
    if not url:
        return ""
    p = urlparse(url)

    # drop fragment
    p = p._replace(fragment="")

    # normalize scheme/netloc a bit
    scheme = (p.scheme or "").lower()
    netloc = (p.netloc or "").strip()

    # remove default ports
    if netloc.endswith(":80") and scheme == "http":
        netloc = netloc[:-3]
    if netloc.endswith(":443") and scheme == "https":
        netloc = netloc[:-4]

    p = p._replace(scheme=scheme, netloc=netloc)
    return urlunparse(p)


def is_http_url(url: str) -> bool:
    p = urlparse(url)
    return p.scheme in ("http", "https")


def is_disallowed_scheme(href: str) -> bool:
    s = (href or "").strip().lower()
    return s.startswith(("mailto:", "tel:", "javascript:", "data:"))


def strip_tracking_query_params(url: str, keep_query_keys: set[str] | None = None, drop_all_query: bool = False) -> str:
    """
    Remove common tracking parameters.
    - keep_query_keys: if provided, keep only these keys (after dropping tracking keys)
    - drop_all_query: if True, remove entire query
    """
    p = urlparse(url)
    if drop_all_query:
        return urlunparse(p._replace(query=""))

    q = parse_qsl(p.query, keep_blank_values=True)

    filtered = []
    for k, v in q:
        kl = (k or "").lower()
        # drop trackers
        if kl in TRACKING_QUERY_KEYS:
            continue
        if any(kl.startswith(pref) for pref in TRACKING_QUERY_PREFIXES):
            continue

        filtered.append((k, v))

    if keep_query_keys is not None:
        keep_lower = {k.lower() for k in keep_query_keys}
        filtered = [(k, v) for (k, v) in filtered if (k or "").lower() in keep_lower]

    new_query = urlencode(filtered, doseq=True)
    return urlunparse(p._replace(query=new_query))


def host_matches_domain(host: str, domain: str) -> bool:
    """Match subdomains too: host == domain or host.endswith('.'+domain)."""
    host = (host or "").lower()
    if ":" in host:
        host = host.split(":", 1)[0]
    domain = (domain or "").lower().lstrip(".")
    if not host or not domain:
        return False

    # handle 'www.' gracefully
    if host.startswith("www."):
        host2 = host[4:]
    else:
        host2 = host

    if domain.startswith("www."):
        domain2 = domain[4:]
    else:
        domain2 = domain

    return host2 == domain2 or host2.endswith("." + domain2)


def is_disallowed_extension(url: str, disallowed_exts: set[str] | None = None) -> bool:
    disallowed_exts = disallowed_exts or DEFAULT_DISALLOWED_EXTS
    path = (urlparse(url).path or "").lower()
    return any(path.endswith(ext) for ext in disallowed_exts)


def matches_excluded_path(url: str, exclude_path_keywords: list[str] | None = None) -> bool:
    exclude_path_keywords = exclude_path_keywords or DEFAULT_EXCLUDE_PATH_KEYWORDS
    path = (urlparse(url).path or "").lower()
    return any(k in path for k in exclude_path_keywords)


def in_allowed_paths(url: str, allowed_path_prefixes: list[str] | None) -> bool:
    if not allowed_path_prefixes:
        return True
    path = (urlparse(url).path or "")
    return any(path.startswith(pref) for pref in allowed_path_prefixes)


def should_follow_url(
    url: str,
    allowed_domains: list[str] | None = None,
    allowed_path_prefixes: list[str] | None = None,
    exclude_path_keywords: list[str] | None = None,
    disallowed_exts: set[str] | None = None,
) -> bool:
    if not url:
        return False
    if not is_http_url(url):
        return False
    if is_disallowed_extension(url, disallowed_exts):
        return False
    if matches_excluded_path(url, exclude_path_keywords):
        return False

    if allowed_domains:
        host = (urlparse(url).hostname or "").lower()
        if not any(host_matches_domain(host, d) for d in allowed_domains):
            return False

    if not in_allowed_paths(url, allowed_path_prefixes):
        return False

    return True


def absolutize_and_clean_url(
    base_url: str,
    href: str,
    keep_query_keys: set[str] | None = None,
    drop_all_query: bool = False,
) -> str:
    """
    base_url + href -> absolute URL -> normalize -> strip trackers
    """
    href = (href or "").strip()
    if not href:
        return ""
    if is_disallowed_scheme(href):
        return ""

    abs_url = urljoin(base_url, href)
    abs_url = normalize_url(abs_url)
    if not abs_url:
        return ""
    abs_url = strip_tracking_query_params(abs_url, keep_query_keys=keep_query_keys, drop_all_query=drop_all_query)
    return abs_url


def looks_js_required(html: str, text: str) -> bool:
    blob = (html + "\n" + text).lower()
    return any(re.search(pat, blob) for pat in JS_REQUIRED_PATTERNS)


def compute_text_sha1(text: str) -> str:
    return sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def extract_title_and_text(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")

    title = ""
    if soup.title and soup.title.get_text(strip=True):
        title = soup.title.get_text(strip=True)

    # remove obvious noise
    for tag in soup(["script", "style", "noscript", "svg", "canvas"]):
        tag.decompose()
    for tag in soup(["header", "footer", "nav", "aside"]):
        tag.decompose()

    text = soup.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return title, text
