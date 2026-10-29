"""Feature extraction module for URL classification.

Extracts 25 lexical and structural URL features in the exact order
specified by feature_names.json. Performs zero network requests.
"""

from __future__ import annotations

import re
import urllib.parse
from typing import Any, Dict, List, Tuple

FEATURE_NAMES: List[str] = [
    "url_length",
    "hostname_length",
    "path_length",
    "query_length",
    "dot_count",
    "hyphen_count",
    "underscore_count",
    "slash_count",
    "question_mark_count",
    "equal_count",
    "at_count",
    "ampersand_count",
    "percent_count",
    "digit_count",
    "letter_count",
    "digit_ratio",
    "letter_ratio",
    "is_https",
    "is_ip_address",
    "subdomain_count",
    "has_suspicious_word",
    "suspicious_word_count",
    "has_port",
    "has_fragment",
    "has_query",
]

SUSPICIOUS_WORDS: List[str] = [
    "login",
    "verify",
    "secure",
    "account",
    "update",
    "bank",
    "signin",
    "password",
    "confirm",
    "payment",
    "wallet",
]

IP_REGEX = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")
SCHEME_REGEX = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")


def extract_features(raw_url: str) -> Tuple[List[float], Dict[str, Any]]:
    """Extract exactly 25 URL features in the order of FEATURE_NAMES.

    Args:
        raw_url: The URL string submitted by the user.

    Returns:
        A tuple of:
          - list of 25 float feature values (exact order for model input)
          - dictionary mapping feature name to formatted feature value
    """
    url = raw_url.strip()
    if not url:
        raise ValueError("URL string cannot be empty.")

    # URL Normalization: If no scheme is present, prepend http:// for parsing
    if not SCHEME_REGEX.match(url):
        parse_url = "http://" + url
    else:
        parse_url = url

    try:
        parsed = urllib.parse.urlparse(parse_url)
    except Exception as exc:
        raise ValueError(f"Failed to parse URL: {exc}") from exc

    hostname = (parsed.hostname or "").lower()
    path = parsed.path or ""
    query = parsed.query or ""
    fragment = parsed.fragment or ""

    # 1. url_length
    url_length = len(url)

    # 2. hostname_length
    hostname_length = len(hostname)

    # 3. path_length
    path_length = len(path)

    # 4. query_length
    query_length = len(query)

    # Character counts on the raw URL string
    dot_count = url.count(".")
    hyphen_count = url.count("-")
    underscore_count = url.count("_")
    slash_count = url.count("/")
    question_mark_count = url.count("?")
    equal_count = url.count("=")
    at_count = url.count("@")
    ampersand_count = url.count("&")
    percent_count = url.count("%")

    # Digits and letters
    digit_count = sum(1 for c in url if c.isdigit())
    letter_count = sum(1 for c in url if c.isalpha())

    # Ratios
    digit_ratio = (digit_count / url_length) if url_length > 0 else 0.0
    letter_ratio = (letter_count / url_length) if url_length > 0 else 0.0

    # 18. is_https (1 if scheme starts with https://, 0 otherwise)
    is_https = 1 if url.lower().startswith("https://") else 0

    # 19. is_ip_address (1 if hostname matches IPv4 pattern)
    is_ip_address = 1 if bool(IP_REGEX.match(hostname)) else 0

    # 20. subdomain_count: max(len(hostname.split(".")) - 2, 0)
    subdomain_count = max(len(hostname.split(".")) - 2, 0)

    # 21 & 22. suspicious words
    url_lower = url.lower()
    suspicious_word_count = sum(url_lower.count(word) for word in SUSPICIOUS_WORDS)
    has_suspicious_word = 1 if suspicious_word_count > 0 else 0

    # 23. has_port
    try:
        has_port = 1 if (parsed.port is not None) else 0
    except Exception:
        has_port = 0

    # 24. has_fragment
    has_fragment = 1 if bool(fragment) else 0

    # 25. has_query
    has_query = 1 if bool(query) else 0

    feature_dict: Dict[str, Any] = {
        "url_length": float(url_length),
        "hostname_length": float(hostname_length),
        "path_length": float(path_length),
        "query_length": float(query_length),
        "dot_count": float(dot_count),
        "hyphen_count": float(hyphen_count),
        "underscore_count": float(underscore_count),
        "slash_count": float(slash_count),
        "question_mark_count": float(question_mark_count),
        "equal_count": float(equal_count),
        "at_count": float(at_count),
        "ampersand_count": float(ampersand_count),
        "percent_count": float(percent_count),
        "digit_count": float(digit_count),
        "letter_count": float(letter_count),
        "digit_ratio": float(round(digit_ratio, 4)),
        "letter_ratio": float(round(letter_ratio, 4)),
        "is_https": float(is_https),
        "is_ip_address": float(is_ip_address),
        "subdomain_count": float(subdomain_count),
        "has_suspicious_word": float(has_suspicious_word),
        "suspicious_word_count": float(suspicious_word_count),
        "has_port": float(has_port),
        "has_fragment": float(has_fragment),
        "has_query": float(has_query),
    }

    feature_vector: List[float] = [feature_dict[name] for name in FEATURE_NAMES]
    return feature_vector, feature_dict
