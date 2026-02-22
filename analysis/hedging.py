import re

# Simple surface patterns for epistemic hedging

HEDGE_PATTERNS = [
    r"\bi\s+think\b",
    r"\bi\s+feel\b",
    r"\bseems?\b",
    r"\bappears?\b",
    r"\bmight\b",
    r"\bmay\b",
    r"\bcould\b",
    r"\bprobably\b",
    r"\bpossibly\b",
    r"\blikely\b",
    r"\bin\s+my\s+experience\b"
]


def count_hedges(text):
    """
    Count how many hedge markers appear in a span.
    """
    text = text.lower()
    count = 0

    for pat in HEDGE_PATTERNS:
        count += len(re.findall(pat, text))

    return count


def has_hedge(text):
    """
    Return True if at least one hedge marker is found.
    """
    return count_hedges(text) > 0