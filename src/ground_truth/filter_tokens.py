"""Token filtering per GlitchCleaner's protocol (following Land & Bartolo 2024):

  SPECIAL      - control tokens (<s>, <unk>, chat markers, added tokens)
  UNDECODABLE  - decoding yields U+FFFD (partial UTF-8 byte tokens)
  UNREACHABLE  - decode->re-encode never reproduces the original id
  candidate    - everything else; these go into the repetition sweep

GlitchCleaner reports 229 filtered for Llama-2 (3 special + 2 undecodable + 224
unreachable). Compare our counts against that in the thesis.
"""
from ..common.model_utils import token_str


def classify_vocab(tok, limit: int | None = None) -> list[tuple[int, str, str]]:
    """Return [(token_id, token_string, category)] for the whole vocabulary."""
    n = len(tok)
    if limit:
        n = min(n, limit)
    special = set(tok.all_special_ids or [])
    try:
        special |= set(tok.get_added_vocab().values())
    except Exception:
        pass

    out = []
    for tid in range(n):
        if tid in special:
            out.append((tid, token_str(tok, tid), "special"))
            continue
        s = token_str(tok, tid)
        if "�" in s:
            out.append((tid, s, "undecodable"))
            continue
        re_ids = tok.encode(s, add_special_tokens=False)
        if tid not in re_ids:
            out.append((tid, s, "unreachable"))
        else:
            out.append((tid, s, "candidate"))
    return out
