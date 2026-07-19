"""Token filtering per GlitchCleaner's protocol.

IMPORTANT PROVENANCE: this mirrors the authors' own implementation
(third_party/GlitchCleaner/Fine-tuning/tokenfilter.py, itself following Land &
Bartolo 2024 "Fishing for Magikarp"). The crucial detail is the CONTEXT-ANCHORED
roundtrip: decode/encode each token behind a single-token prefix "«" so that
SentencePiece leading-space semantics don't distort the check. A naive
decode->re-encode (our first attempt) misclassifies ~39% of a Llama-family vocab
as UNREACHABLE; the anchored version reproduces the papers' scale (~hundreds).

Categories:
  special      - control-looking tokens (<s>, [BOS], ...) + tokenizer-registered specials
  undecodable  - decoding fails / yields U+FFFD
  unreachable  - anchored encode(decode(id)) != [id]: no input string ever produces this id
  candidate    - testable; goes into the repetition sweep
"""

PREFIX = "«"


class ContextualCodec:
    """Decode/encode a single token id in a neutral one-token context."""

    def __init__(self, tok):
        ids = tok.encode(PREFIX, add_special_tokens=False)
        self.tok = tok
        self.prefix_id = ids[0]
        self.single = len(ids) == 1

    def decode(self, token_id: int) -> str:
        decoded = self.tok.decode([self.prefix_id, token_id], skip_special_tokens=False)
        if decoded and decoded[0] == " ":  # e.g. Mistral prepends one (upstream comment)
            decoded = decoded[1:]
        assert decoded.startswith(PREFIX), (
            f"decoded {decoded!r} does not start with prefix for id {token_id}"
        )
        return decoded[len(PREFIX):]

    def encode(self, s: str) -> list[int]:
        tokens = self.tok.encode(PREFIX + s, add_special_tokens=False)
        if not tokens or tokens[0] != self.prefix_id:
            return [0, 1]  # prefix got merged -> counts as not-reproducing (upstream behavior)
        return tokens[1:]


def classify_vocab(tok, limit: int | None = None) -> list[tuple[int, str, str]]:
    """Return [(token_id, token_string, category)] for the whole vocabulary."""
    codec = ContextualCodec(tok)
    registered_special = set(tok.all_special_ids or [])
    n = len(tok)
    if limit:
        n = min(n, limit)

    out = []
    for tid in range(n):
        try:
            s = codec.decode(tid)
        except Exception:
            out.append((tid, "", "undecodable"))
            continue
        if codec.encode(s) == [tid]:
            if tid in registered_special or (
                len(s) >= 3 and s[0] in "[<" and s[-1] in "]>" and any(c.isalpha() for c in s)
            ):
                cat = "special"
            elif "�" in s:
                cat = "undecodable"
            else:
                cat = "candidate"
        else:
            cat = "unreachable"
        out.append((tid, s, cat))
    return out
