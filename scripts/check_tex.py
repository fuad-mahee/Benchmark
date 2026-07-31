"""Structural sanity check for a .tex file, for machines with no TeX toolchain.

  python scripts/check_tex.py docs/benchmark_report.tex

Checks environment nesting, tabular/longtable cell counts against the declared
column spec, undefined \\ref targets, and brace balance. Not a substitute for
pdflatex -- it catches the errors that hand-editing tables actually produces, and
nothing else. Exit code is non-zero if anything failed, so it can gate a commit.
"""
import re
import sys
from pathlib import Path

_default = Path(__file__).resolve().parents[1] / "docs" / "benchmark_report.tex"
path = Path(sys.argv[1]) if len(sys.argv) > 1 else _default
src = path.read_text(encoding="utf-8")
lines = src.split("\n")
print(f"checking {path}")

# 1. environment nesting
stack, errs = [], []
for i, ln in enumerate(lines, 1):
    for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", ln):
        kind, env = m.group(1), m.group(2)
        if kind == "begin":
            stack.append((env, i))
        else:
            if not stack:
                errs.append(f"line {i}: \\end{{{env}}} with empty stack")
            elif stack[-1][0] != env:
                errs.append(f"line {i}: \\end{{{env}}} closes \\begin{{{stack[-1][0]}}} "
                            f"from line {stack[-1][1]}")
                stack.pop()
            else:
                stack.pop()
for env, i in stack:
    errs.append(f"unclosed \\begin{{{env}}} from line {i}")
print("ENVIRONMENTS:", "OK" if not errs else "PROBLEMS")
for e in errs:
    print("  ", e)


def grab_braced(s, start):
    """Return the brace-balanced group beginning at s[start] == '{'."""
    d, i = 0, start
    while i < len(s):
        if s[i] == "{":
            d += 1
        elif s[i] == "}":
            d -= 1
            if d == 0:
                return s[start + 1:i], i + 1
        i += 1
    return None, len(s)


def n_cols(spec):
    spec = re.sub(r"[pmb]\{(?:[^{}]|\{[^{}]*\})*\}", "P", spec)
    spec = re.sub(r"[<>]\{(?:[^{}]|\{[^{}]*\})*\}", "", spec)
    spec = re.sub(r"\*\{(\d+)\}\{([^{}]*)\}", lambda m: m.group(2) * int(m.group(1)), spec)
    spec = re.sub(r"[|@!\s]", "", spec)
    return sum(1 for c in spec if c in "lcrP")


# 2. tabular/longtable row cell counts (rows are logical: joined until \\)
bad = []
for env in ("tabular", "longtable"):
    for m in re.finditer(r"\\begin\{" + env + r"\}(?:\[[^\]]*\])?", src):
        j = m.end()
        while j < len(src) and src[j] != "{":
            j += 1
        spec, k = grab_braced(src, j)
        want = n_cols(spec)
        end = src.find(r"\end{" + env + "}", k)
        body = src[k:end]
        line0 = src[:k].count("\n") + 1
        for raw in body.split(r"\\"):
            if r"\multicolumn" in raw or not raw.strip():
                continue
            clean = re.sub(r"(?<!\\)%.*", "", raw)
            clean = re.sub(r"\\(toprule|midrule|bottomrule|cmidrule)(\([^)]*\))?"
                           r"(\{[^}]*\})?", "", clean)
            if not clean.strip():
                continue
            got = len(re.split(r"(?<!\\)&", clean))
            if got != want:
                ln = line0 + body[:body.find(raw)].count("\n")
                bad.append(f"  ~line {ln}: {got} cells, spec {want} ({spec}) | "
                           f"{' '.join(clean.split())[:70]}")
print("TABULAR COLUMNS:", "OK" if not bad else "PROBLEMS")
for b in bad:
    print(b)

# 3. labels vs refs
labels = set(re.findall(r"\\label\{([^}]+)\}", src))
refs = set(re.findall(r"\\ref\{([^}]+)\}", src))
missing = refs - labels
print("REFS:", "OK" if not missing else f"UNDEFINED: {sorted(missing)}")

# 4. brace balance, reported per top-level block so drift is locatable
depth, first_bad = 0, None
for i, ln in enumerate(lines, 1):
    s = re.sub(r"(?<!\\)%.*$", "", ln)
    s = s.replace(r"\{", "").replace(r"\}", "")
    depth += s.count("{") - s.count("}")
    if depth < 0 and first_bad is None:
        first_bad = i
    if depth == 0 and not ln.strip():
        continue
print("BRACES:", "OK" if depth == 0 else f"UNBALANCED net {depth:+d}"
      + (f", first negative at line {first_bad}" if first_bad else ""))

if depth:
    d = 0
    for i, ln in enumerate(lines, 1):
        s = re.sub(r"(?<!\\)%.*$", "", ln).replace(r"\{", "").replace(r"\}", "")
        before = d
        d += s.count("{") - s.count("}")
        if before == 0 and d != 0 and not re.search(r"\\(begin|end)\{", ln):
            print(f"    line {i} opens {d:+d} and does not close: {ln.strip()[:74]}")

print("PENDING MARKERS:", src.count("[pending]"))
print("lines:", len(lines))
sys.exit(1 if (errs or bad or missing or depth) else 0)
