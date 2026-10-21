"""Compile manuscript.tex with the real IEEEtran template and report the page count.

The camera-ready artifact for ICCICA 2027 must be a true IEEEtran build
(US Letter, two-column, 10pt), not the ReportLab review mock in
src/make_pdf.py.  ICCICA 2027 allows 6 pages including figures, tables and
references; up to 2 extra pages are permitted at INR 1000/page.

Usage:
    python src/compile_ieee.py            # build and report page count
    python src/compile_ieee.py --max 6    # fail loudly if over the limit
    python src/compile_ieee.py --anon     # also build an anonymized copy
"""

import argparse
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, "manuscript", "build")
TEX = os.path.join(ROOT, "manuscript", "manuscript.tex")
FIGDIRS = [os.path.join(ROOT, "assets", "figures"), os.path.join(ROOT, "results", "figures")]


def find_pdflatex():
    for exe in ("pdflatex", "latexmk"):
        path = shutil.which(exe)
        if path:
            return path
    raise SystemExit("pdflatex/latexmk not found on PATH (MiKTeX required)")


def stage(figures=True):
    os.makedirs(BUILD, exist_ok=True)
    shutil.copy2(TEX, os.path.join(BUILD, "manuscript.tex"))
    if figures:
        for d in FIGDIRS:
            if not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                if f.lower().endswith(".png"):
                    shutil.copy2(os.path.join(d, f), os.path.join(BUILD, f))


def run_pdflatex(name="manuscript", passes=2):
    exe = find_pdflatex()
    for _ in range(passes):
        p = subprocess.run(
            [exe, "-interaction=nonstopmode", name + ".tex"],
            cwd=BUILD, capture_output=True, text=True, errors="replace",
        )
    log = os.path.join(BUILD, name + ".log")
    if not os.path.exists(log):
        raise SystemExit("pdflatex produced no log; see stdout:\n" + p.stdout[-3000:])
    return open(log, encoding="utf-8", errors="replace").read()


def page_count(log):
    m = re.findall(r"Output written on .*?\((\d+) pages?", log)
    if m:
        return int(m[-1])
    m = re.findall(r"\[(\d+)\]", log)
    return int(m[-1]) if m else -1


def warnings(log):
    w = []
    for line in log.splitlines():
        if re.search(r"(Overfull|Underfull|LaTeX Warning)", line):
            w.append(line.strip()[:160])
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=6)
    ap.add_argument("--anon", action="store_true",
                    help="build an author-anonymized copy for double-blind review")
    ap.add_argument("--keep", action="store_true", help="keep intermediate build files")
    args = ap.parse_args()

    stage()
    log = run_pdflatex()
    pages = page_count(log)
    out = os.path.join(BUILD, "manuscript.pdf")
    print("IEEEtran build: %s" % out)
    print("pages: %d  (limit %d, over-length charge applies beyond it)" % (pages, args.max))
    over = pages - args.max
    if over > 0:
        print("OVER LIMIT by %d page(s) -> trim needed" % over)
    elif over < 0:
        print("room to spare: %d page(s)" % (-over))
    w = warnings(log)
    if w:
        print("--- typesetting warnings (%d) ---" % len(w))
        for line in w[:40]:
            print("  " + line)

    if args.anon:
        src = os.path.join(BUILD, "manuscript.tex")
        t = open(src, encoding="utf-8").read()
        t = re.sub(r"\\author\{.*?\}\s*", "", t, flags=re.S)
        t = re.sub(r"\\maketitle", r"\\maketitle", t)
        t = t.replace("[Author Name]", "Author").replace("[Department]", "Department")
        t = t.replace("[Institution]", "Institution").replace("[City], [Country]", "City, Country")
        t = t.replace("[email]", "email")
        open(os.path.join(BUILD, "manuscript_anon.tex"), "w", encoding="utf-8").write(t)
        log2 = run_pdflatex("manuscript_anon")
        print("anonymized build: %s (%d pages)" % (
            os.path.join(BUILD, "manuscript_anon.pdf"), page_count(log2)))
        for name in ("manuscript.aux", "manuscript_anon.aux", "manuscript_anon.out",
                     "manuscript_anon.bbl", "manuscript.bbl", "manuscript.out",
                     "manuscript.log", "manuscript_anon.log", "manuscript.toc"):
            p = os.path.join(BUILD, name)
            if os.path.exists(p) and not args.keep:
                try:
                    os.remove(p)
                except OSError:
                    pass
    return 0 if pages <= args.max else 2


if __name__ == "__main__":
    sys.exit(main())
