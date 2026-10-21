"""Build the double-blind (anonymized) submission PDF for ICCICA 2027.

ICCICA 2027 uses anonymous review, so the submitted PDF must not carry author
names, affiliations, or any link that identifies the author. This script
generates `build/manuscript_anon.pdf` from `manuscript.tex` by:

  * replacing the author block with an anonymous placeholder,
  * removing the repository URL (the GitHub account name identifies the author),
  * anonymizing the repository bibliography entry.

The named version (`build/manuscript.pdf`) stays the camera-ready source.
"""
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, "manuscript", "build")
SRC = os.path.join(ROOT, "manuscript", "manuscript.tex")
ANON = os.path.join(BUILD, "manuscript_anon.tex")

ANON_AUTHOR = """\\author{\\IEEEauthorblockN{Anonymous Submission}
\\IEEEauthorblockA{\\textit{Author details withheld for double-blind review}}}"""

REPO_URL = "https://github.com/Aldrin7/brfss-diabetes-screening-iccica2027"


def anonymize(text):
    # 1. author block
    text = re.sub(r"\\author\{.*?\}\n\s*\\maketitle",
                  lambda m: ANON_AUTHOR + "\n\n\\maketitle", text, flags=re.S)
    # 2. repository url in the bibliography
    text = text.replace(REPO_URL, "[repository withheld for double-blind review]")
    # 3. repository bibliography entry -> anonymous
    text = re.sub(
        r"\\bibitem\{repo\}\s*A\. Author,[^\n]*",
        r"\\bibitem{repo} Anonymous, ``BRFSS diabetes screening: code, run manifests, and "
        r"SHA-256 artifact manifest,'' repository withheld for double-blind review.",
        text,
    )
    # 4. data-availability sentence
    text = text.replace(
        "The analysis code, run manifests, logged results, and a SHA-256 hash manifest of every "
        "artifact are held in a private repository for review and will be made public on acceptance "
        "\\cite{repo}.",
        "The analysis code, run manifests, logged results, and a SHA-256 hash manifest of every "
        "artifact are withheld during review and will be released on acceptance \\cite{repo}.",
    )
    # 5. any residual self-identifying strings
    for token in ("Aldrin7", "Aldrin"):
        text = text.replace(token, "Anonymous")
    return text


def main():
    os.makedirs(BUILD, exist_ok=True)
    for d in (os.path.join(ROOT, "assets", "figures"), os.path.join(ROOT, "results", "figures")):
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.lower().endswith(".png"):
                    shutil.copy2(os.path.join(d, f), os.path.join(BUILD, f))

    text = anonymize(open(SRC, encoding="utf-8").read())
    open(ANON, "w", encoding="utf-8").write(text)

    exe = shutil.which("pdflatex")
    if not exe:
        raise SystemExit("pdflatex not found on PATH")
    for _ in range(2):
        p = subprocess.run([exe, "-interaction=nonstopmode", "manuscript_anon.tex"],
                           cwd=BUILD, capture_output=True, text=True, errors="replace")
    log = os.path.join(BUILD, "manuscript_anon.log")
    if not os.path.exists(log):
        raise SystemExit("no log produced:\n" + p.stdout[-2000:])
    log_txt = open(log, encoding="utf-8", errors="replace").read()
    m = re.findall(r"Output written on .*?\((\d+) pages?", log_txt)
    pages = int(m[-1]) if m else -1
    print("anonymized PDF: %s (%d pages)" % (os.path.join(BUILD, "manuscript_anon.pdf"), pages))
    for bad in ("Aldrin7", "Author Name", "[email]", "[Department]"):
        if bad in text:
            print("WARNING: residual identifier %r in anonymized source" % bad)
    for line in log_txt.splitlines():
        if "Warning" in line and "Citation" in line:
            print("  " + line.strip()[:140])
    return 0


if __name__ == "__main__":
    sys.exit(main())
