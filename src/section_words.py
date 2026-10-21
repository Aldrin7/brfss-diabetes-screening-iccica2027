import re
t = open(r"D:\Research\On-Progress\PIET Conference\manuscript.tex", encoding="utf-8").read()
# strip environments we don't count as prose (tables, figures, bibliography)
t2 = re.sub(r"\\begin\{table\}.*?\\end\{table\}", " ", t, flags=re.S)
t2 = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", " ", t2, flags=re.S)
t2 = re.sub(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", " ", t2, flags=re.S)
parts = re.split(r"(\\section\*?\{[^}]*\}|\\subsection\*?\{[^}]*\})", t2)
def wc(s):
    s = re.sub(r"\\[a-zA-Z]+(\{[^{}]*\})*", " ", s)
    s = s.replace("{", " ").replace("}", " ").replace("$", " ")
    return len(s.split())
cur = "PREAMBLE"
tot = 0
for p in parts:
    p = p.strip()
    if not p:
        continue
    if p.startswith("\\section") or p.startswith("\\subsection"):
        cur = re.sub(r"\\section\*?\{|\\subsection\*?\{", "", p).rstrip("}").strip()
    else:
        n = wc(p)
        tot += n
        if n > 5:
            print("%5d  %s" % (n, cur))
print("total prose words (excl. tables/figures/refs):", tot)
