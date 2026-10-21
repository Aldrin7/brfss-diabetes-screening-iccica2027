import re
p = r"D:\Research\On-Progress\PIET Conference\manuscript.tex"
t = open(p, encoding="utf-8").read()
b = re.findall(r"\\begin\{(\w+)\}", t)
e = re.findall(r"\\end\{(\w+)\}", t)
from collections import Counter
cb, ce = Counter(b), Counter(e)
print("env mismatch:", {k: (cb.get(k, 0) - ce.get(k, 0)) for k in set(cb) | set(ce) if cb.get(k, 0) != ce.get(k, 0)})
cites = set(re.findall(r"\\cite\{([\w,]+)\}", t))
keys = set()
for c in cites:
    for k in c.split(","):
        keys.add(k.strip())
bibs = set(re.findall(r"\\bibitem\{(\w+)\}", t))
print("undefined cites:", sorted(keys - bibs))
print("unused bibs:", sorted(bibs - keys))
refs = set(re.findall(r"\\ref\{(\w+)\}", t))
labels = set(re.findall(r"\\label\{(\w+)\}", t))
print("undefined refs:", sorted(refs - labels))
print("unused labels:", sorted(labels - refs))
print("tables:", t.count("\\begin{table}"), "figures:", t.count("\\begin{figure}"), "bibitems:", t.count("\\bibitem"))
plain = re.sub(r"\\[a-zA-Z]+(\{[^{}]*\})*", " ", t)
plain = plain.replace("{", " ").replace("}", " ").replace("&", " ")
print("approx words:", len(plain.split()))
subs = re.findall(r"\\subsection\{([^}]*)\}", t)
print("\nSubsection order:")
for i, s in enumerate(subs):
    print(" ", i + 1, s)
