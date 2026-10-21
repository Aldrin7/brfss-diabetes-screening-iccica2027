import json, urllib.request, time, os, sys

ROOT = r"D:\Research\On-Progress\PIET Conference"
LOGSD = os.path.join(ROOT, "results", "logs")
BASE = "https://paperreview.ai"

tokens = {}
cur = json.load(open(os.path.join(LOGSD, "tempmail.json")))
tokens[int(cur.get("round", 3))] = cur["token"]
for f in os.listdir(LOGSD):
    if f.startswith("paperreview_token_round"):
        d = json.load(open(os.path.join(LOGSD, f)))
        tokens[int(d.get("round", 1))] = d["token"]
print("watching tokens:", tokens, flush=True)

deadline = time.time() + float(sys.argv[1] if len(sys.argv) > 1 else 500)
while time.time() < deadline:
    for rnd, tok in list(tokens.items()):
        try:
            req = urllib.request.Request(BASE + "/api/review/" + tok)
            with urllib.request.urlopen(req) as r:
                data = json.loads(r.read())
            if data.get("content"):
                json.dump(data, open(os.path.join(LOGSD, "paperreview_review_round%d.json" % rnd), "w"), indent=2)
                md = "# Review (round %d): %s\n\n" % (rnd, data.get("title", ""))
                md += "Venue: %s | Submitted: %s | Reviewed: %s\n\n" % (
                    data.get("venue", ""), data.get("submission_date", ""), data.get("review_date", ""))
                md += data.get("content", "")
                open(os.path.join(ROOT, "paperreview_review_round%d.md" % rnd), "w", encoding="utf-8").write(md)
                print("ROUND %d READY -> paperreview_review_round%d.md (%d chars)" % (rnd, rnd, len(md)), flush=True)
                del tokens[rnd]
        except urllib.error.HTTPError as e:
            print("round %d HTTP %s" % (rnd, e.code), flush=True)
        except Exception as e:
            print("round %d err %s" % (rnd, e), flush=True)
    if not tokens:
        print("ALL ROUNDS READY", flush=True)
        raise SystemExit(0)
    time.sleep(40)
print("still pending:", list(tokens.keys()), flush=True)
