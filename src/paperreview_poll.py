import json, urllib.request, time, os, sys

ROOT = r"D:\Research\On-Progress\PIET Conference"
LOGSD = os.path.join(ROOT, "results", "logs")
BASE = "https://paperreview.ai"
tm = json.load(open(os.path.join(LOGSD, "tempmail.json")))
token = tm["token"]
rnd = tm.get("round", 2)
out_json = os.path.join(LOGSD, "paperreview_review_round%d.json" % rnd)
out_md = os.path.join(ROOT, "paperreview_review_round%d.md" % rnd)

deadline = time.time() + float(sys.argv[1] if len(sys.argv) > 1 else 540)
print("polling token", token, flush=True)
while time.time() < deadline:
    try:
        req = urllib.request.Request(BASE + "/api/review/" + token)
        with urllib.request.urlopen(req) as r:
            data = json.loads(r.read())
        if "content" in data and data.get("content"):
            json.dump(data, open(out_json, "w"), indent=2)
            md = "# Review (round %d): %s\n\n" % (rnd, data.get("title", ""))
            md += "Venue: %s | Submitted: %s | Reviewed: %s\n\n" % (
                data.get("venue", ""), data.get("submission_date", ""), data.get("review_date", ""))
            md += data.get("content", "")
            open(out_md, "w", encoding="utf-8").write(md)
            print("READY -- saved", out_md, flush=True)
            raise SystemExit(0)
        else:
            print("waiting:", str(data)[:120], flush=True)
    except urllib.error.HTTPError as e:
        print("HTTP", e.code, flush=True)
    time.sleep(40)
print("NOT READY yet. Token:", token, flush=True)
