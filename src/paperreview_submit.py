import json, urllib.request, os, urllib.parse, time

ROOT = r"D:\Research\On-Progress\PIET Conference"
PDF = os.path.join(ROOT, "manuscript", "manuscript.pdf")
BASE = "https://paperreview.ai"
LOGSD = os.path.join(ROOT, "results", "logs")
tm = json.load(open(os.path.join(LOGSD, "tempmail.json")))
email = tm["email"]
venue = "ICCICA"

data = json.dumps({"filename": "manuscript.pdf", "venue": venue}).encode()
req = urllib.request.Request(BASE + "/api/get-upload-url", data=data, headers={"Content-Type": "application/json"})
resp = json.loads(urllib.request.urlopen(req).read())
print("step1 ok, s3_key:", resp.get("s3_key"))
presigned_url = resp["presigned_url"]
presigned_fields = resp["presigned_fields"]
s3_key = resp["s3_key"]

boundary = "----WebKitFormBoundary" + os.urandom(16).hex()
parts = []
for k, v in presigned_fields.items():
    parts.append("--" + boundary + "\r\n" + 'Content-Disposition: form-data; name="' + k + '"\r\n\r\n' + str(v) + "\r\n")
parts.append("--" + boundary + "\r\n" + 'Content-Disposition: form-data; name="file"; filename="manuscript.pdf"\r\nContent-Type: application/pdf\r\n\r\n')
body = "".join(parts).encode() + open(PDF, "rb").read() + ("\r\n--" + boundary + "--\r\n").encode()
req = urllib.request.Request(presigned_url, data=body, headers={"Content-Type": "multipart/form-data; boundary=" + boundary})
up = urllib.request.urlopen(req)
print("upload status:", up.status)

data = "s3_key=" + urllib.parse.quote(s3_key) + "&venue=" + urllib.parse.quote(venue) + "&email=" + urllib.parse.quote(email)
req = urllib.request.Request(BASE + "/api/confirm-upload", data=data.encode(), headers={"Content-Type": "application/x-www-form-urlencoded"})
resp = json.loads(urllib.request.urlopen(req).read())
print("confirm:", json.dumps(resp))
token = resp.get("token")
rnd = int(tm.get("round", 1)) + 1
if tm.get("token"):
    json.dump({"email": email, "password": tm.get("password"), "token": tm["token"],
               "round": rnd - 1, "s3_key": tm.get("s3_key"),
               "review_url": BASE + "/review?token=" + str(tm["token"])},
              open(os.path.join(LOGSD, "paperreview_token_round%d.json" % (rnd - 1)), "w"), indent=2)
json.dump({"email": email, "password": tm.get("password"), "token": token, "s3_key": s3_key,
           "review_url": BASE + "/review?token=" + str(token), "round": rnd},
          open(os.path.join(LOGSD, "tempmail.json"), "w"), indent=2)
print("TOKEN:", token)
print("URL:", BASE + "/review?token=" + str(token))
print("ROUND:", rnd)
