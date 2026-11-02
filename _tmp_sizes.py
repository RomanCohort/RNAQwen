import json, urllib.request
req=urllib.request.Request("https://huggingface.co/api/models/multimolecule/rnafm?blobs=true", headers={"User-Agent":"Mozilla/5.0"})
with urllib.request.urlopen(req, timeout=45) as r: j=json.loads(r.read().decode())
for f in j.get("siblings",[]):
    print(f"{f.get('size',-1):>12} {f['rfilename']}")
print("sha:", j.get("sha"))
