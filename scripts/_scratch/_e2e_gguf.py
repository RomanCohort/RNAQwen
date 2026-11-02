import importlib.util, json, sys, urllib.request
from pathlib import Path

GG = Path(r"C:\qwen-api\RNAQwen\gguf")
spec = importlib.util.spec_from_file_location("rp", GG / "rna_prompt.py")
rp = importlib.util.module_from_spec(spec)
sys.modules["rp"] = rp
spec.loader.exec_module(rp)

TASK = Path(r"C:\qwen-api\RNAQwen\data\processed\rnaqa_seqtask_full.jsonl")
SEQF = Path(r"C:\qwen-api\RNAQwen\data\processed\rnaqa_n6_merged.jsonl")
WANT = {"What are the first 4 nucleotides of this RNA?",
        "What are the first 6 nucleotides of this RNA?"}
MAX_NT = 120

cand = []
with open(TASK, encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r.get("question") in WANT:
            cand.append(r)
        if len(cand) >= 40:
            break

need = {c["uid"] for c in cand}
seq_of = {}
with open(SEQF, encoding="utf-8") as f:
    for line in f:
        if not need:
            break
        r = json.loads(line)
        if r["uid"] in need:
            seq_of[r["uid"]] = r["sequence"]
            need.discard(r["uid"])

full = [c for c in cand if c["uid"] in seq_of]
short = [c for c in full if len(seq_of[c["uid"]]) <= MAX_NT]
picks = (short if len(short) >= 4 else sorted(full, key=lambda c: len(seq_of[c["uid"]])))[:4]
print(f"[data] cand {len(cand)}  with-seq {len(full)}  <= {MAX_NT}nt {len(short)}  "
      f"picked {len(picks)}", flush=True)


def ask(ids, n=20):
    body = json.dumps({"prompt": ids, "n_predict": n, "temperature": 0.0,
                       "cache_prompt": False, "n_keep": 0}).encode()
    req = urllib.request.Request("http://127.0.0.1:8099/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as f:
        return json.loads(f.read().decode("utf-8"))


CT = json.loads((GG / "code_tokens.json").read_text(encoding="utf-8"))


def tok(text):
    body = json.dumps({"content": text}).encode()
    req = urllib.request.Request("http://127.0.0.1:8099/tokenize", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as f:
        return json.loads(f.read().decode("utf-8"))["tokens"]


def text_form_check(ids):
    code_ids = [i for i in ids if i >= CT["base_vocab"]]
    text = "".join(f"<R{i - CT['base_vocab']}>" for i in code_ids)
    got = tok(text)
    same = got == code_ids
    print(f"[tok] 文本形式 {text[:36]}... -> {len(got)} id，与码信道路径"
          f"{'一致' if same else '** 不一致'}  头 {got[:4]}", flush=True)
    return same


ok = 0
tf = None
print("first call loads ERNIE ...", flush=True)
for c in picks:
    seq = seq_of[c["uid"]].strip().upper()
    ids, n = rp.build(seq, c["question"])
    if tf is None:
        tf = text_form_check(ids)
    out = ask(ids)
    txt = (out.get("content") or "").strip()
    gold = c["answer"].strip().upper()
    hit = bool(txt) and (txt.upper().startswith(gold) or gold in txt.upper())
    ok += hit
    print("=" * 70)
    print("uid   :", c["uid"])
    print("seq   :", seq[:60], f"({len(seq)} nt, {n} codes)")
    print("q     :", c["question"])
    print("gold  :", gold)
    print("gguf  :", repr(txt))
    print("match :", hit, flush=True)
print(f"[e2e] {ok}/{len(picks)} hit  文本形式对齐 {tf}")
