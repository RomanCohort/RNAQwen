import json, shutil, sys
from pathlib import Path

ROOT = Path(r"C:\qwen-api\RNAQwen")
BASE = Path(r"C:\qwen-api\models\Qwen3.6-35B-A3B")
WORK = ROOT / "work/gguf/toktest"
SMALL = ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
         "merges.txt", "chat_template.jinja", "generation_config.json",
         "preprocessor_config.json"]
N_CODES = 4096
BASE_VOCAB = 248320

sys.path.insert(0, str(ROOT / "src"))
import rnaqwen  # noqa: F401
from transformers import AutoTokenizer


def stage(tag):
    d = WORK / tag
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for n in SMALL:
        if (BASE / n).is_file():
            shutil.copyfile(BASE / n, d / n)
    return d


def add(d, ids):
    tj = json.loads((d / "tokenizer.json").read_text(encoding="utf-8"))
    tj["added_tokens"] = [t for t in tj["added_tokens"] if t["id"] < ids[0][0]]
    for i, content in ids:
        tj["added_tokens"].append({"id": i, "content": content, "single_word": False,
                                   "lstrip": False, "rstrip": False,
                                   "normalized": False, "special": False})
    (d / "tokenizer.json").write_text(json.dumps(tj, ensure_ascii=False), encoding="utf-8")
    tcd = json.loads((d / "tokenizer_config.json").read_text(encoding="utf-8"))
    dec = tcd.get("added_tokens_decoder")
    if isinstance(dec, dict):
        for i, content in ids:
            dec[str(i)] = {"content": content, "single_word": False, "lstrip": False,
                           "rstrip": False, "normalized": False, "special": False}
        (d / "tokenizer_config.json").write_text(
            json.dumps(tcd, ensure_ascii=False), encoding="utf-8")


def probe(d, tag, rows):
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    cfg["text_config"]["vocab_size"] = rows
    (d / "config.json").write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    t = AutoTokenizer.from_pretrained(d)
    v = t.vocab
    print(f"[{tag}] config vocab {rows}  len(vocab) {len(v)}  max id {max(v.values())}")
    for c in (0, 1, 242, 243, 4095):
        print(f"    <R{c}> -> {v.get('<R' + str(c) + '>')}")
    return len(v), max(v.values())


n_base = len(AutoTokenizer.from_pretrained(BASE).vocab)
print("base tokenizer len(vocab)", n_base)

a = stage("A")
add(a, [(i, f"<|pad{i}|>") for i in range(n_base, BASE_VOCAB)]
       + [(BASE_VOCAB + c, f"<R{c}>") for c in range(N_CODES)])
probe(a, "A-filler", BASE_VOCAB + N_CODES)

b = stage("B")
add(b, [(n_base + c, f"<R{c}>") for c in range(N_CODES)])
probe(b, "B-contig", n_base + N_CODES)
