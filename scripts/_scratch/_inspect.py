import json, torch, sys
from pathlib import Path
ROOT = Path(r"C:\qwen-api\RNAQwen")
BASE = Path(r"C:\qwen-api\models\Qwen3.6-35B-A3B")
cfg = json.load(open(BASE / "config.json", encoding="utf-8"))
tc = cfg["text_config"]
print("arch", cfg["architectures"], "tie", cfg.get("tie_word_embeddings"), "tie_txt", tc.get("tie_word_embeddings"))
print("vocab_size cfg", cfg.get("vocab_size"), tc.get("vocab_size"))
j = json.load(open(BASE / "tokenizer.json", encoding="utf-8"))
mv = j["model"]["vocab"]
at = j.get("added_tokens", [])
print("model.vocab", len(mv), "added", len(at))
mx = max(mv.values())
ids = sorted(mv.values())
print("vocab id range", ids[0], ids[-1], "contiguous", ids == list(range(len(ids))))
print("added id range", min(t["id"] for t in at), max(t["id"] for t in at))
print("has <RNA>", "<RNA>" in mv, any(t["content"] == "<RNA>" for t in at))
from tokenizers import Tokenizer
tk = Tokenizer.from_file(str(BASE / "tokenizer.json"))
v = tk.get_vocab()
print("tokenizers.get_vocab", len(v), "max id", max(v.values()))
print("enc Q: <RNA>", tk.encode("Q: <RNA>").ids)
print("enc </RNA>hi", tk.encode("</RNA>ACGU").ids)
tkobj = torch.load(ROOT / "checkpoints/fsq_full/tokenizer.pt", map_location="cpu", weights_only=False)
print("tok.pt keys", list(tkobj.keys())[:12] if isinstance(tkobj, dict) else type(tkobj))
ce = torch.load(ROOT / "runs/RNAQwen-preview/code_emb.pt", map_location="cpu", weights_only=False) if (ROOT/"runs/RNAQwen-preview/code_emb.pt").exists() else None
print("code_emb", None if ce is None else (tuple(ce.shape), ce.dtype))
