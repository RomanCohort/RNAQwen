import torch
def stats(path, name):
    d = torch.load(path, map_location="cpu", weights_only=False)
    w = (d["code_emb"] if isinstance(d, dict) else d).float()
    n = w.norm(dim=1)
    wn = w / n.clamp(min=1e-8).unsqueeze(1)
    # sample 512 rows to keep it cheap
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(wn.shape[0], generator=g)[:512]
    c = wn[idx] @ wn[idx].T
    off = c[~torch.eye(len(idx), dtype=torch.bool)]
    print(f"{name}: shape={tuple(w.shape)}")
    print(f"   row norm  mean {n.mean():.4f}  std {n.std():.4f}")
    print(f"   pairwise cos  mean {off.mean():+.4f}  p50 {off.median():+.4f}  p95 {off.quantile(0.95):+.4f}")
stats(r"checkpoints/fsq_stream/code_emb_contentinit.pt", "NEW (stream)")
stats(r"checkpoints/fsq_full/code_emb_contentinit_d2048.pt", "OLD (fsq_full d2048)")
