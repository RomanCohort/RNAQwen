import torch
d = torch.load(r"runs/immune_gene_cls/feats_kmer4_k3_n104920.pt", map_location="cpu", weights_only=False)
n = 20000
sub = dict(X=d["X"][:n], Xmean=d["Xmean"][:n], meta=d["meta"][:n],
           tag="kmer4", k=d["k"], win=d["win"])
out = r"runs/immune_gene_cls/feats_kmer4_20k.pt"
torch.save(sub, out)
print("saved", out, tuple(sub["X"].shape))
