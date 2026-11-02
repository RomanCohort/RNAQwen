import sys, torch
sys.path.insert(0,'src'); sys.path.insert(0,'scripts')
import rnaqwen
import importlib.util
spec=importlib.util.spec_from_file_location("rc","scripts/50_routeC_e2e.py")
rc=importlib.util.module_from_spec(spec); spec.loader.exec_module(rc)
DEV=rc.DEV
tkc=torch.load(rc.FSQ_CKPT, map_location="cpu", weights_only=False)
LAT,LEV=int(tkc["latent"]),int(tkc["levels"])
rc.LATENT,rc.LEVELS,rc.VOCAB=LAT,LEV,LEV**LAT
print(f"[cfg] LATENT={LAT} LEVELS={LEV} VOCAB={rc.VOCAB}")
art="runs/B_seq8/artifacts"
fsq=torch.load(f"{art}/fsq.pt",map_location="cpu",weights_only=False)
emb=torch.load(f"{art}/code_emb.pt",map_location="cpu",weights_only=False)
print(f"[art] fsq keys {list(fsq.keys())}  state 张量 {len(fsq['state'])}")
print(f"[art] code_emb {tuple(emb.shape)}  dtype {emb.dtype}  行范数均值 {float(emb.norm(dim=1).mean()):.4f}")
# 只建 FSQ + 码表，不建 LLM
class Mini(torch.nn.Module):
    def __init__(s):
        super().__init__()
        s.fsq=rc.FSQCode(768,LAT,LEV,4096,pool=4).to(DEV)
        s.rna_emb=torch.nn.Embedding(rc.VOCAB,4096).to(DEV)
        s.base_vocab=151936
    def rna_vectors(s,x,perm=None,tau=6.0):
        code,q_ste,t,q=s.fsq(x)
        idx=perm[code] if perm is not None else code
        v_hard=s.rna_emb(idx)
        grid=s.fsq.grid.to(q_ste.device)
        d2=torch.cdist(q_ste.float(),grid.float()).pow(2)
        soft=torch.softmax(-d2/tau,dim=-1)
        v_soft=soft@s.rna_emb.weight
        return v_hard+v_soft-v_soft.detach(),code,t,q
m=Mini(); m.fsq.load_state_dict(fsq["state"],strict=False)
m.rna_emb.weight.data.copy_(emb.to(m.rna_emb.weight.dtype))
from rnaqwen.models.encoder import RnaEncoder
enc=RnaEncoder("models/ernierna",dtype=torch.float32).to(DEV).eval()
for s in ("AUAAAGGAAGUUCCGGAA", "GGGCUAGCUAGCUA"*3):
    with torch.no_grad():
        o=enc([s]); h=o.hidden.float(); mk=o.mask.squeeze(-1)>0
        if mk.shape[1]!=h.shape[1]: mk=torch.ones(h.shape[:2],dtype=torch.bool,device=h.device)
        hh=h[0][mk[0]]
        if hh.shape[0]==len(s)+2: hh=hh[1:-1]
        v,code,_,_=m.rna_vectors(hh.to(DEV,dtype=torch.float32))
    print(f"  seq len {len(s):>3}  编码位置 {hh.shape[0]:>3}  码数 {v.shape[0]:>3}  "
          f"码 id 范围 [{int(code.min())},{int(code.max())}]  向量 {tuple(v.shape)}")
    assert hh.shape[0]==len(s), "编码位置数 != 序列长度"
print("✅ 序列 -> 码 这一环通了")
