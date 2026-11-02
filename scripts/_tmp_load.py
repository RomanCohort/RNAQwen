import sys, time, torch
sys.path.insert(0,'src')
import rnaqwen
from rnaqwen.qwen36 import load_qwen36_text
t0=time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
try:
    model, text, lmh, stats = load_qwen36_text(mode="int4", smoke=-1, log=log)
    log(f"✅ 全量装载成功 {stats}")
    # 真前向 + 反向
    model.eval()
    dev="cuda"
    ids=torch.tensor([[248045,8678,198,248046,248045,846,198]], device=dev)
    with torch.no_grad():
        o=model(input_ids=ids); l=o.logits.float()
        log(f"input_ids 前向 logits {tuple(l.shape)} 有限 {bool(torch.isfinite(l).all())}")
    emb=model.model.language_model.embed_tokens(ids).detach()
    v=torch.randn(3, emb.shape[-1], device=dev, dtype=emb.dtype, requires_grad=True)
    inp=emb.clone(); inp[0,2:5]=v.to(inp.dtype); inp.requires_grad_(True)
    o2=model(inputs_embeds=inp); l2=o2.logits.float()
    l2.pow(2).mean().backward()
    log(f"inputs_embeds 前向 logits {tuple(l2.shape)} 有限 {bool(torch.isfinite(l2).all())}")
    log(f"反向 g_v 有限 {bool(torch.isfinite(v.grad).all())} |g_v|max {float(v.grad.abs().max()):.3e}")
except Exception as e:
    import traceback; traceback.print_exc()
    log(f"** {type(e).__name__}: {e}")
