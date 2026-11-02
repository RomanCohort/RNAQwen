import sys, time
sys.path.insert(0, r"C:\qwen-api\RNAQwen\src")
import rnaqwen.env.shim as shim
print("[shim] patched:", ", ".join(shim.report()) or "(nothing)")
t0=time.time()
from multimolecule.models.rnafm import RnaFmModel, RnaTokenizer
print("[import] multimolecule OK in %.1fs" % (time.time()-t0))
tok = RnaTokenizer.from_pretrained(r"C:\qwen-api\RNAQwen\models\rnafm")
print("[tok] class:", type(tok).__name__, "vocab:", tok.vocab_size)
enc = tok("ACGUACGUACGU")
print("[tok] ids:", enc["input_ids"])
