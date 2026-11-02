import traceback
print("B: transformers first, then AutoTokenizer")
try:
    import transformers
    from transformers import AutoTokenizer
    print("B OK")
except Exception:
    traceback.print_exc()
print("C: torch.distributed")
try:
    import torch.distributed as d
    print("C OK")
except Exception:
    traceback.print_exc()
