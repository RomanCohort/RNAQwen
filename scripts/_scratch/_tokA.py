import traceback
print("A: import torch then transformers.AutoTokenizer")
try:
    import torch
    from transformers import AutoTokenizer
    print("A OK")
except Exception:
    traceback.print_exc()
