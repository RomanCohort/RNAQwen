import importlib, sys
print(sys.version)
for m in ["aiohttp","requests","datasets","pandas","pyarrow","numpy","tqdm","yaml","huggingface_hub","transformers","httpx"]:
    try:
        mod = importlib.import_module(m)
        print("%-20s %s" % (m, getattr(mod, "__version__", "?")))
    except Exception:
        print("%-20s MISSING" % m)
