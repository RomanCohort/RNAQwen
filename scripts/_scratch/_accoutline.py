import json
from pathlib import Path
p = Path(r"C:\qwen-api\RNAQwen\reports\acc_by_source_D_qwen36_sos_final.json")
d = json.loads(p.read_text(encoding="utf-8"))
def outline(o, pre="", depth=0):
    if depth > 2: return
    if isinstance(o, dict):
        for k, v in list(o.items())[:6]:
            print(f"{pre}{k}: {type(v).__name__}" + (f" len {len(v)}" if isinstance(v, (list, dict)) else f" = {v}" if not isinstance(v, (list, dict)) else ""))
            outline(v, pre + "  ", depth + 1)
    elif isinstance(o, list) and o:
        print(f"{pre}[0]: {type(o[0]).__name__}")
        outline(o[0], pre + "  ", depth + 1)
outline(d)
print("size", p.stat().st_size)
