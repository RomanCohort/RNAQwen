"""本地 multimolecule 支不支持 ernierna？能不能直接接上？"""
import sys, importlib
from pathlib import Path

ROOT = Path(r"C:\qwen-api\RNAQwen")
sys.path.insert(0, str(ROOT / "src"))
import rnaqwen  # noqa: F401  （必须：它给 torch.distributed 打了 shim）

import multimolecule
print("multimolecule 版本 :", getattr(multimolecule, "__version__", "?"))
print("路径              :", Path(multimolecule.__file__).parent)

import pkgutil
mods = sorted(m.name for m in pkgutil.iter_modules(multimolecule.models.__path__))
print("\n=== multimolecule.models 下所有模型 ===")
for m in mods:
    mark = "  <== 我们要的" if "ernie" in m.lower() else ""
    print("   ", m, mark)

print("\n=== 尝试导入 ernierna ===")
for name in ["multimolecule.models.ernierna", "multimolecule.models.ernierna.modeling_ernierna",
             "multimolecule.models.ernierna.configuration_ernierna"]:
    try:
        m = importlib.import_module(name)
        print("  [OK] ", name)
        if hasattr(m, "__all__"):
            print("        ", getattr(m, "__all__"))
        else:
            print("        ", [n for n in dir(m) if n.startswith("Ernie")])
    except Exception as e:
        print("  [--] ", name, "->", type(e).__name__, str(e)[:90])

print("\n=== 本地 models 目录里有什么 ===")
for p in sorted((ROOT / "models").iterdir()):
    if p.is_dir():
        n = sum(1 for _ in p.rglob("*") if _.is_file())
        print("   %-26s %d 个文件" % (p.name, n))
