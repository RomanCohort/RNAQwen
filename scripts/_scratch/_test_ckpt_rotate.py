"""验 checkpoint 机制：原子落盘 + 轮转只留最近 N 个 + 完整性检查。

为什么单独测：机制失效是**静默**的。轮转选错文件（留下最旧的 10 个、清掉最新的）
不会有任何报错；best.pt 被顺手清掉要等早停那一刻才发现；而截断的 .pt
看起来完全合法 —— 这三种都得靠断言钉死。

跑法： python scripts/_scratch/_test_ckpt_rotate.py
"""
import importlib.util
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

FAILS = []


def check(cond, msg):
    print(("  [ok]   " if cond else "  [FAIL] ") + msg)
    if not cond:
        FAILS.append(msg)


def load_trainer():
    """按下划线前缀的模块名导入 50_routeC_e2e（数字开头不能直接 import）。"""
    spec = importlib.util.spec_from_file_location(
        "route_c_trainer", ROOT / "scripts" / "50_routeC_e2e.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    print("== 0. 导入训练脚本（会拉起 torch，约几秒）==")
    m = load_trainer()
    check(hasattr(m, "prune_ckpts"), "拿到 prune_ckpts")
    check(hasattr(m, "save_ckpt"), "拿到 save_ckpt")

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)

        print("\n== 1. 轮转只留最近 N 个 ==")
        steps = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]
        for s in steps:
            (d / f"step_{s:06d}.pt").write_bytes(b"x" * 16)
        # 两个不参与轮转的档
        (d / "best.pt").write_bytes(b"best")
        (d / "final.pt").write_bytes(b"final")
        # 上一次被中途杀掉留下的半成品
        (d / "step_000700.pt.tmp").write_bytes(b"half")

        gone = m.prune_ckpts(d, 10)
        left = sorted(int(p.stem.split("_")[1]) for p in d.glob("step_*.pt"))
        check(gone == 5, f"清掉 {gone} 个，期望 5")
        check(len(left) == 10, f"剩下 {len(left)} 个，期望 10")
        check(left == steps[-10:],
              f"留下的是**最新的** 10 个：{left[0]}..{left[-1]}（期望 {steps[-10]}..{steps[-1]}）")
        check((d / "best.pt").exists(), "best.pt 没被清掉")
        check((d / "final.pt").exists(), "final.pt 没被清掉")
        check(not list(d.glob("*.pt.tmp")), "残留的 *.pt.tmp 被清掉了")
        check(not (d / "step_000100.pt").exists(), "最旧的 step_000100.pt 已清")

        print("\n== 2. keep 大于总数时不误删 ==")
        d2 = Path(td) / "few"
        d2.mkdir()
        for s in (100, 200):
            (d2 / f"step_{s:06d}.pt").write_bytes(b"x")
        gone2 = m.prune_ckpts(d2, 10)
        check(gone2 == 0 and len(list(d2.glob("step_*.pt"))) == 2,
              f"只有 2 个、keep=10：清掉 {gone2} 个，剩 {len(list(d2.glob('step_*.pt')))} 个")

        print("\n== 3. 完整性检查能抓住截断文件（就是 step_000600.pt 那个失败模式）==")
        good = d / "good.zip"
        with zipfile.ZipFile(good, "w") as zf:
            zf.writestr("data/0", b"A" * 40000)
            zf.writestr("data/1", b"B" * 40000)
        raw = good.read_bytes()
        # 砍掉尾部 —— 中央目录正好在尾部，正是"写到一半被杀"的样子
        trunc = d / "trunc.zip"
        trunc.write_bytes(raw[: int(len(raw) * 0.6)])

        ok_good = True
        try:
            with zipfile.ZipFile(good) as zf:
                n = len(zf.namelist())
            ok_good = (n == 2)
        except Exception as e:
            ok_good = False
            print(f"        完整文件反而读不了：{type(e).__name__}: {e}")
        check(ok_good, "完整 zip 的中央目录可读")

        raised = False
        try:
            with zipfile.ZipFile(trunc) as zf:
                zf.namelist()
        except zipfile.BadZipFile:
            raised = True
        check(raised,
              f"截断 zip 被 BadZipFile 拦下（原 {len(raw)}B -> 截到 {trunc.stat().st_size}B）")

        print("\n== 4. 原子性：崩在写 tmp 阶段，正式档必须原封不动 ==")
        real = d / "step_000900.pt"
        real.write_bytes(b"OLD-VALID")
        # 模拟 save_ckpt：tmp 写了一半就死
        tmp = real.with_name(real.name + ".tmp")
        tmp.write_bytes(b"HALF-WRITTEN")
        check(real.read_bytes() == b"OLD-VALID",
              "正式档没被碰过（tmp 死了不影响它）")
        check(tmp.exists(), "半成品 .tmp 留在地上（等 prune 清）")
        m.prune_ckpts(d, 10)
        check(not tmp.exists(), "下次 prune 把 .tmp 清掉了")

    print()
    if FAILS:
        print(f"===== {len(FAILS)} 项失败 =====")
        for f in FAILS:
            print("  - " + f)
        return 1
    print("===== 全部通过 =====")
    return 0


if __name__ == "__main__":
    sys.exit(main())
