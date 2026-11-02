"""验 50_routeC_e2e.py 末尾 _hard_exit 包装的退出语义。

为什么值得单独测：这个包装存在的唯一理由是「解释器收尾会挂」。真挂起来要等 ROCm
上下文销毁，没法在单测里复现。但它的**机制**可以复现 —— 用一个 atexit 钩子模拟
「收尾阶段卡住」：

  raise SystemExit(...)   -> atexit 会跑 -> 卡住（这就是踩到的那个现象）
  os._exit(code)          -> atexit 被跳过 -> 立刻退出

顺带把退出码语义钉死：正常返回 0、返回非 0 照传、未捕获异常 1、SystemExit(n) 传 n。

跑法： python scripts/_scratch/_test_hard_exit.py
"""
import subprocess
import sys
import textwrap
from pathlib import Path

WRAPPER = textwrap.dedent('''
    import atexit, os, sys, time

    # 模拟「收尾阶段会挂」：真凶是 ROCm 的上下文销毁，这里用 atexit 代替。
    atexit.register(lambda: time.sleep(600))

    def _hard_exit(code):
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        os._exit(code)

    def main():
    {body}

    if __name__ == "__main__":
        try:
            _code = main()
        except SystemExit as e:
            _code = e.code if isinstance(e.code, int) else 1
        except BaseException:
            import traceback
            traceback.print_exc()
            _code = 1
        _hard_exit(_code)
''')

# 名字, 主体, 期望退出码, 必须快于（秒）
CASES = [
    ("正常返回",       "    return 0", 0, 15),
    ("返回 1",         "    return 1", 1, 15),
    ("未捕获异常",     "    raise RuntimeError('boom')", 1, 15),
    ("SystemExit(5)",  "    raise SystemExit(5)", 5, 15),
    ("值错误异常",     "    x = 1 / 0", 1, 15),
]


def main() -> int:
    tmp = Path(__file__).resolve().parent / "_tmp_hard_exit_case.py"
    fails = []
    print("每个用例都注册了 atexit(睡 600 秒)。若 os._exit 没跳过 atexit，就会超时。\n")
    for name, body, want_code, limit in CASES:
        tmp.write_text(WRAPPER.format(body=body), encoding="utf-8")
        try:
            p = subprocess.run([sys.executable, "-u", str(tmp)],
                               capture_output=True, timeout=limit,
                               encoding="utf-8", errors="replace")
            got, secs_ok, note = p.returncode, True, ""
        except subprocess.TimeoutExpired:
            got, secs_ok, note = None, False, f"超时 >{limit}s（atexit 没被跳过）"

        code_ok = (got == want_code)
        ok = code_ok and secs_ok
        print(f"  [{'ok' if ok else 'FAIL'}] {name:<16} 退出码 {got}（期望 {want_code}）"
              + (f"  {note}" if note else ""))
        if not ok:
            fails.append(name)

    tmp.unlink(missing_ok=True)
    print()
    if fails:
        print(f"===== {len(fails)} 项失败: {', '.join(fails)} =====")
        return 1
    print("===== 全部通过：os._exit 确实跳过了会挂的收尾，退出码语义也对 =====")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
