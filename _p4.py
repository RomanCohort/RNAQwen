"""加学会的 4x 压缩模块（Conv1d k=4,s=4，初始化为平均池化）。跑完即删。

用户指令：token 压缩，4x（1022 -> 256），且要**学会的模块**而不是固定步长参数。

设计后果：
  · 压缩器可学 -> 码的输入分布随训练变 -> 不能再预训练码表 -> 全端到端
  · 压缩器初始化为平均池化（每 4 个位置取均值），而不是随机
    —— 保住"第 0 步就有信息"这一点，同时保留可学性
  · 序列长度降到 L/4：中位 335->84，最长 1022->256，每条样本总长 ~234
  · token 预算按压缩后的长度算，OOM 消失
"""
import ast
import pathlib

p = pathlib.Path("scripts/50_routeC_e2e.py")
s = p.read_text(encoding="utf-8")

# ---- 1) 压缩器 + FSQ 在同一模块里，forward 直接吃 (L,768) ----
old = """    def __init__(self, dim, latent, levels, d_model):
        super().__init__()
        self.levels, self.latent = levels, latent
        self.enc = nn.Linear(dim, latent)
        self.head = nn.Linear(latent, d_model)          # 可微通路：码 -> 向量
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        t = torch.tanh(self.enc(x))                      # [-1,1]"""
new = """    def __init__(self, dim, latent, levels, d_model, pool=4):
        super().__init__()
        self.levels, self.latent, self.pool = levels, latent, pool
        # 学会的 4x 压缩：Conv1d(k=4, s=4)，初始化为平均池化。
        # 初始成平均池化而不是随机 —— 让第 0 步就有信息，同时保留可学性。
        self.compress = nn.Conv1d(dim, dim, kernel_size=pool, stride=pool, bias=False)
        with torch.no_grad():
            w = torch.zeros(dim, 1, pool)
            w[:, 0, :] = 1.0 / pool
            self.compress.weight.copy_(w.repeat(dim, 1, 1))
        self.enc = nn.Linear(dim, latent)
        self.head = nn.Linear(latent, d_model)          # 可微通路：码 -> 向量
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        L = x.shape[0]
        n = L // self.pool
        if n == 0:
            x = x.repeat(self.pool, 1); n = 1          # 极短序列兜底
        else:
            x = x[:n * self.pool]
        # (L,768) -> (1,768,L) -> conv -> (1,768,n) -> (n,768)
        x = self.compress(x.t().unsqueeze(0)).squeeze(0).t().contiguous()
        t = torch.tanh(self.enc(x))                      # [-1,1]"""
assert old in s, "锚点1"
s = s.replace(old, new, 1)

# ---- 2) clen 改成压缩后的长度；切片用原始长度 ----
old = """    def mk(rs):
        return [(r, qa, len_of[r["uid"]], off_of[r["uid"]]) for r in rs
                for qa in (r.get("qa") or []) if qa.get("question") and qa.get("answer")]"""
new = """    P = args.pool
    def mk(rs):
        # clen 是**压缩后**的 token 数（决定占位符个数）；切片时用 clen*P 的原始隐藏态
        return [(r, qa, len_of[r["uid"]] // P, off_of[r["uid"]]) for r in rs
                for qa in (r.get("qa") or []) if qa.get("question") and qa.get("answer")]"""
assert old in s, "锚点2"
s = s.replace(old, new, 1)

old = """        x = hid[off:off + clen].to(DEV, dtype=torch.float32)"""
new = """        x = hid[off:off + clen * P].to(DEV, dtype=torch.float32)"""
assert old in s, "锚点3"
s = s.replace(old, new, 1)
s = s.replace("""            v, c, t, q = model.rna_vectors(hid[off:off + clen].to(DEV, dtype=torch.float32))""",
              """            v, c, t, q = model.rna_vectors(
                hid[off:off + clen * P].to(DEV, dtype=torch.float32))""")

# ---- 3) 加 --pool，并让 token 预算按压缩后长度走 ----
old = '    ap.add_argument("--token-budget", type=int, default=6000,'
new = ('    ap.add_argument("--pool", type=int, default=4,\n'
       '                    help="学习式压缩的倍率：Conv1d(stride=pool)，初始化为平均池化")\n'
       '    ap.add_argument("--token-budget", type=int, default=6000,')
assert old in s, "锚点4"
s = s.replace(old, new, 1)
s = s.replace('E2EModel(args.lora_r, args.rna_emb)', 'E2EModel(args.lora_r, args.rna_emb, pool=args.pool)')
s = s.replace('def __init__(self, lora_r, rna_emb_mode, latent=LATENT, levels=LEVELS):',
              'def __init__(self, lora_r, rna_emb_mode, latent=LATENT, levels=LEVELS, pool=4):')
s = s.replace('self.fsq = FSQCode(768, latent, levels, d).to(device=dev, dtype=torch.float32)',
              'self.fsq = FSQCode(768, latent, levels, d, pool=pool).to(device=dev, dtype=torch.float32)')

p.write_text(s, encoding="utf-8")
ast.parse(s)
print("patched ok")
