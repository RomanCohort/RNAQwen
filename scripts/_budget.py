# v3 的真实规模 vs 训练预算
rows = 1_060_818
ans_tok = 16.1e6; code_tok = 66.1e6; q_tok = 13.1e6
tot = ans_tok + code_tok + q_tok
print(f"v3 total tokens        {tot/1e6:.1f} M")
print(f"  code                 {code_tok/1e6:.1f} M  ({100*code_tok/tot:.0f}%)")
print(f"  question             {q_tok/1e6:.1f} M  ({100*q_tok/tot:.0f}%)")
print(f"  answer               {ans_tok/1e6:.1f} M  ({100*ans_tok/tot:.0f}%)")
print()
for steps, name in ((500, "pilot"), (6000, "full")):
    tok = steps * 512
    print(f"{name:6s} {steps:>5} steps x 512 tok = {tok/1e6:.2f} M tok "
          f"= {100*tok/tot:.2f}% of one epoch")
print()
print(f"one full epoch at 34 tok/s = {tot/34/86400:.1f} days")
print()
print("如果每序列 10 个问题打包成一个多轮样本：")
packed_code = code_tok / 10
packed = ans_tok + packed_code + q_tok
print(f"  total {packed/1e6:.1f} M, code 占比 {100*packed_code/packed:.0f}%, "
      f"答案占比 {100*ans_tok/packed:.0f}%")
print(f"  one epoch = {packed/34/86400:.1f} days")
