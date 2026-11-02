import sys; t=open(sys.argv[1],encoding="utf-8").read(); bad=[c for c in t if ord(c)>127]; print("non-ascii:",len(bad), "".join(sorted(set(bad)))[:50])
