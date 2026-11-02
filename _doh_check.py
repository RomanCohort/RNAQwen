"""用 DoH 查真实 IP（本地 DNS 被 Clash fake-IP 接管了），再查归属。"""
import json
import urllib.request

UA = {"User-Agent": "Mozilla/5.0", "Accept": "application/dns-json"}


def doh(name, server="https://dns.alidns.com/resolve"):
    url = f"{server}?name={name}&type=A"
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=20) as r:
            d = json.load(r)
        return [a["data"] for a in d.get("Answer", []) if a.get("type") == 1]
    except Exception as e:
        return [f"FAIL {type(e).__name__}: {str(e)[:50]}"]


def geo(ip):
    url = f"http://ip-api.com/json/{ip}?fields=query,country,regionName,city,isp,as"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "curl/8.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            d = json.load(r)
        return "%s %s / %s / %s" % (d.get("country"), d.get("city"), d.get("isp"), d.get("as"))
    except Exception as e:
        return f"查询失败 {type(e).__name__}"


NAMES = ["hf-mirror.com", "huggingface.co", "www.romancohort.cloud",
         "cdn-lfs.huggingface.co", "us.aws.cdn.hf.co", "cas-bridge.xethub.hf.co"]

print("=" * 78)
print("真实解析（阿里 DoH，绕过 Clash fake-IP）")
print("=" * 78)
allips = {}
for n in NAMES:
    ips = doh(n)
    allips[n] = ips
    line = "  %-30s %s" % (n, ", ".join(ips))
    print(line)

print()
print("=" * 78)
print("归属")
print("=" * 78)
seen = set()
for n, ips in allips.items():
    for ip in ips:
        if ip in seen or not ip[0].isdigit():
            continue
        seen.add(ip)
        print("  %-18s <- %-26s %s" % (ip, n, geo(ip)))

print()
print("=" * 78)
print("判定")
print("=" * 78)
hm = allips.get("hf-mirror.com", [])
cn = [ip for ip in hm if ip[0].isdigit()]
if cn:
    print("  hf-mirror.com 解析到:", cn)
    for ip in cn:
        g = geo(ip)
        print("      %s -> %s" % (ip, g))
    print()
    print("  如果上面是「中国」，那 Clash 的 GEOSITE,cn,DIRECT 本该命中它 ——")
    print("  说明 hf-mirror.com 不在 cn 规则集里，被 MATCH,PROXY 丢进了那条快死的隧道。")
