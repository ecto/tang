"""Gate table from `cram bench` CSV: Default level vs the best arm Default tries,
plus Exhaustive vs every arm, plus compress MB/s per level."""
import csv, sys

SNIFF = ("stored", "zstd-3")
rows = list(csv.DictReader(open(sys.argv[1])))
files, by = [], {}
for r in rows:
    f = r["file"]
    if f not in by:
        files.append(f)
        by[f] = {}
    by[f][r["arm"]] = r


def hdr(n):
    v = 1
    while n >= 128:
        n >>= 7
        v += 1
    return 9 + v


LEVELS = ("cram-fast", "cram", "cram-exhaustive")
print("| file | in bytes | Default tries | best tried arm | best bytes | cram (Default) bytes | Δ | gate | Exhaustive bytes | Δ vs best of all arms | c MB/s fast | c MB/s default | c MB/s exhaustive |")
print("|---|---:|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|")
ok_all = True
for f in files:
    d = by[f]
    n = int(d["cram"]["in_bytes"])
    h = hdr(n)
    arms = {k: v for k, v in d.items() if k not in LEVELS}
    sniffed = d["cram"]["pick"] in SNIFF and "zip" in f
    tried = {k: v for k, v in arms.items() if not sniffed or k in SNIFF}
    b = min(tried.values(), key=lambda r: int(r["out_bytes"]))
    ball = min(arms.values(), key=lambda r: int(r["out_bytes"]))
    c = int(d["cram"]["out_bytes"])
    delta = c - int(b["out_bytes"])
    ok = delta <= h
    ok_all &= ok
    e = int(d["cram-exhaustive"]["out_bytes"])
    ok_all &= e - int(ball["out_bytes"]) <= h
    gate = ("tie" if delta == h else "pass") if ok else "FAIL"
    print(f"| {f} | {n} | {'stored, zstd-3 (sniff)' if sniffed else 'all lossless' + (' + lossy' if 'cram-lossy' in arms else '')} | {b['arm']} | {b['out_bytes']} | {c} | {delta:+d} | {gate} | {e} | {e - int(ball['out_bytes']):+d} | {d['cram-fast']['c_MBps']} | {d['cram']['c_MBps']} | {d['cram-exhaustive']['c_MBps']} |")
print("\nALL PASS" if ok_all else "\nSOME FAIL")
