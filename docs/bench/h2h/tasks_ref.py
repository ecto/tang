"""Reference solutions for tasks.py, so `h2h.py --selftest` can check the tests themselves."""
REF = {}
REF["rle_encode"] = """
def rle_encode(s):
    out = []
    for c in s:
        if out and out[-1][0] == c: out[-1] = (c, out[-1][1] + 1)
        else: out.append((c, 1))
    return out
"""
REF["is_balanced"] = """
def is_balanced(s):
    st, m = [], {')': '(', ']': '[', '}': '{'}
    for c in s:
        if c in '([{': st.append(c)
        elif c in m:
            if not st or st.pop() != m[c]: return False
    return not st
"""
REF["merge_intervals"] = """
def merge_intervals(xs):
    out = []
    for a, b in sorted(xs):
        if out and a <= out[-1][1]: out[-1] = (out[-1][0], max(out[-1][1], b))
        else: out.append((a, b))
    return out
"""
REF["roman_to_int"] = """
def roman_to_int(s):
    v = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}; t = 0
    for i, c in enumerate(s):
        t += -v[c] if i + 1 < len(s) and v[s[i+1]] > v[c] else v[c]
    return t
"""
REF["chunk"] = "def chunk(xs, n):\n    return [xs[i:i+n] for i in range(0, len(xs), n)]\n"
REF["word_freq"] = """
import re
from collections import Counter
def word_freq(text, k):
    c = Counter(re.findall('[a-z]+', text.lower()))
    return sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
"""
REF["flatten"] = """
def flatten(x):
    out = []
    for e in x:
        if isinstance(e, (list, tuple)): out.extend(flatten(e))
        else: out.append(e)
    return out
"""
REF["lcs_len"] = """
def lcs_len(a, b):
    p = [0] * (len(b) + 1)
    for ca in a:
        q = [0]
        for j, cb in enumerate(b):
            q.append(p[j] + 1 if ca == cb else max(p[j+1], q[j]))
        p = q
    return p[-1]
"""
REF["valid_ipv4"] = """
def valid_ipv4(s):
    ps = s.split('.')
    if len(ps) != 4: return False
    for p in ps:
        if not p.isdigit() or (len(p) > 1 and p[0] == '0') or int(p) > 255: return False
    return True
"""
REF["spiral"] = """
def spiral(m):
    out = []
    m = [list(r) for r in m]
    while m:
        out += m.pop(0)
        m = [list(r) for r in zip(*m)][::-1]
    return out
"""
REF["dedupe_keep_order"] = "def dedupe_keep_order(xs):\n    return list(dict.fromkeys(xs))\n"
REF["parse_duration"] = """
import re
def parse_duration(s):
    m = re.fullmatch(r'(?:(\\d+)h)?(?:(\\d+)m)?(?:(\\d+)s)?', s)
    h, mi, se = (int(g) if g else 0 for g in m.groups())
    return h * 3600 + mi * 60 + se
"""
REF["matmul"] = "def matmul(a, b):\n    return [[sum(x * y for x, y in zip(r, c)) for c in zip(*b)] for r in a]\n"
REF["caesar"] = """
def caesar(s, k):
    out = []
    for c in s:
        if 'a' <= c <= 'z': out.append(chr((ord(c) - 97 + k) % 26 + 97))
        elif 'A' <= c <= 'Z': out.append(chr((ord(c) - 65 + k) % 26 + 65))
        else: out.append(c)
    return ''.join(out)
"""
REF["top_k_frequent"] = """
from collections import Counter
def top_k_frequent(xs, k):
    return [v for v, _ in sorted(Counter(xs).items(), key=lambda kv: (-kv[1], kv[0]))[:k]]
"""
REF["binary_search"] = """
import bisect
def binary_search(xs, x):
    i = bisect.bisect_left(xs, x)
    return i if i < len(xs) and xs[i] == x else -1
"""
REF["snake_to_camel"] = """
def snake_to_camel(s):
    ws = [w for w in s.split('_') if w]
    return ws[0] + ''.join(w[:1].upper() + w[1:] for w in ws[1:]) if ws else ''
"""
REF["count_islands"] = """
def count_islands(grid):
    seen, n = set(), 0
    for i, row in enumerate(grid):
        for j, c in enumerate(row):
            if c == '#' and (i, j) not in seen:
                n += 1; st = [(i, j)]; seen.add((i, j))
                while st:
                    a, b = st.pop()
                    for x, y in ((a+1,b),(a-1,b),(a,b+1),(a,b-1)):
                        if 0 <= x < len(grid) and 0 <= y < len(grid[x]) and grid[x][y] == '#' and (x, y) not in seen:
                            seen.add((x, y)); st.append((x, y))
    return n
"""
REF["eval_rpn"] = """
def eval_rpn(tokens):
    st = []
    for t in tokens:
        if t in '+-*/' and len(t) == 1:
            b, a = st.pop(), st.pop()
            st.append(a + b if t == '+' else a - b if t == '-' else a * b if t == '*' else int(a / b))
        else: st.append(int(t))
    return st[0]
"""
REF["lru_cache_ops"] = """
from collections import OrderedDict
def lru_cache_ops(capacity, ops):
    c, out = OrderedDict(), []
    for op in ops:
        if op[0] == 'get':
            if op[1] in c: c.move_to_end(op[1]); out.append(c[op[1]])
            else: out.append(-1)
        else:
            c[op[1]] = op[2]; c.move_to_end(op[1])
            if len(c) > capacity: c.popitem(last=False)
    return out
"""
