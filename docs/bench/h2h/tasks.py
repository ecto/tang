"""Twenty small coding tasks with unit tests, for a pass@1 parity check between endpoints.

Each task: a function name, a prompt (signature + spec), and test code that asserts on it.
Written for this harness (not drawn from a published benchmark), so they're not in anyone's
training data verbatim.
"""

TASKS = [
    ("rle_encode", "def rle_encode(s: str) -> list[tuple[str, int]]:\n    \"\"\"Run-length encode s: consecutive equal characters become (char, count) pairs, in order. Empty string -> [].\"\"\"",
     "assert rle_encode('') == []\nassert rle_encode('aaabccdd') == [('a',3),('b',1),('c',2),('d',2)]\nassert rle_encode('x') == [('x',1)]"),
    ("is_balanced", "def is_balanced(s: str) -> bool:\n    \"\"\"True if every (, [ and { in s is closed by the matching bracket in the right order. Other characters are ignored.\"\"\"",
     "assert is_balanced('a(b[c]{d})e')\nassert not is_balanced('(]')\nassert not is_balanced('((')\nassert is_balanced('')\nassert not is_balanced('}{')"),
    ("merge_intervals", "def merge_intervals(xs: list[tuple[int, int]]) -> list[tuple[int, int]]:\n    \"\"\"Merge overlapping or touching closed intervals [a, b]; return them sorted by start.\"\"\"",
     "assert merge_intervals([]) == []\nassert merge_intervals([(5,6),(1,3),(2,4)]) == [(1,4),(5,6)]\nassert merge_intervals([(1,2),(2,3)]) == [(1,3)]\nassert merge_intervals([(1,10),(2,3)]) == [(1,10)]"),
    ("roman_to_int", "def roman_to_int(s: str) -> int:\n    \"\"\"Convert a valid Roman numeral (I, V, X, L, C, D, M, with subtractive pairs) to an integer.\"\"\"",
     "assert roman_to_int('III') == 3\nassert roman_to_int('IV') == 4\nassert roman_to_int('MCMXCIV') == 1994\nassert roman_to_int('LVIII') == 58"),
    ("chunk", "def chunk(xs: list, n: int) -> list[list]:\n    \"\"\"Split xs into consecutive lists of length n (the last may be shorter). n >= 1.\"\"\"",
     "assert chunk([1,2,3,4,5], 2) == [[1,2],[3,4],[5]]\nassert chunk([], 3) == []\nassert chunk([1,2], 5) == [[1,2]]"),
    ("word_freq", "def word_freq(text: str, k: int) -> list[tuple[str, int]]:\n    \"\"\"The k most frequent lowercase words in text (words are runs of letters a-z, case-insensitive), as (word, count), by count descending then word ascending.\"\"\"",
     "assert word_freq('The the cat. A cat, the dog!', 2) == [('the',3),('cat',2)]\nassert word_freq('', 3) == []\nassert word_freq('b a b a c', 3) == [('a',2),('b',2),('c',1)]"),
    ("flatten", "def flatten(x) -> list:\n    \"\"\"Flatten arbitrarily nested lists and tuples into a flat list of the non-list, non-tuple leaves, in order.\"\"\"",
     "assert flatten([1,[2,(3,[4])],5]) == [1,2,3,4,5]\nassert flatten([]) == []\nassert flatten([[[]]]) == []\nassert flatten(['ab',['c']]) == ['ab','c']"),
    ("lcs_len", "def lcs_len(a: str, b: str) -> int:\n    \"\"\"Length of the longest common subsequence of a and b.\"\"\"",
     "assert lcs_len('abcde','ace') == 3\nassert lcs_len('','abc') == 0\nassert lcs_len('abc','def') == 0\nassert lcs_len('AGGTAB','GXTXAYB') == 4"),
    ("valid_ipv4", "def valid_ipv4(s: str) -> bool:\n    \"\"\"True if s is a dotted-quad IPv4 address: four decimal parts 0-255, no leading zeros (except '0' itself), no signs or spaces.\"\"\"",
     "assert valid_ipv4('192.168.0.1')\nassert not valid_ipv4('256.1.1.1')\nassert not valid_ipv4('01.2.3.4')\nassert not valid_ipv4('1.2.3')\nassert valid_ipv4('0.0.0.0')\nassert not valid_ipv4('1.2.3.-4')"),
    ("spiral", "def spiral(m: list[list[int]]) -> list[int]:\n    \"\"\"The elements of a rectangular matrix in clockwise spiral order starting at the top-left.\"\"\"",
     "assert spiral([[1,2,3],[4,5,6],[7,8,9]]) == [1,2,3,6,9,8,7,4,5]\nassert spiral([]) == []\nassert spiral([[1,2,3,4]]) == [1,2,3,4]\nassert spiral([[1],[2],[3]]) == [1,2,3]"),
    ("dedupe_keep_order", "def dedupe_keep_order(xs: list) -> list:\n    \"\"\"Remove duplicates from xs (hashable items), keeping each item's first occurrence, in order.\"\"\"",
     "assert dedupe_keep_order([3,1,3,2,1]) == [3,1,2]\nassert dedupe_keep_order([]) == []"),
    ("parse_duration", "def parse_duration(s: str) -> int:\n    \"\"\"Seconds in a duration like '1h30m', '45s', '2h5s', '10m': units h, m, s in that order, each optional but at least one present.\"\"\"",
     "assert parse_duration('1h30m') == 5400\nassert parse_duration('45s') == 45\nassert parse_duration('2h5s') == 7205\nassert parse_duration('10m') == 600"),
    ("matmul", "def matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:\n    \"\"\"Matrix product of a (n x k) and b (k x m) as nested lists.\"\"\"",
     "assert matmul([[1,2],[3,4]], [[5,6],[7,8]]) == [[19,22],[43,50]]\nassert matmul([[1,2,3]], [[1],[2],[3]]) == [[14]]"),
    ("caesar", "def caesar(s: str, k: int) -> str:\n    \"\"\"Shift each ASCII letter in s by k positions in the alphabet (wrapping, preserving case; k may be negative or large). Other characters are unchanged.\"\"\"",
     "assert caesar('Hello, World!', 3) == 'Khoor, Zruog!'\nassert caesar('abc', -1) == 'zab'\nassert caesar('xyz', 29) == 'abc'"),
    ("top_k_frequent", "def top_k_frequent(xs: list[int], k: int) -> list[int]:\n    \"\"\"The k most frequent values in xs, by frequency descending then value ascending.\"\"\"",
     "assert top_k_frequent([1,1,1,2,2,3], 2) == [1,2]\nassert top_k_frequent([4,4,5,5,6], 2) == [4,5]\nassert top_k_frequent([], 1) == []"),
    ("binary_search", "def binary_search(xs: list[int], x: int) -> int:\n    \"\"\"Index of the first element equal to x in the sorted list xs, or -1 if absent. O(log n).\"\"\"",
     "assert binary_search([1,2,2,2,3], 2) == 1\nassert binary_search([], 1) == -1\nassert binary_search([1,3,5], 4) == -1\nassert binary_search([1,3,5], 5) == 2"),
    ("snake_to_camel", "def snake_to_camel(s: str) -> str:\n    \"\"\"Convert snake_case to camelCase: drop underscores, capitalize the letter after each, keep the first word lowercase. Leading/trailing/double underscores are dropped.\"\"\"",
     "assert snake_to_camel('hello_world') == 'helloWorld'\nassert snake_to_camel('a_b_c') == 'aBC'\nassert snake_to_camel('_private__name_') == 'privateName'\nassert snake_to_camel('x') == 'x'"),
    ("count_islands", "def count_islands(grid: list[str]) -> int:\n    \"\"\"Number of 4-connected groups of '#' cells in a grid of '#' and '.' strings.\"\"\"",
     "assert count_islands(['#.#','#..','..#']) == 3\nassert count_islands([]) == 0\nassert count_islands(['###','#.#','###']) == 1"),
    ("eval_rpn", "def eval_rpn(tokens: list[str]) -> int:\n    \"\"\"Evaluate integer reverse Polish notation with + - * / (division truncates toward zero).\"\"\"",
     "assert eval_rpn(['2','1','+','3','*']) == 9\nassert eval_rpn(['4','13','5','/','+']) == 6\nassert eval_rpn(['7','-3','/']) == -2"),
    ("lru_cache_ops", "def lru_cache_ops(capacity: int, ops: list[tuple]) -> list:\n    \"\"\"Simulate an LRU cache. ops are ('put', k, v) or ('get', k); return the results of the gets in order (-1 for a miss). A get or put makes the key most recent; a put over capacity evicts the least recent.\"\"\"",
     "assert lru_cache_ops(2, [('put',1,1),('put',2,2),('get',1),('put',3,3),('get',2),('get',3),('get',1)]) == [1,-1,3,1]\nassert lru_cache_ops(1, [('get',1)]) == [-1]"),
]
