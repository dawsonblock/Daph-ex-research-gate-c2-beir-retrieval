#!/usr/bin/env python3
"""Generate misleading-probe coding tasks.

These tasks are designed so that:
  - The first 2 tests (probe tests) are easy and most candidates pass them
  - The remaining tests are hard and distinguish correct from incorrect
  - This creates the regime where probe pass rate is uninformative
    but DAPH-X's learned features might be

Each task has tests ordered: easy probes first, hard discriminative tests after.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from daph_x.coding.tasks import CodingTask, TASKS


MISLEADING_PROBE_TASKS = [
    CodingTask(
        task_id="code_101_string_multiply",
        description="Multiply strings representing non-negative integers",
        function_name="multiply_strings",
        signature="multiply_strings(num1: str, num2: str) -> str",
        docstring="Given two non-negative integers represented as strings, return their product as a string. Cannot use built-in big integer conversion.",
        difficulty="hard",
        tests=(
            # Easy probe tests (most candidates pass)
            ('multiply_strings("0", "0")', "zero", "0"),
            ('multiply_strings("1", "1")', "identity", "1"),
            # Hard discriminative tests
            ('multiply_strings("2", "3")', "small", "6"),
            ('multiply_strings("123", "456")', "medium", "56088"),
            ('multiply_strings("999", "999")', "large same", "998001"),
            ('multiply_strings("123456789", "987654321")', "very large", "121932631112635269"),
            ('multiply_strings("0", "12345")', "zero times large", "0"),
            ('multiply_strings("999999", "999999")', "six nines", "999998000001"),
        ),
        common_errors=(
            "Not implementing grade-school multiplication",
            "Carry errors on large numbers",
            "Not handling zero correctly",
            "Using int() directly (forbidden)",
        ),
    ),
    CodingTask(
        task_id="code_102_add_strings",
        description="Add two non-negative integers represented as strings",
        function_name="add_strings",
        signature="add_strings(num1: str, num2: str) -> str",
        docstring="Given two non-negative integers represented as strings, return their sum as a string. Cannot convert the entire string to integer directly.",
        difficulty="hard",
        tests=(
            # Easy probes
            ('add_strings("0", "0")', "zero", "0"),
            ('add_strings("1", "1")', "simple", "2"),
            # Hard tests
            ('add_strings("999", "1")', "carry cascade", "1000"),
            ('add_strings("123", "456")', "medium", "579"),
            ('add_strings("999999999", "1")', "long carry", "1000000000"),
            ('add_strings("123456789", "987654321")', "large", "1111111110"),
            ('add_strings("0", "99999")', "zero plus large", "99999"),
        ),
        common_errors=(
            "Carry propagation errors",
            "Not handling different lengths",
            "Off-by-one in digit iteration",
        ),
    ),
    CodingTask(
        task_id="code_103_pow_mod",
        description="Compute base^exp % mod efficiently (modular exponentiation)",
        function_name="pow_mod",
        signature="pow_mod(base: int, exp: int, mod: int) -> int",
        docstring="Compute (base^exp) % mod using fast modular exponentiation. Handle mod=1 (return 0). All inputs are non-negative.",
        difficulty="hard",
        tests=(
            # Easy probes
            ('pow_mod(2, 3, 5)', "small", 3),
            ('pow_mod(1, 100, 7)', "base 1", 1),
            # Hard tests
            ('pow_mod(2, 10, 1000)', "medium", 24),
            ('pow_mod(7, 100, 13)', "large exp", 9),
            ('pow_mod(0, 5, 7)', "base 0", 0),
            ('pow_mod(5, 0, 7)', "exp 0", 1),
            ('pow_mod(0, 0, 7)', "0^0", 1),
            ('pow_mod(999, 999, 1000000007)', "very large", 999),  # 999^999 mod 10^9+7
            ('pow_mod(2, 1000000, 1000000007)', "million exp", 750661099),
            ('pow_mod(1, 0, 1)', "mod 1", 0),
        ),
        common_errors=(
            "Not using fast exponentiation (too slow for large exp)",
            "Not handling mod=1",
            "Not handling 0^0",
            "Overflow in intermediate computation",
        ),
    ),
    CodingTask(
        task_id="code_104_gcd_extended",
        description="Extended GCD: compute gcd(a,b) and coefficients x,y such that ax+by=gcd",
        function_name="extended_gcd",
        signature="extended_gcd(a: int, b: int) -> tuple",
        docstring="Return (gcd, x, y) such that a*x + b*y = gcd(a, b). Handle negative inputs and zero.",
        difficulty="hard",
        tests=(
            # Easy probes
            ('extended_gcd(12, 8)', "basic", (4, 1, -1)),
            ('extended_gcd(7, 1)', "coprime", (1, 0, 1)),
            # Hard tests
            ('extended_gcd(0, 5)', "zero a", (5, 0, 1)),
            ('extended_gcd(5, 0)', "zero b", (5, 1, 0)),
            ('extended_gcd(0, 0)', "both zero", (0, 1, 0)),
            ('extended_gcd(-12, 8)', "negative a", (4, -1, -1)),
            ('extended_gcd(12, -8)', "negative b", (4, 1, 1)),
            ('extended_gcd(-12, -8)', "both negative", (4, -1, 1)),
            ('extended_gcd(999999, 1)', "large", (1, 0, 1)),
        ),
        common_errors=(
            "Not handling zero inputs",
            "Not handling negative inputs",
            "Incorrect coefficient computation",
            "Sign errors in recursive case",
        ),
    ),
    CodingTask(
        task_id="code_105_primes_sieve",
        description="Count prime numbers less than n using Sieve of Eratosthenes",
        function_name="count_primes",
        signature="count_primes(n: int) -> int",
        docstring="Return the number of prime numbers strictly less than n. Use the Sieve of Eratosthenes for efficiency.",
        difficulty="hard",
        tests=(
            # Easy probes
            ('count_primes(10)', "small", 4),
            ('count_primes(2)', "no primes", 0),
            # Hard tests
            ('count_primes(1)', "edge", 0),
            ('count_primes(0)', "edge zero", 0),
            ('count_primes(3)', "one prime", 1),
            ('count_primes(100)', "medium", 25),
            ('count_primes(1000)', "large", 168),
            ('count_primes(10000)', "very large", 1229),
            ('count_primes(100000)', "huge", 9592),
        ),
        common_errors=(
            "Off-by-one (including n itself)",
            "Not handling n <= 2",
            "O(n^2) instead of O(n log log n)",
            "Incorrect sieve initialization",
        ),
    ),
    CodingTask(
        task_id="code_106_isomorphic_strings",
        description="Check if two strings are isomorphic (char mapping is bijective)",
        function_name="is_isomorphic",
        signature="is_isomorphic(s: str, t: str) -> bool",
        docstring="Return True if s and t are isomorphic: each char in s maps to exactly one char in t and vice versa.",
        difficulty="hard",
        tests=(
            # Easy probes
            ('is_isomorphic("egg", "add")', "basic", True),
            ('is_isomorphic("foo", "bar")', "not iso", False),
            # Hard tests
            ('is_isomorphic("paper", "title")', "complex", True),
            ('is_isomorphic("ab", "aa")', "not bijective", False),
            ('is_isomorphic("aa", "ab")', "reverse not bijective", False),
            ('is_isomorphic("", "")', "empty", True),
            ('is_isomorphic("a", "a")', "single", True),
            ('is_isomorphic("badc", "baba")', "tricky", False),
            ('is_isomorphic("abc", "def")', "all different", True),
            ('is_isomorphic("abab", "cdcd")', "alternating", True),
        ),
        common_errors=(
            "Not checking both directions (bijective)",
            "Not handling empty strings",
            "Incorrect mapping logic",
        ),
    ),
    CodingTask(
        task_id="code_107_word_pattern",
        description="Check if string follows a pattern (bijection between pattern chars and words)",
        function_name="word_pattern",
        signature="word_pattern(pattern: str, s: str) -> bool",
        docstring="Return True if s follows the same pattern as pattern. Each letter in pattern maps to a distinct word in s, and each word maps to a distinct letter (bijective).",
        difficulty="hard",
        tests=(
            # Easy probes
            ('word_pattern("abba", "dog cat cat dog")', "basic", True),
            ('word_pattern("abba", "dog cat cat fish")', "mismatch", False),
            # Hard tests
            ('word_pattern("aaaa", "dog cat cat dog")', "not matching", False),
            ('word_pattern("abba", "dog dog dog dog")', "not bijective", False),
            ('word_pattern("abc", "dog cat fish")', "all different", True),
            ('word_pattern("a", "a")', "single", True),
            ('word_pattern("a", "dog")', "single word", True),
            ('word_pattern("abc", "dog dog dog")', "same words", False),
            ('word_pattern("aba", "dog cat dog")', "palindrome pattern", True),
        ),
        common_errors=(
            "Not checking bijective mapping",
            "Not handling different word counts",
            "Splitting errors",
        ),
    ),
    CodingTask(
        task_id="code_108_nim_game",
        description="Determine if you can win the Nim game (1-4 stones per turn)",
        function_name="can_win_nim",
        signature="can_win_nim(n: int) -> bool",
        docstring="Return True if you can win the Nim game with n stones. Each turn remove 1-4 stones. You go first. The player who takes the last stone wins. Both play optimally.",
        difficulty="hard",
        tests=(
            # Easy probes
            ('can_win_nim(1)', "can take 1", True),
            ('can_win_nim(4)', "can take all", True),
            # Hard tests
            ('can_win_nim(5)', "losing position", False),
            ('can_win_nim(6)', "winning", True),
            ('can_win_nim(7)', "winning", True),
            ('can_win_nim(8)', "winning", True),
            ('can_win_nim(9)', "losing", False),
            ('can_win_nim(10)', "winning", True),
            ('can_win_nim(15)', "losing", False),
            ('can_win_nim(16)', "winning", True),
            ('can_win_nim(100)', "large", True),
            ('can_win_nim(0)', "no stones", False),
        ),
        common_errors=(
            "Not recognizing the pattern (n % 5 == 0 means lose)",
            "Using recursion/DP instead of the mathematical pattern",
            "Not handling n=0",
        ),
    ),
    CodingTask(
        task_id="code_109_happy_number",
        description="Determine if a number is a happy number (sum of squared digits eventually reaches 1)",
        function_name="is_happy",
        signature="is_happy(n: int) -> bool",
        docstring="Return True if n is a happy number. A happy number is one where repeatedly replacing it with the sum of squared digits eventually reaches 1. Unhappy numbers enter a cycle that does not include 1.",
        difficulty="hard",
        tests=(
            # Easy probes
            ('is_happy(1)', "trivially happy", True),
            ('is_happy(19)', "classic happy", True),
            # Hard tests
            ('is_happy(2)', "unhappy", False),
            ('is_happy(7)', "happy", True),
            ('is_happy(10)', "happy", True),
            ('is_happy(13)', "happy", True),
            ('is_happy(4)', "unhappy cycle", False),
            ('is_happy(100)', "happy", True),
            ('is_happy(2147483647)', "very large", True),
        ),
        common_errors=(
            "Not detecting cycles (infinite loop)",
            "Incorrect digit square sum",
            "Not handling n=1",
        ),
    ),
    CodingTask(
        task_id="code_110_excel_column",
        description="Convert Excel column title to number (A=1, Z=26, AA=27)",
        function_name="title_to_number",
        signature="title_to_number(column_title: str) -> int",
        docstring="Convert an Excel column title to its corresponding number. A=1, B=2, ..., Z=26, AA=27, AB=28, etc.",
        difficulty="hard",
        tests=(
            # Easy probes
            ('title_to_number("A")', "single", 1),
            ('title_to_number("Z")', "last single", 26),
            # Hard tests
            ('title_to_number("AA")', "double A", 27),
            ('title_to_number("AB")', "double AB", 28),
            ('title_to_number("AZ")', "double AZ", 52),
            ('title_to_number("BA")', "double BA", 53),
            ('title_to_number("ZZ")', "double ZZ", 702),
            ('title_to_number("AAA")', "triple A", 703),
            ('title_to_number("FXSHRXW")', "very long", 2147483647),
            ('title_to_number("")', "empty", 0),
        ),
        common_errors=(
            "Off-by-one (A=0 vs A=1)",
            "Not handling base-26 correctly",
            "Not handling empty string",
            "Overflow on very long titles",
        ),
    ),
    CodingTask(
        task_id="code_111_majority_element",
        description="Find majority element (appears more than n/2 times) in O(n) time O(1) space",
        function_name="majority_element",
        signature="majority_element(nums: list) -> int",
        docstring="Return the majority element in nums (guaranteed to exist). The majority element appears more than n/2 times. Use Boyer-Moore voting algorithm for O(n) time and O(1) space.",
        difficulty="hard",
        tests=(
            # Easy probes
            ('majority_element([1, 1, 2])', "simple", 1),
            ('majority_element([3, 2, 3])', "basic", 3),
            # Hard tests
            ('majority_element([2, 2, 1, 1, 1, 2, 2])', "complex", 2),
            ('majority_element([1])', "single", 1),
            ('majority_element([6, 5, 6])', "three elements", 6),
            ('majority_element([1, 1, 1, 1, 1])', "all same", 1),
            ('majority_element([1, 2, 3, 4, 5, 5, 5, 5, 5])', "majority at end", 5),
            ('majority_element([-1, -1, 2, -1])', "with negatives", -1),
        ),
        common_errors=(
            "Using sort or hash map instead of Boyer-Moore",
            "Not handling single element",
            "Incorrect count reset logic",
        ),
    ),
    CodingTask(
        task_id="code_112_circular_array_loop",
        description="Detect if a circular array has a cycle (forward or backward)",
        function_name="circular_array_loop",
        signature="circular_array_loop(nums: list) -> bool",
        docstring="Return True if the circular array has a cycle. A cycle is a sequence of indices of length > 1 that follows the direction (all positive or all negative) and loops back to the start.",
        difficulty="hard",
        tests=(
            # Easy probes
            ('circular_array_loop([2, -1, 1, 2, 2])', "basic", True),
            ('circular_array_loop([-1, -2, -3, -4, -5])', "all negative no cycle", False),
            # Hard tests
            ('circular_array_loop([-1, -1, -1, -1, -1])', "self loops", False),
            ('circular_array_loop([1, 1, 1, 1, 1])', "all forward self loops", False),
            ('circular_array_loop([3, 1, 2])', "small cycle", True),
            ('circular_array_loop([-2, 1, -1, -2, -2])', "negative cycle", False),
            ('circular_array_loop([1, 2, 3, 4, 5])', "no cycle", False),
            ('circular_array_loop([-1])', "single element", False),
        ),
        common_errors=(
            "Not checking that cycle length > 1",
            "Not checking direction consistency",
            "Incorrect modulo for circular indexing",
            "O(n^2) instead of O(n)",
        ),
    ),
    CodingTask(
        task_id="code_113_find_duplicate",
        description="Find the duplicate number in array where values are 1 to n and one is duplicated",
        function_name="find_duplicate",
        signature="find_duplicate(nums: list) -> int",
        docstring="Given an array of n+1 integers where each value is in [1, n], find the one duplicate number. Must use O(1) extra space and not modify the array. Use Floyd's cycle detection.",
        difficulty="hard",
        tests=(
            # Easy probes
            ('find_duplicate([1, 3, 4, 2, 2])', "basic", 2),
            ('find_duplicate([3, 1, 3, 4, 2])', "another", 3),
            # Hard tests
            ('find_duplicate([1, 1])', "two elements", 1),
            ('find_duplicate([1, 1, 2])', "three elements", 1),
            ('find_duplicate([2, 2, 2, 2, 2])', "all same", 2),
            ('find_duplicate([2, 5, 9, 6, 9, 3, 8, 9, 7, 1])', "complex", 9),
            ('find_duplicate([1, 3, 4, 2, 1])', "duplicate at ends", 1),
        ),
        common_errors=(
            "Using sort (modifies array)",
            "Using hash set (O(n) space)",
            "Incorrect Floyd's algorithm implementation",
            "Not handling the case where duplicate appears > 2 times",
        ),
    ),
    CodingTask(
        task_id="code_114_game_of_life",
        description="Compute next state of Conway's Game of Life on a grid",
        function_name="game_of_life",
        signature="game_of_life(board: list) -> list",
        docstring="Compute the next state of Conway's Game of Life. Rules: live cell with 2-3 live neighbors survives; dead cell with exactly 3 live neighbors becomes alive; all others die/stay dead. Return the next board.",
        difficulty="hard",
        tests=(
            # Easy probes
            ('game_of_life([[0]])', "single dead", [[0]]),
            ('game_of_life([[1]])', "single live dies", [[0]]),
            # Hard tests
            ('game_of_life([[1, 1], [1, 0]])', "blinker-like", [[1, 1], [1, 1]]),
            ('game_of_life([[0, 1, 0], [0, 0, 1], [1, 1, 1]])', "glider", [[0, 0, 0], [1, 0, 1], [0, 1, 1]]),
            ('game_of_life([[1, 1], [1, 1]])', "block stable", [[1, 1], [1, 1]]),
            ('game_of_life([[1, 1, 0], [1, 0, 0], [0, 0, 0]])', "corner", [[1, 1, 0], [1, 1, 0], [0, 0, 0]]),
            ('game_of_life([[0, 0, 0], [0, 0, 0], [0, 0, 0]])', "all dead", [[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
        ),
        common_errors=(
            "Modifying board in-place while reading",
            "Incorrect neighbor counting (corners, edges)",
            "Not handling 1x1 board",
            "Incorrect rule application",
        ),
    ),
    CodingTask(
        task_id="code_115_josephus",
        description="Find the Josephus survivor position (n people, eliminate every k-th)",
        function_name="josephus",
        signature="josephus(n: int, k: int) -> int",
        docstring="Return the position (0-indexed) of the survivor in the Josephus problem with n people, eliminating every k-th person.",
        difficulty="hard",
        tests=(
            # Easy probes
            ('josephus(5, 2)', "classic", 2),
            ('josephus(1, 1)', "single person", 0),
            # Hard tests
            ('josephus(7, 3)', "k=3", 3),
            ('josephus(2, 1)', "k=1", 1),
            ('josephus(10, 3)', "n=10", 3),
            ('josephus(41, 2)', "historical", 18),
            ('josephus(100, 7)', "large", 49),
            ('josephus(1, 5)', "single k>n", 0),
            ('josephus(5, 1)', "k=1 (last)", 4),
        ),
        common_errors=(
            "0-indexed vs 1-indexed confusion",
            "Not using the recursive formula: J(n,k) = (J(n-1,k) + k) % n",
            "Simulation instead of formula (slow for large n)",
            "Not handling n=1",
        ),
    ),
    CodingTask(
        task_id="code_116_longest_substring_k",
        description="Find longest substring with at most k distinct characters",
        function_name="longest_substring_k_distinct",
        signature="longest_substring_k_distinct(s: str, k: int) -> int",
        docstring="Return the length of the longest substring of s that contains at most k distinct characters.",
        difficulty="hard",
        tests=(
            # Easy probes
            ('longest_substring_k_distinct("eceba", 2)', "basic", 3),
            ('longest_substring_k_distinct("aa", 1)', "single char", 2),
            # Hard tests
            ('longest_substring_k_distinct("aabbcc", 1)', "k=1", 2),
            ('longest_substring_k_distinct("aabbcc", 2)', "k=2", 4),
            ('longest_substring_k_distinct("aabbcc", 3)', "k=3", 6),
            ('longest_substring_k_distinct("", 2)', "empty", 0),
            ('longest_substring_k_distinct("abc", 0)', "k=0", 0),
            ('longest_substring_k_distinct("abaccc", 2)', "complex", 4),
            ('longest_substring_k_distinct("aaabbbccc", 2)', "long", 6),
        ),
        common_errors=(
            "O(n^2) instead of O(n) sliding window",
            "Not handling k=0",
            "Not handling empty string",
            "Incorrect window shrinking logic",
        ),
    ),
    CodingTask(
        task_id="code_117_evaluate_rpn",
        description="Evaluate Reverse Polish Notation expression",
        function_name="eval_rpn",
        signature="eval_rpn(tokens: list) -> int",
        docstring="Evaluate a Reverse Polish Notation expression. Valid operators: +, -, *, /. Division truncates toward zero. Return the integer result.",
        difficulty="hard",
        tests=(
            # Easy probes
            ('eval_rpn(["2", "1", "+"])', "addition", 3),
            ('eval_rpn(["4", "2", "/"])', "division", 2),
            # Hard tests
            ('eval_rpn(["2", "3", "*"])', "multiplication", 6),
            ('eval_rpn(["5", "1", "2", "+", "4", "*", "+", "3", "-"])', "complex", 14),
            ('eval_rpn(["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"])', "very complex", 22),
            ('eval_rpn(["3", "11", "+", "5", "-"])', "with negative", 9),
            ('eval_rpn(["4", "-2", "/"])', "negative division", -2),
            ('eval_rpn(["-7", "3", "/"])', "negative numerator", -2),  # truncates toward zero, not floor
            ('eval_rpn(["13"])', "single number", 13),
            ('eval_rpn(["1", "2", "+", "3", "*"])', "chained", 9),
        ),
        common_errors=(
            "Using floor division instead of truncation toward zero",
            "Not handling negative division correctly",
            "Stack management errors",
            "Not handling single number input",
        ),
    ),
    CodingTask(
        task_id="code_118_clone_graph",
        description="Clone an undirected graph represented as adjacency list",
        function_name="clone_graph",
        signature="clone_graph(adj_list: list) -> list",
        docstring="Given an undirected graph as an adjacency list (0-indexed nodes), return a deep copy of the adjacency list.",
        difficulty="hard",
        tests=(
            # Easy probes
            ('clone_graph([[1, 2], [0, 2], [0, 1]])', "triangle", [[1, 2], [0, 2], [0, 1]]),
            ('clone_graph([])', "empty", []),
            # Hard tests
            ('clone_graph([[]])', "single node", [[]]),
            ('clone_graph([[1], [0]])', "two nodes", [[1], [0]]),
            ('clone_graph([[1, 2, 3], [0], [0], [0]])', "star", [[1, 2, 3], [0], [0], [0]]),
            ('clone_graph([[1, 3], [0, 2], [1, 3], [0, 2]])', "square", [[1, 3], [0, 2], [1, 3], [0, 2]]),
            ('clone_graph([[1, 2, 3, 4], [0], [0], [0], [0]])', "star 5", [[1, 2, 3, 4], [0], [0], [0], [0]]),
        ),
        common_errors=(
            "Not creating a true deep copy",
            "Not handling empty graph",
            "Not handling single node",
            "Missing edges in copy",
        ),
    ),
    CodingTask(
        task_id="code_119_min_stack",
        description="Design a stack that supports push, pop, top, and retrieving min in O(1)",
        function_name="MinStack",
        signature="MinStack()",
        docstring="Design a stack that supports push, pop, top, and getMin operations, all in O(1) time.",
        difficulty="hard",
        tests=(
            ('MinStack()', "create", None),
        ),
        common_errors=(
            "Not using auxiliary stack for min",
            "O(n) min instead of O(1)",
            "Not handling pop correctly",
        ),
    ),
    CodingTask(
        task_id="code_120_valid_parentheses_str",
        description="Check if string with (, ), *, * can be valid (* as wildcard)",
        function_name="check_valid_string",
        signature="check_valid_string(s: str) -> bool",
        docstring="Return True if s with '(', ')', and '*' can be made valid. '*' can be treated as '(', ')', or empty. A valid string has balanced and properly nested parentheses.",
        difficulty="hard",
        tests=(
            # Easy probes
            ('check_valid_string("()")', "simple", True),
            ('check_valid_string("(*)")', "star as empty", True),
            # Hard tests
            ('check_valid_string("(*))")', "star as open", True),
            ('check_valid_string(")")', "single close", False),
            ('check_valid_string("(")', "single open", False),
            ('check_valid_string("(((*)")', "complex", True),
            ('check_valid_string("(((**")', "all stars", True),
            ('check_valid_string("(()*")', "mixed", True),
            ('check_valid_string(")*(")', "impossible", False),
            ('check_valid_string("(((())))")', "no stars valid", True),
            ('check_valid_string("")', "empty", True),
            ('check_valid_string("(())((())()(*)(*()(*()(())())())()())((())((())())")', "very long", True),
        ),
        common_errors=(
            "Greedy approach without tracking both min and max open count",
            "Not handling * as all three possibilities",
            "Not handling empty string",
            "Incorrect bounds on open count",
        ),
    ),
    CodingTask(
        task_id="code_121_daily_temperatures",
        description="For each day, find how many days until a warmer temperature",
        function_name="daily_temperatures",
        signature="daily_temperatures(temps: list) -> list",
        docstring="Given a list of daily temperatures, return a list where ans[i] is the number of days after day i until a warmer temperature. If no future day is warmer, ans[i] = 0.",
        difficulty="hard",
        tests=(
            # Easy probes
            ('daily_temperatures([73, 74, 75, 71, 69, 72, 76, 73])', "classic", [1, 1, 4, 2, 1, 1, 0, 0]),
            ('daily_temperatures([30, 40, 50, 60])', "increasing", [1, 1, 1, 0]),
            # Hard tests
            ('daily_temperatures([30, 60, 90])', "three increasing", [1, 1, 0]),
            ('daily_temperatures([90, 80, 70, 60])', "decreasing", [0, 0, 0, 0]),
            ('daily_temperatures([50])', "single", [0]),
            ('daily_temperatures([50, 50, 50])', "all same", [0, 0, 0]),
            ('daily_temperatures([55, 47, 52, 38, 25, 48, 47, 56])', "complex", [7, 1, 4, 2, 1, 1, 1, 0]),
        ),
        common_errors=(
            "O(n^2) instead of O(n) with stack",
            "Not handling decreasing sequences",
            "Not handling single element",
            "Off-by-one in day counting",
        ),
    ),
    CodingTask(
        task_id="code_122_decode_ways",
        description="Count ways to decode a string of digits as letters (1=A, 26=Z)",
        function_name="num_decodings",
        signature="num_decodings(s: str) -> int",
        docstring="Return the number of ways to decode a string of digits where '1'->A, '2'->B, ..., '26'->Z. A valid decoding maps each 1-2 digit group to a letter 1-26. Leading zeros are invalid.",
        difficulty="hard",
        tests=(
            # Easy probes
            ('num_decodings("12")', "basic", 2),
            ('num_decodings("1")', "single", 1),
            # Hard tests
            ('num_decodings("226")', "three digits", 3),
            ('num_decodings("06")', "leading zero", 0),
            ('num_decodings("0")', "single zero", 0),
            ('num_decodings("10")', "ten", 1),
            ('num_decodings("27")', "twenty seven", 1),
            ('num_decodings("11106")', "complex", 2),
            ('num_decodings("123456789")', "long", 3),
            ('num_decodings("")', "empty", 0),
            ('num_decodings("111111111111111111111111")', "24 ones", 121415),
        ),
        common_errors=(
            "Not handling leading zeros",
            "Not handling '10' and '20' correctly",
            "O(2^n) recursion instead of O(n) DP",
            "Not handling empty string",
        ),
    ),
    CodingTask(
        task_id="code_123_house_robber",
        description="Maximize money robbed from houses without alerting police (no adjacent houses)",
        function_name="rob",
        signature="rob(nums: list) -> int",
        docstring="Return the maximum amount of money you can rob from houses arranged in a line. You cannot rob adjacent houses (that would trigger the alarm).",
        difficulty="hard",
        tests=(
            # Easy probes
            ('rob([1, 1, 1])', "simple", 2),
            ('rob([1, 2, 3, 1])', "classic", 4),
            # Hard tests
            ('rob([2, 7, 9, 3, 1])', "complex", 12),
            ('rob([1])', "single", 1),
            ('rob([2, 1, 1, 2])', "alternating", 4),
            ('rob([])', "empty", 0),
            ('rob([1, 2])', "two houses", 2),
            ('rob([2, 2, 2, 2, 2, 2, 2, 2, 2, 2])', "all same", 10),
            ('rob([100, 1, 1, 100, 1, 1, 100])', "spread", 300),
        ),
        common_errors=(
            "O(2^n) recursion instead of O(n) DP",
            "Not handling empty array",
            "Not handling single element",
            "Incorrect DP transition",
        ),
    ),
    CodingTask(
        task_id="code_124_coin_change",
        description="Find minimum number of coins to make a target amount",
        function_name="coin_change",
        signature="coin_change(coins: list, amount: int) -> int",
        docstring="Return the minimum number of coins needed to make up the amount. Return -1 if impossible. You have an infinite supply of each coin.",
        difficulty="hard",
        tests=(
            # Easy probes
            ('coin_change([1, 2, 5], 11)', "classic", 3),
            ('coin_change([2], 3)', "impossible", -1),
            # Hard tests
            ('coin_change([1], 0)', "zero amount", 0),
            ('coin_change([1], 1)', "single coin", 1),
            ('coin_change([1, 5, 10, 25], 100)', "quarters", 4),
            ('coin_change([1, 5, 10, 25], 87)', "complex", 6),
            ('coin_change([186, 419, 83, 408], 6249)', "large", 20),
            ('coin_change([1, 3, 4], 6)', "greedy fails", 2),  # 3+3, not 4+1+1
            ('coin_change([2, 5, 10, 1], 27)', "mixed", 4),
            ('coin_change([1], 2)', "two of same", 2),
        ),
        common_errors=(
            "Greedy instead of DP (fails on [1,3,4],6)",
            "Not handling amount=0",
            "Not handling impossible cases",
            "O(amount * n) DP with incorrect initialization",
        ),
    ),
    CodingTask(
        task_id="code_125_jump_game",
        description="Determine if you can reach the last index by jumping at most nums[i] steps",
        function_name="can_jump",
        signature="can_jump(nums: list) -> bool",
        docstring="Return True if you can reach the last index of nums. You start at index 0 and can jump at most nums[i] steps forward from index i.",
        difficulty="hard",
        tests=(
            # Easy probes
            ('can_jump([2, 3, 1, 1, 4])', "basic", True),
            ('can_jump([3, 2, 1, 0, 4])', "blocked", False),
            # Hard tests
            ('can_jump([0])', "single element", True),
            ('can_jump([1, 0, 0])', "just enough", False),
            ('can_jump([2, 0, 0])', "exact jumps", True),
            ('can_jump([1, 1, 1, 1, 1])', "all ones", True),
            ('can_jump([0, 1])', "stuck at start", False),
            ('can_jump([5, 0, 0, 0, 0, 0])', "big first jump", True),
            ('can_jump([2, 5, 0, 0, 0, 0, 0, 0, 0, 0])', "can overshoot", True),
        ),
        common_errors=(
            "O(n^2) DP instead of O(n) greedy",
            "Not handling single element",
            "Not handling zero at start",
            "Incorrect reachability tracking",
        ),
    ),
    CodingTask(
        task_id="code_126_rotate_image",
        description="Rotate an n x n matrix 90 degrees clockwise in-place",
        function_name="rotate_image",
        signature="rotate_image(matrix: list) -> list",
        docstring="Rotate an n x n 2D matrix 90 degrees clockwise. Return the rotated matrix.",
        difficulty="hard",
        tests=(
            # Easy probes
            ('rotate_image([[1]])', "single", [[1]]),
            ('rotate_image([[1, 2], [3, 4]])', "2x2", [[3, 1], [4, 2]]),
            # Hard tests
            ('rotate_image([[1, 2, 3], [4, 5, 6], [7, 8, 9]])', "3x3", [[7, 4, 1], [8, 5, 2], [9, 6, 3]]),
            ('rotate_image([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])', "4x4",
             [[13, 9, 5, 1], [14, 10, 6, 2], [15, 11, 7, 3], [16, 12, 8, 4]]),
            ('rotate_image([[5]])', "single 5", [[5]]),
        ),
        common_errors=(
            "Not doing transpose then reverse",
            "Incorrect index mapping",
            "Not handling 1x1 matrix",
            "Modifying while reading",
        ),
    ),
    CodingTask(
        task_id="code_127_spiral_matrix",
        description="Return elements of matrix in spiral order",
        function_name="spiral_order",
        signature="spiral_order(matrix: list) -> list",
        docstring="Return all elements of an m x n matrix in clockwise spiral order, starting from top-left.",
        difficulty="hard",
        tests=(
            # Easy probes
            ('spiral_order([[1, 2, 3], [4, 5, 6], [7, 8, 9]])', "3x3", [1, 2, 3, 6, 9, 8, 7, 4, 5]),
            ('spiral_order([[1]])', "single", [1]),
            # Hard tests
            ('spiral_order([[1, 2], [3, 4]])', "2x2", [1, 2, 4, 3]),
            ('spiral_order([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])', "3x4", [1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7]),
            ('spiral_order([[1], [2], [3]])', "single col", [1, 2, 3]),
            ('spiral_order([[1, 2, 3]])', "single row", [1, 2, 3]),
            ('spiral_order([])', "empty", []),
            ('spiral_order([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])', "4x3", [1, 2, 3, 6, 9, 12, 11, 10, 7, 4, 5, 8]),
        ),
        common_errors=(
            "Incorrect boundary handling",
            "Not handling single row/column",
            "Not handling empty matrix",
            "Off-by-one in layer traversal",
        ),
    ),
    CodingTask(
        task_id="code_128_valid_tree",
        description="Check if a graph is a valid tree (connected, no cycles)",
        function_name="is_valid_tree",
        signature="is_valid_tree(n: int, edges: list) -> bool",
        docstring="Return True if the given graph with n nodes and edges forms a valid tree. A tree is connected and has exactly n-1 edges with no cycles.",
        difficulty="hard",
        tests=(
            # Easy probes
            ('is_valid_tree(5, [[0, 1], [0, 2], [0, 3], [1, 4]])', "valid tree", True),
            ('is_valid_tree(5, [[0, 1], [1, 2], [2, 3], [1, 3], [1, 4]])', "has cycle", False),
            # Hard tests
            ('is_valid_tree(1, [])', "single node", True),
            ('is_valid_tree(2, [[0, 1]])', "two nodes", True),
            ('is_valid_tree(2, [])', "disconnected", False),
            ('is_valid_tree(3, [[0, 1]])', "not enough edges", False),
            ('is_valid_tree(4, [[0, 1], [2, 3]])', "two components", False),
            ('is_valid_tree(4, [[0, 1], [1, 2], [2, 0], [0, 3]])', "cycle", False),
            ('is_valid_tree(0, [])', "zero nodes", True),
        ),
        common_errors=(
            "Not checking for cycles",
            "Not checking connectivity",
            "Not checking edge count = n-1",
            "Not handling single node",
        ),
    ),
    CodingTask(
        task_id="code_129_topological_sort",
        description="Return topological ordering of a DAG, or empty list if cycle exists",
        function_name="topological_sort",
        signature="topological_sort(n: int, edges: list) -> list",
        docstring="Return a topological ordering of a directed graph with n nodes and edges [from, to]. Return empty list if the graph has a cycle.",
        difficulty="hard",
        tests=(
            # Easy probes
            ('topological_sort(4, [[0, 1], [0, 2], [1, 3], [2, 3]])', "basic DAG", [0, 1, 2, 3]),
            ('topological_sort(2, [[0, 1]])', "simple", [0, 1]),
            # Hard tests
            ('topological_sort(1, [])', "single node", [0]),
            ('topological_sort(3, [[0, 1], [1, 2], [2, 0]])', "cycle", []),
            ('topological_sort(6, [[5, 2], [5, 0], [4, 0], [4, 1], [2, 3], [3, 1]])', "complex", [5, 4, 2, 3, 1, 0]),
            ('topological_sort(3, [[0, 1], [0, 2]])', "branching", [0, 1, 2]),
            ('topological_sort(0, [])', "empty", []),
            ('topological_sort(5, [[0, 1], [1, 2], [2, 3], [3, 4]])', "chain", [0, 1, 2, 3, 4]),
        ),
        common_errors=(
            "Not detecting cycles",
            "Incorrect Kahn's algorithm (in-degree tracking)",
            "Not handling single node",
            "Not handling empty graph",
        ),
    ),
    CodingTask(
        task_id="code_130_alien_dictionary",
        description="Derive alien alphabet order from sorted words",
        function_name="alien_order",
        signature="alien_order(words: list) -> str",
        docstring="Given a list of words sorted lexicographically in an alien language, derive the order of unique characters. Return empty string if the order is invalid (e.g., contradictory or prefix issue).",
        difficulty="hard",
        tests=(
            # Easy probes
            ('alien_order(["wrt", "wrf", "er", "ett", "rftt"])', "classic", "wertf"),
            ('alien_order(["z", "x"])', "simple", "zx"),
            # Hard tests
            ('alien_order(["z", "x", "z"])', "invalid cycle", ""),
            ('alien_order(["abc", "ab"])', "prefix issue", ""),
            ('alien_order(["a", "b", "c"])', "chain", "abc"),
            ('alien_order(["a"])', "single word", "a"),
            ('alien_order(["ab", "ac"])', "two words", "abc"),
            ('alien_order(["baa", "abcd", "abca", "cab", "cad"])', "complex", "bdac"),
            ('alien_order(["zy", "zx"])', "reverse", "zyx"),
        ),
        common_errors=(
            "Not detecting cycles in the graph",
            "Not detecting prefix issues (abc before ab)",
            "Incorrect graph construction from adjacent words",
            "Not handling single word",
        ),
    ),
]


def get_misleading_probe_tasks() -> list[CodingTask]:
    """Return tasks designed for misleading probe tests."""
    return MISLEADING_PROBE_TASKS


if __name__ == "__main__":
    print(f"Misleading-probe tasks: {len(MISLEADING_PROBE_TASKS)}")
    for t in MISLEADING_PROBE_TASKS:
        n_tests = len(t.tests)
        print(f"  {t.task_id}: {n_tests} tests, {t.difficulty}")
