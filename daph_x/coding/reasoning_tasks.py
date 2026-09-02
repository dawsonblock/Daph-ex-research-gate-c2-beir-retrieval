#!/usr/bin/env python3
"""Reasoning tasks for DAPH-X authority evaluation.

These tasks have NO execution feedback — the authority must decide
based on reasoning features alone. Correctness is verified by
checking the final answer, but there are no "probe tests" to run.

This is the regime where the original DAPH formulation was designed:
the authority must estimate value from the reasoning process, not
from execution evidence.

Task types:
  - Math word problems (verifiable numeric answer)
  - Logic puzzles (verifiable boolean/string answer)
  - Sequence completion (verifiable next element)
  - Combinatorics (verifiable count)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ReasoningTask:
    task_id: str
    description: str
    prompt: str  # The question to ask the model
    answer: str  # The correct answer (string match)
    answer_type: str  # "int", "float", "string", "bool"
    difficulty: str  # "easy", "medium", "hard"
    category: str  # "math", "logic", "sequence", "combinatorics"
    common_errors: tuple[str, ...]


REASONING_TASKS: list[ReasoningTask] = [
    # ─── Math: arithmetic word problems ───
    ReasoningTask(
        task_id="reason_001",
        description="Train speed problem",
        prompt="A train travels 240 miles in 4 hours. At the same speed, how far will it travel in 7 hours? Answer with just the number.",
        answer="420",
        answer_type="int",
        difficulty="easy",
        category="math",
        common_errors=("Unit confusion", "Rate calculation error"),
    ),
    ReasoningTask(
        task_id="reason_002",
        description="Percentage problem",
        prompt="A store marks up items by 40% then offers a 25% discount. What percentage of the original price does the customer pay? Answer with just the number.",
        answer="105",
        answer_type="int",
        difficulty="medium",
        category="math",
        common_errors=("Adding percentages instead of multiplying", "Not computing compound effect"),
    ),
    ReasoningTask(
        task_id="reason_003",
        description="Compound interest",
        prompt="If you invest $1000 at 10% annual interest compounded annually, how much money will you have after 3 years? Answer with just the number (rounded to nearest integer).",
        answer="1331",
        answer_type="int",
        difficulty="medium",
        category="math",
        common_errors=("Simple interest instead of compound", "Rounding errors"),
    ),
    ReasoningTask(
        task_id="reason_004",
        description="Work rate problem",
        prompt="Alice can paint a room in 6 hours. Bob can paint the same room in 4 hours. Working together, how many hours will it take them to paint the room? Answer with just the number (rounded to 2 decimal places).",
        answer="2.40",
        answer_type="float",
        difficulty="medium",
        category="math",
        common_errors=("Averaging the times", "Not using reciprocal rates"),
    ),
    ReasoningTask(
        task_id="reason_005",
        description="Mixture problem",
        prompt="How many liters of a 20% acid solution must be added to 10 liters of a 50% acid solution to get a 30% acid solution? Answer with just the number.",
        answer="20",
        answer_type="int",
        difficulty="hard",
        category="math",
        common_errors=("Incorrect equation setup", "Solving for wrong variable"),
    ),
    ReasoningTask(
        task_id="reason_006",
        description="Speed/distance/time",
        prompt="Two cars start 300 miles apart and drive toward each other. One drives 60 mph, the other 40 mph. How long until they meet (in hours)? Answer with just the number.",
        answer="3",
        answer_type="int",
        difficulty="easy",
        category="math",
        common_errors=("Dividing by one speed", "Not adding speeds"),
    ),
    ReasoningTask(
        task_id="reason_007",
        description="Geometric sequence sum",
        prompt="What is the sum of the first 10 terms of a geometric sequence with first term 3 and common ratio 2? Answer with just the number.",
        answer="3069",
        answer_type="int",
        difficulty="hard",
        category="math",
        common_errors=("Using arithmetic sum formula", "Off-by-one in term count"),
    ),
    ReasoningTask(
        task_id="reason_008",
        description="Probability problem",
        prompt="A bag contains 5 red marbles and 3 blue marbles. Two marbles are drawn without replacement. What is the probability that both are red? Express as a fraction a/b.",
        answer="5/14",
        answer_type="string",
        difficulty="medium",
        category="math",
        common_errors=("Not using conditional probability", "Computing with replacement"),
    ),
    ReasoningTask(
        task_id="reason_009",
        description="Integer sequence",
        prompt="What is the next number in the sequence: 2, 6, 12, 20, 30, ?",
        answer="42",
        answer_type="int",
        difficulty="medium",
        category="sequence",
        common_errors=("Arithmetic progression assumption", "Not recognizing n(n+1) pattern"),
    ),
    ReasoningTask(
        task_id="reason_010",
        description="Fibonacci variant",
        prompt="What is the next number in the sequence: 1, 1, 2, 3, 5, 8, 13, ?",
        answer="21",
        answer_type="int",
        difficulty="easy",
        category="sequence",
        common_errors=("Arithmetic instead of Fibonacci", "Off-by-one"),
    ),
    ReasoningTask(
        task_id="reason_011",
        description="Geometric sequence",
        prompt="What is the next number in the sequence: 3, 9, 27, 81, ?",
        answer="243",
        answer_type="int",
        difficulty="easy",
        category="sequence",
        common_errors=("Adding instead of multiplying", "Wrong ratio"),
    ),
    ReasoningTask(
        task_id="reason_012",
        description="Triangular numbers",
        prompt="What is the next number in the sequence: 1, 3, 6, 10, 15, ?",
        answer="21",
        answer_type="int",
        difficulty="medium",
        category="sequence",
        common_errors=("Arithmetic progression", "Not recognizing triangular pattern"),
    ),
    # ─── Logic puzzles ───
    ReasoningTask(
        task_id="reason_013",
        description="Knights and knaves",
        prompt="On an island, knights always tell the truth and knaves always lie. You meet two people, A and B. A says 'B is a knave.' B says 'We are both knaves.' What is A? Answer 'knight' or 'knave'.",
        answer="knight",
        answer_type="string",
        difficulty="hard",
        category="logic",
        common_errors=("Not checking consistency", "Assuming B tells truth"),
    ),
    ReasoningTask(
        task_id="reason_014",
        description="Syllogism",
        prompt="All cats are mammals. Some mammals are pets. Therefore: (a) all cats are pets, (b) some cats are pets, (c) no cats are pets, (d) none of the above can be concluded. Answer with just the letter.",
        answer="d",
        answer_type="string",
        difficulty="medium",
        category="logic",
        common_errors=("Choosing (b) incorrectly", "Not understanding valid syllogisms"),
    ),
    ReasoningTask(
        task_id="reason_015",
        description="Truth table",
        prompt="If P is true and Q is false, what is the value of (P AND Q) OR (NOT Q)? Answer 'true' or 'false'.",
        answer="true",
        answer_type="string",
        difficulty="easy",
        category="logic",
        common_errors=("Evaluating AND first incorrectly", "NOT operator error"),
    ),
    ReasoningTask(
        task_id="reason_016",
        description="Conditional logic",
        prompt="If it rains, the picnic is cancelled. The picnic was not cancelled. Did it rain? Answer 'yes', 'no', or 'cannot determine'.",
        answer="no",
        answer_type="string",
        difficulty="medium",
        category="logic",
        common_errors=("Affirming the consequent", "Confusing contrapositive"),
    ),
    # ─── Combinatorics ───
    ReasoningTask(
        task_id="reason_017",
        description="Permutations",
        prompt="How many ways can 5 books be arranged on a shelf? Answer with just the number.",
        answer="120",
        answer_type="int",
        difficulty="easy",
        category="combinatorics",
        common_errors=("Computing combinations instead", "Not using factorial"),
    ),
    ReasoningTask(
        task_id="reason_018",
        description="Combinations",
        prompt="How many ways can you choose 3 items from 7 items? Answer with just the number.",
        answer="35",
        answer_type="int",
        difficulty="medium",
        category="combinatorics",
        common_errors=("Using permutation formula", "Arithmetic error in C(7,3)"),
    ),
    ReasoningTask(
        task_id="reason_019",
        description="Stars and bars",
        prompt="How many ways can you distribute 10 identical candies among 3 children? Answer with just the number.",
        answer="66",
        answer_type="int",
        difficulty="hard",
        category="combinatorics",
        common_errors=("Not using stars and bars", "Treating candies as distinct"),
    ),
    ReasoningTask(
        task_id="reason_020",
        description="Derangements",
        prompt="How many derangements (permutations where no element is in its original position) are there of 4 elements? Answer with just the number.",
        answer="9",
        answer_type="int",
        difficulty="hard",
        category="combinatorics",
        common_errors=("Not knowing derangement formula", "Computing permutations instead"),
    ),
    # ─── More math ───
    ReasoningTask(
        task_id="reason_021",
        description="Modular arithmetic",
        prompt="What is 7^100 mod 11? Answer with just the number.",
        answer="1",
        answer_type="int",
        difficulty="hard",
        category="math",
        common_errors=("Not using Fermat's little theorem", "Direct computation overflow"),
    ),
    ReasoningTask(
        task_id="reason_022",
        description="Number theory",
        prompt="What is the GCD of 1071 and 462? Answer with just the number.",
        answer="3",
        answer_type="int",
        difficulty="medium",
        category="math",
        common_errors=("Not using Euclidean algorithm", "Arithmetic errors"),
    ),
    ReasoningTask(
        task_id="reason_023",
        description="Algebra",
        prompt="If x^2 - 5x + 6 = 0, what is the sum of all possible values of x? Answer with just the number.",
        answer="5",
        answer_type="int",
        difficulty="medium",
        category="math",
        common_errors=("Solving for x instead of using Vieta's", "Not recognizing both roots"),
    ),
    ReasoningTask(
        task_id="reason_024",
        description="Inequality",
        prompt="For what value of x is |2x - 3| = 7? If there are two solutions, give the larger one. Answer with just the number.",
        answer="5",
        answer_type="int",
        difficulty="medium",
        category="math",
        common_errors=("Only finding one solution", "Sign error in absolute value"),
    ),
    ReasoningTask(
        task_id="reason_025",
        description="Logarithm",
        prompt="If log_2(x) = 10, what is x? Answer with just the number.",
        answer="1024",
        answer_type="int",
        difficulty="medium",
        category="math",
        common_errors=("Computing log instead of exponential", "Base confusion"),
    ),
    ReasoningTask(
        task_id="reason_026",
        description="Sum of arithmetic series",
        prompt="What is the sum of all integers from 1 to 100? Answer with just the number.",
        answer="5050",
        answer_type="int",
        difficulty="easy",
        category="math",
        common_errors=("Not using n(n+1)/2", "Arithmetic error"),
    ),
    ReasoningTask(
        task_id="reason_027",
        description="Pythagorean theorem",
        prompt="A right triangle has legs of length 5 and 12. What is the length of the hypotenuse? Answer with just the number.",
        answer="13",
        answer_type="int",
        difficulty="easy",
        category="math",
        common_errors=("Adding legs instead of sqrt(a^2+b^2)", "Computation error"),
    ),
    ReasoningTask(
        task_id="reason_028",
        description="Area problem",
        prompt="A circle has radius 6. What is its area? Use pi = 3.14. Answer with just the number.",
        answer="113.04",
        answer_type="float",
        difficulty="easy",
        category="math",
        common_errors=("Using diameter instead of radius", "Not squaring radius"),
    ),
    ReasoningTask(
        task_id="reason_029",
        description="Ratio problem",
        prompt="The ratio of boys to girls in a class is 3:2. If there are 10 girls, how many boys are there? Answer with just the number.",
        answer="15",
        answer_type="int",
        difficulty="easy",
        category="math",
        common_errors=("Inverting the ratio", "Not scaling properly"),
    ),
    ReasoningTask(
        task_id="reason_030",
        description="Age problem",
        prompt="Tom is currently twice as old as Jerry. Ten years ago, Tom was three times as old as Jerry. How old is Tom now? Answer with just the number.",
        answer="40",
        answer_type="int",
        difficulty="hard",
        category="math",
        common_errors=("Setting up wrong equations", "Solving for wrong variable"),
    ),
    ReasoningTask(
        task_id="reason_031",
        description="Digit sum",
        prompt="What is the digit sum of 2^20? Answer with just the number.",
        answer="31",
        answer_type="int",
        difficulty="hard",
        category="math",
        common_errors=("Computing 2^20 incorrectly", "Digit sum arithmetic error"),
    ),
    ReasoningTask(
        task_id="reason_032",
        description="Remainder problem",
        prompt="What is the remainder when 100! is divided by 97? Answer with just the number.",
        answer="0",
        answer_type="int",
        difficulty="hard",
        category="math",
        common_errors=("Not recognizing 97 is a factor of 100!", "Trying to compute 100! directly"),
    ),
    ReasoningTask(
        task_id="reason_033",
        description="Average speed",
        prompt="A car travels 60 miles at 30 mph and returns 60 miles at 60 mph. What is the average speed for the entire trip? Answer with just the number (rounded to 1 decimal place).",
        answer="40.0",
        answer_type="float",
        difficulty="medium",
        category="math",
        common_errors=("Averaging the speeds (45)", "Not using harmonic mean"),
    ),
    ReasoningTask(
        task_id="reason_034",
        description="LCM problem",
        prompt="What is the LCM of 12, 15, and 20? Answer with just the number.",
        answer="60",
        answer_type="int",
        difficulty="medium",
        category="math",
        common_errors=("Computing GCD instead", "Not factoring all three"),
    ),
    ReasoningTask(
        task_id="reason_035",
        description="Telescoping series",
        prompt="What is 1/(1*2) + 1/(2*3) + 1/(3*4) + ... + 1/(99*100)? Answer as a decimal.",
        answer="0.99",
        answer_type="float",
        difficulty="hard",
        category="math",
        common_errors=("Not recognizing telescoping pattern", "Computing each term individually"),
    ),
    ReasoningTask(
        task_id="reason_036",
        description="Clock angle",
        prompt="What is the angle between the hour and minute hands of a clock at 3:15? Answer with just the number (in degrees).",
        answer="7.5",
        answer_type="float",
        difficulty="hard",
        category="math",
        common_errors=("Saying 0 degrees", "Not accounting for hour hand movement"),
    ),
    ReasoningTask(
        task_id="reason_037",
        description="Coin problem",
        prompt="What is the minimum number of coins needed to make 67 cents using quarters (25c), dimes (10c), nickels (5c), and pennies (1c)? Answer with just the number.",
        answer="6",
        answer_type="int",
        difficulty="medium",
        category="math",
        common_errors=("Greedy doesn't always work (but does for US coins)", "Counting error"),
    ),
    ReasoningTask(
        task_id="reason_038",
        description="Pascal's triangle",
        prompt="What is the value in the 5th row and 3rd position (0-indexed) of Pascal's triangle? Answer with just the number.",
        answer="10",
        answer_type="int",
        difficulty="medium",
        category="combinatorics",
        common_errors=("1-indexed vs 0-indexed confusion", "Wrong row/column"),
    ),
    ReasoningTask(
        task_id="reason_039",
        description="Euler's formula",
        prompt="A polyhedron has 8 vertices and 12 edges. How many faces does it have? Answer with just the number.",
        answer="6",
        answer_type="int",
        difficulty="medium",
        category="math",
        common_errors=("Not knowing V-E+F=2", "Wrong sign"),
    ),
    ReasoningTask(
        task_id="reason_040",
        description="Sum of squares",
        prompt="What is 1^2 + 2^2 + 3^2 + ... + 10^2? Answer with just the number.",
        answer="385",
        answer_type="int",
        difficulty="medium",
        category="math",
        common_errors=("Using n(n+1)/2 instead of n(n+1)(2n+1)/6", "Arithmetic error"),
    ),
    # ─── More logic ───
    ReasoningTask(
        task_id="reason_041",
        description="Conditional with contrapositive",
        prompt="If all Bloops are Razzies and all Razzies are Lazzies, then are all Bloops definitely Lazzies? Answer 'yes' or 'no'.",
        answer="yes",
        answer_type="string",
        difficulty="easy",
        category="logic",
        common_errors=("Not recognizing transitivity", "Overthinking"),
    ),
    ReasoningTask(
        task_id="reason_042",
        description="Set theory",
        prompt="In a class of 30 students, 18 like math and 15 like science. If 8 like both, how many like neither? Answer with just the number.",
        answer="5",
        answer_type="int",
        difficulty="medium",
        category="logic",
        common_errors=("Not using inclusion-exclusion", "Subtracting both instead of intersection"),
    ),
    ReasoningTask(
        task_id="reason_043",
        description="Truth teller puzzle",
        prompt="Three friends — Alex, Bob, and Carl — are standing in a line. Alex says 'Bob is lying.' Bob says 'Carl is lying.' Carl says 'Alex and Bob are both lying.' Who is telling the truth? Answer with the name.",
        answer="Bob",
        answer_type="string",
        difficulty="hard",
        category="logic",
        common_errors=("Not checking all combinations", "Inconsistent reasoning"),
    ),
    ReasoningTask(
        task_id="reason_044",
        description="River crossing",
        prompt="A farmer needs to cross a river with a fox, a chicken, and a bag of grain. The boat can carry the farmer and one item. The fox eats the chicken if left alone; the chicken eats the grain if left alone. What is the minimum number of crossings needed? Answer with just the number.",
        answer="7",
        answer_type="int",
        difficulty="hard",
        category="logic",
        common_errors=("Not bringing the chicken back", "Counting one-way trips only"),
    ),
    ReasoningTask(
        task_id="reason_045",
        description="Bertrand box",
        prompt="There are 3 boxes: one with 2 gold coins, one with 2 silver coins, and one with 1 gold and 1 silver. You pick a random box and draw a gold coin. What is the probability the next coin from the same box is also gold? Answer as a fraction a/b.",
        answer="2/3",
        answer_type="string",
        difficulty="hard",
        category="math",
        common_errors=("Saying 1/2 (conditional probability error)", "Not counting all gold coin scenarios"),
    ),
    # ─── More sequences ───
    ReasoningTask(
        task_id="reason_046",
        description="Square numbers",
        prompt="What is the next number in the sequence: 1, 4, 9, 16, 25, ?",
        answer="36",
        answer_type="int",
        difficulty="easy",
        category="sequence",
        common_errors=("Arithmetic progression", "Not recognizing squares"),
    ),
    ReasoningTask(
        task_id="reason_047",
        description="Cube sequence",
        prompt="What is the next number in the sequence: 1, 8, 27, 64, ?",
        answer="125",
        answer_type="int",
        difficulty="medium",
        category="sequence",
        common_errors=("Arithmetic progression", "Not recognizing cubes"),
    ),
    ReasoningTask(
        task_id="reason_048",
        description="Powers of 2",
        prompt="What is the next number in the sequence: 1, 2, 4, 8, 16, 32, ?",
        answer="64",
        answer_type="int",
        difficulty="easy",
        category="sequence",
        common_errors=("Adding instead of doubling", "Off-by-one"),
    ),
    ReasoningTask(
        task_id="reason_049",
        description="Lucas numbers",
        prompt="What is the next number in the sequence: 2, 1, 3, 4, 7, 11, ?",
        answer="18",
        answer_type="int",
        difficulty="hard",
        category="sequence",
        common_errors=("Not recognizing Lucas pattern", "Arithmetic assumption"),
    ),
    ReasoningTask(
        task_id="reason_050",
        description="Pentagonal numbers",
        prompt="What is the next number in the sequence: 1, 5, 12, 22, 35, ?",
        answer="51",
        answer_type="int",
        difficulty="hard",
        category="sequence",
        common_errors=("Not recognizing pentagonal pattern", "Wrong difference calculation"),
    ),
    # ─── More combinatorics ───
    ReasoningTask(
        task_id="reason_051",
        description="Circular permutations",
        prompt="How many ways can 6 people be seated around a circular table? Answer with just the number.",
        answer="120",
        answer_type="int",
        difficulty="hard",
        category="combinatorics",
        common_errors=("Not dividing by n for circular", "Using 6! instead of 5!"),
    ),
    ReasoningTask(
        task_id="reason_052",
        description="Binomial coefficient",
        prompt="What is C(10,4)? Answer with just the number.",
        answer="210",
        answer_type="int",
        difficulty="medium",
        category="combinatorics",
        common_errors=("Arithmetic error", "Using P(10,4) instead"),
    ),
    ReasoningTask(
        task_id="reason_053",
        description="Inclusion-exclusion",
        prompt="In a group of 50 people, 25 speak English, 20 speak French, and 10 speak both. How many speak neither? Answer with just the number.",
        answer="15",
        answer_type="int",
        difficulty="medium",
        category="combinatorics",
        common_errors=("Not using inclusion-exclusion", "Adding instead of subtracting intersection"),
    ),
    ReasoningTask(
        task_id="reason_054",
        description="Stars and bars variant",
        prompt="How many non-negative integer solutions are there to x + y + z = 8? Answer with just the number.",
        answer="45",
        answer_type="int",
        difficulty="hard",
        category="combinatorics",
        common_errors=("Not using C(10,2)", "Counting manually"),
    ),
    ReasoningTask(
        task_id="reason_055",
        description="Catalan number",
        prompt="What is the 4th Catalan number (0-indexed)? Answer with just the number.",
        answer="14",
        answer_type="int",
        difficulty="hard",
        category="combinatorics",
        common_errors=("Not knowing Catalan formula", "1-indexed vs 0-indexed"),
    ),
    # ─── Advanced math ───
    ReasoningTask(
        task_id="reason_056",
        description="Sum of divisors",
        prompt="What is the sum of all positive divisors of 28? Answer with just the number.",
        answer="56",
        answer_type="int",
        difficulty="medium",
        category="math",
        common_errors=("Missing divisors", "Including 28 twice"),
    ),
    ReasoningTask(
        task_id="reason_057",
        description="Perfect number check",
        prompt="Is 496 a perfect number (sum of proper divisors equals the number)? Answer 'yes' or 'no'.",
        answer="yes",
        answer_type="string",
        difficulty="hard",
        category="math",
        common_errors=("Not computing all divisors", "Arithmetic error"),
    ),
    ReasoningTask(
        task_id="reason_058",
        description="Prime counting",
        prompt="How many prime numbers are there between 1 and 20? Answer with just the number.",
        answer="8",
        answer_type="int",
        difficulty="easy",
        category="math",
        common_errors=("Including 1 as prime", "Missing a prime"),
    ),
    ReasoningTask(
        task_id="reason_059",
        description="Euler totient",
        prompt="What is Euler's totient function phi(12) (count of integers 1-12 coprime to 12)? Answer with just the number.",
        answer="4",
        answer_type="int",
        difficulty="hard",
        category="math",
        common_errors=("Not knowing totient formula", "Counting non-coprime instead"),
    ),
    ReasoningTask(
        task_id="reason_060",
        description="Matrix determinant",
        prompt="What is the determinant of the 2x2 matrix [[3, 4], [2, 5]]? Answer with just the number.",
        answer="7",
        answer_type="int",
        difficulty="medium",
        category="math",
        common_errors=("Wrong formula (ad+bc instead of ad-bc)", "Sign error"),
    ),
]


def get_all_reasoning_tasks() -> list[ReasoningTask]:
    return REASONING_TASKS


def get_reasoning_task(task_id: str) -> ReasoningTask | None:
    for t in REASONING_TASKS:
        if t.task_id == task_id:
            return t
    return None


def check_answer(response: str, correct_answer: str, answer_type: str) -> bool:
    """Check if a response matches the correct answer."""
    response = response.strip().lower()
    correct = correct_answer.strip().lower()

    if answer_type == "int":
        try:
            return int(response) == int(correct)
        except ValueError:
            return False
    elif answer_type == "float":
        try:
            return abs(float(response) - float(correct)) < 0.01
        except ValueError:
            return False
    else:
        return response == correct


if __name__ == "__main__":
    print(f"Reasoning tasks: {len(REASONING_TASKS)}")
    by_cat = {}
    for t in REASONING_TASKS:
        by_cat.setdefault(t.category, []).append(t)
    for cat, tasks in sorted(by_cat.items()):
        print(f"  {cat}: {len(tasks)} tasks")
