from b_tree import BTree
import random


def run_basic_tests(t):
    print(f"\n===== Testing BTree(t={t})  (m={2*t}) =====")

    btree = BTree(t=t)

    values = [10, 20, 5, 6, 12, 30, 7, 17]
    for v in values:
        btree.insert(v)

    for v in values:
        assert btree.search(v), f"Value {v} should be found"

    for v in [99, -1, 1000]:
        assert not btree.search(v), f"Value {v} should NOT be found"

    print("✔ Basic insertion/search passed")


def run_random_tests(t, n=1000, seed=42):
    print(f"--- Random test n={n} ---")
    random.seed(seed)

    btree = BTree(t=t)
    values = random.sample(range(1, n * 10), n)

    for v in values:
        btree.insert(v)

    for v in values:
        assert btree.search(v)

    print("✔ Random test passed")


if __name__ == "__main__":
    for t in [2, 3, 5]:
        run_basic_tests(t)
        run_random_tests(t)
