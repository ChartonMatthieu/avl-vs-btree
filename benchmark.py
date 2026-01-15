import time
import tracemalloc
import random

from avl_tree import AVLTree
from b_tree import BTree


DATASET_SIZES = [1_000, 5_000, 10_000, 50_000]
BTREE_T_VALUES = [2, 3, 5]
SEED = 42


def benchmark_structure(name, structure, values):
    # ---------- INSERT ----------
    tracemalloc.start()
    t0 = time.perf_counter()
    for v in values:
        structure.insert(v)
    insert_time = time.perf_counter() - t0
    _, insert_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # ---------- SEARCH ----------
    t0 = time.perf_counter()
    for v in values:
        structure.search(v)
    search_time = time.perf_counter() - t0

    return insert_time, search_time, insert_mem / 1024  # KB


def benchmark_avl(values):
    avl = AVLTree()
    return benchmark_structure("AVL", avl, values)


def benchmark_btree(values, t):
    btree = BTree(t=t)
    return benchmark_structure(f"BTree(t={t})", btree, values)


def run():
    random.seed(SEED)
    results = []

    for size in DATASET_SIZES:
        values = random.sample(range(size * 10), size)

        # AVL
        avl_res = benchmark_avl(values)
        results.append(("AVL", size, *avl_res))

        # B-Trees
        for t in BTREE_T_VALUES:
            res = benchmark_btree(values, t)
            results.append((f"BTree(t={t})", size, *res))

    return results


if __name__ == "__main__":
    results = run()
    for r in results:
        print(r)
