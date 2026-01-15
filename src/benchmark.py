import sys
import os
import time
import tracemalloc
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from src.avl_tree import AVLTree
from src.b_tree import BTree

# ---------------- CONFIG ----------------

DATASET_SIZES = [1000, 5000, 10000, 50000]
BTREE_T_VALUES = [2, 3, 5]
PLOTS_DIR = os.path.join(ROOT, "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------------- DATA LOADER ----------------

def load_dataset(size):
    path = os.path.join(ROOT, "data", f"data_{size}.txt")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Run: python src/datasets.py first."
        )

    with open(path, encoding="utf-8") as f:
        return [int(x.strip()) for x in f if x.strip()]


# ---------------- MEASURE ----------------

def measure_insert(structure, values):
    tracemalloc.start()
    start = time.perf_counter()

    for v in values:
        structure.insert(v)

    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return elapsed, peak / 1024


def measure_search(structure, values):
    start = time.perf_counter()

    for v in values:
        structure.search(v)

    return time.perf_counter() - start


def measure_delete_avl(avl, values):
    to_delete = values[:len(values)//2]

    start = time.perf_counter()
    for v in to_delete:
        avl.delete(v)

    return time.perf_counter() - start


# ---------------- BENCHMARK ----------------

def benchmark_avl(values):
    avl = AVLTree()

    ins, mem = measure_insert(avl, values)
    sea = measure_search(avl, values)
    dele = measure_delete_avl(avl, values)

    return ins, sea, dele, mem


def benchmark_btree(values, t):
    btree = BTree(t=t)

    ins, mem = measure_insert(btree, values)
    sea = measure_search(btree, values)

    return ins, sea, mem


# ---------------- PLOTS ----------------

def plot_results(results):
    sizes = DATASET_SIZES

    avl_insert = [results[s]["avl"]["insert"] for s in sizes]
    bt2_insert = [results[s]["bt2"]["insert"] for s in sizes]
    bt3_insert = [results[s]["bt3"]["insert"] for s in sizes]
    bt5_insert = [results[s]["bt5"]["insert"] for s in sizes]

    avl_mem = [results[s]["avl"]["mem"] for s in sizes]
    bt2_mem = [results[s]["bt2"]["mem"] for s in sizes]
    bt3_mem = [results[s]["bt3"]["mem"] for s in sizes]
    bt5_mem = [results[s]["bt5"]["mem"] for s in sizes]

    # INSERT TIME
    plt.figure()
    plt.plot(sizes, avl_insert, label="AVL")
    plt.plot(sizes, bt2_insert, label="BTree(t=2)")
    plt.plot(sizes, bt3_insert, label="BTree(t=3)")
    plt.plot(sizes, bt5_insert, label="BTree(t=5)")
    plt.xlabel("Dataset size")
    plt.ylabel("Insert time (s)")
    plt.title("Insertion time comparison")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(PLOTS_DIR, "insert_time.png"))
    plt.close()

    # MEMORY
    plt.figure()
    plt.plot(sizes, avl_mem, label="AVL")
    plt.plot(sizes, bt2_mem, label="BTree(t=2)")
    plt.plot(sizes, bt3_mem, label="BTree(t=3)")
    plt.plot(sizes, bt5_mem, label="BTree(t=5)")
    plt.xlabel("Dataset size")
    plt.ylabel("Memory usage (KB)")
    plt.title("Memory usage comparison")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(PLOTS_DIR, "memory_usage.png"))
    plt.close()

    print(f"Plots generated in: {PLOTS_DIR}")


# ---------------- RUN ----------------

def run():
    results = {}

    print("\n====== BENCHMARK RESULTS ======\n")

    for size in DATASET_SIZES:
        print(f"\n--- Dataset size: {size} ---")
        values = load_dataset(size)

        results[size] = {}

        # AVL
        ai, asr, ad, am = benchmark_avl(values)
        results[size]["avl"] = {
            "insert": ai,
            "search": asr,
            "delete": ad,
            "mem": am
        }

        print("\nAVL")
        print(f"Insert : {ai:.4f}s")
        print(f"Search : {asr:.4f}s")
        print(f"Delete : {ad:.4f}s")
        print(f"Memory : {am:.2f}KB")

        # B-Trees
        for t in BTREE_T_VALUES:
            bi, bs, bm = benchmark_btree(values, t)

            results[size][f"bt{t}"] = {
                "insert": bi,
                "search": bs,
                "mem": bm
            }

            print(f"\nBTree t={t}")
            print(f"Insert : {bi:.4f}s")
            print(f"Search : {bs:.4f}s")
            print(f"Memory : {bm:.2f}KB")

    plot_results(results)


if __name__ == "__main__":
    run()
