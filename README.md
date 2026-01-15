# Performance Comparison: AVL Tree vs B-Tree

This project implements and compares the performance of an AVL tree and B-Trees with different orders.
The comparison focuses on execution time and memory usage for insertion and search operations.

This work was developed as part of a seminar assignment on data structures.
Tested on Windows.

---

## 1. Project Structure

```
avl-vs-btree/
├── src/
│   ├── avl_tree.py        # AVL implementation + visualization
│   ├── b_tree.py          # B-Tree implementation
│   ├── benchmark.py      # Benchmark scripts
│   ├── plot_results.py   # Plot generation
│   └── datasets.py       # Dataset generator
│
├── tests/
│   ├── tests_b_tree.py
│   └── tests_avl.py
│
├── data/
│   ├── data_1000.txt
│   ├── data_5000.txt
│   ├── data_10000.txt
│   └── data_50000.txt
│
├── plots/                # Generated graphs
├── requirements.txt
└── README.md
```

---

## 2. Requirements

- Python 3.9+
- Graphviz (for visualization)

### Install Graphviz (Windows)

Download from:  
https://graphviz.org/download/

⚠ During installation:  
✔ Check **"Add Graphviz to PATH"**

Verify installation:

```bash
dot -V
```

---

## 3. Setup Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## 4. Run Unit Tests

```bash
python -m tests.tests_b_tree
python -m tests.tests_avl
```

---

## 5. Run Benchmarks

```bash
python src/benchmark.py
```

---

## 6. Generate Performance Plots

```bash
python src/plot_results.py
```

Generated graphs will appear in the `plots/` folder.

---

## 7. Visualization

### AVL Tree

```bash
python src/avl_tree.py
```

### B-Tree

```bash
python src/b_tree.py
```

PNG images will be generated showing the tree structures.

---

## 8. Dataset Generation

Datasets are already provided in the `data/` folder.

To regenerate them:

```bash
python src/datasets.py
```

---

## 9. Reproducibility

- A fixed random seed (42) is used
- All datasets are deterministic
- All experiments can be reproduced using the provided scripts

---

## 10. Notes

- AVL deletion is fully implemented
- B-Tree deletion is not benchmarked (complexity reasons, documented in report)
- Only integer values are used as required
- Experiments are performed on incremental dataset sizes

---

## 11. Author

Matthieu Charton  
GitHub: https://github.com/ChartonMatthieu
