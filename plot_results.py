import matplotlib.pyplot as plt
from collections import defaultdict
from benchmark import run

results = run()

data = defaultdict(lambda: {"size": [], "insert": [], "search": [], "mem": []})

for name, size, insert, search, mem in results:
    data[name]["size"].append(size)
    data[name]["insert"].append(insert)
    data[name]["search"].append(search)
    data[name]["mem"].append(mem)

# ---------- INSERT TIME ----------
plt.figure()
for name, d in data.items():
    plt.plot(d["size"], d["insert"], label=name)
plt.xlabel("Dataset size")
plt.ylabel("Insert time (s)")
plt.legend()
plt.title("Insertion time comparison")
plt.show()

# ---------- MEMORY ----------
plt.figure()
for name, d in data.items():
    plt.plot(d["size"], d["mem"], label=name)
plt.xlabel("Dataset size")
plt.ylabel("Memory usage (KB)")
plt.legend()
plt.title("Memory usage comparison")
plt.show()
