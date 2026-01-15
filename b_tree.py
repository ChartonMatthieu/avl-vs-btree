from graphviz import Digraph

class BTreeNode:
    def __init__(self, t, leaf=False):
        self.t = t                  # degré minimal
        self.keys = []
        self.children = []
        self.leaf = leaf


class BTree:
    def __init__(self, t):
        self.t = t                  # t = ceil(m/2)
        self.root = BTreeNode(t, leaf=True)

    # ---------- SEARCH ----------
    def search(self, key, node=None):
        if node is None:
            node = self.root

        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1

        if i < len(node.keys) and key == node.keys[i]:
            return True

        if node.leaf:
            return False

        return self.search(key, node.children[i])

    # ---------- INSERT ----------
    def insert(self, key):
        root = self.root
        if len(root.keys) == 2 * self.t - 1:
            new_root = BTreeNode(self.t)
            new_root.children.append(root)
            self._split_child(new_root, 0)
            self.root = new_root
            self._insert_non_full(new_root, key)
        else:
            self._insert_non_full(root, key)

    def _insert_non_full(self, node, key):
        i = len(node.keys) - 1

        if node.leaf:
            node.keys.append(None)
            while i >= 0 and key < node.keys[i]:
                node.keys[i + 1] = node.keys[i]
                i -= 1
            node.keys[i + 1] = key
        else:
            while i >= 0 and key < node.keys[i]:
                i -= 1
            i += 1

            if len(node.children[i].keys) == 2 * self.t - 1:
                self._split_child(node, i)
                if key > node.keys[i]:
                    i += 1
            self._insert_non_full(node.children[i], key)

    def _split_child(self, parent, i):
        t = self.t
        full_child = parent.children[i]
        new_child = BTreeNode(t, leaf=full_child.leaf)

        parent.keys.insert(i, full_child.keys[t - 1])
        parent.children.insert(i + 1, new_child)

        new_child.keys = full_child.keys[t:]
        full_child.keys = full_child.keys[:t - 1]

        if not full_child.leaf:
            new_child.children = full_child.children[t:]
            full_child.children = full_child.children[:t]

    def visualize(self, filename="btree"):
        dot = Digraph()
        self._add_nodes(dot, self.root)
        dot.render(filename, format="png", cleanup=True)

    def _add_nodes(self, dot, node):
        label = "|".join(str(k) for k in node.keys)
        dot.node(str(id(node)), label=label, shape="record")

        for child in node.children:
            dot.edge(str(id(node)), str(id(child)))
            self._add_nodes(dot, child)


if __name__ == "__main__":
    btree = BTree(t=2)  # ordre m = 4

    values = [10, 20, 5, 6, 12, 30, 7, 17]
    for v in values:
        btree.insert(v)

    print(btree.search(6))   # True
    print(btree.search(99))  # False
    btree.visualize("btree_t2")


