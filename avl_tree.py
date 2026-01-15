from graphviz import Digraph

class AVLNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.height = 1


class AVLTree:
    def __init__(self):
        self.root = None

    def _height(self, node):
        return node.height if node else 0

    def _balance(self, node):
        return self._height(node.left) - self._height(node.right)

    def _update_height(self, node):
        node.height = 1 + max(self._height(node.left), self._height(node.right))

    # ---------- ROTATIONS ----------
    def _rotate_right(self, y):
        x = y.left
        T2 = x.right

        x.right = y
        y.left = T2

        self._update_height(y)
        self._update_height(x)

        return x

    def _rotate_left(self, x):
        y = x.right
        T2 = y.left

        y.left = x
        x.right = T2

        self._update_height(x)
        self._update_height(y)

        return y

    # ---------- INSERTION ----------
    def insert(self, key):
        self.root = self._insert(self.root, key)

    def _insert(self, node, key):
        if not node:
            return AVLNode(key)

        if key < node.key:
            node.left = self._insert(node.left, key)
        elif key > node.key:
            node.right = self._insert(node.right, key)
        else:
            return node

        self._update_height(node)
        balance = self._balance(node)

        if balance > 1 and key < node.left.key:
            return self._rotate_right(node)

        if balance < -1 and key > node.right.key:
            return self._rotate_left(node)

        if balance > 1 and key > node.left.key:
            node.left = self._rotate_left(node.left)
            return self._rotate_right(node)

        if balance < -1 and key < node.right.key:
            node.right = self._rotate_right(node.right)
            return self._rotate_left(node)

        return node

    # ---------- SEARCH ----------
    def search(self, key):
        return self._search(self.root, key)

    def _search(self, node, key):
        if not node:
            return False
        if key == node.key:
            return True
        if key < node.key:
            return self._search(node.left, key)
        return self._search(node.right, key)
    

# ---------- VISUALISATION ----------
    def visualize(self, filename="avl_tree"):
        dot = Digraph()
        self._add_nodes(dot, self.root)
        dot.render(filename, format="png", cleanup=True)

    def _add_nodes(self, dot, node):
        if not node:
            return

        label = f"{node.key}\nh={node.height}"
        dot.node(str(id(node)), label)

        if node.left:
            dot.edge(str(id(node)), str(id(node.left)))
            self._add_nodes(dot, node.left)

        if node.right:
            dot.edge(str(id(node)), str(id(node.right)))
            self._add_nodes(dot, node.right)

 # ---------- DELETE ----------
    def delete(self, key):
        self.root = self._delete(self.root, key)

    def _delete(self, node, key):
        if not node:
            return node

        # BST delete
        if key < node.key:
            node.left = self._delete(node.left, key)
        elif key > node.key:
            node.right = self._delete(node.right, key)
        else:
            if not node.left:
                return node.right
            elif not node.right:
                return node.left

            temp = self._min_value_node(node.right)
            node.key = temp.key
            node.right = self._delete(node.right, temp.key)

        self._update_height(node)
        balance = self._balance(node)

        # LL
        if balance > 1 and self._balance(node.left) >= 0:
            return self._rotate_right(node)

        # LR
        if balance > 1 and self._balance(node.left) < 0:
            node.left = self._rotate_left(node.left)
            return self._rotate_right(node)

        # RR
        if balance < -1 and self._balance(node.right) <= 0:
            return self._rotate_left(node)

        # RL
        if balance < -1 and self._balance(node.right) > 0:
            node.right = self._rotate_right(node.right)
            return self._rotate_left(node)

        return node

    def _min_value_node(self, node):
        current = node
        while current.left:
            current = current.left
        return current


if __name__ == "__main__":
    avl = AVLTree()
    values = [10, 20, 30, 40, 50, 25]

    for v in values:
        avl.insert(v)

    avl.visualize("avl_before_delete")

    avl.delete(40)
    avl.visualize("avl_after_delete")

    assert not avl.search(40)
