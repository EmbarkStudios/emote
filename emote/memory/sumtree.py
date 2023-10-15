class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = [0] * (2 * capacity - 1)
        self.data_pointer = 0

    def update(self, identity, priority):
        tree_idx = self.data_pointer + self.capacity - 1
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate_change(tree_idx, change)
        self.data_pointer = (self.data_pointer + 1) % self.capacity

    def remove(self, identity):
        tree_idx = identity + self.capacity - 1
        self.update(identity, 0)

    def sample(self, value):
        return self._retrieve(0, value)

    def _propagate_change(self, tree_idx, change):
        parent_idx = (tree_idx - 1) // 2
        self.tree[parent_idx] += change
        if parent_idx != 0:
            self._propagate_change(parent_idx, change)

    def _retrieve(self, idx, value):
        left_child_idx = 2 * idx + 1
        right_child_idx = left_child_idx + 1

        if left_child_idx >= len(self.tree):
            return idx - self.capacity + 1  # Adjust to get the data index

        if value <= self.tree[left_child_idx]:
            return self._retrieve(left_child_idx, value)
        else:
            return self._retrieve(right_child_idx, value - self.tree[left_child_idx])

    def total_priority(self):
        return self.tree[0]
