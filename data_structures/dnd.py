import torch
import pickle

from sklearn.neighbors import KDTree
from xxhash import xxh64 as xxhs
from estimators.kernel_density_estimate import KernelDensityEstimate
from torch.autograd import Variable


class DND(object):
    def __init__(self, size=50000, embedding_size=60, knn_no=50):
        self.size = size
        self.knn_no = knn_no

        self.M = {}
        self.keys = torch.FloatTensor(size, embedding_size).fill_(0)
        self.vals = torch.FloatTensor(size, 1).fill_(0)
        self.priority = torch.IntTensor(size, 1).fill_(0)
        self.kde = KernelDensityEstimate()
        self.kd_tree = None
        self.idx = 0
        self.count = 0
        self.old = 0
        self.new = 0

    def write(self, h, v, update_rule=None):
        h = h.squeeze(0)
        key = self._hash(h)
        is_new_key = key not in self.M

        if self.idx < self.size and is_new_key:
            # new experience, memory not full, append to DND
            self._write(key, h, v, self.idx)
            self.idx += 1
            self.new += 1
        elif not is_new_key:
            # old experience, update its value
            idx_to_update = self.M[key]
            old_v = self.vals[idx_to_update]
            self.vals[idx_to_update] = update_rule(old_v, v)
            self.old += 1
        else:
            # new experience, full memory, pop least used and append to DND
            write_idx = self._get_least_used()
            t = self.keys[write_idx]
            old_key = self._hash(t)
            self.M.pop(old_key)
            self._write(key, h, v, write_idx)
            self.new += 1
        self.count += 1
        assert self.idx == len(self.M), "Belele!!"

    def lookup(self, h, training=False):
        volatile = not training
        _, knn_indices = self.kd_tree.query(h.data.numpy(), k=self.knn_no)
        mask = torch.from_numpy(knn_indices).long().squeeze()
        h_i = Variable(self.keys[mask], volatile=volatile)
        v_i = Variable(self.vals[mask], volatile=volatile)
        self._increment_priority(knn_indices)
        if volatile:
            return self._get_q_value(h, h_i, v_i).data
        else:
            return self._get_q_value(h, h_i, v_i)

    def rebuild_tree(self):
        if self.idx < self.size:
            self.kd_tree = KDTree(self.keys[:self.idx].numpy())
        else:
            self.kd_tree = KDTree(self.keys.numpy())

    def _write(self, key, h, v, idx):
        self.M[key] = idx
        self.keys[idx] = h
        self.vals[idx] = v
        self.priority[idx] = 0

    def _get_q_value(self, h, h_i, v_i):
        distances = self.kde.gaussian_kernel(h, h_i)
        weights = self.kde.normalized_kernel(distances)
        return torch.sum(weights * v_i)

    def _increment_priority(self, knn_indices):
        for idx in knn_indices[0]:
            self.priority[idx, 0] += 1

    def _get_least_used(self):
        """ Get the idx of the memory least frequently appearing in nearest
            neighbors searches. I think a smarter data structure is required
            here, maybe a priority queue.
        """
        if self.idx < self.size:
            return self.priority[:self.idx].min(0)[1][0, 0]
        else:
            return self.priority.min(0)[1][0, 0]

    def _hash(self, h):
        """ Not sure about pytorch `__hash__`.
        """
        assert h.ndimension() == 1, "Tensor must be one-dimensional"
        return xxhs(pickle.dumps(h.tolist())).hexdigest()
