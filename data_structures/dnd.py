import torch
import pickle

from torch.nn import Parameter
from sklearn.neighbors import KDTree
from xxhash import xxh64 as xxhs
from estimators.kernel_density_estimate import KernelDensityEstimate


class RandomProjection(object):
    def __init__(self, input_size, embedding_size, sparsity=0.6):
        A = torch.rand(embedding_size, input_size)
        self.A = torch.mul(A, torch.bernoulli(A.clone().fill_(1 - sparsity)))
        max = torch.mv(self.A, torch.ones(input_size))
        self.scalling = 255 / max

    def __call__(self, x):
        return torch.mul(torch.mv(self.A, x), self.scalling)


class DND(object):
    def __init__(self, size=50000, embedding_size=60, knn_no=50):
        self.size = size
        self.knn_no = knn_no

        self.M = {}
        self.keys = Parameter(torch.FloatTensor(size, embedding_size).fill_(0))
        self.vals = Parameter(torch.FloatTensor(size, 1).fill_(0))
        self.stable_keys = torch.FloatTensor(size, embedding_size).fill_(0)
        self.priority = torch.IntTensor(size, 1).fill_(0)

        self.kde = KernelDensityEstimate()
        self.kd_tree = None
        self.random_projection = RandomProjection(24*24, embedding_size)
        self.idx = 0
        self.count = 0
        self.old = 0
        self.new = 0

    def write(self, state, h, v, update_rule=None):
        """ Write to the Differentiable Neural Dictionary.

        Args:
            state (Tensor): Game screen. Used as a stable key for the
                hash function.
            h (Tensor): Feature extractor result. Used in knn computation.
            v (float): Episodic return associated with `state` / `h`
            update_rule (function): function used to update `v`
        """
        h = h.squeeze(0)
        flat_state = state.view(-1, 24 * 24).squeeze()
        stable_h = self.random_projection(flat_state)
        key = self._hash(stable_h)
        is_new_key = key not in self.M

        if self.idx < self.size and is_new_key:
            # new experience, memory not full, append to DND
            self._write(key, stable_h, h, v, self.idx)
            self.idx += 1
            self.new += 1
        elif not is_new_key:
            # old experience, update its value
            idx_to_update = self.M[key]
            old_v = self.vals.data[idx_to_update]
            self.vals.data[idx_to_update] = update_rule(old_v, v)
            self.old += 1
        else:
            # new experience, full memory, pop least used and append to DND
            write_idx = self._get_least_used()
            old_stable_h = self.stable_keys[write_idx]
            old_key = self._hash(old_stable_h)
            try:
                self.M.pop(old_key)
            except KeyError:
                print("Old: ", old_key)
                print(old_stable_h)
                print(self.count, self.idx, write_idx)
            self._write(key, stable_h, h, v, write_idx)
            self.new += 1
        self.count += 1

    def lookup(self, h, training=False):
        _, knn_indices = self.kd_tree.query(h.data.numpy(), k=self.knn_no)
        mask = torch.from_numpy(knn_indices).long().squeeze()
        h_i = self.keys[mask]
        v_i = self.vals[mask]
        self._increment_priority(knn_indices)
        if not training:
            return self._get_q_value(h, h_i, v_i).data
        else:
            return self._get_q_value(h, h_i, v_i)

    def rebuild_tree(self):
        if self.idx < self.size:
            self.kd_tree = KDTree(self.keys.data[:self.idx].numpy())
        else:
            self.kd_tree = KDTree(self.keys.data.numpy())

    def _write(self, key, stable_h, h, v, idx):
        self.M[key] = idx
        self.keys.data[idx] = h
        self.vals.data[idx] = v
        self.stable_keys[idx] = stable_h
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

    def _hash(self, stable_h):
        """ Not sure about pytorch `__hash__`.
        """
        assert stable_h.ndimension() == 1, "Tensor must be one-dimensional"
        return xxhs(pickle.dumps(stable_h.tolist())).hexdigest()
