import torch
import numpy as np
import math
from torch.autograd import Variable
import torch.nn.functional as F

from .base_agent import BaseAgent
from estimators import get_estimator
from data_structures import ReplayMemory, Transition, DND
from utils import TorchTypes


def update_rule(fast_lr):
    def update_q(old, new):
        return old + fast_lr * (new - old)
    return update_q


class NECAgent(BaseAgent):
    def __init__(self, action_space, cmdl):
        BaseAgent.__init__(self, action_space)

        self.name = "NEC_agent"
        self.cmdl = cmdl
        self.dtype = TorchTypes()
        self.slow_lr = slow_lr = cmdl.slow_lr
        self.fast_lr = fast_lr = cmdl.fast_lr
        dnd = cmdl.dnd

        # Feature extractor and embedding size
        FeatureExtractor = get_estimator(cmdl.estimator)
        state_dim = (1, 24) if not cmdl.rescale else (1, 84)
        if dnd.linear_projection:
            self.conv = FeatureExtractor(state_dim, dnd.linear_projection)
        elif dnd.linear_projection is False:
            self.conv = FeatureExtractor(state_dim, None)
        embedding_size = self.conv.get_embedding_size()

        # DNDs, Memory, N-step buffer
        self.dnds = [DND(dnd.size, embedding_size, dnd.knn_no)
                     for i in range(self.action_no)]
        self.replay_memory = ReplayMemory(capacity=cmdl.experience_replay)
        self.N_step = self.cmdl.n_horizon
        self.N_buff = []

        self.optimizer = torch.optim.Adam(self.conv.parameters(), lr=slow_lr)
        self.optimizer.zero_grad()
        self.update_q = update_rule(fast_lr)

        # Temp data, flags, stats, misc
        self._key_tmp = None
        self.knn_ready = False
        self.initial_val = 0.1
        self.max_q = -math.inf

    def evaluate_policy(self, state):
        """ Policy Evaluation.

            Performs a forward operation through the neural net feature
            extractor and uses the resulting representation to compute the k
            nearest neighbors in each of the DNDs associated with each action.

            Returs the action with the highest weighted value between the
            k nearest neighbors.
        """
        state = torch.from_numpy(state).unsqueeze(0).unsqueeze(0)
        h = self.conv(Variable(state, volatile=True))
        self._key_tmp = h

        # corner case, randomly fill the buffers so that we can perform knn.
        if not self.knn_ready:
            return self._heat_up_dnd(h.data)

        # query each DND for q values and pick the largest one.
        if np.random.uniform() > self.cmdl.epsilon:
            v, action = self._query_dnds(h)
            self.max_q = v if self.max_q < v else self.max_q
            return action
        else:
            return self.action_space.sample()

    def improve_policy(self, _state, _action, reward, state, done):
        """ Policy Evaluation.
        """
        self.N_buff.append((_state, self._key_tmp, _action, reward))

        R = 0
        if self.knn_ready and ((len(self.N_buff) == self.N_step) or done):
            if not done:
                # compute Q(t + N)
                state = torch.from_numpy(state).unsqueeze(0).unsqueeze(0)
                h = self.conv(Variable(state, volatile=True))
                R, _ = self._query_dnds(h)

            for i in range(len(self.N_buff) - 1, -1, -1):

                s = self.N_buff[i][0]
                h = self.N_buff[i][1]
                a = self.N_buff[i][2]
                R = self.N_buff[i][3] + 0.99 * R

                # write to DND
                self.dnds[a].write(h.data, R, self.update_q)
                # print("%3d, %3d, %3d  |  %0.3f" % (self.step_cnt, i, a, R))

                # append to experience replay
                self.replay_memory.push(s, a, R)

            self.N_buff.clear()

            for dnd in self.dnds:
                dnd.rebuild_tree()

        if self.cmdl.update_freq is False:
            return

        if (self.step_cnt % self.cmdl.update_freq == 0) and (
                len(self.replay_memory) > self.cmdl.batch_size):
            # get batch of transitions
            transitions = self.replay_memory.sample(self.cmdl.batch_size)
            batch = self._batch2torch(transitions)
            # compute gradients
            self._accumulate_gradient(*batch)
            # optimize
            self._update_model()

    def _query_dnds(self, h):
        q_vals = torch.FloatTensor(self.action_no, 1).fill_(0)
        for i, dnd in enumerate(self.dnds):
            q_vals[i] = dnd.lookup(h)
        return q_vals.max(0)[0][0, 0], q_vals.max(0)[1][0, 0]

    def _accumulate_gradient(self, states, actions, returns):
        """ Compute gradient
            v=Q(s,a), return = QN(s,a)
        """
        states = Variable(states)
        actions = Variable(actions)
        returns = Variable(returns)

        # Compute Q(s, a)
        features = self.conv(states)

        v_variables = []
        for i in range(self.cmdl.batch_size):
            act = actions[i].data[0]
            v = self.dnds[act].lookup(features[i].unsqueeze(0), volatile=False)
            v_variables.append(v)

        q_values = torch.stack(v_variables)

        loss = F.smooth_l1_loss(q_values, returns)
        loss.data.clamp(-1, 1)

        # Accumulate gradients
        loss.backward()

    def _update_model(self):
        for param in self.conv.parameters():
            param.grad.data.clamp(-1, 1)

        self.optimizer.step()
        self.optimizer.zero_grad()

    def _heat_up_dnd(self, h):
        # fill the dnds with knn_no * (action_no + 1)
        action = np.random.randint(self.action_no)
        self.dnds[action].write(h, self.initial_val, self.update_q)
        self.knn_ready = self.step_cnt >= 2 * self.cmdl.dnd.knn_no * \
            (self.action_space.n + 1)
        if self.knn_ready:
            for dnd in self.dnds:
                dnd.rebuild_tree()
        return action

    def _batch2torch(self, batch, batch_sz=None):
        """ List of Transitions to List of torch states, actions, rewards.

            From a batch of transitions (s0, a0, Rt)
            get a batch of the form state=(s0,s1...), action=(a1,a2...),
            Rt=(rt1,rt2...)
            Inefficient. Adds 1.5s~2s for 20,000 steps with 32 agents.
        """
        batch_sz = len(batch) if batch_sz is None else batch_sz
        batch = Transition(*zip(*batch))

        states = [torch.from_numpy(s).unsqueeze(0) for s in batch.state]
        state_batch = torch.stack(states).type(self.dtype.FloatTensor)
        action_batch = self.dtype.LongTensor(batch.action)
        rt_batch = self.dtype.FloatTensor(batch.Rt)
        return [state_batch, action_batch, rt_batch]

    def display_model_stats(self):
        param_abs_mean = 0
        grad_abs_mean = 0
        n_params = 0
        for p in self.conv.parameters():
            param_abs_mean += p.data.abs().sum()
            if p.grad:
                grad_abs_mean += p.grad.data.abs().sum()
            n_params += p.data.nelement()

        print("[NEC_agent] step=%6d, Wm: %.9f" % (
            self.step_cnt, param_abs_mean / n_params))
        print("[NEC_agent] maxQ=%.3f " % (self.max_q))
        for i, dnd in enumerate(self.dnds):
            print("[DND] M=%d, DND.count=%d" % (len(dnd.M), dnd.count))
        for i, dnd in enumerate(self.dnds):
            print("[DND] old=%6d, new=%6d" % (dnd.old, dnd.new))
