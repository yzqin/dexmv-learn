#!/usr/bin/env python3

"""Replay buffer."""

import numpy as np


class ReplayBuffer(object):

    def __init__(self, max_size, ob_dim, ac_dim):
        self.max_size = max_size
        self.obs = np.empty((max_size, ob_dim))
        self.acs = np.empty((max_size, ac_dim))
        self.cur_ind = 0
        self.full = False

    def add_data(self, obs, acs):
        if self.cur_ind + obs.shape[0] <= self.max_size:
            self.obs[self.cur_ind:(self.cur_ind + obs.shape[0])] = obs
            self.acs[self.cur_ind:(self.cur_ind + obs.shape[0])] = acs
            self.cur_ind += obs.shape[0]
        else:
            num_left = self.max_size - self.cur_ind
            self.obs[self.cur_ind:(self.cur_ind + num_left)] = obs[:num_left]
            self.acs[self.cur_ind:(self.cur_ind + num_left)] = acs[:num_left]
            self.cur_ind = 0
            self.obs[self.cur_ind:(obs.shape[0] - num_left)] = obs[num_left:]
            self.acs[self.cur_ind:(obs.shape[0] - num_left)] = acs[num_left:]
            self.cur_ind += (obs.shape[0] - num_left)
            self.full = True
            #from IPython import embed; embed()

    def sample_data(self, batch_size):
        max_ind = self.max_size if self.full else self.cur_ind
        inds = np.random.choice(max_ind, size=batch_size)
        return self.obs[inds], self.acs[inds]

    def get_norm_stats(self):
        max_ind = self.max_size if self.full else self.cur_ind
        return {
            'obs_mean': np.mean(self.obs[:max_ind], axis=0),
            'obs_std': np.std(self.obs[:max_ind], axis=0),
            'acs_mean': np.mean(self.acs[:max_ind], axis=0),
            'acs_std': np.std(self.acs[:max_ind], axis=0)
        }
