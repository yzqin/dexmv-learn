#!/usr/bin/env python3

"""Inverse dynamics trainer."""

import logging
logging.disable(logging.CRITICAL)

import numpy as np
import torch
import torch.nn as nn

from mjrl.utils.logger import DataLog


def normalize(data, mean, std, eps=1e-8):
    """Normalizes the data."""
    return (data - mean) / (std + eps)


def unnormalize(data, mean, std, eps=1e-8):
    """Unnormalizes the data."""
    return data * (std + eps) + mean


class InvDynTrainer:
    """Trains inverse dynamics model."""

    def __init__(
        self, paths, model, num_ep=5, mb_size=64, lr=1e-3
    ):
        # Record the fields
        self.model = model
        self.num_ep = num_ep
        self.mb_size = mb_size
        # Construct the logger
        self.logger = DataLog()
        # Retrieve the consecutive obs
        obs_t = np.concatenate([path['observations'][:-1] for path in paths])
        obs_t1 = np.concatenate([path['observations'][1:] for path in paths])
        # Retrieve the actions
        act_t = np.concatenate([path['actions'][:-1] for path in paths])
        # Concat consecutive obs
        obs_tt1 = np.concatenate([obs_t, obs_t1], axis=1)
        # Normalize obs
        self.obs_mean = np.mean(obs_tt1, axis=0)
        self.obs_std = np.std(obs_tt1, axis=0)
        obs_tt1 = normalize(obs_tt1, self.obs_mean, self.obs_std)
        # Normalize actions
        self.act_mean = np.mean(act_t, axis=0)
        self.act_std = np.std(act_t, axis=0)
        act_t = normalize(act_t, self.act_mean, self.act_std)
        # Save the training inputs and targets
        self.obs_tt1 = obs_tt1
        self.act_t = act_t
        # Construct the optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        # Construct the loss function
        self.loss_fun = torch.nn.MSELoss()

    def train(self):
        """Trains the model."""
        # Compute the number of iters per epoch
        num_samples = self.obs_tt1.shape[0]
        num_iter = int(num_samples / self.mb_size)
        # Enable training mode
        self.model.train()
        # Train the model
        for cur_ep in range(self.num_ep):
            loss_total = 0.0
            for cur_iter in range(num_iter):
                # Sample a mini batch
                mb_inds = np.random.choice(num_samples, size=self.mb_size)
                obs = torch.from_numpy(self.obs_tt1[mb_inds]).float()
                act = torch.from_numpy(self.act_t[mb_inds]).float()
                # Perform the forward pass
                act_pred = self.model(obs)
                # Compute the loss
                loss = self.loss_fun(act_pred, act)
                # Perform the backward pass
                self.optimizer.zero_grad()
                loss.backward()
                # Update the parameters
                self.optimizer.step()
                # Record the loss
                loss_total += (loss * self.mb_size)
            # Log epoch stats
            loss_avg = loss_total / (num_iter * self.mb_size)
            self.logger.log_kv('epoch', cur_ep)
            self.logger.log_kv('loss', loss_avg)
            print('Epoch: {}, Loss: {:.6f}'.format(cur_ep, loss_avg))

    def get_norm_stats(self):
        """Retrieves the normalization stats."""
        return {
            'obs_mean': self.obs_mean,
            'obs_std': self.obs_std,
            'act_mean': self.act_mean,
            'act_std': self.act_std
        }
