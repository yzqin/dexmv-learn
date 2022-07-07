import logging
logging.disable(logging.CRITICAL)
import numpy as np
import scipy
import scipy.sparse.linalg as spLA
import scipy.spatial
import copy
import time as timer
import torch
import torch.nn as nn
from torch.autograd import Variable
import copy
import pickle

# samplers
import mjrl.samplers.trajectory_sampler as trajectory_sampler
import mjrl.samplers.batch_sampler as batch_sampler

# utility functions
import mjrl.utils.process_samples as process_samples
from mjrl.utils.logger import DataLog
from mjrl.utils.cg_solve import cg_solve

# Import Algs
from mjrl.algos.npg_cg import NPG
from mjrl.algos.behavior_cloning import BC

from mjrl.models.invdyn import InvDynMLP
from mjrl.utils.replay_buffer import ReplayBuffer
from tpi.core.config import cfg

import mjrl.algos.invdyn as invdyn


class SOIL(NPG):
    def __init__(self, env, policy, baseline,
                 demo_paths=None,
                 normalized_step_size=0.01,
                 FIM_invert_args={'iters': 10, 'damping': 1e-4},
                 hvp_sample_frac=1.0,
                 seed=None,
                 save_logs=False,
                 kl_dist=None,
                 lam_0=1.0,  # sim coef
                 lam_1=0.95, # decay coef
                 pg_algo='trpo'
                 ):

        self.env = env
        self.policy = policy
        self.baseline = baseline
        self.kl_dist = kl_dist if kl_dist is not None else 0.5*normalized_step_size
        self.seed = seed
        self.save_logs = save_logs
        self.FIM_invert_args = FIM_invert_args
        self.hvp_subsample = hvp_sample_frac
        self.running_score = None
        self.demo_paths = demo_paths
        self.lam_0 = lam_0
        self.lam_1 = lam_1
        self.iter_count = 0.0
        self.pg_algo = pg_algo
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if save_logs:
            self.logger = DataLog()
        # Construct the inverse model
        try:
            self.inverse_model = InvDynMLP(env.spec, mlp_w=cfg.SOIL.MLP_W, seed=seed)
        except:
            self.inverse_model = InvDynMLP(env, mlp_w=cfg.SOIL.MLP_W, seed=seed)
        self.inverse_model = self.inverse_model.to(self.device)
        # Load the inverse model
        if cfg.SOIL.CHECKPOINT:
            with open(cfg.SOIL.CHECKPOINT, 'rb') as f:
                checkpoint_data = pickle.load(f)
            self.inverse_model.load_state_dict(checkpoint_data['model_state'])
        # Construct the replay buffer
        self.replay_buffer = ReplayBuffer(
            max_size=cfg.SOIL.RBS,
            ob_dim=self.inverse_model.obs_dim * 2,
            ac_dim=self.inverse_model.act_dim
        )
        # Construct the inverse model optimizer
        self.inverse_model_optim = torch.optim.Adam(
            self.inverse_model.parameters(),
            lr=cfg.SOIL.LR,
            weight_decay=cfg.SOIL.WD
        )
        # Construct the inverse model loss functions
        self.inverse_model_loss_fun = torch.nn.MSELoss()

    def train_from_paths(self, paths):
        assert self.demo_paths
        # Iter count used for demo w annealing
        self.iter_count += 1

        # Sampled trajectories
        obs = np.concatenate([path["observations"] for path in paths])
        act = np.concatenate([path["actions"] for path in paths])
        adv = np.concatenate([path["advantages"] for path in paths])
        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-6)

        # Measure the time it takes to perform inverse model computations
        ts = timer.time()

        # Retrieve the (st, at, st1) triplets
        obs_t = np.concatenate([path["observations"][:-1] for path in paths])
        obs_t1 = np.concatenate([path["observations"][1:] for path in paths])
        obs_tt1 = np.concatenate([obs_t, obs_t1], axis=1)
        act_t = np.concatenate([path["actions"][:-1] for path in paths])
        # Add data to the replay buffer
        self.replay_buffer.add_data(obs_tt1, act_t)

        # Train the inverse model
        self.inverse_model.train()
        loss_total = 0.0
        for cur_iter in range(cfg.SOIL.NUM_ITER):
            # Sample a mini-batch
            obs_mb, act_mb = self.replay_buffer.sample_data(cfg.SOIL.MB_SIZE)
            # Convert to torch tensors
            obs_mb = torch.from_numpy(obs_mb).float().to(self.device)
            act_mb = torch.from_numpy(act_mb).float().to(self.device)
            # Predict actions
            act_mb_pred = self.inverse_model(obs_mb)
            # Compute the loss
            loss = self.inverse_model_loss_fun(act_mb_pred, act_mb)
            # Compute the gradients
            self.inverse_model_optim.zero_grad()
            loss.backward()
            # Update the inverse model
            self.inverse_model_optim.step()
            # Record the loss
            loss_total += loss.item()
        # Recrod the average loss
        loss_avg = loss_total / cfg.SOIL.NUM_ITER

        # Retrieve the demo trajectories
        demo_obs_t = np.concatenate([path["observations"][:-1] for path in self.demo_paths])
        demo_obs_t1 = np.concatenate([path["observations"][1:] for path in self.demo_paths])
        # Prepare for inverse model input
        demo_obs_tt1 = np.concatenate([demo_obs_t, demo_obs_t1], axis=1)
        demo_obs_tt1 = torch.from_numpy(demo_obs_tt1).float().to(self.device)
        # Predict the actions for demo trajectories
        self.inverse_model.eval()
        with torch.no_grad():
            demo_act_t_pred = self.inverse_model(demo_obs_tt1).cpu().numpy()

        # Compute demo adv
        demo_w = self.lam_0 * (self.lam_1 ** self.iter_count)
        demo_adv = demo_w * np.ones(demo_obs_t.shape[0])
        # Combine sampled and demo trajs
        all_obs = np.concatenate([obs, demo_obs_t])
        all_act = np.concatenate([act, demo_act_t_pred])
        all_adv = np.concatenate([adv, demo_adv])
        # Apply advantage scaling
        all_adv = cfg.SOIL.ADV_W * all_adv

        # Record the total inverse model time
        t_inverse_model = timer.time() - ts

        # cache return distributions for the paths
        path_returns = [sum(p["rewards"]) for p in paths]
        mean_return = np.mean(path_returns)
        std_return = np.std(path_returns)
        min_return = np.amin(path_returns)
        max_return = np.amax(path_returns)
        base_stats = [mean_return, std_return, min_return, max_return]
        self.running_score = mean_return if self.running_score is None else \
                             0.9*self.running_score + 0.1*mean_return  # approx avg of last 10 iters
        if self.save_logs: self.log_rollout_statistics(paths)

        # Keep track of times for various computations
        t_gLL = 0.0
        t_FIM = 0.0

        # Optimization algorithm
        # --------------------------
        surr_before = self.CPI_surrogate(obs, act, adv).data.numpy().ravel()[0]

        # DAPG
        ts = timer.time()
        #sample_coef = 1.0
        sample_coef = all_adv.shape[0] / adv.shape[0]
        dapg_grad = sample_coef * self.flat_vpg(all_obs, all_act, all_adv)
        t_gLL += timer.time() - ts

        # NPG
        ts = timer.time()
        hvp = self.build_Hvp_eval([obs, act],
                                  regu_coef=self.FIM_invert_args['damping'])
        npg_grad = cg_solve(hvp, dapg_grad, x_0=dapg_grad.copy(),
                            cg_iters=self.FIM_invert_args['iters'])
        t_FIM += timer.time() - ts

        # Step size computation
        # --------------------------
        n_step_size = 2.0*self.kl_dist
        alpha = np.sqrt(np.abs(n_step_size / (np.dot(dapg_grad.T, npg_grad) + 1e-20)))

        # Policy update
        # --------------------------
        # NPG
        if self.pg_algo == 'npg':
            print('update by npg')
            curr_params = self.policy.get_param_values()
            new_params = curr_params + alpha * npg_grad
            self.policy.set_param_values(new_params, set_new=True, set_old=False)
            surr_after = self.CPI_surrogate(obs, act, adv).data.numpy().ravel()[0]
            kl_dist = self.kl_old_new(obs, act).data.numpy().ravel()[0]
            self.policy.set_param_values(new_params, set_new=True, set_old=True)


        # TRPO
        else:
            #print((npg_grad*hvp(npg_grad)))
            #print(type(npg_grad), type(hvp(npg_grad)))
            shs = 0.5 * (npg_grad * hvp(npg_grad)).sum(0, keepdims=True)
            lm = np.sqrt(shs / 1e-2)
            full_step = npg_grad / lm[0]
            grads = torch.autograd.grad(self.CPI_surrogate(all_obs, all_act, all_adv), self.policy.trainable_params)
            loss_grad = torch.cat([grad.view(-1)for grad in grads]).detach().numpy()
            print(loss_grad.shape, npg_grad.shape)
            neggdotstepdir = (loss_grad * npg_grad).sum(0, keepdims=True)
            print(f'dot value: {neggdotstepdir}')
            print('update by trpo')
            curr_params = self.policy.get_param_values()
            alpha = 1 # new implementation
            for k in range(10):
                new_params = curr_params + alpha * full_step
                self.policy.set_param_values(new_params, set_new=True, set_old=False)
                surr_after = self.CPI_surrogate(obs, act, adv).data.numpy().ravel()[0]
                kl_dist = self.kl_old_new(obs, act).data.numpy().ravel()[0]
                
                actual_improve = (surr_after - surr_before)
                expected_improve = neggdotstepdir / lm[0] * alpha
                ratio = actual_improve / expected_improve
                print(f'ratio: {ratio}, lm: {lm}')
                
                #if kl_dist < self.kl_dist:
                if ratio.item() > .1 and actual_improve > 0:
                    break
                else:
                    alpha = 0.5 * alpha
                    print('step size too high. backtracking. | kl = %f | suff diff = %f' % \
                            (kl_dist, surr_after-surr_before))
            
                if k == 9:
                    alpha = 0

            new_params = curr_params + alpha * full_step
            self.policy.set_param_values(new_params, set_new=True, set_old=False)
            surr_after = self.CPI_surrogate(obs, act, adv).data.numpy().ravel()[0]
            kl_dist = self.kl_old_new(obs, act).data.numpy().ravel()[0]
            self.policy.set_param_values(new_params, set_new=True, set_old=True)


        # Log information
        if self.save_logs:
            self.logger.log_kv('alpha', alpha)
            self.logger.log_kv('delta', n_step_size)
            self.logger.log_kv('time_vpg', t_gLL)
            self.logger.log_kv('time_npg', t_FIM)
            self.logger.log_kv('kl_dist', kl_dist)
            self.logger.log_kv('surr_improvement', surr_after - surr_before)
            self.logger.log_kv('running_score', self.running_score)
            self.logger.log_kv('time_invdyn', t_inverse_model)
            self.logger.log_kv('invdyn_train_err', loss_avg)
            try:
                self.env.env.env.evaluate_success(paths, self.logger)
            except:
                # nested logic for backwards compatibility. TODO: clean this up.
                try:
                    success_rate = self.env.env.env.evaluate_success(paths)
                    self.logger.log_kv('success_rate', success_rate)
                except:
                    pass
        return base_stats
