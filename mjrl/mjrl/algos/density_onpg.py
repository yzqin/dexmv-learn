import logging

import numpy as np
import time as timer
import torch

# Import Algs
from mjrl.algos.npg_cg import NPG
from mjrl.models.density import DensityMLP
from mjrl.utils.cg_solve import cg_solve
from mjrl.utils.logger import DataLog
from tpi.core.config import cfg

# samplers

logging.disable(logging.CRITICAL)


class DensityONPG(NPG):
    def __init__(self, env, policy, baseline,
                 demo_paths=None,
                 normalized_step_size=0.01,
                 FIM_invert_args={'iters': 10, 'damping': 1e-4},
                 hvp_sample_frac=1.0,
                 seed=None,
                 save_logs=False,
                 kl_dist=None,
                 lam_0=1.0,  # sim coef
                 lam_1=0.95,  # decay coef
                 pg_algo='trpo'
                 ):

        self.env = env
        self.policy = policy
        self.baseline = baseline
        self.kl_dist = kl_dist if kl_dist is not None else 0.5 * normalized_step_size
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
        # Construct the density model
        self.density_model = DensityMLP(
            env, ws=cfg.DENSITY_ONPG_WS, seed=cfg.RNG_SEED
        ).to(self.device)
        # Record the density model fields
        self.density_num_iter = cfg.DENSITY_ONPG_NUM_ITER
        self.density_mb_size = cfg.DENSITY_ONPG_MB_SIZE
        # Compute the number of pos and neg per density model mini batch
        self.density_pos_mb = int(round(
            cfg.DENSITY_ONPG_POS_FRAC * self.density_mb_size
        ))
        self.density_neg_mb = self.density_mb_size - self.density_pos_mb
        # Construct the density model optimizer
        self.density_optimizer = torch.optim.Adam(
            self.density_model.parameters(), lr=cfg.DENSITY_ONPG_LR
        )
        # Construct the density model loss function
        self.density_loss_fun = torch.nn.BCELoss()

    def train_from_paths(self, paths):
        assert self.demo_paths

        # Iter count used for sim w annealing
        self.iter_count += 1

        # Sampled trajectories
        obs = np.concatenate([path["observations"] for path in paths])
        act = np.concatenate([path["actions"] for path in paths])
        oa = np.concatenate((obs, act), axis=1)
        adv = np.concatenate([path["advantages"] for path in paths])
        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-6)

        # Demo trajectories
        demo_obs = np.concatenate([path["observations"] for path in self.demo_paths])
        demo_act = np.concatenate([path['actions'] for path in self.demo_paths])
        demo_oa = np.concatenate((demo_obs, demo_act), axis=1)
        # Update the density model
        ts = timer.time()
        # Retrieve the total number of pos and neg
        pos_num = demo_oa.shape[0]
        neg_num = oa.shape[0]
        # Construct the targets
        pos_targets = torch.from_numpy(np.ones(self.density_pos_mb)).float()
        neg_targets = torch.from_numpy(np.zeros(self.density_neg_mb)).float()
        targets = torch.cat([pos_targets, neg_targets]).to(self.device)
        # Enable training mode
        self.density_model.train()
        # Train the model
        loss_total = 0.0
        for cur_iter in range(self.density_num_iter):
            # Sample a mini batch
            pos_inds = np.random.choice(pos_num, size=self.density_pos_mb)
            neg_inds = np.random.choice(neg_num, size=self.density_neg_mb)
            pos_oa_mb = torch.from_numpy(demo_oa[pos_inds]).float().to(self.device)
            neg_oa_mb = torch.from_numpy(oa[neg_inds]).float().to(self.device)
            oa_mb = torch.cat([pos_oa_mb, neg_oa_mb])
            # Perform the forward pass
            preds = self.density_model(oa_mb)
            # Compute the loss
            targets = targets.reshape(preds.shape)
            loss = self.density_loss_fun(preds, targets)
            # Perform the backward pass
            self.density_optimizer.zero_grad()
            loss.backward()
            # Update the parameters
            self.density_optimizer.step()
            # Record the loss
            loss_total += (loss.item() * self.density_mb_size)
        # Log the loss
        loss_avg = loss_total / (self.density_num_iter * self.density_mb_size)

        # Prepare the oa for density model
        oa_prep = torch.from_numpy(oa).float().to(self.device)
        # Compute the similarity scores
        self.density_model.eval()
        with torch.no_grad():
            sim_scores = self.density_model(oa_prep)
        sim_scores = sim_scores.detach().cpu().numpy().squeeze()

        # Record the total density time
        t_density = timer.time() - ts

        # Incorporate the similarity in the advantage
        sim_w = self.lam_0 * (self.lam_1 ** self.iter_count)
        adv = adv + sim_w * sim_scores
        adv = cfg.DENSITY_ONPG_ADV_W * adv

        # cache return distributions for the paths
        path_returns = [sum(p["rewards"]) for p in paths]
        mean_return = np.mean(path_returns)
        std_return = np.std(path_returns)
        min_return = np.amin(path_returns)
        max_return = np.amax(path_returns)
        base_stats = [mean_return, std_return, min_return, max_return]
        self.running_score = mean_return if self.running_score is None else \
            0.9 * self.running_score + 0.1 * mean_return  # approx avg of last 10 iters
        if self.save_logs: self.log_rollout_statistics(paths)

        # Keep track of times for various computations
        t_gLL = 0.0
        t_FIM = 0.0

        # Optimization algorithm
        # --------------------------
        surr_before = self.CPI_surrogate(obs, act, adv).data.numpy().ravel()[0]

        # DAPG
        ts = timer.time()
        sample_coef = 1.0
        dapg_grad = sample_coef * self.flat_vpg(obs, act, adv)
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
        n_step_size = 2.0 * self.kl_dist
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
            print('update by trpo')
            shs = 0.5 * (npg_grad * hvp(npg_grad)).sum(0, keepdims=True)
            lm = np.sqrt(shs / 1e-2)
            full_step = npg_grad / lm[0]
            grads = torch.autograd.grad(self.CPI_surrogate(obs, act, adv), self.policy.trainable_params)
            loss_grad = torch.cat([grad.view(-1) for grad in grads]).detach().numpy()
            neggdotstepdir = (loss_grad * npg_grad).sum(0, keepdims=True)
            curr_params = self.policy.get_param_values()
            alpha = 1  # new implementation
            for k in range(10):
                new_params = curr_params + alpha * full_step
                self.policy.set_param_values(new_params, set_new=True, set_old=False)
                surr_after = self.CPI_surrogate(obs, act, adv).data.numpy().ravel()[0]
                kl_dist = self.kl_old_new(obs, act).data.numpy().ravel()[0]

                actual_improve = (surr_after - surr_before)
                expected_improve = neggdotstepdir / lm[0] * alpha
                ratio = actual_improve / expected_improve
                print(f'ratio: {ratio}, lm: {lm}')

                if ratio.item() > .1 and actual_improve > 0:
                    break
                else:
                    alpha = 0.5 * alpha
                    print('step size too high. backtracking. | kl = %f | surr diff = %f' % \
                          (kl_dist, surr_after - surr_before))

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
            self.logger.log_kv('time_density', t_density)
            self.logger.log_kv('density_train_loss', loss_avg)
            self.logger.log_kv('sim_max', np.max(sim_scores))
            self.logger.log_kv('sim_mean', np.mean(sim_scores))
            self.logger.log_kv('sim_min', np.min(sim_scores))
            try:
                self.env.env.env.evaluate_success(paths, self.logger)
            except AttributeError:
                pass

        return base_stats
