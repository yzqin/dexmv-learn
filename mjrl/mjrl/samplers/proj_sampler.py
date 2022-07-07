import logging
logging.disable(logging.CRITICAL)

import copy
import numpy as np

from mjrl.utils import tensor_utils
from mjrl.utils.get_environment import get_environment
from tpi.core.config import cfg

from hand_manipulation.hand_project.env.relocation_env import AdroitReloationEnv, AdroitRelocationScene

scene = AdroitRelocationScene(ycb_object_id=-1,
                                primitive="sphere",
                                random_pose=False,
                                density=1000,
                                size=0.04)
env = AdroitReloationEnv(scene, 20, 200, False)

# Single core rollout to sample trajectories
# =======================================================
def do_rollout(N,
    policy,
    T=1e6,
    env=None,
    env_name=None,
    pegasus_seed=None):
    """
    params:
    N               : number of trajectories
    learning          : learning to be used to sample the data
    T               : maximum length of trajectory
    env             : env object to sample from
    env_name        : name of env to be sampled from 
                      (one of env or env_name must be specified)
    pegasus_seed    : seed for environment (numpy speed must be set externally)
    """

    env.seed(pegasus_seed)
    paths = []

    for ep in range(N):

        # Set pegasus seed if asked
        if pegasus_seed is not None:
            seed = pegasus_seed + ep
            env.seed(seed)
            np.random.seed(seed)
        
        observations=[]
        actions=[]
        rewards=[]
        agent_infos = []
        env_infos = []

        o = env.reset()
        done = False
        t = 0

        while t < T and done != True:
            a, agent_info = policy.get_action(o)
            next_o, r, done, env_info = env.step(a)
            #observations.append(o.ravel())
            observations.append(o)
            actions.append(a)
            rewards.append(r)
            agent_infos.append(agent_info)
            env_infos.append(env_info)
            o = next_o
            t += 1

        path = dict(
            observations=np.array(observations),
            actions=np.array(actions),
            rewards=np.array(rewards),
            agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
            env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
            terminated=done
        )

        if cfg.SAMPLER_INIT_STATE:
            path['init_state_dict'] = init_state

        paths.append(path)

    #print("====== Worker finished ======")
    del(env)
    return paths

def do_rollout_star(args_list):
    return do_rollout(*args_list)