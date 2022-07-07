"""
    Add the environment you wish to train here
"""

from hand_imitation.env.environments.dapg_env.dapg_wrapper import DAPGWrapper
from hand_imitation.env.environments.mug_place_object_env import MugPlaceObjectEnv
from hand_imitation.env.environments.mug_pour_water_env import WaterPouringEnv
from hand_imitation.env.environments.ycb_relocate_env import YCBRelocate

from mjrl.utils.gym_env import GymEnv


def get_environment(env_name=None):
    if env_name is None:
        print("Need to specify environment name")
        return
    # env format task-obj-size
    env_info = env_name.split('-')
    task = env_info[0]
    obj_name = env_info[1]
    obj_scale = float(env_info[2])
    friction = (1, 0.5, 0.01)
    if task == 'relocate':
        if obj_name != "mustard_bottle":
            env = YCBRelocate(has_renderer=False, object_name=obj_name,
                              friction=friction, object_scale=obj_scale,
                              solref='-6000 -300', randomness_scale=0.25)
        else:
            env = YCBRelocate(has_renderer=False, object_name=obj_name,
                              friction=friction, object_scale=obj_scale,
                              solref='-6000 -300', randomness_scale=0.1)
        return DAPGWrapper(env)
    if task == 'pour':
        env = WaterPouringEnv(has_renderer=False, scale=obj_scale, tank_size=(0.15, 0.15, 0.12))
        return DAPGWrapper(env)
    if task == 'place':
        env = MugPlaceObjectEnv(has_renderer=False, object_scale=obj_scale, mug_scale=1.5, large_force=True)
        return DAPGWrapper(env)

    return GymEnv(env_name)
