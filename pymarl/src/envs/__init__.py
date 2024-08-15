from functools import partial
# from smac.env import MultiAgentEnv, StarCraft2Env
# from ..env import SnakeEatBeans
import sys
import os

def env_fn(env, **kwargs):
    return env(**kwargs)

REGISTRY = {}
# REGISTRY["snake"] = partial(env_fn,SnakeEatBeans)
# REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)


# if sys.platform == "linux":
#     os.environ.setdefault("SC2PATH",
#                           os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
