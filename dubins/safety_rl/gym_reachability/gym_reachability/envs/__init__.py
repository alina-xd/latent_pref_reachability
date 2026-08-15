"""
Please contact the author(s) of this library if you have any questions.
Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )
         Vicenc Rubies Royo ( vrubies@berkeley.edu )
"""

# LunarLander envs require Box2D and are unrelated to the Dubins tasks;
# make them optional so a missing Box2D doesn't block the Dubins env.
try:
    from .multi_player_lunar_lander_reachability import (
        MultiPlayerLunarLanderReachability
    )
    from .one_player_reach_avoid_lunar_lander import OnePlayerReachAvoidLunarLander
except ModuleNotFoundError:
    pass

from .dubins_car_one import DubinsCarOneEnv
from .dubins_car_avoid import DubinsCarAvoidEnv
from .dubins_car_latent_avoid import DubinsCarAvoidLatentEnv

from .dubins_car_pe import DubinsCarPEEnv

from .point_mass import PointMassEnv

from .zermelo_show import ZermeloShowEnv
