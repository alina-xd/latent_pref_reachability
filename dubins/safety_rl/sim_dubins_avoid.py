"""
Please contact the author(s) of this library if you have any questions.
Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

This experiment runs double deep Q-network with the discounted reach-avoid
Bellman equation (DRABE) proposed in [RSS21] on a 3-dimensional Dubins car
problem. We use this script to generate Fig. 5 in the paper.

Examples:
    RA: python3 sim_car_one.py -sf -of scratch -w -wi 5000 -g 0.9999 -n 9999
    test: python3 sim_car_one.py -sf -of scratch -w -wi 50 -mu 1000 -cp 400
        -n tmp
"""

import os
import argparse
import time
from warnings import simplefilter
import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch

from RARL.DDQNSingleAvoid import DDQNSingleAvoid as DDQNSingle
from RARL.config import dqnConfig
from RARL.utils import save_obj
from gym_reachability import gym_reachability  # Custom Gym env.

matplotlib.use('Agg')
simplefilter(action='ignore', category=FutureWarning)
timestr = time.strftime("%Y-%m-%d-%H_%M")

import ruamel.yaml as yaml
import pathlib
import sys
from datetime import datetime
from termcolor import cprint
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
dreamer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dreamerv3-torch'))
sys.path.append(dreamer_dir)
print(sys.path)
import tools
def RARL(args): 
    
    # == CONFIGURATION ==
    env_name = "dubins_car_avoid-v1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    maxUpdates = args.maxUpdates
    updateTimes = args.updateTimes
    updatePeriod = int(maxUpdates / updateTimes)
    updatePeriodHalf = int(updatePeriod / 2)
    maxSteps = 25

    fn = args.name + '-g' + str(args.gamma) + '-' + args.doneType
    if args.showTime:
        fn = fn + '-' + timestr

    outFolder = os.path.join(args.outFolder, 'car-DDQN', fn)
    print(outFolder)
    figureFolder = os.path.join(outFolder, 'figure')
    os.makedirs(figureFolder, exist_ok=True)

    # == Environment ==
    print("\n== Environment Information ==")
    if args.doneType == 'toEnd':
        sample_inside_obs = True
    elif args.doneType == 'TF' or args.doneType == 'fail':
        sample_inside_obs = False

    env = gym.make(
        env_name, device=device, config=args, mode=args.mode, doneType=args.doneType,
        sample_inside_obs=sample_inside_obs
    )

    stateDim = env.state.shape[0]
    actionNum = env.action_space.n
    actionList = np.arange(actionNum)
    print(
        "State Dimension: {:d}, ActionSpace Dimension: {:d}".format(
            stateDim, actionNum
        )
    )

    # == Setting in this Environment ==
    env.set_speed(speed=args.speed)
    env.set_constraint(radius=args.consRadius)
    env.set_radius_rotation(R_turn=(args.speed / args.turnRate))
    print("Dynamic parameters:")
    print("  CAR", end='\n    ')
    print(
        "Constraint: {:.1f} ".format(env.car.constraint_radius)
        + "Turn: {:.2f} ".format(env.car.R_turn)
        + "Max speed: {:.2f} ".format(env.car.speed)
        + "Max angular speed: {:.3f}".format(env.car.max_turning_rate)
    )
    print("  ENV", end='\n    ')
    print(
        "Constraint: {:.1f} ".format(env.constraint_radius)
        + "Turn: {:.2f} ".format(env.R_turn)
        + "Max speed: {:.2f} ".format(env.speed)
    )
    print(env.car.discrete_controls)

    env.set_seed(args.randomSeed)

    # == Get and Plot g_x ==
    if args.plotFigure or args.storeFigure:
        nx, ny = 101, 101
        vmin = -1
        vmax = 1

    v = np.zeros((nx, ny))
    g_x = np.zeros((nx, ny))
    xs = np.linspace(env.bounds[0, 0], env.bounds[0, 1], nx)
    ys = np.linspace(env.bounds[1, 0], env.bounds[1, 1], ny)

    it = np.nditer(v, flags=['multi_index'])

    while not it.finished:
        idx = it.multi_index
        x = xs[idx[0]]
        y = ys[idx[1]]

        g_x[idx] = env.safety_margin(np.array([x, y]))

        v[idx] = g_x[idx]
        it.iternext()

    axStyle = env.get_axes()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))


    ax = axes[0]
    im = ax.imshow(
        g_x.T, interpolation='none', extent=axStyle[0], origin="lower",
        cmap="seismic", vmin=vmin, vmax=vmax, zorder=-1
    )
    cbar = fig.colorbar(
        im, ax=ax, pad=0.01, fraction=0.05, shrink=.95, ticks=[vmin, 0, vmax]
    )
    cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=24)
    ax.set_title(r'$g(x)$', fontsize=18)

    ax = axes[1]
    im = ax.imshow(
        v.T, interpolation='none', extent=axStyle[0], origin="lower",
        cmap="seismic", vmin=vmin, vmax=vmax, zorder=-1
    )
    cbar = fig.colorbar(
        im, ax=ax, pad=0.01, fraction=0.05, shrink=.95, ticks=[vmin, 0, vmax]
    )
    cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=24)
    ax.set_title(r'$v(x)$', fontsize=18)

    for ax in axes:
        env.plot_failure_set(ax=ax)
        env.plot_formatting(ax=ax)

    fig.tight_layout()
    if args.storeFigure:
        figurePath = os.path.join(figureFolder, 'env.png')
        fig.savefig(figurePath)
    if args.plotFigure:
        plt.show()
        plt.pause(0.001)
    plt.close()

    # == Agent CONFIG ==
    print("\n== Agent Information ==")
    if args.annealing:
        GAMMA_END = 0.9999
        EPS_PERIOD = int(updatePeriod / 10)
        EPS_RESET_PERIOD = updatePeriod
    else:
        GAMMA_END = args.gamma
        EPS_PERIOD = updatePeriod
        EPS_RESET_PERIOD = maxUpdates

    CONFIG = dqnConfig(
        DEVICE=device, ENV_NAME=env_name, SEED=args.randomSeed,
        MAX_UPDATES=maxUpdates, MAX_EP_STEPS=maxSteps, BATCH_SIZE=64,
        MEMORY_CAPACITY=args.memoryCapacity, ARCHITECTURE=args.architecture,
        ACTIVATION=args.actType, GAMMA=args.gamma, GAMMA_PERIOD=updatePeriod,
        GAMMA_END=GAMMA_END, EPS_PERIOD=EPS_PERIOD, EPS_DECAY=0.7,
        EPS_RESET_PERIOD=EPS_RESET_PERIOD, LR_C=args.learningRate,
        LR_C_PERIOD=updatePeriod, LR_C_DECAY=0.75, MAX_MODEL=50
    )

    # == AGENT ==
    dimList = [stateDim] + list(CONFIG.ARCHITECTURE) + [actionNum]    
    agent = DDQNSingle(
        CONFIG, actionNum, actionList, dimList=dimList, mode=args.mode,
        terminalType=args.terminalType
    )
    print("We want to use: {}, and Agent uses: {}".format(device, agent.device))
    print("Critic is using cuda: ", next(agent.Q_network.parameters()).is_cuda)

    vmin = -1
    vmax = 1
    if args.warmup:
        print("\n== Warmup Q ==")
        lossList = agent.initQ(
            env, args.warmupIter, outFolder, num_warmup_samples=200, vmin=vmin,
            vmax=vmax, plotFigure=args.plotFigure, storeFigure=args.storeFigure
        )

        if args.plotFigure or args.storeFigure:
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            tmp = np.arange(500, args.warmupIter)
            # tmp = np.arange(args.warmupIter)
            ax.plot(tmp, lossList[tmp], 'b-')
            ax.set_xlabel('Iteration', fontsize=18)
            ax.set_ylabel('Loss', fontsize=18)
            plt.tight_layout()

            if args.storeFigure:
                figurePath = os.path.join(figureFolder, 'initQ_Loss.png')
                fig.savefig(figurePath)
            if args.plotFigure:
                plt.show()
                plt.pause(0.001)
                plt.close()

    print("\n== Training Information ==")
    vmin = -1
    vmax = 1
    trainRecords, trainProgress = agent.learn(
        env, MAX_UPDATES=maxUpdates, MAX_EP_STEPS=maxSteps, warmupQ=False,
        doneTerminate=True, vmin=vmin, vmax=vmax, showBool=False,
        checkPeriod=args.checkPeriod, outFolder=outFolder,
        plotFigure=args.plotFigure, storeFigure=args.storeFigure
    )

    trainDict = {}
    trainDict['trainRecords'] = trainRecords
    trainDict['trainProgress'] = trainProgress
    filePath = os.path.join(outFolder, 'train')

    save_obj(trainDict, filePath)

def recursive_update(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and key in base:
            recursive_update(base[key], value)
        else:
            base[key] = value

if __name__ == "__main__":

    # == ARGS ==
    parser = argparse.ArgumentParser()

    parser.add_argument("--configs", nargs="+")
    parser.add_argument("--expt_name", type=str, default=None)
    parser.add_argument("--resume_run", type=bool, default=False)
    parser.add_argument("--annealing", type=bool, default=False)
    # environment parameters
    config, remaining = parser.parse_known_args()
    annealing = config.annealing
    if not config.resume_run:
        curr_time = datetime.now().strftime("%m%d/%H%M%S")
        config.expt_name = (
            f"{curr_time}_{config.expt_name}" if config.expt_name else curr_time
        )
    else:
        assert config.expt_name, "Need to provide experiment name to resume run."

    yaml = yaml.YAML(typ="safe", pure=True)
    configs = yaml.load(
        (pathlib.Path(sys.argv[0]).parent / "../configs.yaml").read_text()
    )

    name_list = ["defaults", *config.configs] if config.configs else ["defaults"]

    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    print(defaults.keys())
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    final_config = parser.parse_args(remaining)

    final_config.logdir = f"{final_config.logdir}/{config.expt_name}"
    #final_config.time_limit = HORIZONS[final_config.task.split("_")[-1]]

    print("---------------------")
    cprint(f"Experiment name: {config.expt_name}", "red", attrs=["bold"])
    cprint(f"Task: {final_config.task}", "cyan", attrs=["bold"])
    cprint(f"Logging to: {final_config.logdir}", "cyan", attrs=["bold"])
    print("---------------------")

    
    if annealing:
        final_config.annealing = True
        final_config.name = 'privileged_state_annealing'
    else:
        final_config.annealing = False
        final_config.name = 'privileged_state'
    print(final_config.annealing)
    RARL(final_config)
  
