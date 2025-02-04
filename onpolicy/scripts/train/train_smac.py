#!/usr/bin/env python
import argparse
import os
import socket
import sys
from pathlib import Path
from pprint import pprint

import numpy as np
import setproctitle
import torch
import yaml

from onpolicy.config import get_config
from onpolicy.envs.env_wrappers import ShareDummyVecEnv, ShareSubprocVecEnv
from onpolicy.envs.starcraft2.smac_maps import get_map_params
from onpolicy.envs.starcraft2.StarCraft2_Env import StarCraft2Env
from onpolicy.exp_utils import SacredAimExperiment
from onpolicy.exp_utils.args_utils import args_str2bool
from onpolicy.train_utils import setup_seed
from onpolicy.utils.util import generate_random_str
"""Train script for SMAC."""


def make_train_env(all_args):

    def get_env_fn(rank):

        def init_env():
            if all_args.env_name == "StarCraft2":
                env = StarCraft2Env(all_args)
            else:
                print("Can not support the " + all_args.env_name +
                      "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv(
            [get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):

    def get_env_fn(rank):

        def init_env():
            if all_args.env_name == "StarCraft2":
                env = StarCraft2Env(all_args)
            else:
                print("Can not support the " + all_args.env_name +
                      "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv(
            [get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser: argparse.ArgumentParser):
    parser.add_argument("--map_name",
                        type=str,
                        required=True,
                        help="Which smac map to run on")
    parser.add_argument("--add_move_state", type=args_str2bool, default=False)
    parser.add_argument("--add_local_obs", type=args_str2bool, default=False)
    parser.add_argument("--add_distance_state",
                        type=args_str2bool,
                        default=False)
    parser.add_argument("--add_enemy_action_state",
                        type=args_str2bool,
                        default=False)
    parser.add_argument("--add_agent_id", type=args_str2bool, default=False)
    parser.add_argument("--add_visible_state",
                        type=args_str2bool,
                        default=False)
    parser.add_argument("--add_xy_state", type=args_str2bool, default=False)
    parser.add_argument("--use_state_agent", type=args_str2bool, default=True)
    parser.add_argument("--use_mustalive", type=args_str2bool, default=True)
    parser.add_argument("--add_center_xy", type=args_str2bool, default=True)

    all_args = parser.parse_args(args)

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    if bool(all_args.use_mappo):
        yaml_file = '/../config/smac_mappo.yaml'
    else:
        yaml_file = "/../config/smac.yaml"
    yaml_path = Path(
        os.path.abspath(
            os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] +
            yaml_file))
    with open(yaml_path, "r") as yaml_file:
        map_configs = yaml.safe_load(yaml_file)
        for k, v in map_configs[all_args.map_name].items():
            assert hasattr(all_args, k), "error input in yaml config"
            setattr(all_args, k, v)

    setattr(all_args, "use_" + all_args.adv, True)
    setattr(all_args, all_args.loop_order + "_loop_first", True)
    setattr(all_args, all_args.seq_strategy + "_sequence", True)

    pprint(all_args.__dict__)

    sanity_check(all_args)

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    result_dir = (Path(
        os.path.abspath(
            os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] +
            "/../../results")) / all_args.env_name / all_args.map_name /
                  all_args.algorithm_name / all_args.experiment_name)

    exp_name = f"{all_args.map_name}_{all_args.experiment_name}"
    code_dir = Path(
        os.path.abspath(
            os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] +
            "/../"))
    all_args.code_dir = str(code_dir)
    all_args.host = socket.gethostname()

    if not result_dir.exists():
        os.makedirs(str(result_dir))

    if all_args.use_wandb:
        import wandb
        run = wandb.init(config=all_args,
                         project=all_args.project_name,
                         entity=all_args.wandb_usr,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                              str(all_args.experiment_name) +
                              "_seed" + str(all_args.seed),
                        #  group=all_args.map_name,
                         dir=str(result_dir),
                         job_type="training",
                         reinit=True)
        exp_dir = Path(wandb.run.dir)
    else:
        while True:
            curr_exp = 'exp_' + generate_random_str(length=8)
            exp_dir = result_dir / curr_exp
            if not exp_dir.exists():
                os.makedirs(str(exp_dir))
                break

    all_args.exp_dir = str(exp_dir)
    print(f"the results are saved in {exp_dir}")

    logger = SacredAimExperiment(
        exp_name,
        code_dir,
        all_args.use_sacred,
        exp_dir / "logs",
        all_args.use_aim,
        all_args.aim_repo,
        not all_args.use_aim,
    )

    setproctitle.setproctitle(
        str(all_args.algorithm_name) + "-" + str(all_args.env_name) + "-" +
        str(all_args.experiment_name) + "-" + str(all_args.map_name))

    for seed in range(all_args.seed, all_args.seed + all_args.n_run):

        logger.reset()
        config = all_args.__dict__
        config.update({"argv": args})
        logger.set_config(config)
        run_name = f"{all_args.experiment_name}_seed_{all_args.seed}"
        logger.set_tag(run_name)

        # seed
        setup_seed(seed)

        # env
        envs = make_train_env(all_args)
        eval_envs = make_eval_env(all_args) if all_args.use_eval else None
        num_agents = get_map_params(all_args.map_name)["n_agents"]

        setattr(all_args, "num_agents", num_agents)
        setattr(all_args, "device", device)
        setattr(all_args, "run_dir", result_dir)

        n_episode = (int(all_args.num_env_steps) // all_args.episode_length //
                     all_args.n_rollout_threads)
        setattr(all_args, "n_episode", n_episode)

        config = {
            "all_args": all_args,
            "envs": envs,
            "eval_envs": eval_envs,
            "num_agents": num_agents,
            "device": device,
            "run_dir": exp_dir,
            "logger": logger,
        }

        # run experiments
        from onpolicy.runner.smac_runner import SMACRunner as Runner
        try:
            if all_args.use_sacred:
                import sacred

                sacred_exp = logger.get_sacred_exp()

                @sacred_exp.main
                def sacred_run(_run):
                    config["logger"].set_sacred_run(_run)
                    runner = Runner(config)
                    runner.run()
                    runner.logger.close()

                sacred_exp.run()
            else:
                runner = Runner(config)
                runner.run()
                runner.logger.close()
        except Exception as e:
            import traceback
            print("Error occurred:")
            traceback.print_exc()
        finally:
            # post process
            envs.close()
            if all_args.use_eval and eval_envs is not envs:
                eval_envs.close()
    
    if all_args.use_wandb:
        run.finish()

def sanity_check(all_args):
    if all_args.algorithm_name == "rmappo":
        assert (all_args.use_recurrent_policy or
                all_args.use_naive_recurrent_policy), "check recurrent policy!"
    elif all_args.algorithm_name == "mappo":
        assert (all_args.use_recurrent_policy == False
                and all_args.use_naive_recurrent_policy
                == False), "check recurrent policy!"
    else:
        raise NotImplementedError

    assert not (all_args.use_gae and
                all_args.use_gae_trace), "advantage esitmation option error!"

    assert (sum([all_args.use_sequential, all_args.use_MA_ratio]) <=
            1), "agent by agent or joint update, exclusively"

    if all_args.use_MA_ratio:
        assert (all_args.clip_param >
                all_args.others_clip_param), "check clip params for MA ratio"

    if all_args.use_sequential:
        assert (sum([all_args.agent_loop_first, all_args.ppo_loop_first
                     ]) == 1), "update order in sequential update error!"

        assert not (all_args.agent_loop_first and "_r" in all_args.seq_strategy
                    ), "agent loop first can not be extreme greedy"

    assert (all_args.episode_length %
            all_args.data_chunk_length == 0), "chunk length requirement!"


if __name__ == "__main__":
    main(sys.argv[1:])
