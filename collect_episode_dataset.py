import argparse
import collections
import concurrent.futures
import multiprocessing
import os
import yaml
from pathlib import Path

import isaacgym

import numpy as np
from PIL import Image

from stable_baselines3.common.env_checker import check_env
from tqdm import tqdm

from td3_agent import TD3HER

from isaac_panda_push_env import IsaacPandaPush
from isaac_env_wrappers import IsaacPandaPushGoalSB3Wrapper

from utils import load_pretrained_rep_model, load_latent_classifier, check_config


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--num_train_episodes', type=int, required=True)
    parser.add_argument('--num_val_episodes', type=int, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--epsilon_initial', type=float, required=True)
    parser.add_argument('--epsilon_final', type=float, default=0)
    parser.add_argument('--action_noise_sigma_initial', type=float, required=True)
    parser.add_argument('--action_noise_sigma_final', type=float, default=0)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--max_episode_length', type=int, required=True)
    parser.add_argument('--num_objects', type=int, required=True)

    return parser.parse_args()


def store(observations, actions, dataset_path, split, episode_id):
    dir_name = os.path.join(dataset_path, split, f'{episode_id:05d}')
    os.makedirs(dir_name, exist_ok=True)
    np.save(file=os.path.join(dir_name, 'actions.npy'), arr=np.asarray(actions))
    for i, observation in enumerate(observations):
        path = os.path.join(dir_name, f'{i:04d}.png')
        Image.fromarray(observation).save(path)

    return len(observations)


if __name__ == '__main__':
    """
    Script for evaluating trained policies on a given environment.
    """
    args = parser_args()

    #################################
    #            Config             #
    #################################

    model_path = args.checkpoint_path
    stat_save_path = None

    # load config files
    config = yaml.safe_load(Path('config/n_cubes/Config.yaml').read_text())
    isaac_env_cfg = yaml.safe_load(Path('config/n_cubes/IsaacPandaPushConfig.yaml').read_text())
    isaac_env_cfg['env']['numObjects'] = args.num_objects

    check_config(config, isaac_env_cfg)

    # random seed
    seed = np.random.randint(args.seed)
    print(f"Random seed: {seed}")

    #################################
    #         Representation        #
    #################################

    latent_rep_model = load_pretrained_rep_model(dir_path=config['Model']['latentRepPath'], model_type=config['Model']['obsMode'])
    latent_classifier = load_latent_classifier(config, num_objects=isaac_env_cfg["env"]["numObjects"])

    #################################
    #          Environment          #
    #################################

    # create environments
    envs = IsaacPandaPush(
        cfg=isaac_env_cfg,
        rl_device=f"cuda:{config['cudaDevice']}",
        sim_device=f"cuda:{config['cudaDevice']}",
        graphics_device_id=config['cudaDevice'],
        headless=True,
        virtual_screen_capture=False,
        force_render=False,
    )

    # wrap enviroments for GoalEnv and SB3 compatibility
    env = IsaacPandaPushGoalSB3Wrapper(
        env=envs,
        obs_mode=config['Model']['obsMode'],
        n_views=config['Model']['numViews'],
        latent_rep_model=latent_rep_model,
        latent_classifier=latent_classifier,
        reward_cfg=config['Reward']['GT'],
        smorl=(config['Model']['method'] == 'SMORL'),
    )

    if config['envCheck']:
        check_env(env, warn=True, skip_render_check=True)  # verify SB3 compatibility

    print(f"Finished setting up environment")

    #################################
    #        Model Analysis       #
    #################################

    model = TD3HER.load(model_path, env,
                        custom_objects=dict(
                            seed=seed,
                            eval_max_episode_length=args.max_episode_length,
                         ),
                        )

    # collect data in environment
    num_episodes = args.num_train_episodes + args.num_val_episodes
    proba_val = args.num_val_episodes / num_episodes
    counter = collections.Counter()
    futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers,
                                                mp_context=multiprocessing.get_context('forkserver')) as executor:

        def store_episode_callback(images, actions):
            split = 'val' if np.random.random() < proba_val else 'train'
            futures.append(executor.submit(store, images, actions, args.dataset_path, split, counter[split]))
            counter[split] += 1

        print(f"Collecting dataset with model {model_path} on {env.num_objects} objects")
        eval_stat_dict = model.collect_dataset(store_episode_callback, num_episodes=num_episodes,
                                               epsilon_initial=args.epsilon_initial, epsilon_final=args.epsilon_final,
                                               action_noise_sigma_initial=args.action_noise_sigma_initial,
                                               action_noise_sigma_final=args.action_noise_sigma_final)

        print(eval_stat_dict)
        for i in tqdm(range(len(futures))):
            futures[i].result()
