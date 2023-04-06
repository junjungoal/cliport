"""Data collection script."""

import os
import hydra
import cv2
import numpy as np
import random
import pickle

from cliport import tasks
from cliport.dataset import RavensDataset
from cliport.environments.environment import Environment
import h5py


@hydra.main(config_path='./cfg', config_name='data')
def main(cfg):
    # Initialize environment and task.
    env = Environment(
        cfg['assets_root'],
        disp=cfg['disp'],
        shared_memory=cfg['shared_memory'],
        hz=480,
        record_cfg=cfg['record']
    )
    task = tasks.names[cfg['task']]()
    task.mode = cfg['mode']
    record = cfg['record']['save_video']
    save_frames = cfg['record']['save_frames']
    save_data = cfg['save_data']

    # Initialize scripted oracle agent and dataset.
    agent = task.oracle(env)
    data_path = os.path.join(cfg['data_dir'], "{}-{}".format(cfg['task'], task.mode))
    dataset = RavensDataset(data_path, cfg, n_demos=0, augment=False)
    print(f"Saving to: {data_path}")
    print(f"Mode: {task.mode}")

    # Train seeds are even and val/test seeds are odd. Test seeds are offset by 10000
    seed = dataset.max_seed
    if seed < 0:
        if task.mode == 'train':
            seed = -2
        elif task.mode == 'val': # NOTE: beware of increasing val set to >100
            seed = -1
        elif task.mode == 'test':
            seed = -1 + 10000
        else:
            raise Exception("Invalid mode. Valid options: train, val, test")

    # Collect training data from oracle demonstrations.
    while dataset.n_episodes < cfg['n']:
        episode, total_reward = [], 0
        seed += 2

        # Set seeds.
        np.random.seed(seed)
        random.seed(seed)

        print('Oracle demo: {}/{} | Seed: {}'.format(dataset.n_episodes + 1, cfg['n'], seed))

        env.set_task(task)
        obs = env.reset()
        info = env.info
        reward = 0

        # Unlikely, but a safety check to prevent leaks.
        if task.mode == 'val' and seed > (-1 + 10000):
            raise Exception("!!! Seeds for val set will overlap with the test set !!!")

        # Start video recording (NOTE: super slow)
        if record:
            env.start_rec(f'{dataset.n_episodes+1:06d}')

        # if save_frames:
        #     env.start_frame_save()

        # Rollout expert policy

        task_goal = info['lang_goal']
        for _ in range(task.max_steps):
            act = agent.act(obs, info)
            episode.append((obs, act, reward, info))
            lang_goal = info['lang_goal']
            obs, reward, done, info = env.step(act)
            total_reward += reward
            print(f'Total Reward: {total_reward:.3f} | Done: {done} | Goal: {lang_goal}')
            if done:
                break
        episode.append((obs, None, reward, info))

        # End video recording
        if record:
            env.end_rec()
        # if save_frames:
        #     env.end_frame_save()

        # Only save completed demonstrations.
        if save_data and total_reward > 0.99:
            dataset.add(seed, episode)


            if not os.path.exists(os.path.join(data_path, 'episodes')):
                os.makedirs(os.path.join(data_path, 'episodes'))
            colors = env.colors
            depths = env.depth
            new_colors = []
            new_depths = []
            for color, depth in zip(colors, depths):
                new_colors.append(cv2.resize(color[:, 80:-80], (cfg['record']['frame_height'], cfg['record']['frame_width'])))
                new_depths.append(cv2.resize(depth[:, 80:-80], (cfg['record']['frame_height'], cfg['record']['frame_width'])))

            with h5py.File(os.path.join(data_path, 'episodes/episode_{}.h5'.format(dataset.n_episodes)), 'w') as F:
                F['traj_per_file'] = 1
                F['traj0/images'] = np.array(new_colors).astype(np.uint8)
                F['traj0/depths'] = np.array(new_depths)
                F['traj0/pad_mask'] = np.ones(len(new_colors))
                F['traj0/objects'] = info['present_objects']
                F['traj0/task_goal'] = task_goal

            with open(os.path.join(data_path, 'episodes/episode_{}.pkl'.format(dataset.n_episodes)), 'wb') as f:
                pickle.dump(env.predicates, f)


if __name__ == '__main__':
    main()
