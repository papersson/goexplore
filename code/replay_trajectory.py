from gym.envs.classic_control import rendering
import gym
import argparse
import numpy as np
import pickle

p = argparse.ArgumentParser()
default_demo = 'demo.trajectory'
p.add_argument('--path', type=str, default=default_demo)
args = p.parse_args()


def replay(actions_taken, env):
    env.reset()
    done = False
    for action in actions_taken:
        _, _, done, _ = env.step(action)

        rgb = env.render('rgb_array')
        img = repeat_upsample(rgb, 8, 8)
        viewer.imshow(img)

        if done:
            break
    env.close()


def read(file):
    for s in file.split('_'):
        if 'Deterministic' in s:
            env = s
        else:
            env = 'MontezumaRevengeDeterministic-v4'
    with open(file, 'rb') as f:
        actions = pickle.load(f)
    return env, actions


def repeat_upsample(rgb_array, k=1, l=1, err=[]):
    if k <= 0 or l <= 0:
        if not err:
            err.append('logged')
        return rgb_array
    return np.repeat(np.repeat(rgb_array, k, axis=0), l, axis=1)


viewer = rendering.SimpleImageViewer()

env_name, actions = read(args.path)
env = gym.make(env_name)
replay(actions, env)
