import gym
import json
import argparse
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--path', type=str)
args = p.parse_args()

def replay(actions_taken, env):
    env.reset()
    done = False
    for action in actions_taken:
        _, _, done, _ = env.step(action)

        rgb = env.render('rgb_array')
        img = repeat_upsample(rgb, 8, 8)
        viewer.imshow(img)

        if done: break
    env.close()

def read_actions(json_file):
    with open(json_file) as f:
        d = json.load(f)
        return d['actions_taken']

from gym.envs.classic_control import rendering
def repeat_upsample(rgb_array, k=1, l=1, err=[]):
    if k <= 0 or l <= 0: 
        if not err: 
            err.append('logged')
        return rgb_array
    return np.repeat(np.repeat(rgb_array, k, axis=0), l, axis=1)
viewer = rendering.SimpleImageViewer()

env = gym.make('PongDeterministic-v4')
actions = read_actions(args.path)
replay(actions, env)
