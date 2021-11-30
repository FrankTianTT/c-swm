import argparse
import torch
import utils
import os
import pickle
import gym
import envs
from torch.utils import data
import numpy as np
from collections import defaultdict
import modules
import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2

input_shape = None


def load_env():
    global input_shape
    env = gym.make("ShapesEval-v0")
    (state, obs) = env.reset()
    input_shape = obs.shape
    return env


def load_model(meta_file, model_file, cuda):
    args = pickle.load(open(meta_file, 'rb'))['args']

    args.batch_size = 100
    args.seed = 0

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if args.cuda else 'cpu')

    model = modules.ContrastiveSWM(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        action_dim=args.action_dim,
        input_dims=input_shape,
        num_objects=args.num_objects,
        sigma=args.sigma,
        hinge=args.hinge,
        ignore_action=args.ignore_action,
        copy_action=args.copy_action,
        encoder=args.encoder).to(device)

    model.load_state_dict(torch.load(model_file))
    model.eval()

    return model


def visual_rollout(env, model, render_folder):
    latent_render_folder = os.path.join(render_folder, "latent")
    os.makedirs(latent_render_folder, exist_ok=True)
    obs_render_folder = os.path.join(render_folder, "obs")
    os.makedirs(obs_render_folder, exist_ok=True)

    cnames = ['blue', 'black', 'green', 'red', 'c', 'm', 'y', 'k', 'w']

    os.makedirs(render_folder, exist_ok=True)
    (state, obs) = env.reset()

    timer = 0

    while True:
        cv2.imwrite(os.path.join(obs_render_folder, "{}.png".format(timer)), obs.transpose() * 255)
        with torch.no_grad():
            obs = torch.Tensor(obs)
            obs = torch.unsqueeze(obs, 0)
            z = model(obs)
            z = torch.squeeze(z).numpy()
        plt.scatter(z[:, 0], z[:, 1], c=cnames[:len(z)], s=100, marker="s")
        plt.savefig(os.path.join(latent_render_folder, "{}.png".format(timer)))
        plt.close()
        timer += 1

        (state, obs), reward, done, _ = env.step(env.action_space.sample())

        if done:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-folder', type=str,
                        default='checkpoints/shapes',
                        help='Path to checkpoints.')
    parser.add_argument('--render-folder', type=str,
                        default='render/shapes',
                        help='Path to save render result.')
    parser.add_argument('--num-steps', type=int, default=1,
                        help='Number of prediction steps to evaluate.')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disable CUDA training.')

    args_eval = parser.parse_args()

    meta_file = os.path.join(args_eval.save_folder, 'metadata.pkl')
    model_file = os.path.join(args_eval.save_folder, 'model.pt')

    env = load_env()

    cuda = not args_eval.no_cuda and torch.cuda.is_available()
    model = load_model(meta_file, model_file, cuda)

    visual_rollout(env, model, args_eval.render_folder)
