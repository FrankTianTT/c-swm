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
import ffmpeg

input_shape = None


def load_env():
    global input_shape
    env = gym.make("ShapesTrain-v0")
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
    merge_render_folder = os.path.join(render_folder, "merge")
    os.makedirs(merge_render_folder, exist_ok=True)

    cnames = ['blue', 'black', 'green', 'red', 'cyan', 'magenta', 'navy', 'lime', 'gold', 'coral']

    os.makedirs(render_folder, exist_ok=True)
    (state, obs) = env.reset()

    timer = 0
    latent_render_range = None

    while True:
        with torch.no_grad():
            torch_obs = torch.unsqueeze(torch.Tensor(obs), 0)
            torch_z = model(torch_obs)
            numpy_z = torch.squeeze(torch_z).numpy()
        if latent_render_range is None:
            x_min, y_min = numpy_z.min(0) - 1
            x_max, y_max = numpy_z.max(0) + 1
            latent_render_range = [x_min, x_max, y_min, y_max]
        # obs render
        plt.imshow(obs.transpose(), interpolation='nearest')
        plt.savefig(os.path.join(obs_render_folder, "img{:04d}.png".format(timer)))
        plt.close()
        # latent render
        plt.axis(latent_render_range)
        plt.scatter(numpy_z[:, 0], numpy_z[:, 1], c=cnames[:len(numpy_z)], s=100, marker="s")
        plt.savefig(os.path.join(latent_render_folder, "img{:04d}.png".format(timer)))
        plt.close()
        # merge render
        _, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(obs.transpose(), interpolation='nearest')
        plt.axis(latent_render_range)
        axes[1].scatter(numpy_z[:, 0], numpy_z[:, 1], c=cnames[:len(numpy_z)], s=100, marker="s")
        plt.savefig(os.path.join(merge_render_folder, "img{:04d}.png".format(timer)))
        plt.close()

        timer += 1

        (state, obs), reward, done, _ = env.step(env.action_space.sample())

        if done:
            break
    # video
    # examples:
    # ffmpeg -y -f image2 -i render/shapes_double_num/merge/img%04d.png render/shapes_double_num/output.mp4
    command = "ffmpeg -r 5 -y -f image2 -i {} {}".format(os.path.join(merge_render_folder, "img%04d.png"),
                                                    os.path.join(render_folder, "output.mp4"))
    os.system(command)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints-folder', type=str,
                        default='checkpoints',
                        help='Path to checkpoints.')
    parser.add_argument('--render-folder', type=str,
                        default='render',
                        help='Path to save render result.')
    parser.add_argument('--name', type=str,
                        default='shapes',
                        help='Experiment name.')
    parser.add_argument('--num-steps', type=int, default=1,
                        help='Number of prediction steps to evaluate.')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disable CUDA training.')

    args_eval = parser.parse_args()
    save_folder = os.path.join(args_eval.checkpoints_folder, args_eval.name)

    meta_file = os.path.join(save_folder, 'metadata.pkl')
    model_file = os.path.join(save_folder, 'model.pt')

    env = load_env()

    cuda = not args_eval.no_cuda and torch.cuda.is_available()
    model = load_model(meta_file, model_file, cuda)

    render_folder = os.path.join(args_eval.render_folder, args_eval.name)
    visual_rollout(env, model, render_folder)
