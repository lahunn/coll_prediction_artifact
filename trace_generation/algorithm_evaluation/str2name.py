from model import EncoderProcessDecoder
from model_smoother import ModelSmoother

# import model_smoother2
import torch
from trace_generation.core.robot.modular_env import ModularEnv
import numpy as np
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def str2name(str, get_data=False, use_obstacle=True, load=False, env=None):
    if "maze2" in str:
        # env = MazeEnv(dim=2)
        # Assuming maze also uses ModularEnv now, but let's focus on kuka7 first.
        # If maze still needs MazeEnv, we might need to keep it or adapt it.
        # For now, I'll only change what's necessary for kuka7.
        if env is None:
            from environment import MazeEnv
            env = MazeEnv(dim=2)
        model_explore = EncoderProcessDecoder(
            workspace_size=2, config_size=2, embed_size=32, obs_size=2
        ).to(device)
        model_explore_path = os.path.join(BASE_DIR, "data/weights/weights_maze.pt")
        model_smooth = ModelSmoother(
            workspace_size=env.dim,
            config_size=env.config_dim,
            embed_size=128,
            obs_size=6,
        ).to(device)
        model_smooth_path = os.path.join(BASE_DIR, "data/weights/smooth_2d_attv3.pt")
        data_path = os.path.join(BASE_DIR, "data/pkl/maze_prm_4000.pkl")

    elif str == "maze3":
        if env is None:
            from environment import MazeEnv

            env = MazeEnv(dim=3)
        model_explore = EncoderProcessDecoder(
            workspace_size=2, config_size=3, embed_size=32, obs_size=2
        ).to(device)
        model_explore_path = os.path.join(BASE_DIR, "data/weights/weights_maze_3.pt")
        model_smooth = ModelSmoother(
            workspace_size=env.dim,
            config_size=env.config_dim,
            embed_size=128,
            obs_size=6,
        ).to(device)
        model_smooth_path = os.path.join(BASE_DIR, "data/weights/smooth_3d_attv3.pt")
        data_path = os.path.join(BASE_DIR, "data/pkl/maze_prm_3.pkl")

    elif str == "kuka7":
        if env is None:
            data_path = os.path.join(BASE_DIR, "maze_files/kukas_7_4000.pkl")
            env = ModularEnv(robot_name="iiwa", map_file=data_path)
        model_explore = EncoderProcessDecoder(
            workspace_size=3, config_size=7, embed_size=64, obs_size=6
        ).to(device)
        model_explore_path = os.path.join(BASE_DIR, "data/weights/weights_kuka.pt")
        model_smooth = ModelSmoother(
            workspace_size=env.dim,
            config_size=env.config_dim,
            embed_size=128,
            obs_size=6,
        ).to(device)
        model_smooth_path = os.path.join(BASE_DIR, "data/weights/smooth_7d_attv3.pt")
        data_path = os.path.join(BASE_DIR, "maze_files/kukas_7_4000.pkl")

    elif str == "ur5":
        if env is None:
            data_path = os.path.join(BASE_DIR, "maze_files/ur5s_6_3000.pkl")
            env = ModularEnv(robot_name="ur5e", map_file=data_path)  # Assuming ur5e
            env.dim = 3
        model_explore = EncoderProcessDecoder(
            workspace_size=3, config_size=6, embed_size=32, obs_size=6
        ).to(device)
        model_explore_path = os.path.join(BASE_DIR, "data/weights/weights_ur5.pt")
        model_smooth = ModelSmoother(
            workspace_size=3,
            config_size=6,
            embed_size=128,
            obs_size=6,
            scale=np.max(env.bound),
        ).to(device)
        model_smooth_path = os.path.join(BASE_DIR, "data/weights/smooth_ur5_attv3.pt")
        data_path = os.path.join(BASE_DIR, "maze_files/ur5s_6_3000.pkl")

    elif str == "snake7":
        # env = SnakeEnv(map_file='maze_files/snakes_15_2_3000.npz')
        # Snake might need more work if it's not a standard robot
        if env is None:
            from environment import SnakeEnv

            env = SnakeEnv(
                map_file=os.path.join(BASE_DIR, "maze_files/snakes_15_2_3000.npz")
            )
        model_explore = EncoderProcessDecoder(
            workspace_size=3, config_size=7, embed_size=32, obs_size=2
        ).to(device)
        model_explore_path = os.path.join(BASE_DIR, "data/weights/weights_snake.pt")
        model_smooth = ModelSmoother(
            workspace_size=env.dim,
            config_size=env.config_dim,
            embed_size=128,
            obs_size=6,
        ).to(device)
        model_smooth_path = os.path.join(BASE_DIR, "data/weights/smooth_snake_attv3.pt")
        data_path = os.path.join(BASE_DIR, "data/pkl/snake_prm_3000.pkl")

    elif str == "kuka13":
        if env is None:
            data_path = os.path.join(BASE_DIR, "maze_files/kukas_13_3000.pkl")
            env = ModularEnv(
                robot_name="iiwa_allegro", map_file=data_path
            )  # 14 DOF actually? Or something else
            env.dim = 3
        model_explore = EncoderProcessDecoder(
            workspace_size=3, config_size=13, embed_size=32, obs_size=6
        ).to(device)
        model_explore_path = os.path.join(BASE_DIR, "data/weights/weights_kuka_13.pt")
        model_smooth = ModelSmoother(
            workspace_size=env.dim,
            config_size=env.config_dim,
            embed_size=128,
            obs_size=6,
        ).to(device)
        model_smooth_path = os.path.join(BASE_DIR, "data/weights/smooth_13d_attv3.pt")
        data_path = os.path.join(BASE_DIR, "maze_files/kukas_13_3000.pkl")

    elif str == "kuka14":
        if env is None:
            data_path = os.path.join(BASE_DIR, "maze_files/kukas_14_3000.pkl")
            env = ModularEnv(robot_name="iiwa_allegro", map_file=data_path)
            env.dim = 3
        model_explore = EncoderProcessDecoder(
            workspace_size=3, config_size=14, embed_size=32, obs_size=6
        ).to(device)
        model_explore_path = os.path.join(BASE_DIR, "data/weights/kuka_14.pt")
        model_smooth = ModelSmoother(
            workspace_size=env.dim,
            config_size=env.config_dim,
            embed_size=128,
            obs_size=6,
        ).to(device)
        model_smooth_path = os.path.join(BASE_DIR, "data/weights/smooth_14d_attv3.pt")
        data_path = os.path.join(BASE_DIR, "maze_files/kukas_14_3000.pkl")

    if not use_obstacle:
        model_explore_path = model_explore_path.replace(".pt", "_pure.pt")

    if load:
        model_explore.load_state_dict(
            torch.load(model_explore_path, map_location=device, weights_only=True)
        )
        model_explore.to(device)

        model_smooth.load_state_dict(
            torch.load(model_smooth_path, map_location=device, weights_only=True)
        )
        model_smooth.to(device)

    if get_data:
        return (
            env,
            model_explore,
            model_explore_path,
            model_smooth,
            model_smooth_path,
            data_path,
        )
    else:
        return env, model_explore, model_explore_path, model_smooth, model_smooth_path
