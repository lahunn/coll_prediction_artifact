import os
import torch
from model import EncoderProcessDecoder
from model_smoother import ModelSmoother
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Get base directory for relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


MODEL_CONFIGS = {
    'maze2': {
        'workspace_size': 2,
        'config_size': 2,
        'embed_size': 32,
        'obs_size': 2,
        'dim': 2,
        'model_explore_path': 'data/weights/weights_maze.pt',
        'model_smooth_path': 'data/weights/smooth_2d_attv3.pt',
        'smooth_embed_size': 128,
        'smooth_obs_size': 6,
    },
    'maze3': {
        'workspace_size': 2,
        'config_size': 3,
        'embed_size': 32,
        'obs_size': 2,
        'dim': 3,
        'model_explore_path': 'data/weights/weights_maze_3.pt',
        'model_smooth_path': 'data/weights/smooth_3d_att.pt',
        'smooth_embed_size': 128,
        'smooth_obs_size': 6,
    },
    'kuka7': {
        'workspace_size': 3,
        'config_size': 7,
        'embed_size': 64,
        'obs_size': 6,
        'dim': 3,
        'model_explore_path': 'data/weights/weights_kuka.pt',
        'model_smooth_path': 'data/weights/smooth_7d_attv3.pt',
        'smooth_embed_size': 128,
        'smooth_obs_size': 6,
    },
    'ur5': {
        'workspace_size': 3,
        'config_size': 6,
        'embed_size': 32,
        'obs_size': 6,
        'dim': 3,
        'model_explore_path': 'data/weights/weights_ur5.pt',
        'model_smooth_path': 'data/weights/smooth_ur5_attv3.pt',
        'smooth_embed_size': 128,
        'smooth_obs_size': 6,
        'smooth_scale': None,
    },
    'snake7': {
        'workspace_size': 3,
        'config_size': 7,
        'embed_size': 32,
        'obs_size': 2,
        'dim': 3,
        'model_explore_path': 'data/weights/weights_snake.pt',
        'model_smooth_path': 'data/weights/smooth_snake_attv3.pt',
        'smooth_embed_size': 128,
        'smooth_obs_size': 6,
    },
    'kuka13': {
        'workspace_size': 3,
        'config_size': 13,
        'embed_size': 32,
        'obs_size': 6,
        'dim': 3,
        'model_explore_path': 'data/weights/weights_kuka_13.pt',
        'model_smooth_path': 'data/weights/smooth_13d_attv3.pt',
        'smooth_embed_size': 128,
        'smooth_obs_size': 6,
    },
    'kuka14': {
        'workspace_size': 3,
        'config_size': 14,
        'embed_size': 32,
        'obs_size': 6,
        'dim': 3,
        'model_explore_path': 'data/weights/kuka_14.pt',
        'model_smooth_path': 'data/weights/smooth_14d_attv3.pt',
        'smooth_embed_size': 128,
        'smooth_obs_size': 6,
    },
}


def str2model(model_key, use_obstacle=True, load=False):
    """
    Load model and smoother instances based on model key.
    
    Args:
        model_key: String identifier for the model configuration
        use_obstacle: If False, load pure model weights (no obstacle)
        load: If True, load pretrained weights into models
        
    Returns:
        tuple: (model_explore, model_explore_path, model_smooth, model_smooth_path)
    """
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model_key: {model_key}. Available: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[model_key]
    
    model_explore = EncoderProcessDecoder(
        workspace_size=config['workspace_size'],
        config_size=config['config_size'],
        embed_size=config['embed_size'],
        obs_size=config['obs_size']
    ).to(device)
    
    model_explore_path = os.path.join(BASE_DIR, config['model_explore_path'])
    
    smooth_kwargs = {
        'workspace_size': config['dim'],
        'config_size': config['config_size'],
        'embed_size': config['smooth_embed_size'],
        'obs_size': config['smooth_obs_size'],
    }
    
    if 'smooth_scale' in config and config['smooth_scale'] is not None:
        smooth_kwargs['scale'] = config['smooth_scale']
    
    model_smooth = ModelSmoother(**smooth_kwargs).to(device)
    model_smooth_path = os.path.join(BASE_DIR, config['model_smooth_path'])
    
    if not use_obstacle:
        model_explore_path = model_explore_path.replace('.pt', '_pure.pt')
    
    if load:
        model_explore.load_state_dict(torch.load(model_explore_path, map_location=device))
        model_explore.to(device)
        
        model_smooth.load_state_dict(torch.load(model_smooth_path, map_location=device))
        model_smooth.to(device)
    
    return model_explore, model_explore_path, model_smooth, model_smooth_path
