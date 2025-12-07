"""
Configuration and CLI argument handling for Snake Game AI.
"""
import argparse
from pathlib import Path


def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Snake Game AI - Train or play Snake')
    
    # Mode selection
    parser.add_argument(
        '--mode',
        choices=['train', 'play', 'play-ai'],
        default='train',
        help='Mode: train (AI training), play (human), play-ai (trained agent)'
    )
    
    # Game parameters
    parser.add_argument(
        '--width',
        type=int,
        default=640,
        help='Game window width in pixels (default: 640)'
    )
    parser.add_argument(
        '--height',
        type=int,
        default=480,
        help='Game window height in pixels (default: 480)'
    )
    parser.add_argument(
        '--block-size',
        type=int,
        default=20,
        help='Size of each game block in pixels (default: 20)'
    )
    parser.add_argument(
        '--speed',
        type=int,
        default=40,
        help='Game speed for training (FPS, default: 40)'
    )
    parser.add_argument(
        '--speed-play',
        type=int,
        default=20,
        help='Game speed for human play (FPS, default: 20)'
    )
    
    # Training parameters
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (optional)'
    )
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Disable display/plotting (save to PNG instead)'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='./model/model.pth',
        help='Path to save/load trained model (default: ./model/model.pth)'
    )
    parser.add_argument(
        '--max-games',
        type=int,
        default=None,
        help='Maximum number of games to train (optional)'
    )
    
    # Hyperparameters
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate (default: 0.001)'
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.9,
        help='Discount factor (default: 0.9)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='Batch size for training (default: 1000)'
    )
    
    return parser.parse_args()


def configure_defaults(args):
    """Update game and agent global constants based on args."""
    import game
    import agent
    
    # Update game constants
    game.BLOCK_SIZE = args.block_size
    game.SPEED = args.speed
    
    # Update agent constants
    agent.LR = args.lr
    agent.BATCH_SIZE = args.batch_size
    
    # Model path
    args.model_path = args.model_path or './model/model.pth'
    
    return args
