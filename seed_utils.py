"""
Utility functions for reproducible random seeding across all libraries.
"""
import random
import numpy as np


def set_seed(seed):
    """
    Set random seed for reproducibility across random, numpy, torch, and pygame.
    
    Args:
        seed: Integer seed value
    """
    if seed is None:
        return
    
    # Python's random module
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch (if available)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # Disable cuDNN auto-tuning for full reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    
    # Pygame doesn't have a direct seed function, but random.seed covers it
    
    print(f"[Seeding] Random seed set to {seed}")
