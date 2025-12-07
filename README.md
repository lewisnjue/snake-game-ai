# Snake Game AI

**Reinforcement Learning**: An area of machine learning concerned with how software agents ought to take actions in an environment in order to maximize the notion of cumulative reward. Or: RL is teaching a software agent how to behave in an environment by telling it how good it's doing.

## Model Used

This game uses **Deep Q-Network (DQN)** for Reinforcement Learning.

### What is DQN?

- **Deep Q Networks** are a type of reinforcement learning model where an agent learns to take actions in an environment to maximize cumulative rewards over time.
- Instead of maintaining a Q table (used in traditional Q-learning), the agent uses a neural network to approximate the Q-values for state-action pairs.

### Components of the Code

#### `Linear_QNet` Class
Defines the neural network architecture for Q-function approximation.

**Attributes:**
- `linear1`: A fully connected (dense) layer that maps `input_size` to `hidden_size`
- `linear2`: Another fully connected layer that maps the hidden layer to `output_size` (Q-values for all actions)

**Forward Pass:**
- Input `x` passes through `linear1` with ReLU activation
- Output passed to `linear2` to compute final Q-values

#### `QTrainer` Class
Handles the training process using Q-learning principles.

**Attributes:**
- `model`: The neural network (instance of `Linear_QNet`)
- `lr`: Learning rate for the optimizer
- `gamma`: Discount factor for future rewards
- `optimizer`: Adam optimizer for parameter updates
- `criterion`: Mean Squared Error loss function

## Installation

### Requirements
- Python 3.7+
- GPU recommended (CUDA for PyTorch) but CPU will work

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/lewisnjue/snake-game-ai.git
   cd snake-game-ai
   ```

2. Install dependencies:

## Usage
```bash
   # Snake Game AI

   This repository contains a simple Snake game and a Deep Q-Network (DQN) agent that learns to play it using
   reinforcement learning. The project is intended as a learning/demo project for reinforcement learning concepts.

   Overview
   --------
   - Model: Deep Q-Network (DQN) to approximate Q-values for actions given a game state.
   - Languages / libs: Python, PyTorch, pygame, NumPy, matplotlib.

   Quick links
   -----------
   - Code: `game.py`, `agent.py`, `model.py`, `helper.py`, `snake_game_human.py`
   - Tests and CI: `.github/workflows/ci.yml`, `tests/`

   Requirements
   ------------
   - Python 3.8+
   - See `requirements.txt` for the main Python packages (PyTorch, pygame, numpy, matplotlib, ipython).

   Installation
   ------------
   Clone and install dependencies:

   ```bash
   git clone https://github.com/lewisnjue/snake-game-ai.git
   cd snake-game-ai
   pip install -r requirements.txt
   ```

   If you want GPU support, install a CUDA-enabled PyTorch build following the instructions on https://pytorch.org/.

   Running the project
   -------------------
   There are two main usage modes: human play and training the AI agent.

   1) Play as a human

   ```bash
   python snake_game_human.py
   ```

   Use the arrow keys to control the snake. Close the window or press the window close button to quit.

   2) Train the AI agent

   The project now includes a small CLI. The recommended way to run training is via `agent.py`.

   Examples:

   ```bash
   # Train (default settings)
   python agent.py --mode train

   # Train headless (no interactive plotting), run 100 games max, with a fixed seed
   python agent.py --mode train --headless --max-games 100 --seed 42

   # Train with custom hyperparameters
   python agent.py --mode train --lr 0.0005 --batch-size 512 --gamma 0.95
   ```

   3) Human play via CLI (alias)

   ```bash
   python agent.py --mode play
   ```

   Note: `--mode play` currently launches the human play script. `--mode play-ai` is reserved for running a trained agent (future work).

   CLI options (selected)
   ----------------------
   - `--mode`: `train` (default), `play`, `play-ai`
   - `--block-size`: game grid cell size in pixels (default: 20)
   - `--speed`: game speed (FPS) used during training (default: 40)
   - `--speed-play`: game speed for human play (default: 20)
   - `--headless`: disable interactive plotting and save PNG snapshots instead
   - `--seed`: integer seed for reproducibility
   - `--max-games`: stop training after N games
   - `--model-path`: path to save/load the model

   Project structure
   -----------------

   ```
   . 
   ├── agent.py               # DQN agent and training loop (entry point for training)
   ├── game.py                # Core game engine (SnakeGameAI)
   ├── model.py               # Neural network and training helper
   ├── snake_game_human.py    # Human-playable version using pygame
   ├── helper.py              # Plotting utilities (supports headless mode)
   ├── requirements.txt       # Python dependencies
   ├── tests/                 # Unit tests (pytest)
   └── .github/workflows/ci.yml # CI pipeline (pytest + flake8)
   ```

   How the agent works (brief)
   ---------------------------
   - State: 11 features including danger indicators, current direction, and relative food position.
   - Actions: 3 discrete actions (straight, right, left).
   - Rewards: +10 for eating food, -10 for collision, 0 otherwise.
   - Training: DQN with experience replay and a small feed-forward network.

   Development notes and tips
   -------------------------
   - If you run on a headless server (CI or remote machine) use `--headless` to avoid interactive plotting.
   - Use `--seed` to get repeatable runs (note: GPU floating point nondeterminism can still cause small differences).
   - CI runs unit tests on multiple Python versions and checks imports and basic functionality.

   Contributing
   ------------
   Open issues or PRs for improvements. Suggested next steps:

   - Add a `play-ai` runner to load a saved model and let the agent play interactively.
   - Add more informative logging and model checkpointing during training.
   - Provide a short demo notebook showing training results and sample gameplay.

   License
   -------
   MIT License
