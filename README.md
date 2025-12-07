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
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Play the Game (Human Mode)
```bash
python snake_game_human.py
```
Use arrow keys to control the snake. Try to eat food and avoid hitting walls or yourself!

### Train the AI Agent
```bash
python agent.py
```
The agent will train on multiple games. Training progress is plotted in real-time (requires a display).

**Training Details:**
- The agent learns by playing many games
- Training saves the best model to `./model/model.pth`
- Average scores are plotted after each game
- Training takes ~20-30 minutes on CPU, ~5-10 minutes on GPU

### Run Trained Agent
To play with a trained model (coming soon - feature to add).

## Project Structure

```
.
├── game.py                 # Core game engine (SnakeGameAI class)
├── agent.py               # DQN agent and training loop
├── model.py               # Neural network and trainer
├── snake_game_human.py    # Human playable version
├── helper.py              # Plotting utilities
├── requirements.txt       # Python dependencies
└── arial.ttf              # Font file (optional)
```

## How It Works

1. **State Representation**: The agent observes 11 features:
   - 3 danger signals (straight, left, right)
   - 4 direction signals (current movement direction)
   - 4 food location signals (relative to head)

2. **Actions**: The agent chooses one of 3 actions:
   - Go straight
   - Turn right
   - Turn left

3. **Rewards**:
   - +10 for eating food
   - -10 for colliding with wall or itself
   - 0 for each step

4. **Training**: Uses experience replay and Q-learning to improve policy

## Performance

The agent typically achieves scores of 10-30+ after sufficient training, compared to human performance of 5-15+ typically.

## Known Issues / TODO

- [ ] Large `arial.ttf` file bloats repo
- [ ] No CLI flags for configuration
- [ ] Plotting requires display (not suitable for headless training)
- [ ] No reproducibility/seeding support
- [ ] Missing unit tests and CI

## Contributing

Feel free to open issues or submit pull requests!

## License

MIT License
