"""
Unit tests for the agent module.
"""
import pytest
import numpy as np
from game import SnakeGameAI
from agent import Agent


class TestAgent:
    """Tests for the Agent class."""
    
    def test_agent_initialization(self):
        """Test that agent initializes correctly."""
        agent = Agent()
        assert agent.n_games == 0
        assert agent.gamma == 0.9
        assert agent.model is not None
        assert agent.trainer is not None
    
    def test_get_state_shape(self):
        """Test that agent state has correct shape (11 features)."""
        game = SnakeGameAI()
        agent = Agent()
        state = agent.get_state(game)
        assert len(state) == 11, f"Expected state size 11, got {len(state)}"
        assert state.dtype == np.int32 or state.dtype == int
    
    def test_get_state_values(self):
        """Test that state values are binary (0 or 1)."""
        game = SnakeGameAI()
        agent = Agent()
        state = agent.get_state(game)
        assert all(s in (0, 1) for s in state), "State should be binary (0/1)"
    
    def test_remember(self):
        """Test that agent remembers experiences."""
        agent = Agent()
        game = SnakeGameAI()
        state = agent.get_state(game)
        action = [1, 0, 0]
        reward = 10
        next_state = agent.get_state(game)
        done = False
        
        agent.remember(state, action, reward, next_state, done)
        assert len(agent.memory) == 1
    
    def test_memory_limit(self):
        """Test that memory doesn't exceed MAX_MEMORY."""
        agent = Agent()
        game = SnakeGameAI()
        
        # Add many experiences (more than MAX_MEMORY)
        for _ in range(agent.memory.maxlen + 100):
            state = agent.get_state(game)
            action = [1, 0, 0]
            reward = 0
            next_state = agent.get_state(game)
            done = False
            agent.remember(state, action, reward, next_state, done)
        
        assert len(agent.memory) <= agent.memory.maxlen
    
    def test_get_action_shape(self):
        """Test that action has correct shape (3 elements)."""
        agent = Agent()
        game = SnakeGameAI()
        state = agent.get_state(game)
        action = agent.get_action(state)
        assert len(action) == 3
        assert sum(action) == 1, "Only one action should be active"
    
    def test_get_action_valid(self):
        """Test that action is valid (one-hot encoded)."""
        agent = Agent()
        game = SnakeGameAI()
        
        for _ in range(100):
            state = agent.get_state(game)
            action = agent.get_action(state)
            assert action in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


class TestAgentLearning:
    """Tests for agent learning components."""
    
    def test_train_short_memory(self):
        """Test that short memory training doesn't crash."""
        agent = Agent()
        game = SnakeGameAI()
        state = agent.get_state(game)
        action = [1, 0, 0]
        reward = 10
        next_state = agent.get_state(game)
        done = False
        
        # Should not raise exception
        agent.train_short_memory(state, action, reward, next_state, done)
    
    def test_train_long_memory(self):
        """Test that long memory training doesn't crash with small batch."""
        agent = Agent()
        game = SnakeGameAI()
        
        # Add a few experiences
        for _ in range(10):
            state = agent.get_state(game)
            action = [1, 0, 0]
            reward = 0
            next_state = agent.get_state(game)
            done = False
            agent.remember(state, action, reward, next_state, done)
        
        # Should not raise exception
        agent.train_long_memory()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
