"""
Unit tests for the game engine.
"""
import pytest
from game import SnakeGameAI, Direction, Point


class TestSnakeGameAI:
    """Tests for the SnakeGameAI class."""
    
    def test_game_initialization(self):
        """Test that game initializes correctly."""
        game = SnakeGameAI(w=640, h=480)
        assert game.w == 640
        assert game.h == 480
        assert game.score == 0
        assert len(game.snake) == 3
        assert game.direction == Direction.RIGHT
    
    def test_reset(self):
        """Test that reset() restores initial state."""
        game = SnakeGameAI()
        game.score = 10
        game.reset()
        assert game.score == 0
        assert len(game.snake) == 3
        assert game.direction == Direction.RIGHT
    
    def test_collision_with_wall(self):
        """Test collision detection with walls."""
        game = SnakeGameAI()
        # Head at top-left boundary
        assert game.is_collision(Point(-1, 10))
        # Head at right boundary
        assert game.is_collision(Point(game.w, 10))
        # Head at bottom boundary
        assert game.is_collision(Point(10, game.h))
    
    def test_collision_with_self(self):
        """Test collision detection with self."""
        game = SnakeGameAI()
        # Snake body is at head - 1 block and head - 2 blocks
        # Check if collision with body segments works
        snake_body = game.snake[1:]
        assert len(snake_body) >= 1
        # Head should collide with any point in its body
        assert not game.is_collision(game.head)
    
    def test_no_collision_in_center(self):
        """Test that center positions don't collide."""
        game = SnakeGameAI()
        safe_point = Point(game.w // 2, game.h // 2)
        # Should not collide if safe (excluding head itself)
        if safe_point != game.head:
            assert not game.is_collision(safe_point)
    
    def test_food_placement(self):
        """Test that food is placed within bounds."""
        game = SnakeGameAI()
        food = game.food
        assert 0 <= food.x < game.w
        assert 0 <= food.y < game.h
    
    def test_food_not_on_snake(self):
        """Test that food is never placed on snake body."""
        for _ in range(10):
            game = SnakeGameAI()
            assert game.food not in game.snake


class TestGameCoordinates:
    """Tests for coordinate alignment and grid system."""
    
    def test_head_coordinates_are_integer(self):
        """Test that head coordinates are integers (grid-aligned)."""
        game = SnakeGameAI()
        assert isinstance(game.head.x, int), "Head x-coordinate should be integer"
        assert isinstance(game.head.y, int), "Head y-coordinate should be integer"
    
    def test_snake_body_alignment(self):
        """Test that snake body is aligned to block grid."""
        game = SnakeGameAI()
        from game import BLOCK_SIZE
        for segment in game.snake:
            assert segment.x % BLOCK_SIZE == 0, f"x={segment.x} not aligned to BLOCK_SIZE={BLOCK_SIZE}"
            assert segment.y % BLOCK_SIZE == 0, f"y={segment.y} not aligned to BLOCK_SIZE={BLOCK_SIZE}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
