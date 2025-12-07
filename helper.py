import matplotlib
import matplotlib.pyplot as plt

# Use non-interactive backend by default for headless environments
matplotlib.use('Agg')

def plot(scores, mean_scores, headless=False):
    """
    Plot training progress.
    
    Args:
        scores: List of game scores
        mean_scores: List of mean scores over time
        headless: If True, save to file instead of displaying
    """
    if headless:
        # Save to file instead of displaying
        plt.figure(figsize=(10, 6))
        plt.title('Training Progress')
        plt.xlabel('Number of Games')
        plt.ylabel('Score')
        plt.plot(scores, label='Score')
        plt.plot(mean_scores, label='Mean Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save PNG with game count as filename
        filename = f'training_progress_{len(scores)}.png'
        plt.savefig(filename, dpi=100)
        plt.close()
        print(f"[Headless] Saved plot to {filename}")
    else:
        # Interactive plotting (requires display)
        try:
            from IPython import display
            display.clear_output(wait=True)
            display.display(plt.gcf())
        except (ImportError, RuntimeError):
            # Fallback if not in Jupyter or display not available
            pass
        
        plt.clf()
        plt.title('Training...')
        plt.xlabel('Number of Games')
        plt.ylabel('Score')
        plt.plot(scores)
        plt.plot(mean_scores)
        plt.ylim(ymin=0)
        plt.text(len(scores)-1, scores[-1], str(scores[-1]))
        plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
        
        try:
            plt.show(block=False)
            plt.pause(0.1)
        except:
            # Fallback for non-interactive environments
            pass
