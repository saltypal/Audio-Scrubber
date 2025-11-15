import torch
import itertools
from train import Config, train_model
import json
from pathlib import Path

"""
Created by Satya with Copilot @ 15/11/25

Hyperparameter tuning for 1D U-Net Audio Denoiser
- Tests different combinations of hyperparameters
- Saves results for comparison
- Helps find the best configuration
"""

# Define hyperparameter search space
HYPERPARAM_GRID = {
    'BATCH_SIZE': [4, 8, 16],
    'LEARNING_RATE': [0.0001, 0.001, 0.01],
    'NUM_EPOCHS': [30, 50],  # Reduced for faster tuning
}


def hyperparameter_search():
    """
    Perform grid search over hyperparameters.
    
    Tests all combinations and saves results.
    """
    print(f"\n{'='*60}")
    print(f"Hyperparameter Tuning for 1D U-Net")
    print(f"{'='*60}\n")
    
    # Generate all combinations
    keys = HYPERPARAM_GRID.keys()
    values = HYPERPARAM_GRID.values()
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Total combinations to test: {len(combinations)}\n")
    
    results = []
    
    for i, params in enumerate(combinations, 1):
        print(f"\n{'='*60}")
        print(f"Experiment {i}/{len(combinations)}")
        print(f"{'='*60}")
        print(f"Parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        print(f"{'='*60}\n")
        
        # Create config with these hyperparameters
        config = Config()
        for key, value in params.items():
            setattr(config, key, value)
        
        # Update model save path to include hyperparameters
        model_name = f"unet1d_bs{params['BATCH_SIZE']}_lr{params['LEARNING_RATE']}_ep{params['NUM_EPOCHS']}.pth"
        config.MODEL_SAVE_PATH = f"saved_models/tuning/{model_name}"
        
        try:
            # Train model
            train_losses, val_losses, model = train_model(config)
            
            # Record results
            result = {
                'experiment': i,
                'hyperparameters': params,
                'final_train_loss': train_losses[-1],
                'final_val_loss': val_losses[-1],
                'best_val_loss': min(val_losses),
                'model_path': config.MODEL_SAVE_PATH
            }
            results.append(result)
            
            print(f"\n✅ Experiment {i} completed!")
            print(f"   Final validation loss: {result['final_val_loss']:.6f}")
            print(f"   Best validation loss: {result['best_val_loss']:.6f}")
            
        except Exception as e:
            print(f"\n❌ Experiment {i} failed: {e}")
            results.append({
                'experiment': i,
                'hyperparameters': params,
                'error': str(e)
            })
    
    # Save results
    results_path = "saved_models/tuning/hyperparameter_results.json"
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Hyperparameter Tuning Completed!")
    print(f"{'='*60}\n")
    
    # Find best configuration
    successful_results = [r for r in results if 'error' not in r]
    if successful_results:
        best_result = min(successful_results, key=lambda x: x['best_val_loss'])
        
        print("🏆 Best Configuration:")
        print(f"   Experiment: {best_result['experiment']}")
        print(f"   Best validation loss: {best_result['best_val_loss']:.6f}")
        print(f"   Hyperparameters:")
        for key, value in best_result['hyperparameters'].items():
            print(f"     {key}: {value}")
        print(f"   Model saved to: {best_result['model_path']}")
    
    print(f"\nResults saved to: {results_path}")
    print(f"{'='*60}\n")
    
    return results


def quick_tune():
    """
    Quick hyperparameter tuning with fewer combinations.
    Good for testing before full grid search.
    """
    print(f"\n{'='*60}")
    print(f"Quick Hyperparameter Tuning")
    print(f"{'='*60}\n")
    
    # Smaller search space for quick testing
    quick_grid = {
        'BATCH_SIZE': [8],
        'LEARNING_RATE': [0.0001, 0.001],
        'NUM_EPOCHS': [20],
    }
    
    global HYPERPARAM_GRID
    HYPERPARAM_GRID = quick_grid
    
    return hyperparameter_search()


if __name__ == "__main__":
    # Uncomment the one you want to run:
    
    # Full grid search (takes longer)
    hyperparameter_search()
    
    # Quick tuning (faster, fewer combinations)
    # quick_tune()
