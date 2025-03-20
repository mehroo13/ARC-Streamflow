import os

# Create necessary directories
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Initialize empty __init__.py files for proper imports
with open(os.path.join('utils', '__init__.py'), 'w') as f:
    pass
