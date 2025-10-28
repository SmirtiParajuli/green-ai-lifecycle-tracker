# Configuration file for project paths and CIFAR-10 training settings
# Context: src/config.py

import os

# Project paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data', 'cifar10')
LOGS_DIR = os.path.join(ROOT_DIR, 'logs')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# CIFAR-10 training config
CIFAR10_CONFIG = {
    'batch_size': 128,
    'epochs': 20,
    'learning_rate': 0.001,
    'seed': 42
}

CARBON_CONFIG = {
    'log_level': 'error',
    'save_to_file': True
}
