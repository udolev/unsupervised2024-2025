import numpy as np

# Random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Data settings
NUM_SAMPLES = 5000  # Match actual dataset size
NUM_CV_RUNS = 10    # Number of cross-validation runs

# Categorical and numerical column definitions
CATEGORICAL_COLS = [
    'GENDER', 'SMOKING', 'FINGER_DISCOLORATION', 'MENTAL_STRESS',
    'EXPOSURE_TO_POLLUTION', 'LONG_TERM_ILLNESS', 'BREATHING_ISSUE',
    'ALCOHOL_CONSUMPTION', 'THROAT_DISCOMFORT', 'CHEST_TIGHTNESS',
    'FAMILY_HISTORY', 'SMOKING_FAMILY_HISTORY', 'STRESS_IMMUNE',
    'IMMUNE_WEAKNESS'
]
NUMERICAL_COLS = ['AGE', 'ENERGY_LEVEL', 'OXYGEN_SATURATION']
TARGET_COL = 'PULMONARY_DISEASE'

# Clustering parameters
CLUSTER_RANGE = range(2, 15)
MCA_DIMENSIONS = [1, 2, 3, 5, 8]  # More appropriate range

# DBSCAN parameters
DBSCAN_EPS_RANGE = np.linspace(0.3, 0.8, 6)
DBSCAN_MIN_SAMPLES_RANGE = range(3, 8)

# Output directory
OUTPUT_DIR = 'results'