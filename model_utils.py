import numpy as np

def generate_sample_features(n_features=20):
    """Generate sample features for testing"""
    return np.random.randn(n_features).tolist()

def validate_features(features, expected_length=20):
    """Validate input features"""
    if not isinstance(features, list):
        raise ValueError("Features must be a list")
    
    if len(features) != expected_length:
        raise ValueError(f"Expected {expected_length} features, got {len(features)}")
    
    return True
