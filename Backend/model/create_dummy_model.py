#!/usr/bin/env python
"""
Create dummy model for testing
"""

import pickle
import numpy as np

class DummyPredictor:
    def predict(self, X):
        # Always predict 0 (no mental health issue)
        return np.array([0])
    
    def predict_proba(self, X):
        # Return probabilities with 30% chance of mental health issue
        return np.array([[0.7, 0.3]])

# Save the model to disk
with open('mental_health_model.xgb', 'wb') as f:
    pickle.dump(DummyPredictor(), f)

print("Dummy model created successfully.") 