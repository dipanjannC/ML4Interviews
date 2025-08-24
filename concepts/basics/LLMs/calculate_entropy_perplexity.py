"""
1. Entropy (H)

Entropy measures the uncertainty of the model’s predictions.

Lower entropy = model is more confident (skewed distribution).
Higher entropy = model is less confident (more uniform distribution).
"""


import numpy as np

# Model predicts probabilities for next word
probs = np.array([0.7, 0.2, 0.1])  # toy distribution for 3 tokens

# Entropy calculation
entropy = -np.sum(probs * np.log(probs))
print("Entropy:", entropy)
