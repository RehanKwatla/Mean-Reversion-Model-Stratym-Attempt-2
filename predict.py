"""
Standalone Prediction Script for SP-02 Mean-Reversion Policy
Usage: python predict.py
"""
import torch
import numpy as np
from pathlib import Path
from sp02_model import MeanReversionPolicy

def run_inference():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = MeanReversionPolicy.load(".", device=device)
    print("Loaded SP-02 policy successfully.")
    
    # Example input: 30 bars of 40 features
    dummy_seq = np.random.randn(30, 40).astype(np.float32)
    state = {
        "sequence": dummy_seq,
        "position": 0, # Flat
        "profile": {"risk_tolerance": 0.5, "patience": 0.5, "aggressiveness": 0.5}
    }
    out = policy.predict(state)
    print("Inference Result (Flat State):")
    print(f"  Action: {out['action_name']} (id {out['action']}) | Confidence: {out['confidence']:.3f}")
    print(f"  Probabilities: {out['action_probs']}")
    
    # Held Long Position
    held_state = {
        "sequence": dummy_seq,
        "position": 1, # Long
        "profile": (0.5, 0.5, 0.5),
        "unrealized_pnl": -0.02,
        "holding_bars": 5
    }
    held_out = policy.predict(held_state)
    print("\nInference Result (Held Long):")
    print(f"  Action: {held_out['action_name']} (id {held_out['action']}) | Confidence: {held_out['confidence']:.3f}")

if __name__ == "__main__":
    run_inference()
