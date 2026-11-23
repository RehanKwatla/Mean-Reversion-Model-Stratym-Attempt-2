# SP-02 Mean-Reversion Policy (Inference & Deployment)

This repository contains the standalone weights, causal feature pipeline, preprocessors, configuration, and inference scripts to run the trained **SP-02 Mean-Reversion Policy** on any machine.

---

## 1. What's Included

- `mean_reversion_policy.pt`: Trained PyTorch weights (~214 KB)
- `mean_reversion_config.json`: Model hyperparameters, sequence length ($L=30$), and architecture configuration
- `feature_schema.json`: Complete specification of the 40 sequential market features and 8 static features
- `preprocessing.pkl`: Fitted feature scalers (medians, means, standard deviations, and clipping thresholds)
- `sp02_features.py`: Causal feature calculation library (strictly past-only indicators)
- `sp02_model.py`: PyTorch model definition (`MeanReversionPolicyNet`) and inference wrapper (`MeanReversionPolicy`)
- `predict.py`: Self-contained verification and prediction demo script
- `requirements.txt`: Minimal runtime dependencies for CPU/GPU execution

---

## 2. Quickstart

### Prerequisites
Python 3.9+ with PyTorch, NumPy, and Pandas.

```bash
pip install -r requirements.txt
```

### Run Verification / Test Inference
```bash
python predict.py
```

### Python API Usage
```python
import numpy as np
from sp02_model import MeanReversionPolicy

# 1. Load policy (automatically detects CPU / CUDA)
policy = MeanReversionPolicy.load(".", device="cpu")

# 2. Feed sequence of 30 bars (with 40 features each)
# Features can be raw - the wrapper automatically scales them with fitted parameters
dummy_seq = np.random.randn(30, 40).astype(np.float32)

state = {
    "sequence": dummy_seq,
    "position": 0,  # 0: flat, 1: long, -1: short
    "profile": {"risk_tolerance": 0.5, "patience": 0.5, "aggressiveness": 0.5},
    "unrealized_pnl": 0.0,
    "holding_bars": 0
}

# 3. Predict action (0: HOLD, 1: BUY, 2: SELL, 3: EXIT)
output = policy.predict(state)
print(f"Action: {output['action_name']} (id {output['action']})")
print(f"Probabilities: {output['action_probs']}")
```

---

## 3. Action Space
| ID | Action | Description | Valid When |
|---|---|---|---|
| `0` | **HOLD** | Maintain current exposure or remain flat | Always |
| `1` | **BUY** | Open long position | Flat (`position=0`) |
| `2` | **SELL** | Open short position (if shorting enabled) | Flat (`position=0`) |
| `3` | **EXIT** | Liquidate position to flat | Long or Short (`position != 0`) |
