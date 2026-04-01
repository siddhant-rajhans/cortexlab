from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from cortexlab.analysis.temporal_dynamics import TemporalDynamicsAnalyzer
from cortexlab.viz.temporal_plots import (
    plot_peak_latencies,
    plot_response_curves,
    plot_lag_correlations,
)

# ── Synthetic data ──────────────────────────────────────────────
np.random.seed(42)
n_timepoints = 100
n_vertices = 500
tr_seconds = 1.0

predictions = np.random.randn(n_timepoints, n_vertices)
model_features = np.sin(np.linspace(0, 4 * np.pi, n_timepoints))

roi_indices = {
    "V1": np.arange(0, 100),
    "V2": np.arange(100, 200),
    "MT": np.arange(200, 300),
}

analyzer = TemporalDynamicsAnalyzer(roi_indices, tr_seconds=tr_seconds)
result = analyzer.analyze(predictions, model_features)

# ── Create Visualizations ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Peak Latency Bar Chart
plot_peak_latencies(axes[0], result)

# Plot 2: Response Curves (Sustained vs Transient)
plot_response_curves(axes[1], result, "V1", tr_seconds=tr_seconds)

# Plot 3: Lag-Correlation Plot
plot_lag_correlations(axes[2], result, tr_seconds=tr_seconds)

plt.tight_layout()
plt.savefig("temporal_dynamics_visualization.png", dpi=150, bbox_inches="tight")
plt.show()
print(" Visualization saved!")