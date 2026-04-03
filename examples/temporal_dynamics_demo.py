from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cortexlab.analysis.temporal_dynamics import TemporalDynamicsAnalyzer
from cortexlab.viz.temporal_plots import (
    plot_lag_correlations,
    plot_peak_latencies,
    plot_response_curves,
)


def main() -> None:
    np.random.seed(42)
    n_timepoints = 100
    n_vertices = 500
    tr_seconds = 1.0

    # Generate stimulus-evoked signal with hemodynamic response
    stimulus_onsets = [10, 30, 55, 75]
    signal = np.zeros(n_timepoints)
    for onset in stimulus_onsets:
        if onset < n_timepoints:
            signal[onset] = 1.0

    # Simple HRF approximation
    hrf_t = np.arange(0, 20, tr_seconds)
    hrf = hrf_t * np.exp(-hrf_t / 1.5)
    hrf = hrf / hrf.max()
    bold = np.convolve(signal, hrf)[:n_timepoints]

    roi_indices = {
        "V1": np.arange(0, 100),
        "V2": np.arange(100, 200),
        "MT": np.arange(200, 300),
    }

    predictions = np.random.randn(n_timepoints, n_vertices) * 0.3
    for v in roi_indices["V1"]:
        predictions[:, v] += bold * 1.0
    for v in roi_indices["MT"]:
        predictions[:, v] += np.roll(bold, 2) * 0.7

    model_features = np.sin(np.linspace(0, 4 * np.pi, n_timepoints))

    analyzer = TemporalDynamicsAnalyzer(roi_indices, tr_seconds=tr_seconds)
    result = analyzer.analyze(predictions, model_features)

    _, axes = plt.subplots(1, 3, figsize=(15, 5))

    plot_peak_latencies(axes[0], result)
    plot_response_curves(axes[1], result, "V1", tr_seconds=tr_seconds)
    plot_lag_correlations(axes[2], result, tr_seconds=tr_seconds)

    plt.tight_layout()
    output_path = Path(__file__).with_name("temporal_dynamics_visualization.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Visualization saved to {output_path.resolve()}")


if __name__ == "__main__":
    main()
