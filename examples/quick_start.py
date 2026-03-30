"""CortexLab Quick Start Example.

Demonstrates loading a pretrained model, running inference,
and using the brain-alignment benchmark and cognitive load scorer.
"""

import numpy as np


def run_benchmark_demo():
    """Demonstrate the brain-alignment benchmark with synthetic data."""
    from cortexlab.analysis import BrainAlignmentBenchmark

    n_stimuli = 50
    model_features = np.random.randn(n_stimuli, 768)  # e.g. CLIP features
    brain_predictions = np.random.randn(n_stimuli, 20484)  # fsaverage5 vertices

    roi_indices = {
        "V1": np.arange(0, 100),
        "MT": np.arange(500, 600),
        "A1": np.arange(1000, 1100),
        "Broca": np.arange(2000, 2100),
    }

    bench = BrainAlignmentBenchmark(brain_predictions, roi_indices=roi_indices)

    for method in ["rsa", "cka", "procrustes"]:
        result = bench.score_model(model_features, method=method)
        print(f"[{method.upper()}] Aggregate: {result.aggregate_score:.4f}")
        for roi, score in sorted(result.roi_scores.items()):
            print(f"  {roi}: {score:.4f}")
    print()


def run_cognitive_load_demo():
    """Demonstrate the cognitive load scorer with synthetic predictions."""
    from cortexlab.analysis import CognitiveLoadScorer

    roi_indices = {
        "46": np.arange(0, 10),
        "FEF": np.arange(10, 20),
        "V1": np.arange(100, 120),
        "V2": np.arange(120, 140),
        "MT": np.arange(140, 160),
        "A1": np.arange(200, 220),
        "LBelt": np.arange(220, 230),
        "44": np.arange(300, 310),
        "45": np.arange(310, 320),
    }

    scorer = CognitiveLoadScorer(roi_indices, baseline_activation=0.5)

    # Simulate 30 seconds of predictions
    predictions = np.random.randn(30, 500) * 0.5
    # Add high visual activation
    predictions[:, 100:160] *= 3.0

    result = scorer.score_predictions(predictions, tr_seconds=1.0)
    print(f"Overall cognitive load:  {result.overall_load:.2f}")
    print(f"Visual complexity:       {result.visual_complexity:.2f}")
    print(f"Auditory demand:         {result.auditory_demand:.2f}")
    print(f"Language processing:     {result.language_processing:.2f}")
    print(f"Executive load:          {result.executive_load:.2f}")
    print(f"Timeline points:         {len(result.timeline)}")


if __name__ == "__main__":
    print("=== Brain-Alignment Benchmark ===")
    run_benchmark_demo()

    print("=== Cognitive Load Scorer ===")
    run_cognitive_load_demo()
