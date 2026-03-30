"""CortexLab - Enhanced multimodal fMRI brain encoding toolkit.

Built on Meta's TRIBE v2 foundation model, CortexLab adds streaming
inference, modality attribution, cross-subject adaptation,
brain-alignment benchmarking, and cognitive load scoring.
"""

__version__ = "0.1.0"

from cortexlab.core.model import FmriEncoder, FmriEncoderModel
from cortexlab.core.attention import AttentionExtractor, attention_to_roi_scores
from cortexlab.core.subject import SubjectAdapter

__all__ = [
    "FmriEncoder",
    "FmriEncoderModel",
    "AttentionExtractor",
    "attention_to_roi_scores",
    "SubjectAdapter",
]
