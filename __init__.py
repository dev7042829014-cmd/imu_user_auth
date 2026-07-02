"""
ContinAuth recreation package
=============================
A self-contained recreation of dynobo/ContinAuth (Buech 2019) — a Siamese FCN
trained with contrastive loss whose deep features feed a per-user one-class SVM —
applied to the Apple-Watch IMU dataset in three data regimes (A / B / C).

See continauth/README.md for the full description and run commands.
"""

from .config import CONFIGS, VERSION_A, VERSION_B, VERSION_C, ExperimentConfig  # noqa: F401

__all__ = ["CONFIGS", "VERSION_A", "VERSION_B", "VERSION_C", "ExperimentConfig"]
