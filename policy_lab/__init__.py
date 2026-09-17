"""Versioned enterprise policy workbench."""
import os

# This workbench uses PyTorch. Avoid loading an unrelated system TensorFlow/Keras
# installation through transformers' optional integrations.
os.environ.setdefault("USE_TF", "0")
