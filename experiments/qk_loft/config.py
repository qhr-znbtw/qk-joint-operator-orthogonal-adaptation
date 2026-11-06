"""Controlled configuration for the 3x3 QK-LOFT factorial study."""

QK_LOFT_RANK = 46
SUPPORT_TYPES = ("principal", "grad_svd", "skew_grad")
SIDES = ("left", "right", "bilateral")
NUM_EPOCHS = 100
WARMUP_EPOCHS = 10
ADAPTER_LEARNING_RATE = 5e-4
CLASSIFIER_LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
CALIBRATION_BATCHES = 1
