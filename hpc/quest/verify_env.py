"""Import check run by bootstrap_env.sh after uv sync.

Confirms torch/CUDA are visible and that the locally added hybrid MC-MD
modules (nvalchemi.hybrid, nvalchemi.mc, nvalchemi.scheduling) import
cleanly in the environment the campaign will actually run in.
"""

import torch

import nvalchemi
import nvalchemi.hybrid
import nvalchemi.mc
import nvalchemi.scheduling

print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
print(f"nvalchemi {nvalchemi.__version__}")
print("nvalchemi.hybrid / nvalchemi.mc / nvalchemi.scheduling import OK")
