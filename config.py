"""Central configuration for GeoCDL.

All paths can be overridden through environment variables so that the
repository can be run on a machine other than the original development one.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# Directory containing the CSV tables and, by default, the processed feature
# directories below.
DATA_ROOT = Path(os.environ.get("GEOCDL_DATA_ROOT", REPO_ROOT / "data")).resolve()

# Directory that contains the processed protein feature folders. It defaults to
# DATA_ROOT so the repository can point directly at the original dataset root.
FEATURE_ROOT = Path(os.environ.get("GEOCDL_FEATURE_ROOT", DATA_ROOT)).resolve()

TRAIN_CSV = DATA_ROOT / "5fold.csv"
TEST_CSV = DATA_ROOT / "ide_test.csv"
DRUG_CSV = DATA_ROOT / "drug_SMILES.csv"

PROT_DIRS = {
    "mut_gvp": FEATURE_ROOT / "mut" / "old" / "pocket_embeddings1",
    "mut_esm": FEATURE_ROOT / "mut" / "old" / "esm2_features",
    "wild_gvp": FEATURE_ROOT / "wild" / "pocket_embeddings1",
    "wild_esm": FEATURE_ROOT / "wild" / "esm2_features",
}

CHECKPOINT_DIR = Path(
    os.environ.get("GEOCDL_CHECKPOINT_DIR", REPO_ROOT / "checkpoints")
).resolve()

# Optional local path to an ESM-2 checkpoint used by feature extraction.
ESM2_PATH = os.environ.get("GEOCDL_ESM2_PATH", "")

# Optional local path to the pretrained GVP encoder checkpoint.
GVP_WEIGHTS = os.environ.get("GEOCDL_GVP_WEIGHTS", str(FEATURE_ROOT / "gvp_pretrained_encoder.pt"))
