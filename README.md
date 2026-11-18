# GeoCDL

GeoCDL is a multi-view framework for predicting missense mutation-drug effects
using geometric deep learning and contrastive learning. It jointly models
wild-type and mutant proteins together with multi-view drug representations.

## Overview

The model has three main components:

- **Protein encoder** (`protein_train_cl3.py`): a Siamese module that processes
  mutant and wild-type proteins. It projects GVP-based 3D binding-pocket
  features and ESM-2 sequence features into a shared space, learns the
  mutation-induced delta between wild-type and mutant states, and uses an
  InfoNCE contrastive objective to align structural and sequence
  representations.

- **Drug encoder** (`GNN.py`): encodes each drug SMILES from three views, a
  BRICS substructure sequence (Transformer), a molecular graph (two-layer GCN),
  and a Morgan fingerprint (MLP). A gated attention mechanism fuses the three
  views, and a contrastive objective aligns the heterogeneous molecular views.

- **Interaction model** (`interaction_model5.py`): predicts the mutation-drug
  response. It splits protein features into Context and Delta streams, applies
  multi-head cross-attention (Drug-to-Context, Drug-to-Delta, Prot-to-Drug),
  then uses bilinear pooling and a Wide&Deep predictor.

The three components are assembled in `JointInteractionModel.py`. The main
training script `train_joint3.py` runs five-fold cross-validation with a Focal
Loss plus the two contrastive objectives, using AdamW and mixed precision, and
reports AUC, AUPR, F1, accuracy, precision, and recall.

## Repository structure

```text
GeoCDL/
|-- GNN.py                     # Multi-view drug encoder and drug contrastive loss
|-- protein_train_cl3.py       # Siamese protein encoder and protein contrastive loss
|-- interaction_model5.py      # Cross-attention mutation-drug interaction model
|-- JointInteractionModel.py   # Joint drug-protein model
|-- joint_dataset.py           # Dataset, collate, and residue alignment
|-- train_joint3.py            # Main five-fold cross-validation training
|-- config.py                  # Central path configuration
|-- preprocessing/             # PDB pocket extraction and feature generation
|-- gvp/                       # GVP implementation
`-- data/                      # Training/test tables and drug SMILES
```

## Requirements

```text
torch
torch-geometric
torch-scatter
torch-cluster
rdkit
numpy
pandas
scikit-learn
scipy
matplotlib
biopython
biotite
transformers
tqdm
```

Install the dependencies with:

```bash
pip install torch torch-geometric torch-scatter torch-cluster rdkit numpy pandas scikit-learn scipy matplotlib biopython biotite transformers tqdm
```

## Data

`data/` contains:

- `5fold.csv`: the training table used for five-fold cross-validation.
- `ide_test.csv`: the independent test table containing unseen mutations and
  UniProt identifiers.
- `drug_SMILES.csv`: drug identifiers and SMILES strings.
- `train_test_split.py`: script that constructs the independent test split by
  selecting whole genes.

## Run the main experiment

Train with five-fold cross-validation:

```bash
python train_joint3.py
```

The script saves the best checkpoint of each fold to `checkpoints/` as
`best_model_fold1.pth` through `best_model_fold5.pth`.

## Path configuration

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `GEOCDL_DATA_ROOT` | `data/` | Root containing CSV tables |
| `GEOCDL_CHECKPOINT_DIR` | `checkpoints/` | Directory for trained checkpoints |

## License

See [LICENSE](LICENSE).
