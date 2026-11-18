# Data

This directory contains the small CSV tables required to run the model:

- `train.csv`: training records with columns `Mutation_ID`, `Relation`, `Drug`,
  and `SMILES`.
- `test.csv`: independent test records with the same schema.
- `drug_SMILES.csv`: drug identifiers and SMILES strings.
- `relation_with_smiles.csv`: the relation records after joining drug SMILES.
- `train_test_split.py`: script used to construct the independent test split by
  selecting whole genes whose unique mutations are absent from training.

The processed protein features are intentionally not committed to GitHub. To
use the repository, either download/place the feature directories under this
directory, or set `GEOCDL_FEATURE_ROOT` (or `GEOCDL_DATA_ROOT`) to the root of
an existing processed dataset.

Expected feature layout:

```text
mut/old/pocket_embeddings1/*_pocket.pt
mut/old/esm2_features/*.pt
wild/pocket_embeddings1/*_WT_pocket.pt
wild/esm2_features/*_WT.pt
```

See the repository-level README for the preprocessing commands that generate
these files from PDB structures and pretrained ESM-2/GVP checkpoints.
