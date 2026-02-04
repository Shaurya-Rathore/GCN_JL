### GCN + DropEdge++ + JL (Cora/Citeseer/PubMed)

**Entrypoint:** `src/train_new.py`  
This repo compares a **baseline** run (projection disabled) against **DropEdge++ + JL** (JL projection enabled). There is **no separate test script**: `src/train_new.py` trains using the dataset’s train/val/test masks and prints the final test metric during the run.

**Baseline (projection OFF):**
- `python src/train_new.py --dataset cora --proj none --seed 42`
- `python src/train_new.py --dataset citeseer --proj none --seed 42`
- `python src/train_new.py --dataset pubmed --proj none --seed 42`

**DropEdge++ + JL (projection ON):**
- `python src/train_new.py --dataset cora --proj jl --proj-dim 512 --jl-seed 0 --jl-orth --seed 42`
- `python src/train_new.py --dataset citeseer --proj jl --proj-dim 512 --jl-seed 0 --jl-orth --seed 42`
- `python src/train_new.py --dataset pubmed --proj jl --proj-dim 512 --jl-seed 0 --jl-orth --seed 42`

**Optional sweeps:**
- Seeds: run the above commands for `--seed 0 1 2 3 4` and report mean ± std.  
- JL dimension: vary `--proj-dim` (e.g., `64 128 256 512 768 1024`) to study accuracy/efficiency trade-offs.
