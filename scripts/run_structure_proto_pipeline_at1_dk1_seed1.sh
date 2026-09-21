#!/usr/bin/env bash
set -euo pipefail

bash scripts/train_structure_proto_source_at1_seed1.sh
bash scripts/run_structure_proto_da_at1_dk1_seed1.sh
