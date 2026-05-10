# Reference UID lists

## toys4k_uids.txt

3229 SHA-256 hashes — one per line — covering every asset in the Toys4k
distribution. Sourced from the TRELLIS-500K Toys4k metadata at
`/inspire/hdd/global_public/public_datas/TRELLIS-500k/Toys4k/metadata.csv`
(column `sha256`).

**Use:** training scripts must assert that every training-set UID is
disjoint from this list before starting. Toys4k is reserved for
evaluation only and must never appear in the training distribution.

The list is regenerable from the upstream metadata.csv:
```bash
awk -F, "NR>1 {print \$1}" \
  /inspire/hdd/global_public/public_datas/TRELLIS-500k/Toys4k/metadata.csv \
  > distill_methods/data/toys4k_uids.txt
```
