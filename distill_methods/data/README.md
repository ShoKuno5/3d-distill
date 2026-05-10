# Reference UID lists for Toys4k eval-only enforcement

Two equivalent lists describe every Toys4k asset, intended for the
training-time guard in `distill_methods/src/train.py`
(`assert_no_toys4k_in_train`). Source for both is the TRELLIS-500K
Toys4k metadata at
`/inspire/hdd/global_public/public_datas/TRELLIS-500k/Toys4k/metadata.csv`.

## toys4k_uids.txt (3229 lines)

SHA-256 hashes from the `sha256` column. Used when training data
manifests use TRELLIS-500K-style sha256 IDs.

```bash
awk -F, "NR>1 {print \$1}" \
  /inspire/hdd/global_public/public_datas/TRELLIS-500k/Toys4k/metadata.csv \
  > distill_methods/data/toys4k_uids.txt
```

## toys4k_names.txt (3229 lines)

Legacy `category_NNN` names parsed from the `file_identifier`
column. Used when training data manifests use Toys4k-native names
(e.g. as in the historical exp/unified-eval-matrix runs).

```bash
awk -F, "NR>1 {n = split(\$2, p, \"/\"); if (n>=2) print p[2]}" \
  /inspire/hdd/global_public/public_datas/TRELLIS-500k/Toys4k/metadata.csv \
  | sort -u > distill_methods/data/toys4k_names.txt
```

The training entry-point checks every training UID against **both**
files. Any overlap (sha256 or name form) is a hard failure.
