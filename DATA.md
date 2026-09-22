# Dataset archive manifest

The experiments use **SC2EGSet: StarCraft II Esport Replay and Game-state Dataset**. Its Scientific Data article was published in 2023, while the current [Zenodo release](https://doi.org/10.5281/zenodo.17829625) includes later archive additions. The benchmark selects 19 archives labelled 2018–2024 from that release; the release also contains archives not used here. See the [dataset article](https://doi.org/10.1038/s41597-023-02510-7) for the dataset description.

The Zenodo record identifies the dataset license as [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). That license covers SC2EGSet; it does not determine the license for this repository's source code.

The repository does not redistribute SC2EGSet. Download the following 19 outer archives from Zenodo and extract each one into a same-named directory at the repository root. Keep the nested `*_data.zip` file inside each directory; `code/prepare.py` reads that ZIP directly.

| Directory / Zenodo archive stem |
| --- |
| `2018_IEM_Katowice` |
| `2019_IEM_Katowice` |
| `2020_Dreamhack_SC2_Masters_Fall` |
| `2020_Dreamhack_SC2_Masters_Summer` |
| `2020_Dreamhack_SC2_Masters_Winter` |
| `2020_IEM_Katowice` |
| `2021_Dreamhack_SC2_Masters_Fall` |
| `2021_Dreamhack_SC2_Masters_Summer` |
| `2021_Dreamhack_SC2_Masters_Winter` |
| `2021_IEM_Katowice` |
| `2022_03_DH_SC2_Masters_Atlanta` |
| `2022_Dreamhack_SC2_Masters_Last_Chance2021` |
| `2022_Dreamhack_SC2_Masters_Valencia` |
| `2022_IEM_Katowice` |
| `2023_01_IEM_Katowice` |
| `2023_04_ESL_SC2_Masters_Summer_Finals` |
| `2023_07_ESL_SC2_Masters_Winter_Finals` |
| `2024_01_IEM_Katowice` |
| `2024_03_ESL_SC2_Masters_Spring_Finals` |

These labels are copied verbatim from `artifacts/preprocessing.json` and match filenames in the linked Zenodo release. Raw archives, processed events, and training outputs are intentionally excluded from Git because of their size.

## Build a small smoke dataset

This reads five replay records from each selected archive and is intended only to verify that the pipeline works:

```bash
python code/prepare.py \
  --root . \
  --outdir artifacts_smoke \
  --hierarchy-taxonomy legacy8 \
  --split-mode replay \
  --smoke
```

## Build the full paper dataset

```bash
python code/prepare.py \
  --root . \
  --outdir artifacts \
  --hierarchy-taxonomy legacy8 \
  --split-mode replay
```

Both commands write preprocessing metadata, an action vocabulary, split audits, and `processed_events.parquet` to the selected artifact directory.
