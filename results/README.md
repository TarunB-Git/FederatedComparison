# Result provenance

`paper_results.csv` is a machine-readable transcription of the final tables in the IEEE GEM 2026 manuscript:

- 9 centralized runs: 3 encoders × 3 races
- 9 FedAvg runs: 3 encoders × 3 races
- 9 FedProx runs: 3 encoders × 3 races
- 3 personalized backbone–head runs trained jointly across all races

The values were copied from the current accepted manuscript's centralized, federated, and backbone–head test-set tables. They are checked against the stored `final_test.json` artifacts where those artifacts are available locally.

The backbone–head rows have `race=All` and `scope=combined-all-races`. They must not be interpreted as per-race measurements or compared directly with a single-race row without this qualification.

Each configuration was trained once with a fixed seed. These results are point estimates and do not support claims of statistical significance.

## Experiment protocol

`paper_protocol.json` is a normalized record of the saved `config.json` files for the reported runs. The full pipeline profile now resolves to these settings.

| Setting | Centralized | FedAvg / FedProx | Backbone–Head |
| --- | ---: | ---: | ---: |
| Batch size | 512 | 512 | 512 |
| Epochs / rounds | 20 epochs | 50 rounds | 150 rounds |
| Clients per round | — | 25 | 25 |
| Local epochs | — | 1 | 1 |
| Per-client sample cap | — | 2,000 | None (`0`) |
| Local batch ceiling | — | 75 | 75 |
| Weight decay | `1e-5` | `1e-5` | `0.0` |
| Validation clients per round | — | — | 50 |

The backbone–head runs are an ablation with a different training budget and local-training configuration. They must not be presented as a controlled personalization gain over the race-specific FedAvg or FedProx runs. The manuscript source should state the backbone exceptions: no per-client sample cap and zero local weight decay.

The centralized saved configurations use early-stopping patience 100. With a 20-epoch maximum, this records validation-based checkpoint selection without normally terminating training early.

## Units and timing provenance

`communication_mib` is total upload plus download divided by `1024²`. It is mebibytes (MiB), not decimal megabytes.

The reported backbone Transformer runtime of approximately 167,116 seconds comes from the preserved cumulative round history plus the final resumed evaluation: `sum(round_metrics.round_time_sec) = 166,639.53` seconds and the last invocation's `final_test.total_wall_clock_sec = 476.77` seconds. The latter file alone records only the resumed/final invocation because resume mode reset its in-process timers; it is not the cumulative training runtime.

## Dataset count clarification

The processed artifact contains 1,314 unique player identifiers across the union of all splits. The training split contains 1,305 unique player identifiers, which form the candidate federated client pool. The paper's 1,305 figure therefore describes the training-client population, not the all-split union.

Regenerate the repository figures with:

```bash
python scripts/generate_readme_figures.py
```
