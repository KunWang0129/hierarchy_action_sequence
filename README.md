# Hierarchical Action-Sequence (Star-Making) Experiments

LLM agents here practice a hierarchical action-sequence task: four keypresses form items, item pairs form stars, and agents must learn the mappings in a learning phase before transferring them to new goal stars. The repo hosts simulators, experiment drivers (local vLLM and API-based), and analysis notebooks/scripts for measuring learning, transfer, and sequence reuse.

## Repository Map
- `inference/`: Star-making simulators (`star_making`, `star_making_no_item`) plus batch/single-text environments. Experiment runners cover the local vLLM baseline (`experiment.py`), Gemini API (`experiment_gemini.py`), generalization to novel stars (`experiment_generalize.py`), and the item-feedback ablation (`experiment_no_item.py`). `attention_prob.py` replays trials to probe token-level attention.
- `config/`: Hydra configs for experiments, inference batches, and attention replay; prefer CLI overrides to keep runs reproducible.
- `analysis/`: Post-hoc analysis. `analyze_result.py` parses legacy conversation-history logs. `original_experiments/` covers the base learning/transfer mixed-models and plots. `new_experiments/` adds item-feedback effects, generalize-vs-original transfer, sequence reuse, and success/failure splits. `attention_experiments/star_attention.py` aggregates attention on star tokens. `experiment_summary.md` summarizes recent findings.
- `results/` and `outputs/`: JSON logs, CSVs, and figures from runs; Hydra also mirrors configs under timestamped folders.
- `init_env.sh`, `setup_env.sh`, `requirements.txt`: Environment setup notes for the shared conda env.

## Running Experiments
1) Load the env: `bash init_env.sh`.  
2) Local vLLM baseline: `python inference/experiment.py experiment.n_trials=60 experiment.transfer_rule_type=transfer_high output.dir=results/debug`.  
   - Generalization: `python inference/experiment_generalize.py experiment.transfer_rule_type=transfer_low ...`  
   - No-item ablation: `python inference/experiment_no_item.py ...`  
   - Gemini API: export `GOOGLE_API_KEY` then `python inference/experiment_gemini.py experiment.n_trials=60 ...`  
3) Attention replay (prototype): `python inference/attention_prob.py input.path=<path-to-experiment.json> output.dir=results/attention_debug`.

## Analyzing Results
- Legacy parser/plots: `python analysis/analyze_result.py path/to/*_experiment.json`.
- Baseline learning/transfer stats: `python analysis/original_experiments/learning_phase.py <json> -o plot.png --trials 20` and `python analysis/original_experiments/transfer_phase.py <json> ...`.
- New experiments: scripts in `analysis/new_experiments/` accept multiple JSON paths (original vs generalize, item vs no-item, etc.; see docstrings for arguments) and emit mixed-model summaries plus plots under `results/<model>/experiments/...`.
- Attention aggregation: `python analysis/attention_experiments/star_attention.py <attention_analysis.json> --output-dir <dir>`.

Keep configs in CLI overrides instead of editing YAML directly, and stash outputs under unique `output.dir` folders per run.
