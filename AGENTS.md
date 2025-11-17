# Repository Guidelines

## Project Structure & Module Organization
`inference/` contains the runnable experiment (`inference/inference.py`), model wrappers (`inference/llm`), and the text-based environment (`inference/star_making`). Hydra configs sit in `config/inference.yaml`; prefer CLI overrides rather than editing the file so runs stay reproducible. Logs and JSON artifacts land under `results/` and the timestamped folders in `outputs/`. Batch-launch assets live in `slurm/`, and `init_env.sh` documents the expected Anaconda module plus the `hanoi` conda env.

## Build, Test, and Development Commands
- `bash init_env.sh` – load the cluster modules and activate the shared conda env.
- `python inference/inference.py experiment.n_trials=10 output.dir=results/debug` – run the local Llama client with custom trial counts and output folders.
- `python inference/inference.py model.provider=gemini model.model_name=gemini-1.5-flash-latest` – target the Gemini API (requires `GEMINI_API_KEY`).
- `python inference/star_making/test.py` – replay the success/failure walkthrough to confirm simulator logic.
- `python inference/star_making/test_parsing_failure.py` – validate the action-parsing retry behavior.

## Coding Style & Naming Conventions
Stick to Python 3.11, four-space indentation, and PEP 8 snake_case names. Classes remain PascalCase (`LocalLLMClient`, `StarMakingEnvText`), while Hydra/YAML keys stay lowercase with underscores. Add type hints for public APIs and keep prompts, rule tables, and config defaults in helper modules or YAML rather than inline literals. Use `logging` for runtime reporting; reserve `print` for the standalone harnesses and scripted demos.

## Testing Guidelines
The repo favors human-readable scenario scripts over pytest. Update `test.py` whenever you add new rule types or interaction flows so both success and failure narratives exist. Parser or control-loop changes should extend `test_parsing_failure.py` to prove retries, default actions, and conversation-history inserts still behave. Run both scripts locally before sharing results; they finish quickly and their ASCII summaries double as evidence for reviewers.

## Commit & Pull Request Guidelines
Working copies here rarely retain Git metadata, so follow an imperative, ≤72-character subject line that calls out the scope (`feat: tune transfer rules`). Provide a short body when context is non-obvious. PRs or review requests should list: intent, Hydra overrides or config additions, tests/commands executed, and links to `results/` or `outputs/` artifacts. Include screenshots or log snippets only when they clarify success criteria.

## Environment & Configuration Tips
Source `init_env.sh` before running anything; it loads the proper modules and conda env. Never hard-code secrets—inject `GEMINI_API_KEY` via scheduler or shell exports. When launching on Slurm, copy `slurm/inference/action_seq_local.slurm`, adjust `#SBATCH` resources plus Hydra overrides, and point `output.dir` to a unique folder.
