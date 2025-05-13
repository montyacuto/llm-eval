# evaluate_outputs.py

import argparse
# import os # Already imported
import json
import logging
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import sys
import statistics
import math # For checking isnan

# Configure logging (Keep existing)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Keep DEFAULT_EVAL_CONFIG ---
DEFAULT_EVAL_CONFIG = {
    "output_dir": "Output",
    "models_file": "Input/models.txt",
    "prompt_corpora_file": "Input/corpora.txt", # Lists basenames
    "prompts_dir": "Input/Prompts",
    "truth_dir": "Input/Truth",
    "delimiter": "---"
    # Add visualize default here IF needed, but getting from config is better
    # "visualize": False
}

# --- Keep NLTK checks ---
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        logger.info("NLTK 'punkt' not found. Downloading...")
        nltk.download('punkt', quiet=True)
    from nltk import word_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    sentence_bleu, SmoothingFunction, word_tokenize = None, None, None
    logger.warning("nltk not found. BLEU scores will not be calculated.")

# --- Keep ROUGE checks ---
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    rouge_scorer = None
    logger.warning("rouge-score not found. ROUGE scores will not be calculated.")

# --- Keep Pandas checks ---
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None
    logger.warning("pandas not found. Aggregated performance metrics DataFrame cannot be easily saved to CSV.")

# --- Add Matplotlib/Seaborn checks ---
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np # Often used with plotting
    PLOTTING_AVAILABLE = True
    # Import the visualization class
    from model_visualizations import ModelVisualizations
except ImportError:
    PLOTTING_AVAILABLE = False
    plt, sns, np = None, None, None
    ModelVisualizations = None # Set class to None if unavailable
    logger.warning("matplotlib, seaborn or numpy not found. Visualizations will not be generated.")


# --- Keep load_config_from_file, get_list_from_file, read_prompts_and_truths ---
def load_config_from_file(config_path_str: str) -> Dict:
    config_path = Path(config_path_str)
    if not config_path.exists():
        logger.error(f"Config file not found: '{config_path}'")
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_from_file = json.load(f)
        logger.info(f"Loaded configuration for evaluation from {config_path}")
        return config_from_file
    except Exception as e:
        logger.error(f"Error loading config file {config_path}: {e}")
        return {}

def get_list_from_file(file_path_str: str) -> List[str]:
    """Reads a list of strings (e.g., model names, corpus basenames) from a file."""
    file_path = Path(file_path_str)
    logger.info(f"Reading list from {file_path}")
    if not file_path.exists():
        logger.error(f"List file not found: '{file_path}'")
        return []
    try:
        # Handles both directory path (e.g., Models/Gemma/...) and direct GGUF path
        return [line.strip() for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith('#')]
    except Exception as e:
        logger.error(f"Error reading list from {file_path}: {e}")
        return []

def read_prompts_and_truths(
    prompt_corpus_full_path: Path, # Full path to the prompt file
    truth_base_dir: Path,        # Base directory for truth files (e.g., Input/Truth)
    delimiter: str = "---"
) -> Tuple[List[str], Optional[List[str]], Optional[str]]:
    """
    Reads prompts from prompt_corpus_full_path.
    Constructs truth file path as <truth_base_dir>/<prompt_corpus_stem>_truth.txt.
    """
    if not prompt_corpus_full_path.exists():
        logger.warning(f"Prompts corpus file not found: {prompt_corpus_full_path}")
        return [], None, None

    prompts: List[str] = []
    try:
        content = prompt_corpus_full_path.read_text(encoding="utf-8")
        if delimiter and delimiter in content:
            prompts = [p.strip() for p in content.split(delimiter) if p.strip()]
        else:
            if delimiter: logger.warning(f"Delimiter '{delimiter}' not found in {prompt_corpus_full_path.name}. Falling back to newline splitting.")
            prompts = [p.strip() for p in content.splitlines() if p.strip()]
        logger.info(f"Read {len(prompts)} prompts from {prompt_corpus_full_path.name}")
    except Exception as e:
        logger.error(f"Error reading prompts from {prompt_corpus_full_path}: {e}")
        return [], None, None

    corpus_stem = prompt_corpus_full_path.stem # e.g., "example_corpus_A"
    truth_file_name = f"{corpus_stem}_truth.txt"
    truth_file_path = truth_base_dir / truth_file_name # e.g., /path/to/Input/Truth/example_corpus_A_truth.txt
    truths: Optional[List[str]] = None

    logger.debug(f"Attempting to read truth file from: {truth_file_path}")

    if truth_file_path.exists():
        logger.info(f"Found ground truth file: {truth_file_path}")
        try:
            truth_content = truth_file_path.read_text(encoding="utf-8")
            if delimiter and delimiter in truth_content:
                truths = [t.strip() for t in truth_content.split(delimiter) if t.strip()]
            else:
                if delimiter: logger.warning(f"Delimiter '{delimiter}' not found in {truth_file_name}. Falling back to newline splitting.")
                truths = [t.strip() for t in truth_content.splitlines() if t.strip()]

            if len(prompts) != len(truths):
                logger.warning(f"Mismatch: {len(prompts)} prompts in {prompt_corpus_full_path.name} vs {len(truths)} truths in {truth_file_name}. Skipping scoring.")
                truths = None
            else:
                logger.info(f"Read {len(truths)} ground truths from {truth_file_name}")
        except Exception as e:
            logger.error(f"Error reading truths from {truth_file_path}: {e}")
            truths = None
    else:
        logger.info(f"No ground truth file found at {truth_file_path} for {prompt_corpus_full_path.name}. Skipping scoring.")

    return prompts, truths, str(truth_file_path) if truths is not None else None

# --- Keep calculate_bleu, calculate_rouge ---
def calculate_bleu(reference: str, candidate: str) -> Optional[float]:
    if not NLTK_AVAILABLE or not reference or not candidate: return None
    try:
        ref_tokens = [word_tokenize(reference.lower())] if reference else []
        cand_tokens = word_tokenize(candidate.lower()) if candidate else []
        if not ref_tokens or not cand_tokens: return 0.0
        try:
            score = sentence_bleu(ref_tokens, cand_tokens, smoothing_function=SmoothingFunction().method7)
            return score if not math.isnan(score) else 0.0
        except ZeroDivisionError:
            logger.warning("ZeroDivisionError during BLEU calculation, likely due to empty candidate after tokenization. Returning 0.0.")
            return 0.0
    except Exception as e:
        logger.error(f"BLEU calculation error: {e}")
        return None

def calculate_rouge(reference: str, candidate: str) -> Optional[Dict[str, float]]: # Return only f-measures
    if not ROUGE_AVAILABLE or not reference or not candidate: return None
    try:
        if not candidate.strip():
             logger.warning("Empty candidate string provided to ROUGE. Returning None.")
             return None
        scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer_instance.score(reference, candidate)
        return {
            "rouge1": scores['rouge1'].fmeasure,
            "rouge2": scores['rouge2'].fmeasure,
            "rougeL": scores['rougeL'].fmeasure
        }
    except Exception as e:
        logger.error(f"ROUGE calculation error: {e}")
        return None

# --- Keep process_model_outputs ---
def process_model_outputs(
    output_dir_base: Path,      # Base Output directory (e.g., Output/)
    model_name_entry: str,      # Entry from models.txt (e.g., Models/model.gguf or model.gguf)
    corpus_basenames: List[str],# List of corpus basenames (e.g., ["corpus_A.txt"])
    prompts_dir: Path,          # Directory of prompt files (e.g., Input/Prompts/)
    truth_base_dir: Path,       # Base directory for truth files (e.g., Input/Truth/)
    delimiter: str
) -> List[Dict[str, Any]]:
    """
    Processes output JSONs for a given model across specified corpora.
    Adds model name, calculates scores, and returns a list of dictionaries
    containing all relevant data per prompt+response for this model.
    """
    model_path_obj = Path(model_name_entry)
    model_name_stem = model_path_obj.stem

    model_specific_output_dir = output_dir_base / model_name_stem
    logger.info(f"Processing outputs for model: {model_name_stem} in {model_specific_output_dir}")

    if not model_specific_output_dir.is_dir():
        logger.warning(f"Output dir for model '{model_name_stem}' not found: {model_specific_output_dir}. Skipping.")
        return []

    model_results = [] # Will store results for *this* model across all its corpora

    for corpus_basename in corpus_basenames: # e.g., "corpus_A.txt"
        corpus_stem = Path(corpus_basename).stem # e.g., "corpus_A"
        full_prompt_corpus_path = prompts_dir / corpus_basename

        logger.info(f"  Processing corpus: {corpus_basename}")

        output_json_path = model_specific_output_dir / f"{corpus_stem}_out.json"

        prompts, truths, actual_truth_path = read_prompts_and_truths(full_prompt_corpus_path, truth_base_dir, delimiter)

        if not output_json_path.exists():
            logger.warning(f"  Output JSON for {corpus_basename} not found at {output_json_path}. Skipping.")
            continue

        try:
            corpus_outputs_json = json.loads(output_json_path.read_text(encoding="utf-8"))
            logger.info(f"  Read {len(corpus_outputs_json)} outputs from {output_json_path}")

            if len(corpus_outputs_json) != len(prompts):
                 logger.warning(f"  Prompt/output count mismatch for {corpus_basename}: {len(prompts)} prompts vs {len(corpus_outputs_json)} outputs.")

            for i, output_data in enumerate(corpus_outputs_json):
                output_data['model_name'] = model_name_stem
                output_data['response_length_tokens'] = output_data.get('tokens')
                output_data['bleu_score'] = None
                output_data['rouge_scores'] = None # Stores dict of fmeasures

                if truths and i < len(truths):
                    reference = truths[i]
                    candidate = output_data.get("response", "")
                    output_data['bleu_score'] = calculate_bleu(reference, candidate)
                    # Calculate ROUGE expects dict of fmeasures or None
                    output_data['rouge_scores'] = calculate_rouge(reference, candidate)
                elif not truths:
                    logger.debug(f"No truths for {corpus_basename}, skipping scoring for prompt {output_data.get('prompt_id', i+1)}")
                else: # truths exist but i >= len(truths) due to mismatch
                     logger.debug(f"Truth index out of bounds for {corpus_basename} prompt {output_data.get('prompt_id', i+1)}, skipping scoring.")

            model_results.extend(corpus_outputs_json)

        except Exception as e:
            logger.error(f"  Error processing {output_json_path} for {corpus_basename}: {e}", exc_info=True)

    return model_results


# --- Keep aggregate_performance_metrics ---
def aggregate_performance_metrics(all_results: List[Dict]) -> List[Dict]:
    """Aggregates results to calculate average and std dev metrics per model."""
    if not all_results: return []

    aggregated_data = {}

    for item in all_results:
        model_name = item.get("model_name")
        if not model_name: continue

        if model_name not in aggregated_data:
            aggregated_data[model_name] = {
                "tokens_per_second": [], "response_length_tokens": [],
                "bleu_score": [], "rouge1": [], "rouge2": [], "rougeL": [],
                "count": 0
            }

        # Append metrics if they exist and are numeric
        def append_if_numeric(key, value):
            if isinstance(value, (int, float)) and not math.isnan(value):
                aggregated_data[model_name][key].append(value)

        append_if_numeric("tokens_per_second", item.get("tokens_per_second"))
        append_if_numeric("response_length_tokens", item.get("response_length_tokens"))
        append_if_numeric("bleu_score", item.get("bleu_score"))

        rouge = item.get("rouge_scores") # Dict of fmeasures or None
        if isinstance(rouge, dict):
            append_if_numeric("rouge1", rouge.get("rouge1"))
            append_if_numeric("rouge2", rouge.get("rouge2"))
            append_if_numeric("rougeL", rouge.get("rougeL"))

        aggregated_data[model_name]["count"] += 1

    # Calculate averages and standard deviations
    final_metrics = []
    for model_name, metrics in aggregated_data.items():
        summary = {"model_name": model_name, "total_prompts": metrics["count"]}
        for key, values in metrics.items():
            if key != "model_name" and key != "count" and values:
                try:
                    summary[f"avg_{key}"] = statistics.mean(values)
                    # Calculate std dev only if more than one data point
                    if len(values) > 1:
                        summary[f"std_{key}"] = statistics.stdev(values)
                    else:
                        summary[f"std_{key}"] = 0.0 # Or None, depending on preference
                except statistics.StatisticsError:
                     summary[f"avg_{key}"] = None
                     summary[f"std_{key}"] = None
            elif key != "model_name" and key != "count": # Keep key even if no valid data
                summary[f"avg_{key}"] = None
                summary[f"std_{key}"] = None

        final_metrics.append(summary)

    return final_metrics


# --- Keep save_results ---
def save_results(output_dir_base: Path,
                 all_eval_data: List[Dict],
                 aggregated_perf_metrics: List[Dict]):
    """
    Saves the filtered evaluation results and aggregated performance metrics.
    """
    output_dir_base.mkdir(parents=True, exist_ok=True)

    # --- Save Filtered Evaluation Results ---
    filtered_eval_data = []
    for item in all_eval_data:
        filtered_item = {k: v for k, v in item.items() if k not in ['prompt', 'response']}
        if 'model_name' not in filtered_item:
             filtered_item['model_name'] = item.get('model_name', 'Unknown')
        if 'response_length_tokens' not in filtered_item:
             filtered_item['response_length_tokens'] = item.get('tokens')

        if 'rouge_scores' in filtered_item and not isinstance(filtered_item['rouge_scores'], (dict, type(None))):
            logger.warning(f"Unexpected type for rouge_scores: {type(filtered_item['rouge_scores'])}. Converting to str.")
            filtered_item['rouge_scores'] = str(filtered_item['rouge_scores'])

        filtered_eval_data.append(filtered_item)

    eval_path_json = output_dir_base / "all_evaluation_results.json"
    eval_path_csv = output_dir_base / "all_evaluation_results.csv"
    logger.info(f"Saving combined (filtered) evaluation results to {eval_path_json} and {eval_path_csv}")

    try:
        with open(eval_path_json, "w", encoding="utf-8") as f:
            json.dump(filtered_eval_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to save filtered evaluation results to JSON: {e}")

    if filtered_eval_data and PANDAS_AVAILABLE and pd:
        try:
            df_eval = pd.DataFrame(filtered_eval_data)
            if 'rouge_scores' in df_eval.columns:
                 rouge_data = df_eval['rouge_scores'].apply(lambda x: pd.Series(x) if isinstance(x, dict) else pd.Series({}))
                 rouge_data.columns = [f"{col}_fmeasure" for col in rouge_data.columns]
                 rouge_data = rouge_data.rename(columns={'rouge1_fmeasure': 'rouge1_fmeasure', 'rouge2_fmeasure': 'rouge2_fmeasure', 'rougeL_fmeasure': 'rougeL_fmeasure'})
                 df_eval = pd.concat([df_eval.drop('rouge_scores', axis=1), rouge_data], axis=1)

            df_eval.to_csv(eval_path_csv, index=False)
            logger.info(f"Filtered evaluation results saved to {eval_path_csv}")
        except Exception as e:
            logger.error(f"Failed to save filtered evaluation results to CSV: {e}")
    elif filtered_eval_data:
         logger.warning("Pandas not available. Skipping saving filtered evaluation results as CSV.")

    # --- Save Aggregated Performance Metrics ---
    perf_path_base = output_dir_base / "all_performance_metrics"
    if aggregated_perf_metrics and PANDAS_AVAILABLE and pd:
        try:
            df_perf = pd.DataFrame(aggregated_perf_metrics)
            df_perf.to_csv(perf_path_base.with_suffix(".csv"), index=False)
            df_perf.to_json(perf_path_base.with_suffix(".json"), orient="records", indent=2)
            logger.info(f"Aggregated performance metrics saved to {perf_path_base}.[csv/json]")
        except Exception as e:
            logger.error(f"Failed to save aggregated performance metrics DataFrame: {e}")
    elif aggregated_perf_metrics:
        try:
            with open(perf_path_base.with_suffix(".json"), "w", encoding="utf-8") as f:
                 json.dump(aggregated_perf_metrics, f, indent=2, ensure_ascii=False)
            logger.info(f"Aggregated performance metrics saved to {perf_path_base}.json (Pandas not available for CSV).")
        except Exception as e:
            logger.error(f"Failed to save aggregated performance metrics to JSON: {e}")
    else:
        logger.info("No aggregated performance metrics data to save.")

# --- Keep generate_visualizations function ---
def generate_visualizations(visualizer: 'ModelVisualizations',
                            aggregated_metrics: List[Dict],
                            overall_results: List[Dict],
                            output_dir_base: Path):
    """Generates all plots using the ModelVisualizations class."""
    if not PLOTTING_AVAILABLE or not PANDAS_AVAILABLE or visualizer is None:
        logger.info("Plotting libraries not available or visualizer not initialized. Skipping visualization generation.")
        return
    if not aggregated_metrics and not overall_results:
        logger.info("No data available for visualization. Skipping.")
        return

    logger.info("Starting visualization generation...")

    # Convert data to DataFrames
    try:
        aggregated_df = pd.DataFrame(aggregated_metrics) if aggregated_metrics else pd.DataFrame()
        all_results_df = pd.DataFrame(overall_results) if overall_results else pd.DataFrame()

        # Expand ROUGE dict in all_results_df for easier plotting
        if not all_results_df.empty and 'rouge_scores' in all_results_df.columns:
            rouge_fmeasures = all_results_df['rouge_scores'].apply(
                lambda x: pd.Series(x, dtype=float) if isinstance(x, dict) else pd.Series({}, dtype=float)
            )
            rouge_fmeasures = rouge_fmeasures.rename(columns={'rouge1': 'rouge1_fmeasure', 'rouge2': 'rouge2_fmeasure', 'rougeL': 'rougeL_fmeasure'})
            all_results_df = pd.concat([all_results_df.drop('rouge_scores', axis=1), rouge_fmeasures], axis=1)

    except Exception as e:
        logger.error(f"Error converting data to DataFrame for plotting: {e}. Skipping visualizations.")
        return

    # --- Generate Plots ---
    visualizer.plot_avg_speed_length(aggregated_df)
    visualizer.plot_avg_scores(aggregated_df)
    visualizer.plot_speed_vs_scores(aggregated_df)
    visualizer.plot_corpus_perf_by_prompt(all_results_df)

    if not all_results_df.empty:
        all_results_df['prompt_id'] = pd.to_numeric(all_results_df['prompt_id'], errors='coerce')
        all_results_df = all_results_df.dropna(subset=['prompt_id'])
        all_results_df['prompt_id'] = all_results_df['prompt_id'].astype(int)

        models = all_results_df["model_name"].unique()
        for model_name in models:
            model_df = all_results_df[all_results_df["model_name"] == model_name].copy()
            model_output_dir = output_dir_base / model_name

            if model_df.empty: continue

            visualizer.plot_model_speed_vs_bleu(model_df, model_output_dir)
            visualizer.plot_model_speed_vs_rouge(model_df, model_output_dir)
            visualizer.plot_model_perf_by_prompt(model_df, model_output_dir)
            visualizer.plot_model_scores_by_prompt(model_df, model_output_dir)

    logger.info("Visualization generation finished.")


# --- Modify main function ---
def main():
    parser = argparse.ArgumentParser(description="Evaluate GGUF model outputs and generate visualizations.")
    parser.add_argument("--config_file", type=str, required=True, help="Path to JSON configuration file.")
    # Add visualization flag - still useful for CLI override
    parser.add_argument("--visualize", action="store_true",
                        help="Generate plots after evaluation (overrides config setting if false).")
    parser.add_argument("--no-visualize", action="store_false", dest="visualize",
                        help="Prevent plot generation (overrides config setting if true).")
    # Set default=None so we know if the flag was used explicitly
    parser.set_defaults(visualize=None)
    args = parser.parse_args()

    config_abs_path = Path(args.config_file).resolve()
    config = load_config_from_file(str(config_abs_path))
    if not config: sys.exit(1)

    # Determine whether to visualize, prioritizing CLI flags over config file
    should_visualize = config.get("visualize", False) # Default to False if not in config
    if args.visualize is True: # --visualize flag used
        should_visualize = True
    elif args.visualize is False: # --no-visualize flag used
        should_visualize = False
    # If args.visualize is None (neither flag used), the value from config (or its default) is kept.

    project_root = config_abs_path.parent

    # --- Get other config values ---
    output_dir = project_root / config.get("output_dir", DEFAULT_EVAL_CONFIG["output_dir"])
    models_list_file = project_root / config.get("models_file", DEFAULT_EVAL_CONFIG["models_file"])
    corpora_list_file = project_root / config.get("prompt_corpora_file", DEFAULT_EVAL_CONFIG["prompt_corpora_file"])
    prompts_dir = project_root / config.get("prompts_dir", DEFAULT_EVAL_CONFIG["prompts_dir"])
    truth_dir = project_root / config.get("truth_dir", DEFAULT_EVAL_CONFIG["truth_dir"])
    delimiter = config.get("delimiter", DEFAULT_EVAL_CONFIG["delimiter"])

    logger.info(f"Evaluation using config: {config_abs_path}")
    # (Keep other log messages)

    # (Keep file existence checks)
    if not models_list_file.exists(): logger.error(f"Models list file not found: {models_list_file}"); sys.exit(1)
    if not corpora_list_file.exists(): logger.error(f"Corpora list file not found: {corpora_list_file}"); sys.exit(1)
    if not prompts_dir.is_dir(): logger.error(f"Prompts directory not found: {prompts_dir}"); sys.exit(1)
    if not truth_dir.is_dir(): logger.error(f"Truth directory not found: {truth_dir}"); sys.exit(1)


    model_name_entries = get_list_from_file(str(models_list_file))
    corpus_basenames = get_list_from_file(str(corpora_list_file))

    if not model_name_entries: logger.error("No models listed to evaluate."); sys.exit(1)
    if not corpus_basenames: logger.error("No corpora listed to evaluate."); sys.exit(1)

    overall_results_all_models: List[Dict] = []
    for model_name_entry in model_name_entries:
        model_specific_results = process_model_outputs(
            output_dir, model_name_entry, corpus_basenames,
            prompts_dir, truth_dir, delimiter
        )
        overall_results_all_models.extend(model_specific_results)

    aggregated_metrics = aggregate_performance_metrics(overall_results_all_models)

    save_results(output_dir, overall_results_all_models, aggregated_metrics)

    # --- Updated Visualization Call ---
    if should_visualize:
        if PLOTTING_AVAILABLE and ModelVisualizations is not None:
            visualizer = ModelVisualizations(output_dir)
            generate_visualizations(visualizer, aggregated_metrics, overall_results_all_models, output_dir)
        else:
            logger.warning("Visualization requested but plotting libraries are not available. Skipping.")
    else:
        logger.info("Visualization generation not requested or disabled.")

    logger.info("Evaluation script finished.")

if __name__ == "__main__":
    # Update warning message
    if not (NLTK_AVAILABLE or ROUGE_AVAILABLE):
         logger.warning("Neither NLTK nor rouge-score library is available. Scoring will be limited or skipped.")
    if not PANDAS_AVAILABLE:
        logger.warning("Pandas library not found. CSV output and plotting input prep will be skipped.")
    if not PLOTTING_AVAILABLE:
         logger.warning("Plotting libraries (matplotlib, seaborn, numpy) not found. Visualizations will be skipped.")

    main()