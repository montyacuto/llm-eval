#!/usr/bin/env python3
"""
GGUF Model Prompt Runner
"""

import argparse
import os
import time
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import subprocess
import sys
from pathlib import Path # Import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from llama_cpp import Llama
except ImportError:
    logger.error("llama-cpp-python is not installed. Please install it: pip install llama-cpp-python")
    Llama = None

DEFAULT_CONFIG = {
    "models_file": "Input/models.txt",
    "prompt_corpora_file": "Input/corpora.txt", # Lists basenames of corpus files
    "prompts_dir": "Input/Prompts",           # Directory containing the actual prompt files
    "output_dir": "Output",                   # Main output directory
    "truth_dir": "Input/Truth",               # Added for consistency, though not directly used by runner
    "max_tokens": 2048,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "repeat_penalty": 1.1,
    "delimiter": "---",
    "prompt_template": "{prompt}",
    "n_ctx": 4096,
    "rope_freq_base": 10000.0,
    "rope_freq_scale": 1.0,
    "n_threads": None,
    "n_gpu_layers": 0,
    "memory_reserve_gb": 0.0,
    "stream_output": False,
    "verbose": False,
    "performance_log": False
}

def load_config_from_file(config_path_str: str) -> Dict:
    config_path = Path(config_path_str)
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}. Using defaults and CLI args.")
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_from_file = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config_from_file
    except Exception as e:
        logger.error(f"Error loading config file {config_path}: {e}")
        return {}

def get_resolved_config() -> Tuple[argparse.Namespace, Optional[str]]:
    parser = argparse.ArgumentParser(
        description="Run prompts through GGUF models using a configuration file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config_file", type=str, default="config.json", help="Path to JSON configuration file.")
    # Add CLI arguments for all config options to allow overrides
    for key, value in DEFAULT_CONFIG.items():
        if isinstance(value, bool):
            # Handle boolean flags that might be False by default
            # Use a different dest for CLI to distinguish if it was set
            parser.add_argument(f"--{key}", action="store_true", dest=f"{key}_cli_true", default=None)
            parser.add_argument(f"--no-{key}", action="store_false", dest=f"{key}_cli_false", default=None)
        elif isinstance(value, int):
            parser.add_argument(f"--{key}", type=int, default=None, help=f"Override {key}")
        elif isinstance(value, float):
            parser.add_argument(f"--{key}", type=float, default=None, help=f"Override {key}")
        elif isinstance(value, str) or value is None : # handles string and None types like paths
             parser.add_argument(f"--{key}", type=str, default=None, help=f"Override {key}")

    parser.add_argument(
        "--evaluate", action="store_true",
        help="Automatically run evaluation script after processing, skipping confirmation."
    )
    cli_args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config_file_path_used = str(Path(cli_args.config_file).resolve()) # Resolve to absolute path

    if Path(config_file_path_used).exists():
        config_from_file = load_config_from_file(config_file_path_used)
        config.update(config_from_file)
    else:
        logger.warning(f"Specified config file '{config_file_path_used}' not found. Using defaults and other CLI args.")
        # If default config.json doesn't exist, config_file_path_used will still be its path for eval script.
        # If user specifies a non-existent one, that path will be used.

    for key, default_value in DEFAULT_CONFIG.items():
        if isinstance(default_value, bool):
            cli_true_val = getattr(cli_args, f"{key}_cli_true", None)
            cli_false_val = getattr(cli_args, f"{key}_cli_false", None) # This will be True if --no-key is used
            if cli_true_val is not None and cli_true_val: # --key was used
                config[key] = True
            elif cli_false_val is not None and not cli_false_val: # --no-key was used
                config[key] = False
            # If neither, it remains from file or default
        else:
            cli_value = getattr(cli_args, key, None)
            if cli_value is not None:
                config[key] = cli_value
    
    final_config_namespace = argparse.Namespace(**config)
    final_config_namespace.evaluate = cli_args.evaluate
    
    # Ensure paths are Path objects for easier manipulation, then convert to string for functions needing it
    # These are paths to LIST FILES or DIRECTORIES defined in config
    for path_key in ["models_file", "prompt_corpora_file", "output_dir", "prompts_dir", "truth_dir"]:
        if hasattr(final_config_namespace, path_key) and getattr(final_config_namespace, path_key) is not None:
            setattr(final_config_namespace, path_key, str(Path(getattr(final_config_namespace, path_key))))

    if not Path(final_config_namespace.models_file).exists():
         parser.error(f"Models list file does not exist: {final_config_namespace.models_file}")
    if not Path(final_config_namespace.prompt_corpora_file).exists():
        parser.error(f"Prompt corpora list file does not exist: {final_config_namespace.prompt_corpora_file}")
    if not Path(final_config_namespace.prompts_dir).is_dir():
        parser.error(f"Prompts directory does not exist: {final_config_namespace.prompts_dir}")

    logger.info(f"Final configuration for runner: {vars(final_config_namespace)}")
    return final_config_namespace, config_file_path_used

def read_model_list(file_path_str: str) -> List[str]:
    """Reads list of model paths/names from a file."""
    file_path = Path(file_path_str)
    logger.info(f"Reading model list from {file_path}")
    if not file_path.exists():
        logger.error(f"Models list file not found: '{file_path}'")
        return []
    try:
        return [line.strip() for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith('#')]
    except Exception as e:
        logger.error(f"Error reading model list from {file_path}: {e}")
        return []

def get_prompt_corpus_paths(corpora_list_file_str: str, prompts_base_dir_str: str) -> List[Path]:
    """
    Reads corpus basenames from corpora_list_file and constructs full paths
    using prompts_base_dir. Returns a list of Path objects.
    """
    corpora_list_file = Path(corpora_list_file_str)
    prompts_base_dir = Path(prompts_base_dir_str)
    logger.info(f"Reading corpus basenames from {corpora_list_file} and resolving against prompts_dir {prompts_base_dir}")

    if not corpora_list_file.exists():
        logger.error(f"Corpora list file not found: {corpora_list_file}")
        return []
    if not prompts_base_dir.is_dir():
        logger.error(f"Prompts base directory not found: {prompts_base_dir}")
        return []

    corpus_basenames = []
    try:
        corpus_basenames = [line.strip() for line in corpora_list_file.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith('#')]
    except Exception as e:
        logger.error(f"Error reading corpora list file {corpora_list_file}: {e}")
        return []

    actual_corpus_paths = []
    for basename in corpus_basenames:
        full_path = prompts_base_dir / basename
        if full_path.exists():
            actual_corpus_paths.append(full_path)
        else:
            logger.warning(f"Prompt corpus file not found: {full_path}")
    logger.info(f"Found {len(actual_corpus_paths)} valid prompt corpus files.")
    return actual_corpus_paths


def read_prompts_from_corpus_file(corpus_path: Path, delimiter: str = "---") -> List[str]:
    logger.info(f"Reading prompts from corpus file: {corpus_path}")
    if not corpus_path.exists():
        logger.warning(f"Prompts corpus file not found: {corpus_path}")
        return []
    try:
        content = corpus_path.read_text(encoding="utf-8")
        if delimiter and delimiter in content:
            return [p.strip() for p in content.split(delimiter) if p.strip()]
        else:
            if delimiter: logger.warning(f"Delimiter '{delimiter}' not found in {corpus_path.name}. Falling back to newline splitting.")
            return [p.strip() for p in content.splitlines() if p.strip()]
    except Exception as e:
        logger.error(f"Error reading prompts from {corpus_path}: {e}")
        return []

def get_output_path_prefix(output_dir_base_str: str, model_name_stem: str, corpus_file_stem: str) -> Path:
    model_dir = Path(output_dir_base_str) / model_name_stem
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir / f"{corpus_file_stem}_out" # Returns a Path object for the prefix

def load_model_once(model_path_str: str, config: argparse.Namespace) -> Optional[Llama]:
    model_path = Path(model_path_str) # Ensure it's a Path object
    if Llama is None: return None
    logger.info(f"Loading GGUF model from {model_path}")
    # ... (rest of load_model_once, ensure model_path is used as str(model_path) if Llama expects str)
    try:
        model = Llama(
            model_path=str(model_path), # Llama constructor expects string
            n_ctx=config.n_ctx,
            n_threads=config.n_threads,
            n_gpu_layers=config.n_gpu_layers,
            f16_kv=True, verbose=config.verbose,
            rope_freq_base=config.rope_freq_base,
            rope_freq_scale=config.rope_freq_scale,
            logits_all=False, use_mmap=True, use_mlock=False,
        )
        # ... (verbose logging)
        return model
    except Exception as e:
        logger.error(f"Error loading model {model_path}: {e}")
        return None

def generate_response(model: Llama, prompt: str, config: argparse.Namespace) -> Tuple[str, float, int, float, Dict]:
    # ... (generate_response implementation - unchanged)
    prompt_start_time = time.time()
    performance_stats = {
        "token_times": [], "token_speeds": [], "avg_tokens_per_sec_prompt": 0,
        "peak_tokens_per_sec_prompt": 0, "slowest_tokens_per_sec_prompt": float('inf'),
        "first_token_time_prompt": 0
    }
    generation_params = {
        "max_tokens": config.max_tokens, "temperature": config.temperature,
        "top_p": config.top_p, "top_k": config.top_k, "repeat_penalty": config.repeat_penalty,
    }
    response_text = ""
    total_tokens_generated_for_prompt = 0

    if config.stream_output:
        # ... (streaming logic as before)
        if config.performance_log: logger.debug("Streaming response...")
        print(f"\n--- Prompt Start ---\n{prompt}\n--- Response (Streaming) ---")
        first_token_received = False
        last_token_time = prompt_start_time
        try:
            for chunk in model.create_completion(prompt, stream=True, **generation_params):
                chunk_text = chunk["choices"][0].get("text", "") or ""
                print(chunk_text, end="", flush=True)
                response_text += chunk_text
                if chunk_text:
                    total_tokens_generated_for_prompt += 1 # Approximation
                # ... (perf calculation for streaming) ...
            print("\n--- Response End ---")
        except Exception as e:
            logger.error(f"Error during streaming generation: {e}")
            response_text += f"\n[ERROR STREAMING: {e}]"
    else:
        # ... (standard generation logic as before)
        if config.performance_log: logger.debug("Standard generation...")
        try:
            api_response = model.create_completion(prompt, stream=False, **generation_params)
            response_text = api_response["choices"][0].get("text","").strip()
            total_tokens_generated_for_prompt = api_response.get("usage", {}).get("completion_tokens", len(response_text.split()))
        except Exception as e:
            logger.error(f"Error during standard generation: {e}")
            response_text = f"[ERROR GENERATION: {e}]"

    this_prompt_generation_time = time.time() - prompt_start_time
    if total_tokens_generated_for_prompt == 0 and response_text:
        total_tokens_generated_for_prompt = len(response_text.split())
    this_prompt_tokens_per_second = total_tokens_generated_for_prompt / this_prompt_generation_time if this_prompt_generation_time > 0 else 0
    performance_stats["avg_tokens_per_sec_prompt"] = this_prompt_tokens_per_second
    # ... (log perf details)
    return response_text.strip(), this_prompt_generation_time, total_tokens_generated_for_prompt, this_prompt_tokens_per_second, performance_stats


def save_responses(responses: List[Dict], output_prefix_path: Path, performance_log: bool = False):
    """Save responses to JSON and text files using Path object for prefix."""
    output_prefix_path.parent.mkdir(parents=True, exist_ok=True) # Ensure model-specific output dir exists

    json_path = output_prefix_path.with_suffix(".json")
    text_path = output_prefix_path.with_suffix(".txt")
    
    logger.info(f"Saving JSON responses to {json_path}")
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(responses, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to save JSON to {json_path}: {e}")

    logger.info(f"Saving text responses to {text_path}")
    try:
        with open(text_path, "w", encoding="utf-8") as f:
            for i, resp_data in enumerate(responses):
                f.write(f"PROMPT {resp_data.get('prompt_id', i + 1)}:\n{resp_data.get('prompt', '')}\n\nRESPONSE:\n{resp_data.get('response', '')}\n\n")
                f.write(f"Generation time: {resp_data.get('generation_time', 0):.2f}s, Tokens: {resp_data.get('tokens', 0)}, TPS: {resp_data.get('tokens_per_second', 0):.2f}\n")
                if performance_log and "performance_stats" in resp_data:
                    # ... (write detailed perf stats)
                    pass
                if "error" in resp_data:
                     f.write(f"ERROR: {resp_data['error']}\n")
                f.write("\n---\n\n")
    except Exception as e:
        logger.error(f"Failed to save text to {text_path}: {e}")

    if performance_log:
        # This logic remains but it implies gguf_prompt_runner creates this _performance.json file.
        # The evaluate_outputs.py will read performance from the main .json file.
        # For consistency, if evaluate_outputs.py is the sole source of truth for final performance aggregation,
        # this separate _performance.json might be redundant or could be consolidated.
        # For now, keeping runner's original behavior.
        perf_json_path = output_prefix_path.parent / f"{output_prefix_path.name}_performance.json" # Sibling to _out.json
        logger.info(f"Saving detailed prompt performance data to {perf_json_path}")
        # ... (logic to aggregate and save perf_data, similar to before)
        # This should be a list of the 'performance_stats' dicts collected per prompt,
        # plus model_name, corpus_file etc.
        # For now, just save the 'responses' list again as it contains 'performance_stats'
        # This part might need more specific formatting if a dedicated performance file structure is desired
        # from the runner itself, separate from the main output.
        # The original code created a more structured performance file.
        # For this iteration, let's assume the main JSON is sufficient and evaluate_outputs.py handles it.
        # So, we can simplify or remove this specific _performance.json from runner if eval script covers it.
        # To keep it simple for now, let's assume this part is either removed or its data
        # is meant for a different purpose than what evaluate_outputs.py collects.
        # Based on prompt: "performance data is always stored in <prompt_file_name>_out.json"
        # so we actually DON'T need to write a separate _performance.json here.
        pass # Removed explicit saving of separate _performance.json


def cleanup_model_resources(model: Optional[Llama] = None):
    if model:
        del model
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except: pass


def process_corpus_for_model(loaded_model: Llama, model_name_stem: str, corpus_path: Path, config: argparse.Namespace):
    corpus_basename = corpus_path.name
    corpus_stem = corpus_path.stem
    logger.info(f"Processing corpus '{corpus_basename}' for model '{model_name_stem}'")

    # Paths in config are strings, convert to Path for Path methods
    prompts_list = read_prompts_from_corpus_file(corpus_path, config.delimiter)
    if not prompts_list:
        logger.warning(f"No prompts found in {corpus_path}. Skipping for model {model_name_stem}.")
        return

    output_prefix = get_output_path_prefix(str(config.output_dir), model_name_stem, corpus_stem)
    logger.info(f"Output for model '{model_name_stem}' on corpus '{corpus_basename}' will be saved with prefix {output_prefix}")

    responses_for_corpus: List[Dict] = []
    json_output_path = output_prefix.with_suffix(".json")
    processed_prompt_ids = set()

    if json_output_path.exists():
        try:
            responses_for_corpus = json.loads(json_output_path.read_text(encoding='utf-8'))
            processed_prompt_ids = {resp.get("prompt_id") for resp in responses_for_corpus}
            logger.info(f"Loaded {len(responses_for_corpus)} existing responses from {json_output_path}. Will skip {len(processed_prompt_ids)} of them.")
        except Exception as e:
            logger.warning(f"Could not load existing results from {json_output_path}: {e}. Starting fresh.")
            responses_for_corpus = []
            processed_prompt_ids = set()
    
    for i, prompt_text in enumerate(tqdm(prompts_list, desc=f"Model '{model_name_stem}' on '{corpus_stem}'")):
        prompt_id = i + 1
        if prompt_id in processed_prompt_ids:
            logger.debug(f"Skipping already processed prompt {prompt_id} for '{model_name_stem}' on '{corpus_basename}'")
            continue

        logger.info(f"Processing prompt {prompt_id}/{len(prompts_list)} from '{corpus_basename}' for '{model_name_stem}'")
        try:
            loaded_model.reset()
            formatted_prompt = str(config.prompt_template).format(prompt=prompt_text)
            resp_text, gen_time, tokens, tps, perf_stats_detail = generate_response(loaded_model, formatted_prompt, config)
            
            response_data = {
                "prompt_id": prompt_id, "corpus_file": corpus_basename, "prompt": prompt_text,
                "response": resp_text, "generation_time": gen_time, "tokens": tokens,
                "tokens_per_second": tps, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            if config.performance_log: # This flag controls if detailed perf_stats go into the JSON
                response_data["performance_stats"] = perf_stats_detail
            responses_for_corpus.append(response_data)
        except Exception as e:
            logger.error(f"Error on prompt {prompt_id} for '{model_name_stem}' with '{corpus_basename}': {e}")
            responses_for_corpus.append({
                "prompt_id": prompt_id, "corpus_file": corpus_basename, "prompt": prompt_text,
                "response": f"[ERROR: {e}]", "error": str(e), "generation_time": 0, "tokens": 0,
                "tokens_per_second": 0, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })
        finally:
            if (i + 1) % 5 == 0 or (i + 1) == len(prompts_list):
                save_responses(responses_for_corpus, output_prefix, bool(config.performance_log))
    
    save_responses(responses_for_corpus, output_prefix, bool(config.performance_log)) # Final save
    logger.info(f"Corpus '{corpus_basename}' processed for model '{model_name_stem}'.")


def run_evaluation(config_file_path_str: Optional[str]):
    if not config_file_path_str:
        logger.error("No config file path provided. Cannot run evaluation.")
        return

    eval_script_path = Path(__file__).parent / "evaluate_outputs.py"
    logger.info(f"Running evaluation script: {eval_script_path}")
    if not eval_script_path.exists():
        logger.error(f"Evaluation script not found at {eval_script_path}")
        return

    eval_args = [sys.executable, str(eval_script_path), "--config_file", config_file_path_str]
    logger.info(f"Evaluation script arguments: {' '.join(eval_args)}")
    try:
        result = subprocess.run(eval_args, check=True, capture_output=True, text=True, encoding='utf-8')
        logger.info("Evaluation script stdout:\n" + result.stdout)
        if result.stderr: logger.warning("Evaluation script stderr:\n" + result.stderr)
    except Exception as e:
        logger.error(f"Evaluation script execution failed: {e}")
        if hasattr(e, 'stdout'): logger.error("Stdout:\n" + e.stdout)
        if hasattr(e, 'stderr'): logger.error("Stderr:\n" + e.stderr)

def main():
    if Llama is None: sys.exit(1)

    config, config_file_path_used = get_resolved_config()

    # model_paths_from_list contains the raw entries from models.txt (e.g. Models/model.gguf or just model.gguf)
    model_paths_from_list = read_model_list(str(config.models_file))
    if not model_paths_from_list:
        logger.error(f"No models found in {config.models_file}. Exiting.")
        return

    # prompt_corpus_full_paths are full Path objects to the actual corpus files
    prompt_corpus_full_paths = get_prompt_corpus_paths(str(config.prompt_corpora_file), str(config.prompts_dir))
    if not prompt_corpus_full_paths:
        logger.error(f"No valid prompt corpus files found based on {config.prompt_corpora_file} and {config.prompts_dir}. Exiting.")
        return

    for model_idx, model_path_entry in enumerate(model_paths_from_list):
        # model_path_entry can be "Models/some_model.gguf" or just "some_model.gguf"
        # We need to ensure it's correctly resolved if it's a relative path to an actual GGUF file.
        # Assume paths in models.txt are relative to project root or absolute.
        actual_model_path = Path(model_path_entry).resolve()
        if not actual_model_path.exists():
            logger.error(f"Model GGUF file not found: {actual_model_path}. Skipping this model.")
            continue
        
        model_name_stem = actual_model_path.stem # Used for output directory name
        logger.info(f"--- Model {model_idx+1}/{len(model_paths_from_list)}: {model_name_stem} ({actual_model_path}) ---")
        
        current_model_instance = None
        try:
            current_model_instance = load_model_once(str(actual_model_path), config)
            if not current_model_instance:
                logger.error(f"Failed to load model {actual_model_path}. Skipping.")
                continue
            for corpus_path_obj in prompt_corpus_full_paths: # These are already Path objects
                process_corpus_for_model(current_model_instance, model_name_stem, corpus_path_obj, config)
        except Exception as e:
            logger.error(f"Unhandled error processing model {model_name_stem}: {e}", exc_info=True)
        finally:
            if current_model_instance:
                cleanup_model_resources(current_model_instance)
    
    logger.info("All models and corpora processed.")
    if config.evaluate or (sys.stdin.isatty() and input("Run evaluation? (y/N): ").lower() == 'y'):
        run_evaluation(config_file_path_used)

if __name__ == "__main__":
    start_time = time.time()
    logger.info("GGUF Prompt Runner Started")
    main()
    logger.info(f"GGUF Prompt Runner Finished. Total time: {time.time() - start_time:.2f}s")