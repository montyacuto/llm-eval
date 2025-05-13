# GGUF Prompt Runner & Evaluation Pipeline

## Overview

This Python-based pipeline efficiently runs prompts from text corpora through various GGUF (GPT-Generated Unified Format) language models using `llama-cpp-python`. It captures model outputs and performance statistics. An accompanying script evaluates these outputs, calculating BLEU and ROUGE scores against ground truth references and generating aggregated performance metrics per model.

A key feature of the generation script (`gguf_prompt_runner.py`) is its ability to load each model only once, resetting its context for each prompt to ensure clean, isolated responses without the overhead of frequent model reloading. The evaluation script (`evaluate_outputs.py`) then processes the generated outputs to provide comparable metrics across models.

## Features

* **Model Processing**: Runs prompts through multiple GGUF models specified in a list file.
* **Corpus Handling**: Accepts multiple prompt corpora, also specified in a list file.
* **Efficient Generation**: Loads each GGUF model once per run, resetting context for individual prompts.
* **Output Generation**: Saves detailed generation outputs (prompt, response, tokens, time) in JSON format per model and corpus. Human-readable text outputs are also generated.
* **Performance Logging**: Captures generation time and tokens-per-second for each prompt.
* **Evaluation Metrics**: Calculates BLEU and ROUGE scores for model responses against provided ground truth files (optional).
* **Aggregated Results**: Generates summary files comparing models based on average performance (TPS, response length) and evaluation scores (BLEU, ROUGE).
* **Hardware Acceleration**: Supports NVIDIA CUDA, AMD ROCm, and Apple Metal via `llama-cpp-python` build flags.
* **Flexible Configuration**: Uses a central JSON configuration file (`config.json`) for both generation and evaluation, with overrides possible via command-line arguments.

## Prerequisites

* **Python**: Version 3.9 or newer is recommended.
* **Conda**: Recommended for managing Python environments. (Miniconda or Anaconda)
* **Git**: May be required for specific `llama-cpp-python` installations.
* **C++ Compiler**: Necessary for building `llama-cpp-python` (GCC/G++, Clang, MSVC).
* **Optional Acceleration Hardware**: CUDA Toolkit / ROCm Drivers / Metal for GPU support.

## Setup and Installation

1.  **Navigate to the Pipeline Directory**:
    Ensure you have the `gguf_prompt_runner.py` and `evaluate_outputs.py` scripts and navigate to their directory.
    ```bash
    cd /path/to/gguf_pipeline
    ```

2.  **Create a Conda Environment** (Recommended):
    ```bash
    conda create -n gguf_pipeline python=3.10 -y # Or your preferred Python 3.9+
    conda activate gguf_pipeline
    ```

3.  **Install Core Dependencies**:
    ```bash
    pip install tqdm llama-cpp-python # Install llama-cpp-python first (see below)
    ```

4.  **Install `llama-cpp-python` (Crucial Step)**:
    Follow the instructions in the original README section based on your hardware (CPU, NVIDIA CUDA, Apple Metal). Correct installation is vital for model execution.

    * **Example (CPU Only)**: `pip install llama-cpp-python`
    * **Example (NVIDIA CUDA)**: `CMAKE_ARGS="-DLLAMA_CUBLAS=ON" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade`
    * **Example (Apple Metal)**: `CMAKE_ARGS="-DGGML_METAL=on" pip install -U llama-cpp-python --no-cache-dir`

5.  **Install Evaluation Dependencies** (Optional but Recommended for Scoring):
    ```bash
    pip install nltk rouge-score pandas
    ```
    * `nltk` and `rouge-score` are needed for calculating BLEU and ROUGE scores. The script will download necessary NLTK data (`punkt`) on first run if needed.
    * `pandas` is used for convenient handling and saving of results tables (.csv output).

## Directory Structure (Recommended)

Organize your files for clarity:

gguf_pipeline/
├── gguf_prompt_runner.py       # The generation script
├── evaluate_outputs.py       # The evaluation script
├── config.json                 # Central configuration file
│
├── Input/
│   ├── models.txt              # List of paths to GGUF model files
│   ├── corpora.txt             # List of basenames for prompt corpus files
│   ├── Prompts/                # Directory containing prompt files
│   │   ├── corpus_A.txt
│   │   └── corpus_B.txt
│   └── Truth/                  # Directory containing ground truth files
│       ├── corpus_A_truth.txt  # Ground truths for corpus_A.txt
│       └── corpus_B_truth.txt  # Ground truths for corpus_B.txt
│
├── Models/                     # Store your GGUF model files here (or list paths in models.txt)
│   ├── Gemma/
│   │   └── gemma-3-1b-it-GGUF/
│   │       └── gemma-3-1b-it-Q4_K_M.gguf
│   └── Llama/
│       └── Llama-3.2-1B-Instruct-Q8_0-GGUF/
│           └── llama-3.2-1b-instruct-q8_0.gguf
│
└── Output/                     # Default output directory (created by scripts)
├── ModelNameStem1/         # Sub-directory for model 1 outputs
│   ├── corpus_A_out.json
│   ├── corpus_A_out.txt
│   └── ...
├── ModelNameStem2/         # Sub-directory for model 2 outputs
│   ├── corpus_A_out.json
│   └── ...
│
├── all_evaluation_results.json # Filtered per-prompt results across all models
├── all_evaluation_results.csv  # (Same data as JSON, requires pandas)
├── all_performance_metrics.json# Aggregated average metrics per model
└── all_performance_metrics.csv # (Same data as JSON, requires pandas)

## Input Files Format

1.  **Models List File (`models.txt`)**:
    * Plain text file.
    * Each line contains the path (relative to the pipeline directory or absolute) to a GGUF model file.
    * Example:
        ```txt
        Models/Gemma/gemma-3-1b-it-GGUF/gemma-3-1b-it-Q4_K_M.gguf
        Models/Llama/Llama-3.2-1B-Instruct-Q8_0-GGUF/llama-3.2-1b-instruct-q8_0.gguf
        # Lines starting with # are ignored
        ```

2.  **Prompt Corpora List File (`corpora.txt`)**:
    * Plain text file.
    * Each line should contain the **basename** (filename) of a prompt corpus file located in the `prompts_dir` specified in `config.json`.
    * Example:
        ```txt
        corpus_A.txt
        corpus_B.txt
        ```

3.  **Prompt Corpus Files (e.g., `Input/Prompts/corpus_A.txt`)**:
    * Plain text files containing the actual prompts.
    * Prompts are separated by the `delimiter` string defined in `config.json` (default is `---`).
    * If the delimiter is not found, each non-empty line is treated as a separate prompt.

4.  **Ground Truth Files (e.g., `Input/Truth/corpus_A_truth.txt`)**:
    * Optional, but required for BLEU/ROUGE scoring in the evaluation step.
    * Must be located in the `truth_dir` specified in `config.json`.
    * The filename must match the corresponding prompt corpus file with `_truth` appended before the extension (e.g., `corpus_A.txt` -> `corpus_A_truth.txt`).
    * Contains the reference "correct" answers or desired outputs for each prompt in the corresponding corpus file.
    * The reference answers should be separated by the **same delimiter** used in the prompt corpus file.
    * The number of reference answers must exactly match the number of prompts in the corresponding corpus file for scoring to occur.

## Configuration (`config.json`)

The pipeline uses a central `config.json` file to manage settings for both generation and evaluation. Command-line arguments can override settings in the config file.

* **Example `config.json`**:
    ```json
    {
        "models_file": "Input/models.txt",
        "prompt_corpora_file": "Input/corpora.txt",
        "output_dir": "Output",
        "prompts_dir": "Input/Prompts",
        "truth_dir": "Input/Truth",  // Directory for ground truth files
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
        "n_threads": null,           // null for auto-detect (CPU)
        "n_gpu_layers": -1,          // -1 for max GPU offload, 0 for CPU
        "memory_reserve_gb": 0.0,    // Optional: Reserve system RAM (requires psutil)
        "stream_output": false,      // Stream runner output to console?
        "verbose": true,             // Verbose logging from llama.cpp?
        "performance_log": false     // Include detailed perf stats in _out.json?
    }
    ```

* **Key Shared Parameters**: `models_file`, `prompt_corpora_file`, `output_dir`, `prompts_dir`, `truth_dir`, `delimiter`.
* **Generation Parameters**: `max_tokens`, `temperature`, `top_p`, `top_k`, `repeat_penalty`, `prompt_template`, `n_ctx`, `rope_freq_base`, `rope_freq_scale`, `n_threads`, `n_gpu_layers`, `memory_reserve_gb`, `stream_output`, `verbose`, `performance_log`.
* **Evaluation Parameters**: The evaluation script primarily uses the directory/file paths (`output_dir`, `models_file`, `prompt_corpora_file`, `prompts_dir`, `truth_dir`) and the `delimiter` to locate and process the generated outputs and truth files.

## Running the Pipeline

Ensure your conda environment is active (`conda activate gguf_pipeline`).

There are three main ways to run the pipeline:

1.  **Generate Outputs and Evaluate Immediately**:
    Run the `gguf_prompt_runner.py` script. You can either use the `--evaluate` flag or answer 'y' when prompted after generation finishes (if run interactively). This uses the configuration specified in `config.json`.
    ```bash
    # Option 1: Use the flag
    python gguf_prompt_runner.py --config_file config.json --evaluate

    # Option 2: Run without flag and answer 'y' when prompted
    python gguf_prompt_runner.py --config_file config.json
    # ... (wait for generation) ...
    # Run evaluation? (y/N): y
    ```
    You can still override specific generation parameters via CLI:
    ```bash
    python gguf_prompt_runner.py --config_file config.json --temperature 0.8 --evaluate
    ```

2.  **Generate Outputs Only**:
    Run the `gguf_prompt_runner.py` script *without* the `--evaluate` flag and answer 'n' (or press Enter) if prompted.
    ```bash
    python gguf_prompt_runner.py --config_file config.json
    # ... (wait for generation) ...
    # Run evaluation? (y/N): n
    ```

3.  **Run Evaluation Only** (Assumes outputs already exist):
    If you have already generated the outputs using `gguf_prompt_runner.py`, you can run the evaluation script directly. It will read the `config.json` to find the outputs and truth files.
    ```bash
    python evaluate_outputs.py --config_file config.json
    ```
    This is useful if you want to re-run the analysis, adjusted the truth files, or if the generation step was interrupted.

## Output Structure

The pipeline generates outputs in the specified `output_dir` (default: `Output/`):

1.  **Per-Model/Per-Corpus Outputs** (Inside `Output/<ModelNameStem>/`):
    * **`<corpus_stem>_out.json`**: Detailed JSON list. Each entry corresponds to a prompt and includes:
        * `prompt_id`, `corpus_file`, `prompt` (original text)
        * `response` (model's generated text)
        * `generation_time`, `tokens` (response tokens), `tokens_per_second`
        * `timestamp`, `model_name` (Added by evaluation script phase)
        * `response_length_tokens` (Added by evaluation script phase)
        * `bleu_score` (float, if truth available, added by evaluation script)
        * `rouge_scores` (dict of f-measures, if truth available, added by evaluation script)
        * `error` (if applicable)
        * `performance_stats` (dict, if `performance_log: true` in config during generation)
    * **`<corpus_stem>_out.txt`**: Human-readable text file showing prompts, responses, and basic stats.

2.  **Aggregated Evaluation Outputs** (Inside `Output/`):
    * **`all_evaluation_results.json`**: JSON list containing per-prompt results across **all models**. Excludes the full `prompt` and `response` text to keep the file size manageable. Includes identifiers, model name, token counts, and scores (BLEU/ROUGE).
    * **`all_evaluation_results.csv`**: Same data as the JSON counterpart, in CSV format (requires `pandas`). Useful for spreadsheet analysis.
    * **`all_performance_metrics.json`**: JSON list containing aggregated results, with one entry **per model**. Shows average metrics like `avg_tokens_per_second`, `avg_response_length_tokens`, `avg_bleu_score`, `avg_rouge1`, etc., calculated across all prompts processed for that model.
    * **`all_performance_metrics.csv`**: Same aggregated data as the JSON counterpart, in CSV format (requires `pandas`). Ideal for comparing model performance summaries.

## Troubleshooting Common Issues

* **`llama-cpp-python` Installation Failures**: See original README section. Ensure compilers and GPU toolkits (if used) are correctly installed and configured.
* **Model Not Using GPU**: See original README section. Verify build flags and `n_gpu_layers` setting. Check `verbose` output and system monitoring tools.
* **File Not Found Errors**: Double-check paths in `config.json` and the list files (`models.txt`, `corpora.txt`). Ensure `prompts_dir` and `truth_dir` exist and contain the necessary files. Check that `corpora.txt` lists **basenames**, not full paths. Use absolute paths if relative paths are ambiguous.
* **Evaluation Script Errors**:
    * **Missing Truth Files**: Ensure truth files exist in the specified `truth_dir`, are correctly named (`<corpus_stem>_truth.txt`), and use the same delimiter as the prompt files.
    * **Prompt/Truth Mismatch**: Scoring is skipped if the number of prompts and truths in corresponding files doesn't match. Check the files and delimiters.
    * **Missing Libraries**: Ensure `nltk`, `rouge-score`, `pandas` are installed in the environment (`pip install nltk rouge-score pandas`).
* **Slow Performance**: See original README section. GPU acceleration (`n_gpu_layers > 0`) is crucial for speed. Check CPU thread count (`n_threads`) if running on CPU.