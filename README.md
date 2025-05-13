# GGUF Prompt Runner & Evaluation Pipeline

## Overview

This Python-based pipeline runs prompts from text corpora through various GGUF (GPT-Generated Unified Format) language models using `llama-cpp-python`. It captures model outputs and performance statistics, then evaluates these outputs by calculating BLEU and ROUGE scores against ground truth references. The pipeline also includes extensive visualization capabilities to analyze model performance.

A key feature of the generation script (`gguf_prompt_runner.py`) is its ability to load each model only once, resetting its context for each prompt to ensure clean, isolated responses without the overhead of frequent model reloading. The evaluation script (`evaluate_outputs.py`) processes the generated outputs to provide comparable metrics across models, and the visualization component (`model_visualizations.py`) generates comprehensive plots for performance analysis.

## Features

* **Model Processing**
* **Corpus Handling**
* **Output Capture**
* **Performance Logging**
* **Evaluation Metrics**
* **Aggregated Results**
* **Visualization**
* **Hardware Acceleration**
* **Configuration**

## Prerequisites

* **Python**: Version 3.9 or newer
* **Conda**: Recommended for managing Python environments (Miniconda or Anaconda)
* **Git**: Required for some installation methods
* **C++ Compiler**: Necessary for building `llama-cpp-python`
* **Hardware-specific requirements**:
  * **Windows with CUDA**: CUDA Toolkit, Visual Studio Build Tools
  * **Linux with CUDA**: CUDA Toolkit, GCC/G++
  * **Mac with Apple Silicon**: Xcode Command Line Tools

## Installation

### 1. Set Up the Environment

```bash
# Create a conda environment
conda create -n llamaconda=<python_version>
conda activate gguf_pipeline

# Clone the repository (if applicable)
git clone https://github.com/montyacuto/llm-eval.git
cd llm-eval
```

### 2. Install llama-cpp-python

#### Option 1: Prebuilt Wheels (Easiest)

For basic CPU support:

```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

For CUDA-based GPU acceleration:

```bash
# For specific CUDA versions (example: CUDA 11.8)
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/<cuda-version>
#replace <cuda-version> with cu121 for CUDA 12.1, cu122 for CUDA 12.2, etc...
```

Available CUDA versions are typically CUDA 12.1-12.4 - find additional information and support for other backends at [llama-cpp-python](https://llama-cpp-python.readthedocs.io/en/latest/)

#### Option 2: Build from Source
##### Windows CUDA:

1. **Install Visual Studio Build Tools**:
   - Download and install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   - Select "Desktop development with C++" workload
   - Ensure you install the MSVC compiler and Windows SDK

2. **Install CUDA Toolkit**:
   - Download and install [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (version 11.7 or later recommended)
   - Make sure CUDA_PATH environment variable is set (usually done automatically by installer)

3. **Build and install llama-cpp-python**:
```bash
# For CUDA installation
set CMAKE_ARGS=-DLLAMA_CUBLAS=on
set FORCE_CMAKE=1
pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade
```

##### Linux CUDA

1. **Install CUDA Toolkit**:
```bash
# For Ubuntu/Debian
sudo apt update
sudo apt install -y build-essential
# Install CUDA Toolkit - version depends on your GPU and driver
# Visit https://developer.nvidia.com/cuda-downloads for instructions
```

2. **Build and install llama-cpp-python**:
```bash
# For CUDA installation
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade
```

##### Mac with Apple Silicon

1. **Install Xcode Command Line Tools**:
```bash
xcode-select --install
```

2. **Build and install llama-cpp-python with Metal**:
```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade
```

### 3. Install Additional Dependencies

```bash
# Core dependencies
pip install tqdm

# Evaluation dependencies
pip install nltk rouge-score pandas

# Visualization dependencies
pip install matplotlib seaborn numpy
```

## Directory Structure

The default directory structure the pipeline will expect (can be customized using config files):

```
gguf_pipeline/
├── gguf_prompt_runner.py       # The generation script
├── evaluate_outputs.py         # The evaluation script
├── model_visualizations.py     # The visualization module
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
├── Models/                     # Store your GGUF model files here
│   ├── Gemma/
│   │   └── gemma-3-1b-it-GGUF/
│   │       └── gemma-3-1b-it-Q4_K_M.gguf
│   └── Llama/
│       └── Llama-3.2-1B-Instruct-Q8_0-GGUF/
│           └── llama-3.2-1b-instruct-q8_0.gguf
│
└── Output/                     # Default output directory (created by scripts)
    ├── plots/                  # Overall visualization plots
    │   ├── avg_performance_per_model.png
    │   ├── avg_scores_per_model.png
    │   └── avg_speed_vs_avg_scores.png
    │
    ├── ModelNameStem1/         # Sub-directory for model 1 outputs
    │   ├── plots/              # Model-specific visualizations
    │   │   ├── ModelNameStem1_speed_vs_bleu.png
    │   │   └── ...
    │   ├── corpus_A_out.json
    │   ├── corpus_A_out.txt
    │   └── ...
    ├── ModelNameStem2/         # Sub-directory for model 2 outputs
    │   └── ...
    │
    ├── all_evaluation_results.json # Filtered per-prompt results across all models
    ├── all_evaluation_results.csv  # CSV format of evaluation results
    ├── all_performance_metrics.json# Aggregated average metrics per model
    └── all_performance_metrics.csv # CSV format of performance metrics
```

## Input Files Format

### 1. Models List File (`models.txt`)
Plain text file with each line containing a path to a GGUF model file:
```
Models/Gemma/gemma-3-1b-it-GGUF/gemma-3-1b-it-Q4_K_M.gguf
Models/Llama/Llama-3.2-1B-Instruct-Q8_0-GGUF/llama-3.2-1b-instruct-q8_0.gguf
# Lines starting with # are ignored
```

### 2. Prompt Corpora List File (`corpora.txt`)
Plain text file with each line containing the name of a prompt corpus file:
```
corpus_A.txt
corpus_B.txt
```

### 3. Prompt Corpus Files (e.g., `Input/Prompts/corpus_A.txt`)
Text files containing prompts separated by a delimiter (default is `---`, can be set in config):
```
What is the capital of France?
---
Explain the concept of machine learning.
---
Summarize the plot of Hamlet.
```

### 4. Ground Truth Files (e.g., `Input/Truth/<prompt_corpus>_truth.txt` - use this naming convention)
Text files containing reference answers, using the same delimiter:
```
The capital of France is Paris.
---
Machine learning is a branch of artificial intelligence...
---
Hamlet is a tragedy by William Shakespeare...
```

## Configuration (`config.json`)

The pipeline uses a central `config.json` file:

```
{
    "models_file": "Input/models.txt",
    "prompt_corpora_file": "Input/corpora.txt",
    "output_dir": "Output",
    "prompts_dir": "Input/Prompts",
    "truth_dir": "Input/Truth",
    "delimiter": "---",
    "prompt_template": "{prompt}",    # For standardized input tuning across every prompts
    "max_tokens": 1024,           # Maximum response length
    "temperature": 0.75,          # Lower == more deterministic, higher == more random
    "top_p": 0.95,                # Desired sum of scores
    "top_k": 40,                  # Number of tokens to sample
    "repeat_penalty": 1.2,        # Penalty for repeating n-gram tokens, reduces chance of output looping
    "n_ctx": 8192,                # Context size (Recommend >20GB VRAM for 8192 token context)
    "rope_freq_base": 10000.0,    # Rope Frequency Base (for advanced users - usually leave default)
    "rope_freq_scale": 1.0,       # Rope Frequency Scale (for advanced users - usually leave default)
    "n_threads": null,            # How many CPU threads to use, null for automatic assignment
    "n_gpu_layers": -1,           # How many compute layers to offload to the GPU, -1 for as many as possible
    "memory_reserve_gb": 0.0,     # Will provide advance warning for OOM issues
    "stream_output": true,        # Show live LLM output
    "verbose": true,              # Show llama-cpp-python verbose output
    "performance_log": false,     # Shows per-token performance metrics (WARNING: Increases performance overhead and output file size)
    "visualize": true             # Automatically calls visualizer after evaluation step
}
```

Key Parameters:
- **Path Settings**: `models_file`, `prompt_corpora_file`, `output_dir`, `prompts_dir`, `truth_dir` - ensure that these are set correctly

## Running the Pipeline

### 1. Generate Outputs and Evaluate
```bash
# Run with default config.json
python gguf_prompt_runner.py --config_file config.json --evaluate # will automatically begin evaluation after generation is complete

# Override specific parameters
python gguf_prompt_runner.py --config_file config.json --temperature 0.8 --max_tokens 2048 --evaluate
```

### 2. Generate Outputs Only
```bash
python gguf_prompt_runner.py --config_file config.json
# Answer 'n' when prompted about evaluation
```

### 3. Run Evaluation Only
```bash
python evaluate_outputs.py --config_file config.json

# To explicitly enable visualization
python evaluate_outputs.py --config_file config.json --visualize

# To explicitly disable visualization
python evaluate_outputs.py --config_file config.json --no-visualize
```

## Output and Visualizations

### JSON and Text Outputs
- **Model-specific outputs**: `Output/<ModelName>/<corpus>_out.json` and `<corpus>_out.txt`
- **Aggregated results**: `all_evaluation_results.json/csv` and `all_performance_metrics.json/csv`

### Visualizations
When the `visualize` config option is enabled, the pipeline generates:

1. **Overall Performance Plots** (in `Output/plots/`):
   - `avg_performance_per_model.png`: Bar charts showing average tokens-per-second and response length
   - `avg_scores_per_model.png`: Bar chart comparing BLEU and ROUGE scores across models
   - `avg_speed_vs_avg_scores.png`: Scatter plot showing speed vs. accuracy tradeoffs

2. **Model-Specific Plots** (in `Output/<ModelName>/plots/`):
   - `<ModelName>_speed_vs_bleu.png`: Scatter plot of tokens-per-second vs. BLEU scores
   - `<ModelName>_speed_vs_rouge.png`: Scatter plot of tokens-per-second vs. ROUGE scores
   - `<ModelName>_perf_vs_prompt_id.png`: Line graph showing performance across prompts
   - `<ModelName>_scores_vs_prompt_id.png`: Line graph showing evaluation scores across prompts

3. **Corpus-Level Plots**:
   - Performance trends for each corpus showing variation across prompts

## Troubleshooting

### Installation Issues
- **CUDA Compilation Errors**: 
  - Ensure CUDA Toolkit version matches your GPU driver
  - For Windows, check that Visual Studio Build Tools is properly installed
  - Try: `nvcc --version` to verify CUDA compiler is available

- **Build Errors**:
  - Windows: Check environment variables (CUDA_PATH)
  - Linux: Ensure correct development packages are installed (`build-essential`, `python3-dev`)
  - Mac: Run `xcode-select --install` to ensure command-line tools are available

- **Pre-built Wheel Issues**:
  - Make sure to match CUDA version with your installed CUDA Toolkit

### Runtime Issues
- **Model Not Loading**: Check the path in `models.txt` and ensure it's an actual GGUF file
- **GPU Not Being Used**: 
  - Verify that `n_gpu_layers` is set to -1 or a positive number
  - Check `verbose` output for GPU layer loading confirmation
  - Monitor GPU utilization with tools (NVIDIA SMI, Windows Task Manager)

- **Performance Too Slow**:
  - Increase `n_threads` for CPU usage
  - Set `n_gpu_layers` to -1 for maximum GPU offload
  - Consider using models with fewer parameters and higher quantization for better performance

- **Out of Memory Errors**:
  - Reduce `n_ctx` value
  - Use more aggressively quantized models (e.g., Q4 instead of Q8)
  - Set `memory_reserve_gb` to a non-zero value to assist in debugging

### Evaluation Issues
- **Missing Scores**:
  - Ensure truth files exist in the correct location and match prompt file names
  - Check that the number of entries in truth files matches the prompt files
  - Verify that evaluation dependencies are installed (`nltk`, `rouge-score`)

- **Visualization Errors**:
  - Install visualization dependencies: `pip install matplotlib seaborn numpy`
  - Ensure the `Output` directory is writable

## Credits
This README file was created with assistance from Claude Sonnet 3.7
