# model_visualizations.py

import logging
from pathlib import Path
import warnings
from typing import Optional # Ensure this is at the top

# Optional imports for plotting
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    pd, plt, sns, np = None, None, None, None

logger = logging.getLogger(__name__)

MODEL_PALETTE = sns.color_palette("husl", 17) # Increased for 17 models
METRIC_PALETTE = sns.color_palette("viridis", 4)

class ModelVisualizations:
    def __init__(self, output_dir: Path):
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting libraries not found. Skipping visualization.")
            return
        self.output_dir = Path(output_dir)
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Visualization output directory: {self.plots_dir}")
        sns.set_theme(style="whitegrid")

    def _save_plot(self, fig, filename_prefix: str, subdirectory: Optional[Path] = None):
        if not PLOTTING_AVAILABLE: return
        save_dir = subdirectory if subdirectory else self.plots_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        filepath = save_dir / f"{filename_prefix}.png"
        try:
            fig.savefig(filepath, bbox_inches='tight', dpi=150)
            logger.info(f"Saved plot: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save plot {filepath}: {e}")
        finally:
            plt.close(fig)

    def plot_avg_speed_length(self, aggregated_df: pd.DataFrame):
        if not PLOTTING_AVAILABLE: return
        if aggregated_df.empty:
            logger.warning("Aggregated data is empty, skipping plot_avg_speed_length.")
            return

        # Filter out models that have NaN for both metrics to avoid empty slots if possible
        # However, seaborn will typically just not plot them if data is NaN.
        # It's better to ensure the dataframe passed has valid entries where possible.
        # df_to_plot = aggregated_df.dropna(subset=["avg_tokens_per_second", "avg_response_length_tokens"], how='all').copy()
        # if df_to_plot.empty:
        #     logger.warning("No models with valid data for avg_tokens_per_second or avg_response_length_tokens. Skipping plot.")
        #     return
        # For bar plots, seaborn handles missing categories gracefully by not plotting them.
        # The main issue is label overlap.

        # Increase figure width for more label space
        fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=False) # Increased width and height
        fig.suptitle("Average Performance Metrics per Model", fontsize=16)

        metrics = ["avg_tokens_per_second", "avg_response_length_tokens"]
        titles = ["Tokens per Second", "Response Length (Tokens)"]
        std_dev_cols = ["std_tokens_per_second", "std_response_length_tokens"]

        for i, (metric, title, std_dev_col) in enumerate(zip(metrics, titles, std_dev_cols)):
            ax = axes[i]
            if metric not in aggregated_df.columns or aggregated_df[metric].isnull().all():
                logger.warning(f"Metric '{metric}' not found or all values are null in aggregated data. Skipping subplot '{title}'.")
                ax.set_title(f"{title}\n(No data available)")
                ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
                continue

            # Filter data for this specific metric to ensure no NaN values are passed to barplot y
            # though seaborn typically handles this.
            plot_data_metric = aggregated_df.dropna(subset=[metric])

            if plot_data_metric.empty:
                logger.warning(f"No valid (non-NaN) data for metric '{metric}'. Skipping subplot '{title}'.")
                ax.set_title(f"{title}\n(No valid data)")
                ax.text(0.5, 0.5, "No valid data", ha='center', va='center', transform=ax.transAxes)
                continue

            use_error_bars = std_dev_col in plot_data_metric.columns and plot_data_metric[std_dev_col].notna().any()
            
            # Use a slice of MODEL_PALETTE if number of models in plot_data_metric is less
            current_palette = MODEL_PALETTE[:len(plot_data_metric['model_name'].unique())]


            sns.barplot(ax=ax, x="model_name", y=metric, data=plot_data_metric,
                        palette=current_palette, errorbar=('ci', 95) if not use_error_bars else None)

            if use_error_bars:
                # Create a mapping from model_name to its index for error bars
                name_to_x_idx = {name: j for j, name in enumerate(plot_data_metric["model_name"])}
                x_coords = [name_to_x_idx[name] for name in plot_data_metric["model_name"]]
                errors = plot_data_metric[std_dev_col].fillna(0)
                
                ax.errorbar(x=x_coords, y=plot_data_metric[metric],
                                 yerr=errors, fmt='none', c='black', capsize=3, elinewidth=0.8, capthick=0.8)


            ax.set_title(title, fontsize=14)
            ax.set_xlabel("Model", fontsize=12)
            ax.set_ylabel(f"Average {title.split('(')[0].strip()}", fontsize=12) # Shorten Y label
            
            # Adjust x-tick labels
            ax.tick_params(axis='x', labelrotation=65, labelsize=8) # Increased rotation, smaller font
            ax.set_xticklabels(ax.get_xticklabels(), ha="right") # Align rotated labels better

        plt.tight_layout(rect=[0, 0.03, 1, 0.95], pad=2.0) # Add padding
        self._save_plot(fig, "avg_performance_per_model")

    def plot_avg_scores(self, aggregated_df: pd.DataFrame):
        if not PLOTTING_AVAILABLE: return
        if aggregated_df.empty:
            logger.warning("Aggregated data is empty, skipping plot_avg_scores.")
            return

        score_cols = ["avg_bleu_score", "avg_rouge1", "avg_rouge2", "avg_rougeL"]
        metrics_to_plot = [col for col in score_cols if col in aggregated_df.columns and not aggregated_df[col].isnull().all()]

        if not metrics_to_plot:
            logger.info("No average BLEU or ROUGE scores with valid data found. Skipping score plot.")
            return
        
        # Filter aggregated_df to only include models that have at least one valid score to plot
        # This helps if some models have no scores at all.
        df_to_plot = aggregated_df.dropna(subset=metrics_to_plot, how='all').copy()
        if df_to_plot.empty:
            logger.warning("No models with any valid score data. Skipping plot_avg_scores.")
            return

        melted_df = df_to_plot.melt(
            id_vars=["model_name"],
            value_vars=metrics_to_plot,
            var_name="Metric",
            value_name="Average Score"
        ).dropna(subset=["Average Score"]) # Drop rows where the specific score is NaN

        if melted_df.empty:
            logger.info("Melted score data is empty after dropping NaNs. Skipping plot_avg_scores.")
            return
        
        # STD Dev data (optional)
        std_dev_map = {}
        use_manual_error_bars = False # Simpler to rely on seaborn's CI for grouped, will not add manual std dev here
        # for metric in metrics_to_plot:
        #     std_col = f"std_{metric.split('avg_')[1]}"
        #     if std_col in df_to_plot.columns and df_to_plot[std_col].notna().any():
        #         std_dev_map[metric] = df_to_plot.set_index('model_name')[std_col].to_dict()
        #         use_manual_error_bars = True


        # Increase figure size
        fig, ax = plt.subplots(figsize=(18, 9)) # Increased width and height
        
        # Use a slice of METRIC_PALETTE if number of metrics is less
        current_metric_palette = METRIC_PALETTE[:len(melted_df['Metric'].unique())]

        sns.barplot(ax=ax, x="model_name", y="Average Score", hue="Metric", data=melted_df,
                    palette=current_metric_palette, errorbar=('ci', 95)) # Seaborn handles CI

        ax.set_title("Average Evaluation Scores per Model", fontsize=16)
        ax.set_xlabel("Model", fontsize=12)
        ax.set_ylabel("Average Score", fontsize=12)
        ax.tick_params(axis='x', labelrotation=65, labelsize=8) # Increased rotation, smaller font
        ax.set_xticklabels(ax.get_xticklabels(), ha="right") # Align rotated labels better
        ax.legend(title="Metric Type", bbox_to_anchor=(1.02, 1), loc='upper left', title_fontsize='10', fontsize='9')
        
        plt.tight_layout(rect=[0, 0, 0.9, 1], pad=2.0) # Adjust for legend, add padding
        self._save_plot(fig, "avg_scores_per_model")

    # ... (rest of the methods: plot_speed_vs_scores, plot_corpus_perf_by_prompt, etc.)
    # Consider similar adjustments (figsize, label rotation, font size, ha="right") for other plots
    # if they also display many model names on an axis.

    def plot_speed_vs_scores(self, aggregated_df: pd.DataFrame):
        if not PLOTTING_AVAILABLE: return
        if aggregated_df.empty:
             logger.warning("Aggregated data is empty, skipping plot_speed_vs_scores.")
             return
        if "avg_tokens_per_second" not in aggregated_df.columns:
             logger.warning("Metric 'avg_tokens_per_second' not found. Skipping speed vs scores plot.")
             return
        
        score_cols = ["avg_bleu_score", "avg_rouge1", "avg_rouge2", "avg_rougeL"]
        metrics_to_plot = [col for col in score_cols if col in aggregated_df.columns and not aggregated_df[col].isnull().all()]

        if not metrics_to_plot:
            logger.info("No average BLEU or ROUGE scores with valid data found. Skipping speed vs scores plot.")
            return
        
        df_to_plot = aggregated_df.dropna(subset=["avg_tokens_per_second"] + metrics_to_plot, how='all').copy()
        if df_to_plot.empty:
            logger.warning("No models with valid data for avg_tokens_per_second or scores. Skipping speed vs scores plot.")
            return

        melted_df = df_to_plot.melt(
            id_vars=["model_name", "avg_tokens_per_second"],
            value_vars=metrics_to_plot,
            var_name="Score Metric",
            value_name="Average Score"
        ).dropna(subset=["Average Score", "avg_tokens_per_second"])
        
        if melted_df.empty:
             logger.info("No valid average scores or TPS found after filtering NaNs. Skipping speed vs scores plot.")
             return

        fig, ax = plt.subplots(figsize=(12, 8)) # Adjusted size
        sns.scatterplot(ax=ax, data=melted_df, x="avg_tokens_per_second", y="Average Score",
                        hue="model_name", style="Score Metric", s=120, palette=MODEL_PALETTE[:len(melted_df['model_name'].unique())])

        ax.set_title("Average Inference Speed vs. Average Evaluation Scores", fontsize=16)
        ax.set_xlabel("Average Tokens per Second", fontsize=12)
        ax.set_ylabel("Average Score", fontsize=12)
        ax.legend(title="Model / Metric", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, title_fontsize='10')
        plt.tight_layout(rect=[0, 0, 0.83, 1]) # Adjust for legend
        self._save_plot(fig, "avg_speed_vs_avg_scores")


    def plot_corpus_perf_by_prompt(self, all_results_df: pd.DataFrame):
        """Plots 4 & 5: Tokens/sec and response length vs prompt ID per corpus."""
        if not PLOTTING_AVAILABLE: return
        if all_results_df.empty:
             logger.warning("Results data is empty, skipping plot_corpus_perf_by_prompt.")
             return

        required_cols = ["corpus_file", "prompt_id", "model_name", "tokens_per_second", "response_length_tokens"]
        if not all(col in all_results_df.columns for col in required_cols):
             logger.warning(f"Missing one or more required columns ({required_cols}) for corpus performance plot. Skipping.")
             return

        corpora = all_results_df["corpus_file"].unique()
        num_models = len(all_results_df['model_name'].unique())

        for corpus in corpora:
            corpus_df = all_results_df[all_results_df["corpus_file"] == corpus].copy()
            # Ensure prompt_id is numeric before plotting
            corpus_df['prompt_id'] = pd.to_numeric(corpus_df['prompt_id'], errors='coerce')
            corpus_df.dropna(subset=['prompt_id'], inplace=True)


            corpus_name_safe = Path(corpus).stem

            # Plot 4: Tokens per second vs Prompt ID
            if 'tokens_per_second' in corpus_df.columns and corpus_df['tokens_per_second'].notna().any():
                fig1, ax1 = plt.subplots(figsize=(15, 7)) # Adjusted size
                sns.lineplot(ax=ax1, data=corpus_df.dropna(subset=["tokens_per_second"]), x="prompt_id", y="tokens_per_second",
                                hue="model_name", palette=MODEL_PALETTE[:num_models], marker="o", estimator=None, lw=1.5)
                ax1.set_title(f"Tokens per Second vs Prompt ID (Corpus: {corpus})", fontsize=15)
                ax1.set_xlabel("Prompt ID", fontsize=12)
                ax1.set_ylabel("Tokens per Second", fontsize=12)
                ax1.legend(title="Model", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9, title_fontsize='10')
                plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust for legend
                self._save_plot(fig1, f"corpus_{corpus_name_safe}_tps_vs_prompt_id")
            else:
                logger.info(f"No valid tokens_per_second data for corpus '{corpus}' in plot_corpus_perf_by_prompt. Skipping TPS plot.")


            # Plot 5: Response Length vs Prompt ID
            if 'response_length_tokens' in corpus_df.columns and corpus_df['response_length_tokens'].notna().any():
                fig2, ax2 = plt.subplots(figsize=(15, 7)) # Adjusted size
                sns.lineplot(ax=ax2, data=corpus_df.dropna(subset=["response_length_tokens"]), x="prompt_id", y="response_length_tokens",
                                hue="model_name", palette=MODEL_PALETTE[:num_models], marker="o", estimator=None, lw=1.5)
                ax2.set_title(f"Response Length vs Prompt ID (Corpus: {corpus})", fontsize=15)
                ax2.set_xlabel("Prompt ID", fontsize=12)
                ax2.set_ylabel("Response Length (Tokens)", fontsize=12)
                ax2.legend(title="Model", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9, title_fontsize='10')
                plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust for legend
                self._save_plot(fig2, f"corpus_{corpus_name_safe}_length_vs_prompt_id")
            else:
                logger.info(f"No valid response_length_tokens data for corpus '{corpus}' in plot_corpus_perf_by_prompt. Skipping length plot.")


    def plot_model_speed_vs_bleu(self, model_results_df: pd.DataFrame, model_output_dir: Path):
        if not PLOTTING_AVAILABLE: return
        model_name = model_results_df['model_name'].iloc[0] if not model_results_df.empty else "Unknown"
        plot_subdir = model_output_dir / "plots"

        required_cols = ["tokens_per_second", "bleu_score", "corpus_file"]
        if not all(col in model_results_df.columns for col in required_cols) or \
           model_results_df["bleu_score"].isnull().all() or \
           model_results_df["tokens_per_second"].isnull().all():
            logger.warning(f"Missing or all-null BLEU/TPS data for model {model_name}. Skipping speed vs BLEU plot.")
            return

        plot_df = model_results_df.dropna(subset=required_cols)
        if plot_df.empty:
            logger.info(f"No valid data points for model {model_name} speed vs BLEU plot after dropna. Skipping.")
            return

        fig, ax = plt.subplots(figsize=(10, 7)) # Adjusted size
        sns.scatterplot(ax=ax, data=plot_df, x="tokens_per_second", y="bleu_score", hue="corpus_file", s=70)
        ax.set_title(f"Model: {model_name} - Tokens/Sec vs BLEU Score", fontsize=14)
        ax.set_xlabel("Tokens per Second", fontsize=12)
        ax.set_ylabel("BLEU Score", fontsize=12)
        ax.legend(title="Corpus", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, title_fontsize='10')
        plt.tight_layout(rect=[0, 0, 0.83, 1])
        self._save_plot(fig, f"{model_name}_speed_vs_bleu", subdirectory=plot_subdir)


    def plot_model_speed_vs_rouge(self, model_results_df: pd.DataFrame, model_output_dir: Path):
        if not PLOTTING_AVAILABLE: return
        model_name = model_results_df['model_name'].iloc[0] if not model_results_df.empty else "Unknown"
        plot_subdir = model_output_dir / "plots"

        rouge_cols = ["rouge1_fmeasure", "rouge2_fmeasure", "rougeL_fmeasure"]
        value_vars = [col for col in rouge_cols if col in model_results_df.columns and model_results_df[col].notna().any()]

        if not value_vars or "tokens_per_second" not in model_results_df.columns or model_results_df["tokens_per_second"].isnull().all():
            logger.warning(f"No ROUGE f-measure columns with data or missing TPS for model {model_name}. Skipping speed vs ROUGE plot.")
            return

        id_vars = ["prompt_id", "tokens_per_second", "corpus_file"]
        melted_df = model_results_df.melt(
            id_vars=id_vars,
            value_vars=value_vars,
            var_name="ROUGE Type",
            value_name="F-Measure"
        ).dropna(subset=["tokens_per_second", "F-Measure"])

        if melted_df.empty:
            logger.info(f"No valid data points for model {model_name} speed vs ROUGE plot after melt/dropna. Skipping.")
            return

        fig, ax = plt.subplots(figsize=(10, 7)) # Adjusted size
        sns.scatterplot(ax=ax, data=melted_df, x="tokens_per_second", y="F-Measure",
                        hue="ROUGE Type", style="corpus_file", s=70, palette=METRIC_PALETTE[1:len(value_vars)+1])
        ax.set_title(f"Model: {model_name} - Tokens/Sec vs ROUGE F-Measure", fontsize=14)
        ax.set_xlabel("Tokens per Second", fontsize=12)
        ax.set_ylabel("ROUGE F-Measure", fontsize=12)
        ax.legend(title="ROUGE / Corpus", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, title_fontsize='10')
        plt.tight_layout(rect=[0, 0, 0.83, 1])
        self._save_plot(fig, f"{model_name}_speed_vs_rouge", subdirectory=plot_subdir)


    def plot_model_perf_by_prompt(self, model_results_df: pd.DataFrame, model_output_dir: Path):
        if not PLOTTING_AVAILABLE: return
        model_name = model_results_df['model_name'].iloc[0] if not model_results_df.empty else "Unknown"
        plot_subdir = model_output_dir / "plots"

        model_results_df['prompt_id'] = pd.to_numeric(model_results_df['prompt_id'], errors='coerce')
        required_cols = ["prompt_id", "response_length_tokens", "tokens_per_second", "corpus_file"]
        
        plot_df = model_results_df.dropna(subset=required_cols, how='any') # Drop if any of these are NaN

        if plot_df.empty:
            logger.info(f"No valid data points for model {model_name} performance vs prompt ID plot after dropna. Skipping.")
            return

        fig, ax1 = plt.subplots(figsize=(14, 7)) # Adjusted size
        fig.suptitle(f"Model: {model_name} - Performance vs Prompt ID", fontsize=16)

        color1 = 'tab:blue'
        ax1.set_xlabel('Prompt ID', fontsize=12)
        ax1.set_ylabel('Response Length (Tokens)', color=color1, fontsize=12)
        sns.lineplot(ax=ax1, data=plot_df, x='prompt_id', y='response_length_tokens',
                        hue='corpus_file', palette='coolwarm', estimator=None, lw=1, marker='o', markersize=4)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.get_legend().remove()

        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('Tokens per Second', color=color2, fontsize=12)
        sns.lineplot(ax=ax2, data=plot_df, x='prompt_id', y='tokens_per_second',
                        hue='corpus_file', palette='viridis', estimator=None, lw=1, marker='x', markersize=5)
        ax2.tick_params(axis='y', labelcolor=color2)
        
        # Create combined legend
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        
        # To avoid duplicate labels for corpus_file if they are the same
        from collections import OrderedDict
        unique_handles_labels = OrderedDict()
        for h, l in zip(handles1, labels1): unique_handles_labels[f"{l} (Length)"] = h
        for h, l in zip(handles2, labels2): unique_handles_labels[f"{l} (TPS)"] = h
        
        ax2.legend(unique_handles_labels.values(), unique_handles_labels.keys(), title="Corpus / Metric", bbox_to_anchor=(1.18, 1), loc='upper left', fontsize=9, title_fontsize='10')

        plt.tight_layout(rect=[0, 0.03, 0.82, 0.95]) # Adjust for legend
        self._save_plot(fig, f"{model_name}_perf_vs_prompt_id", subdirectory=plot_subdir)


    def plot_model_scores_by_prompt(self, model_results_df: pd.DataFrame, model_output_dir: Path):
        if not PLOTTING_AVAILABLE: return
        model_name = model_results_df['model_name'].iloc[0] if not model_results_df.empty else "Unknown"
        plot_subdir = model_output_dir / "plots"

        model_results_df['prompt_id'] = pd.to_numeric(model_results_df['prompt_id'], errors='coerce')
        score_cols_available = ["bleu_score"] + [col for col in ["rouge1_fmeasure", "rouge2_fmeasure", "rougeL_fmeasure"] if col in model_results_df.columns]
        value_vars = [col for col in score_cols_available if col in model_results_df.columns and model_results_df[col].notna().any()]

        if not value_vars or "prompt_id" not in model_results_df.columns or model_results_df['prompt_id'].isnull().all():
            logger.warning(f"No valid score columns with data or missing prompt_id for model {model_name}. Skipping scores vs prompt ID plot.")
            return

        id_vars = ["prompt_id", "corpus_file"]
        melted_df = model_results_df.melt(
            id_vars=id_vars,
            value_vars=value_vars,
            var_name="Score Type",
            value_name="Score"
        ).dropna(subset=["Score", "prompt_id"])

        if melted_df.empty:
            logger.info(f"No valid score data points for model {model_name} vs prompt ID plot after melt/dropna. Skipping.")
            return

        fig, ax = plt.subplots(figsize=(14, 7)) # Adjusted size
        sns.lineplot(ax=ax, data=melted_df, x="prompt_id", y="Score",
                        hue="Score Type", style="corpus_file", palette=METRIC_PALETTE[:len(value_vars)], estimator=None, lw=1, marker='o', markersize=4)
        ax.set_title(f"Model: {model_name} - Evaluation Scores vs Prompt ID", fontsize=15)
        ax.set_xlabel("Prompt ID", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.legend(title="Score / Corpus", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, title_fontsize='10')
        plt.tight_layout(rect=[0, 0, 0.83, 1])
        self._save_plot(fig, f"{model_name}_scores_vs_prompt_id", subdirectory=plot_subdir)