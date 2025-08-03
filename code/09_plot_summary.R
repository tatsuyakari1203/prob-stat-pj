# =============================================================================
# 09_plot_summary.R
# Summary of All Generated Plots
# =============================================================================

cat("=== PLOT SUMMARY REPORT ===\n")

# List all plot files in graphics directory
graphics_dir <- "../graphics"
plot_files <- list.files(graphics_dir, pattern = "\\.png$", full.names = FALSE)

cat("\n--- GENERATED PLOTS ---\n")
cat("Total plots generated:", length(plot_files), "\n\n")

# Categorize plots
eda_plots <- plot_files[grepl("boxplots|histograms|scatterplots|target_distribution", plot_files)]
model_plots <- plot_files[grepl("knn_|dt_|rf_", plot_files)]
comparison_plots <- plot_files[grepl("model_performance|all_confusion", plot_files)]
other_plots <- plot_files[!plot_files %in% c(eda_plots, model_plots, comparison_plots)]

# EDA Plots
cat("1. EXPLORATORY DATA ANALYSIS PLOTS:\n")
for(plot in eda_plots) {
  cat("   -", plot, "\n")
}
cat("   Purpose: Understanding data distribution and relationships\n\n")

# Model-specific plots
cat("2. MODEL-SPECIFIC PLOTS:\n")

# k-NN plots
knn_plots <- model_plots[grepl("knn_", model_plots)]
if(length(knn_plots) > 0) {
  cat("   k-NN Model:\n")
  for(plot in knn_plots) {
    if(grepl("k_tuning", plot)) {
      cat("     -", plot, "(k-value optimization)\n")
    } else if(grepl("confusion", plot)) {
      cat("     -", plot, "(performance evaluation)\n")
    }
  }
}

# Decision Tree plots
dt_plots <- model_plots[grepl("dt_", model_plots)]
if(length(dt_plots) > 0) {
  cat("   Decision Tree Model:\n")
  for(plot in dt_plots) {
    if(grepl("confusion", plot)) {
      cat("     -", plot, "(performance evaluation)\n")
    } else if(grepl("feature_importance", plot)) {
      cat("     -", plot, "(feature analysis)\n")
    }
  }
}

# Random Forest plots
rf_plots <- model_plots[grepl("rf_", model_plots)]
if(length(rf_plots) > 0) {
  cat("   Random Forest Model:\n")
  for(plot in rf_plots) {
    if(grepl("confusion", plot)) {
      cat("     -", plot, "(performance evaluation)\n")
    } else if(grepl("feature_importance", plot)) {
      cat("     -", plot, "(feature analysis)\n")
    }
  }
}
cat("\n")

# Comparison plots
cat("3. MODEL COMPARISON PLOTS:\n")
for(plot in comparison_plots) {
  if(grepl("model_performance", plot)) {
    cat("   -", plot, "(metrics comparison across all models)\n")
  } else if(grepl("all_confusion", plot)) {
    cat("   -", plot, "(side-by-side confusion matrices)\n")
  }
}
cat("\n")

# Other plots
if(length(other_plots) > 0) {
  cat("4. OTHER PLOTS:\n")
  for(plot in other_plots) {
    cat("   -", plot, "\n")
  }
  cat("\n")
}

# Plot usage recommendations
cat("--- PLOT USAGE RECOMMENDATIONS ---\n")
cat("\nFor LaTeX Report Integration:\n")
cat("1. Use EDA plots in the 'Data Exploration' section\n")
cat("2. Include model-specific plots in respective methodology sections\n")
cat("3. Use comparison plots in the 'Results and Analysis' section\n")
cat("4. All plots are saved as PNG files for easy LaTeX inclusion\n")

cat("\nLaTeX Integration Example:\n")
cat("\\begin{figure}[h]\n")
cat("  \\centering\n")
cat("  \\includegraphics[width=0.8\\textwidth]{graphics/model_performance_comparison.png}\n")
cat("  \\caption{Model Performance Comparison}\n")
cat("  \\label{fig:model_comparison}\n")
cat("\\end{figure}\n\n")

# Technical details
cat("--- TECHNICAL DETAILS ---\n")
cat("Plot Resolution: 800x600 (standard plots), 1000x600 (comparison plots)\n")
cat("File Format: PNG (suitable for LaTeX)\n")
cat("Color Scheme: Professional colors (blues, greens, corals)\n")
cat("Font Size: Optimized for readability in academic reports\n")

cat("\n=== PLOT SUMMARY COMPLETED ===\n")