# ============================================================================
# PHASE 2: DESCRIPTIVE STATISTICS
# ============================================================================
# Project: 3D Printer Dataset Analysis
# Phase: Descriptive Statistics (Histogram, Boxplot, Scatter Plot)
# Author: Statistical Analysis Team
# Date: 2025-08-11
# ============================================================================

# Load required libraries
library(ggplot2)      # For advanced plotting
library(dplyr)        # For data manipulation
library(GGally)       # For correlation matrix and pair plots
library(gridExtra)    # For arranging multiple plots
library(corrplot)     # For correlation visualization
library(reshape2)     # For data reshaping

cat("=== PHASE 2: DESCRIPTIVE STATISTICS ===\n")
cat("Loading cleaned dataset...\n")

# Load cleaned dataset from Phase 1
data_cleaned <- read.csv("/home/kari/prob-stat-pj/3dprintdata/data_cleaned.csv")

# Convert categorical variables back to factors
data_cleaned$infill_pattern <- as.factor(data_cleaned$infill_pattern)
data_cleaned$material <- as.factor(data_cleaned$material)

cat("Dataset loaded successfully!\n")
cat("Dataset dimensions:", nrow(data_cleaned), "rows x", ncol(data_cleaned), "columns\n\n")

# Create graphics directory if it doesn't exist
if (!dir.exists("/home/kari/prob-stat-pj/graphics")) {
  dir.create("/home/kari/prob-stat-pj/graphics", recursive = TRUE)
}

# ============================================================================
# 1. DESCRIPTIVE STATISTICS SUMMARY
# ============================================================================
cat("1. CALCULATING DESCRIPTIVE STATISTICS\n")
cat("=====================================\n")

# Function to calculate comprehensive descriptive statistics
calc_descriptive_stats <- function(x, var_name) {
  # Calculate skewness manually
  n <- length(x)
  x_mean <- mean(x, na.rm = TRUE)
  x_sd <- sd(x, na.rm = TRUE)
  skewness <- sum((x - x_mean)^3, na.rm = TRUE) / ((n - 1) * x_sd^3)
  
  # Calculate kurtosis manually
  kurtosis <- sum((x - x_mean)^4, na.rm = TRUE) / ((n - 1) * x_sd^4) - 3
  
  stats <- data.frame(
    Variable = var_name,
    Count = length(x),
    Mean = round(mean(x, na.rm = TRUE), 3),
    Median = round(median(x, na.rm = TRUE), 3),
    SD = round(sd(x, na.rm = TRUE), 3),
    Variance = round(var(x, na.rm = TRUE), 3),
    Min = round(min(x, na.rm = TRUE), 3),
    Max = round(max(x, na.rm = TRUE), 3),
    Range = round(max(x, na.rm = TRUE) - min(x, na.rm = TRUE), 3),
    Q1 = round(quantile(x, 0.25, na.rm = TRUE), 3),
    Q3 = round(quantile(x, 0.75, na.rm = TRUE), 3),
    IQR = round(IQR(x, na.rm = TRUE), 3),
    Skewness = round(skewness, 3),
    Kurtosis = round(kurtosis, 3)
  )
  return(stats)
}

# Calculate descriptive statistics for all continuous variables
continuous_vars <- c("layer_height", "wall_thickness", "infill_density", 
                    "nozzle_temperature", "bed_temperature", "print_speed", 
                    "fan_speed", "roughness", "tension_strenght", "elongation")

descriptive_stats <- data.frame()
for (var in continuous_vars) {
  stats <- calc_descriptive_stats(data_cleaned[[var]], var)
  descriptive_stats <- rbind(descriptive_stats, stats)
}

print(descriptive_stats)

# ============================================================================
# 2. HISTOGRAM ANALYSIS
# ============================================================================
cat("\n2. CREATING HISTOGRAMS\n")
cat("======================\n")

# Create histograms for output variables (quality metrics)
output_vars <- c("roughness", "tension_strenght", "elongation")
output_labels <- c("Roughness (μm)", "Tension Strength (MPa)", "Elongation (%)")

for (i in 1:length(output_vars)) {
  var <- output_vars[i]
  label <- output_labels[i]
  
  p <- ggplot(data_cleaned, aes_string(x = var)) +
    geom_histogram(bins = 10, fill = "steelblue", color = "black", alpha = 0.7) +
    geom_density(aes(y = ..density.. * nrow(data_cleaned) * diff(range(data_cleaned[[var]])) / 10), 
                 color = "red", size = 1) +
    labs(title = paste("Distribution of", label),
         subtitle = paste("Mean =", round(mean(data_cleaned[[var]]), 2), 
                         ", SD =", round(sd(data_cleaned[[var]]), 2)),
         x = label,
         y = "Frequency") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
          plot.subtitle = element_text(hjust = 0.5, size = 12))
  
  filename <- paste0("/home/kari/prob-stat-pj/graphics/04-histogram_", var, ".png")
  ggsave(filename, plot = p, width = 8, height = 6, dpi = 300)
  cat("Saved:", filename, "\n")
}

# Create histograms for key input variables
input_vars <- c("layer_height", "nozzle_temperature", "print_speed", "infill_density")
input_labels <- c("Layer Height (mm)", "Nozzle Temperature (°C)", "Print Speed (mm/s)", "Infill Density (%)")

for (i in 1:length(input_vars)) {
  var <- input_vars[i]
  label <- input_labels[i]
  
  p <- ggplot(data_cleaned, aes_string(x = var)) +
    geom_histogram(bins = 8, fill = "lightgreen", color = "black", alpha = 0.7) +
    geom_density(aes(y = ..density.. * nrow(data_cleaned) * diff(range(data_cleaned[[var]])) / 8), 
                 color = "darkgreen", size = 1) +
    labs(title = paste("Distribution of", label),
         subtitle = paste("Mean =", round(mean(data_cleaned[[var]]), 2), 
                         ", SD =", round(sd(data_cleaned[[var]]), 2)),
         x = label,
         y = "Frequency") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
          plot.subtitle = element_text(hjust = 0.5, size = 12))
  
  filename <- paste0("/home/kari/prob-stat-pj/graphics/04-histogram_", var, ".png")
  ggsave(filename, plot = p, width = 8, height = 6, dpi = 300)
  cat("Saved:", filename, "\n")
}

# ============================================================================
# 3. BOXPLOT ANALYSIS
# ============================================================================
cat("\n3. CREATING BOXPLOTS\n")
cat("====================\n")

# Boxplots for output variables by infill_pattern
for (i in 1:length(output_vars)) {
  var <- output_vars[i]
  label <- output_labels[i]
  
  p <- ggplot(data_cleaned, aes_string(x = "infill_pattern", y = var, fill = "infill_pattern")) +
    geom_boxplot(alpha = 0.7, outlier.color = "red", outlier.size = 2) +
    geom_jitter(width = 0.2, alpha = 0.5, size = 1.5) +
    scale_fill_manual(values = c("grid" = "lightblue", "honeycomb" = "lightcoral")) +
    labs(title = paste(label, "by Infill Pattern"),
         subtitle = "Comparison between Grid and Honeycomb patterns",
         x = "Infill Pattern",
         y = label,
         fill = "Infill Pattern") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
          plot.subtitle = element_text(hjust = 0.5, size = 12),
          legend.position = "bottom")
  
  filename <- paste0("/home/kari/prob-stat-pj/graphics/04-boxplot_", var, "_by_infill_pattern.png")
  ggsave(filename, plot = p, width = 8, height = 6, dpi = 300)
  cat("Saved:", filename, "\n")
}

# Boxplots for output variables by material
for (i in 1:length(output_vars)) {
  var <- output_vars[i]
  label <- output_labels[i]
  
  p <- ggplot(data_cleaned, aes_string(x = "material", y = var, fill = "material")) +
    geom_boxplot(alpha = 0.7, outlier.color = "red", outlier.size = 2) +
    geom_jitter(width = 0.2, alpha = 0.5, size = 1.5) +
    scale_fill_manual(values = c("abs" = "gold", "pla" = "lightgreen")) +
    labs(title = paste(label, "by Material Type"),
         subtitle = "Comparison between ABS and PLA materials",
         x = "Material Type",
         y = label,
         fill = "Material") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
          plot.subtitle = element_text(hjust = 0.5, size = 12),
          legend.position = "bottom")
  
  filename <- paste0("/home/kari/prob-stat-pj/graphics/04-boxplot_", var, "_by_material.png")
  ggsave(filename, plot = p, width = 8, height = 6, dpi = 300)
  cat("Saved:", filename, "\n")
}

# ============================================================================
# 4. SCATTER PLOT ANALYSIS
# ============================================================================
cat("\n4. CREATING SCATTER PLOTS\n")
cat("==========================\n")

# Key input variables for scatter plots
key_inputs <- c("layer_height", "nozzle_temperature", "print_speed", "infill_density")
key_input_labels <- c("Layer Height (mm)", "Nozzle Temperature (°C)", "Print Speed (mm/s)", "Infill Density (%)")

# Create scatter plots for each output vs key inputs
for (i in 1:length(output_vars)) {
  output_var <- output_vars[i]
  output_label <- output_labels[i]
  
  for (j in 1:length(key_inputs)) {
    input_var <- key_inputs[j]
    input_label <- key_input_labels[j]
    
    p <- ggplot(data_cleaned, aes_string(x = input_var, y = output_var)) +
      geom_point(aes(color = material, shape = infill_pattern), size = 3, alpha = 0.7) +
      geom_smooth(method = "lm", se = TRUE, color = "blue", linetype = "dashed") +
      scale_color_manual(values = c("abs" = "red", "pla" = "green")) +
      scale_shape_manual(values = c("grid" = 16, "honeycomb" = 17)) +
      labs(title = paste(output_label, "vs", input_label),
           subtitle = paste("Correlation:", round(cor(data_cleaned[[input_var]], data_cleaned[[output_var]]), 3)),
           x = input_label,
           y = output_label,
           color = "Material",
           shape = "Infill Pattern") +
      theme_minimal() +
      theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
            plot.subtitle = element_text(hjust = 0.5, size = 12),
            legend.position = "bottom")
    
    filename <- paste0("/home/kari/prob-stat-pj/graphics/04-scatter_", output_var, "_vs_", input_var, ".png")
    ggsave(filename, plot = p, width = 8, height = 6, dpi = 300)
    cat("Saved:", filename, "\n")
  }
}

# ============================================================================
# 5. CORRELATION MATRIX
# ============================================================================
cat("\n5. CREATING CORRELATION MATRIX\n")
cat("===============================\n")

# Calculate correlation matrix for continuous variables
corr_data <- data_cleaned[continuous_vars]
corr_matrix <- cor(corr_data)

# Create correlation heatmap
png("/home/kari/prob-stat-pj/graphics/04-correlation_matrix.png", width = 12, height = 10, units = "in", res = 300)
corrplot(corr_matrix, method = "color", type = "upper", order = "hclust",
         tl.cex = 0.8, tl.col = "black", tl.srt = 45,
         addCoef.col = "black", number.cex = 0.7,
         title = "Correlation Matrix of 3D Printing Parameters",
         mar = c(0,0,2,0))
dev.off()
cat("Saved: /home/kari/prob-stat-pj/graphics/04-correlation_matrix.png\n")

# Print correlation matrix
cat("\nCorrelation Matrix:\n")
print(round(corr_matrix, 3))

# ============================================================================
# 6. SUMMARY STATISTICS BY GROUPS
# ============================================================================
cat("\n6. SUMMARY STATISTICS BY GROUPS\n")
cat("================================\n")

# Summary by infill_pattern
cat("\nSummary by Infill Pattern:\n")
summary_by_infill <- data_cleaned %>%
  group_by(infill_pattern) %>%
  summarise(
    Count = n(),
    Roughness_Mean = round(mean(roughness), 2),
    Roughness_SD = round(sd(roughness), 2),
    Tension_Mean = round(mean(tension_strenght), 2),
    Tension_SD = round(sd(tension_strenght), 2),
    Elongation_Mean = round(mean(elongation), 2),
    Elongation_SD = round(sd(elongation), 2),
    .groups = 'drop'
  )
print(summary_by_infill)

# Summary by material
cat("\nSummary by Material:\n")
summary_by_material <- data_cleaned %>%
  group_by(material) %>%
  summarise(
    Count = n(),
    Roughness_Mean = round(mean(roughness), 2),
    Roughness_SD = round(sd(roughness), 2),
    Tension_Mean = round(mean(tension_strenght), 2),
    Tension_SD = round(sd(tension_strenght), 2),
    Elongation_Mean = round(mean(elongation), 2),
    Elongation_SD = round(sd(elongation), 2),
    .groups = 'drop'
  )
print(summary_by_material)

# ============================================================================
# EXPORT LOG TO TXT FILE
# ============================================================================
# Create logs directory if it doesn't exist
if (!dir.exists("/home/kari/prob-stat-pj/logs")) {
  dir.create("/home/kari/prob-stat-pj/logs", recursive = TRUE)
}

# Capture all output and save to log file
log_file <- "/home/kari/prob-stat-pj/logs/02_descriptive_statistics_log.txt"
sink(log_file)

cat("=== DESCRIPTIVE STATISTICS LOG ===")
cat("\nGenerated on:", as.character(Sys.time()))
cat("\n\n")

cat("1. COMPREHENSIVE DESCRIPTIVE STATISTICS:\n")
print(descriptive_stats)

cat("\n\n2. CORRELATION MATRIX:\n")
print(round(corr_matrix, 3))

cat("\n\n3. SUMMARY BY INFILL PATTERN:\n")
print(summary_by_infill)

cat("\n\n4. SUMMARY BY MATERIAL:\n")
print(summary_by_material)

cat("\n\n5. GRAPHICS GENERATED:\n")
cat("   - 7 Histograms (3 output + 4 input variables)\n")
cat("   - 6 Boxplots (3 output variables × 2 grouping factors)\n")
cat("   - 12 Scatter plots (3 output × 4 input variables)\n")
cat("   - 1 Correlation matrix heatmap\n")
cat("   Total: 26 graphics files\n")

cat("\n\n6. KEY FINDINGS:\n")
cat("   - All graphics saved to ../graphics/ with prefix '04-'\n")
cat("   - Correlation analysis completed for all continuous variables\n")
cat("   - Group comparisons performed for categorical variables\n")
cat("   - Comprehensive descriptive statistics calculated\n")

sink()

cat("\nLog exported to:", log_file, "\n")

# ============================================================================
# PHASE 2 COMPLETED: DESCRIPTIVE STATISTICS
# ============================================================================
cat("\n=== PHASE 2 COMPLETED SUCCESSFULLY ===\n")
cat("Descriptive statistics analysis finished.\n")
cat("Generated 26 graphics files and comprehensive statistical summaries.\n")
cat("Ready for Phase 3: Statistical Inference (Multiple Linear Regression)\n")