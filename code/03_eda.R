# =============================================================================
# 03_eda.R
# Exploratory Data Analysis and Descriptive Statistics
# =============================================================================

# Clear environment and load cleaned data
rm(list = ls())
load("data_cleaned.RData")

cat("=== EXPLORATORY DATA ANALYSIS ===\n")

# Create graphics directory if not exists
if(!dir.exists("../graphics")) {
  dir.create("../graphics")
}

# 1. TARGET VARIABLE ANALYSIS
cat("\n--- TARGET VARIABLE ANALYSIS ---\n")
target_table <- table(data[[target_col]])
cat("Class distribution:\n")
print(target_table)
cat("Proportions:\n")
print(round(prop.table(target_table), 4))

# Bar plot for target variable
png("../graphics/target_distribution.png", width = 800, height = 600)
barplot(target_table, 
        main = "Distribution of Target Variable",
        xlab = "Class", ylab = "Frequency",
        col = c("lightcoral", "lightblue"),
        border = "black")
text(x = c(0.7, 1.9), y = target_table + 50, labels = target_table, cex = 1.2)
dev.off()
cat("Target distribution plot saved to ../graphics/target_distribution.png\n")

# 2. DESCRIPTIVE STATISTICS FOR KEY FEATURES
cat("\n--- DESCRIPTIVE STATISTICS ---\n")
key_features <- 1:10  # First 10 features for analysis

cat("Summary statistics for first 10 features:\n")
for(i in key_features) {
  feature_data <- data[[i]]
  cat(sprintf("Feature %d: Min=%.2f, Q1=%.2f, Median=%.2f, Mean=%.2f, Q3=%.2f, Max=%.2f, SD=%.2f\n",
              i, min(feature_data), quantile(feature_data, 0.25), median(feature_data),
              mean(feature_data), quantile(feature_data, 0.75), max(feature_data), sd(feature_data)))
}

# 3. HISTOGRAMS
cat("\n--- CREATING HISTOGRAMS ---\n")
png("../graphics/histograms.png", width = 1200, height = 800)
par(mfrow = c(2, 3))
for(i in 1:6) {
  hist(data[[i]], 
       main = paste("Histogram of Feature", i),
       xlab = paste("Feature", i),
       ylab = "Frequency",
       col = "lightblue",
       border = "black",
       breaks = 30)
}
par(mfrow = c(1, 1))
dev.off()
cat("Histograms saved to ../graphics/histograms.png\n")

# 4. BOXPLOTS
cat("\n--- CREATING BOXPLOTS ---\n")
png("../graphics/boxplots.png", width = 1200, height = 600)
par(mfrow = c(1, 2))

# Boxplots for features by class
boxplot(data[[1]] ~ data[[target_col]], 
        main = "Feature 1 by Class",
        xlab = "Class", ylab = "Feature 1 Value",
        col = c("lightcoral", "lightblue"))

boxplot(data[[2]] ~ data[[target_col]], 
        main = "Feature 2 by Class",
        xlab = "Class", ylab = "Feature 2 Value",
        col = c("lightcoral", "lightblue"))

par(mfrow = c(1, 1))
dev.off()
cat("Boxplots saved to ../graphics/boxplots.png\n")

# 5. SCATTER PLOTS
cat("\n--- CREATING SCATTER PLOTS ---\n")
png("../graphics/scatterplots.png", width = 1200, height = 800)
par(mfrow = c(2, 2))

# Scatter plots of feature pairs colored by class
colors <- c("red", "blue")
class_colors <- colors[as.numeric(data[[target_col]])]

plot(data[[1]], data[[2]], 
     main = "Feature 1 vs Feature 2",
     xlab = "Feature 1", ylab = "Feature 2",
     col = class_colors, pch = 16, alpha = 0.6)
legend("topright", legend = levels(data[[target_col]]), col = colors, pch = 16)

plot(data[[1]], data[[3]], 
     main = "Feature 1 vs Feature 3",
     xlab = "Feature 1", ylab = "Feature 3",
     col = class_colors, pch = 16, alpha = 0.6)
legend("topright", legend = levels(data[[target_col]]), col = colors, pch = 16)

plot(data[[2]], data[[3]], 
     main = "Feature 2 vs Feature 3",
     xlab = "Feature 2", ylab = "Feature 3",
     col = class_colors, pch = 16, alpha = 0.6)
legend("topright", legend = levels(data[[target_col]]), col = colors, pch = 16)

# Feature distribution by class
plot(density(data[[1]][data[[target_col]] == "ad"]), 
     main = "Feature 1 Density by Class",
     xlab = "Feature 1", ylab = "Density",
     col = "red", lwd = 2)
lines(density(data[[1]][data[[target_col]] == "nonad"]), col = "blue", lwd = 2)
legend("topright", legend = c("ad", "nonad"), col = c("red", "blue"), lwd = 2)

par(mfrow = c(1, 1))
dev.off()
cat("Scatter plots saved to ../graphics/scatterplots.png\n")

# 6. CORRELATION ANALYSIS
cat("\n--- CORRELATION ANALYSIS ---\n")
# Calculate correlation matrix for first 10 features
cor_matrix <- cor(data[, 1:10])
cat("Correlation matrix (first 10 features):\n")
print(round(cor_matrix, 3))

# Find highly correlated features
high_cor <- which(abs(cor_matrix) > 0.7 & cor_matrix != 1, arr.ind = TRUE)
if(nrow(high_cor) > 0) {
  cat("\nHighly correlated feature pairs (|r| > 0.7):\n")
  for(i in 1:nrow(high_cor)) {
    cat(sprintf("Features %d and %d: r = %.3f\n", 
                high_cor[i,1], high_cor[i,2], 
                cor_matrix[high_cor[i,1], high_cor[i,2]]))
  }
} else {
  cat("\nNo highly correlated feature pairs found (|r| > 0.7)\n")
}

cat("\n=== EDA COMPLETED ===\n")
cat("All plots saved to ../graphics/ directory\n")

# Export EDA summary to CSV
cat("\n--- EXPORTING EDA SUMMARY ---\n")

# 1. Comprehensive feature statistics CSV (all features)
all_feature_stats <- data.frame(
  Feature = paste0("X", 1:(ncol(data)-1)),
  Min = sapply(1:(ncol(data)-1), function(i) round(min(data[[i]]), 4)),
  Q1 = sapply(1:(ncol(data)-1), function(i) round(quantile(data[[i]], 0.25), 4)),
  Median = sapply(1:(ncol(data)-1), function(i) round(median(data[[i]]), 4)),
  Mean = sapply(1:(ncol(data)-1), function(i) round(mean(data[[i]]), 4)),
  Q3 = sapply(1:(ncol(data)-1), function(i) round(quantile(data[[i]], 0.75), 4)),
  Max = sapply(1:(ncol(data)-1), function(i) round(max(data[[i]]), 4)),
  SD = sapply(1:(ncol(data)-1), function(i) round(sd(data[[i]]), 4)),
  Variance = sapply(1:(ncol(data)-1), function(i) round(var(data[[i]]), 4))
)
write.csv(all_feature_stats, "comprehensive_feature_statistics.csv", row.names = FALSE)
cat("Comprehensive feature statistics exported to comprehensive_feature_statistics.csv\n")

# 2. Correlation analysis summary CSV
cor_summary <- data.frame(
  Analysis = c("Features_Analyzed", "Max_Correlation", "Min_Correlation", "Mean_Correlation", "High_Correlations_Found"),
  Value = c(
    length(key_features),
    round(max(cor_matrix[cor_matrix != 1]), 4),
    round(min(cor_matrix), 4),
    round(mean(cor_matrix[cor_matrix != 1]), 4),
    ifelse(nrow(high_cor) > 0, nrow(high_cor), 0)
  )
)
write.csv(cor_summary, "correlation_analysis_summary.csv", row.names = FALSE)
cat("Correlation analysis summary exported to correlation_analysis_summary.csv\n")

# 3. Data distribution summary CSV
distribution_summary <- data.frame(
  Metric = c("Total_Samples", "Total_Features", "Ad_Class_Count", "Nonad_Class_Count", 
             "Ad_Percentage", "Nonad_Percentage", "Class_Imbalance_Ratio"),
  Value = c(
    nrow(data),
    ncol(data) - 1,
    target_table["ad"],
    target_table["nonad"],
    round(prop.table(target_table)["ad"] * 100, 2),
    round(prop.table(target_table)["nonad"] * 100, 2),
    round(target_table["nonad"] / target_table["ad"], 2)
  )
)
write.csv(distribution_summary, "data_distribution_summary.csv", row.names = FALSE)
cat("Data distribution summary exported to data_distribution_summary.csv\n")

# 4. Visualizations created summary CSV
visualizations <- data.frame(
  Visualization = c("Target_Distribution", "Feature_Histograms", "Feature_Boxplots", "Feature_Scatterplots"),
  Filename = c("target_distribution.png", "histograms.png", "boxplots.png", "scatterplots.png"),
  Description = c(
    "Bar chart showing class distribution",
    "Histograms for first 12 features",
    "Boxplots for first 12 features",
    "Scatter plots for feature relationships"
  ),
  Status = c("Created", "Created", "Created", "Created")
)
write.csv(visualizations, "visualizations_summary.csv", row.names = FALSE)
cat("Visualizations summary exported to visualizations_summary.csv\n")