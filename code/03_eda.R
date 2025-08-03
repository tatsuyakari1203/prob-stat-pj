# =============================================================================
# 03_eda.R
# Exploratory Data Analysis and Descriptive Statistics
# =============================================================================

# --- SETUP ---
# Clear environment, load cleaned data, and set up graphics directory
rm(list = ls())
load("data_cleaned.RData")
if(!dir.exists("../graphics")) {
  dir.create("../graphics")
}


# --- CORE LOGIC: PLOT GENERATION ---
# This section contains the core logic for creating all visualizations for the report.

# 1. Bar plot for target variable distribution
png("../graphics/03-eda-target_distribution.png", width = 800, height = 600)
target_table <- table(data[[target_col]])
barplot(target_table, 
        main = "Distribution of Target Variable",
        xlab = "Class", ylab = "Frequency",
        col = c("lightcoral", "lightblue"),
        border = "black")
text(x = c(0.7, 1.9), y = target_table + 50, labels = target_table, cex = 1.2)
dev.off()

# 2. Histograms for the first 6 features
png("../graphics/03-eda-histograms.png", width = 1200, height = 800)
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

# 3. Boxplots of features 1 and 2, grouped by target class
png("../graphics/03-eda-boxplots.png", width = 1200, height = 600)
par(mfrow = c(1, 2))
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

# 4. Scatter plots and density plots
png("../graphics/03-eda-scatter_density_plots.png", width = 1200, height = 800)
par(mfrow = c(2, 2))
colors <- c("red", "blue")
class_colors <- colors[as.numeric(data[[target_col]])]
plot(data[[1]], data[[2]], main = "Feature 1 vs Feature 2", xlab = "Feature 1", ylab = "Feature 2", col = class_colors, pch = 16)
legend("topright", legend = levels(data[[target_col]]), col = colors, pch = 16)
plot(data[[1]], data[[3]], main = "Feature 1 vs Feature 3", xlab = "Feature 1", ylab = "Feature 3", col = class_colors, pch = 16)
plot(density(data[[1]][data[[target_col]] == "ad"]), main = "Feature 1 Density by Class", xlab = "Feature 1", col = "red", lwd = 2)
lines(density(data[[1]][data[[target_col]] == "nonad"]), col = "blue", lwd = 2)
legend("topright", legend = c("ad", "nonad"), col = c("red", "blue"), lwd = 2)
plot(density(data[[2]][data[[target_col]] == "ad"]), main = "Feature 2 Density by Class", xlab = "Feature 2", col = "red", lwd = 2)
lines(density(data[[2]][data[[target_col]] == "nonad"]), col = "blue", lwd = 2)
par(mfrow = c(1, 1))
dev.off()

# 5. Correlation heatmap for the first 10 numeric features
numeric_data <- data[, sapply(data, is.numeric)]
cor_matrix <- cor(numeric_data[, 1:10], use = "complete.obs")
png("../graphics/03-eda-correlation_heatmap.png", width = 800, height = 800)
heatmap(cor_matrix, 
        symm = TRUE, 
        main = "Correlation Heatmap of First 10 Features",
        col = colorRampPalette(c("blue", "white", "red"))(100))
dev.off()


# --- SUPPLEMENTARY: LOGGING & DATA EXPORT ---
# This section contains code for printing summaries to the console and exporting
# data tables to CSV files.

cat("=== EXPLORATORY DATA ANALYSIS ===\n")

cat("\n--- TARGET VARIABLE ANALYSIS ---\n")
cat("Class distribution:\n")
print(table(data[[target_col]]))
cat("Target distribution plot saved to ../graphics/03-eda-target_distribution.png\n")

cat("\n--- DESCRIPTIVE STATISTICS ---\n")
key_features <- 1:10
all_feature_stats <- t(sapply(numeric_data[, key_features], function(x) {
  c(Min = min(x), Q1 = quantile(x, 0.25), Median = median(x), 
    Mean = mean(x), Q3 = quantile(x, 0.75), Max = max(x), SD = sd(x))
}))
cat("Summary statistics for first 10 features:\n")
print(all_feature_stats)

cat("\n--- PLOT GENERATION LOG ---\n")
cat("Histograms saved to ../graphics/03-eda-histograms.png\n")
cat("Boxplots saved to ../graphics/03-eda-boxplots.png\n")
cat("Scatter and density plots saved to ../graphics/03-eda-scatter_density_plots.png\n")
cat("Correlation heatmap saved to ../graphics/03-eda-correlation_heatmap.png\n")

cat("\n--- CORRELATION ANALYSIS ---\n")
cor_matrix_full <- cor(numeric_data, use = "complete.obs")
cor_matrix_full[lower.tri(cor_matrix_full, diag = TRUE)] <- NA
cor_flat <- as.data.frame(as.table(cor_matrix_full))
cor_flat <- na.omit(cor_flat)
names(cor_flat) <- c("Feature1", "Feature2", "Correlation")
cor_flat <- cor_flat[order(-abs(cor_flat$Correlation)), ]
cat("Top 10 highest absolute correlations:\n")
print(head(cor_flat, 10))

cat("\n--- EXPORTING EDA SUMMARY ---\n")
write.csv(all_feature_stats, "descriptive_statistics.csv", row.names = TRUE)
cat("Descriptive statistics exported to descriptive_statistics.csv\n")
write.csv(head(cor_flat, 20), "top_correlations.csv", row.names = FALSE)
cat("Top correlations exported to top_correlations.csv\n")

save(target_table, all_feature_stats, cor_flat, file = "eda_results.RData")
cat("\nEDA results saved to eda_results.RData\n")

cat("\n=== EDA COMPLETED ===\n")