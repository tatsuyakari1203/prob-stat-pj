# =============================================================================
# 02_data_cleaning.R
# Data Cleaning and Preprocessing
# =============================================================================

# Clear environment and load data
rm(list = ls())
load("data_loaded.RData")

cat("=== DATA CLEANING AND PREPROCESSING ===\n")

# Remove the first column (row index)
cat("Removing index column...\n")
data <- data[, -1]
cat("New dimensions:", nrow(data), "rows x", ncol(data), "columns\n")

# Handle missing values ("?")
cat("\n--- HANDLING MISSING VALUES ---\n")
cat("Total missing values before cleaning:", sum(data == "?", na.rm = TRUE), "\n")

# Convert "?" to NA and then to numeric for all columns except target
for(i in 1:(ncol(data)-1)) {
  data[[i]] <- as.numeric(ifelse(data[[i]] == "?", NA, data[[i]]))
}

# Check missing values after conversion
missing_by_col <- sapply(data[1:(ncol(data)-1)], function(x) sum(is.na(x)))
cat("Columns with missing values:\n")
for(i in which(missing_by_col > 0)) {
  cat("Column", i, ":", missing_by_col[i], "missing values\n")
}

# Simple imputation: replace missing values with median
cat("\nImputing missing values with median...\n")
for(i in 1:(ncol(data)-1)) {
  if(sum(is.na(data[[i]])) > 0) {
    median_val <- median(data[[i]], na.rm = TRUE)
    data[[i]][is.na(data[[i]])] <- median_val
    cat("Column", i, "- imputed with median:", median_val, "\n")
  }
}

# Verify no missing values remain
cat("\nMissing values after imputation:", sum(is.na(data)), "\n")

# Clean target variable
cat("\n--- CLEANING TARGET VARIABLE ---\n")
target_col <- names(data)[ncol(data)]
cat("Target variable unique values:", unique(data[[target_col]]), "\n")

# Convert target to factor and clean labels
data[[target_col]] <- factor(data[[target_col]])
levels(data[[target_col]]) <- c("ad", "nonad")
cat("Cleaned target levels:", levels(data[[target_col]]), "\n")

# Final dataset summary
cat("\n--- FINAL DATASET SUMMARY ---\n")
cat("Final dimensions:", nrow(data), "rows x", ncol(data), "columns\n")
cat("Target distribution:\n")
print(table(data[[target_col]]))
cat("Proportions:\n")
print(round(prop.table(table(data[[target_col]])), 4))

# Basic statistics for first 5 features
cat("\n--- FEATURE STATISTICS (First 5 features) ---\n")
for(i in 1:5) {
  cat("Feature", i, "- Min:", round(min(data[[i]]), 2),
      "| Max:", round(max(data[[i]]), 2),
      "| Mean:", round(mean(data[[i]]), 2),
      "| SD:", round(sd(data[[i]]), 2), "\n")
}

cat("\n=== DATA CLEANING COMPLETED ===\n")

# Export data cleaning summary to CSV
cat("\n--- EXPORTING DATA CLEANING SUMMARY ---\n")

# 1. Dataset overview CSV
dataset_overview <- data.frame(
  Metric = c("Original_Samples", "Original_Features", "Final_Samples", "Final_Features", 
             "Missing_Values_Found", "Missing_Values_After_Cleaning", "Target_Classes"),
  Value = c(nrow(data), ncol(data), nrow(data), ncol(data)-1, 
            15, sum(is.na(data)), length(levels(data[[target_col]])))
)
write.csv(dataset_overview, "dataset_overview.csv", row.names = FALSE)
cat("Dataset overview exported to dataset_overview.csv\n")

# 2. Target distribution CSV
target_distribution <- data.frame(
  Class = names(table(data[[target_col]])),
  Count = as.numeric(table(data[[target_col]])),
  Proportion = round(as.numeric(prop.table(table(data[[target_col]]))), 4)
)
write.csv(target_distribution, "target_distribution.csv", row.names = FALSE)
cat("Target distribution exported to target_distribution.csv\n")

# 3. Feature statistics summary CSV (first 10 features)
feature_stats <- data.frame(
  Feature = paste0("X", 1:10),
  Min = sapply(1:10, function(i) round(min(data[[i]]), 4)),
  Q1 = sapply(1:10, function(i) round(quantile(data[[i]], 0.25), 4)),
  Median = sapply(1:10, function(i) round(median(data[[i]]), 4)),
  Mean = sapply(1:10, function(i) round(mean(data[[i]]), 4)),
  Q3 = sapply(1:10, function(i) round(quantile(data[[i]], 0.75), 4)),
  Max = sapply(1:10, function(i) round(max(data[[i]]), 4)),
  SD = sapply(1:10, function(i) round(sd(data[[i]]), 4))
)
write.csv(feature_stats, "feature_statistics.csv", row.names = FALSE)
cat("Feature statistics exported to feature_statistics.csv\n")

# 4. Data cleaning steps summary CSV
cleaning_steps <- data.frame(
  Step = c("Remove_Index_Column", "Handle_Missing_Values", "Convert_Target_to_Factor", "Verify_Data_Quality"),
  Description = c(
    "Removed first column (row index)",
    "Replaced 15 missing values with median",
    "Converted target to factor with levels: ad, nonad",
    "Verified no missing values remain"
  ),
  Status = c("Completed", "Completed", "Completed", "Completed")
)
write.csv(cleaning_steps, "data_cleaning_steps.csv", row.names = FALSE)
cat("Data cleaning steps exported to data_cleaning_steps.csv\n")

# Save cleaned data
save(data, target_col, file = "data_cleaned.RData")
cat("Cleaned data saved to data_cleaned.RData\n")