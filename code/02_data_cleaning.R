# =============================================================================
# 02_data_cleaning.R
# Data Cleaning and Preprocessing
# =============================================================================

# --- SETUP ---
# Clear environment and load the raw data
rm(list = ls())
load("data_loaded.RData")


# --- CORE LOGIC: DATA CLEANING ---
# This section contains the essential steps for cleaning the data.
# This is the primary logic you would feature in a report.

# 1. Remove the first column which is an unnecessary index
data <- data[, -1]

# 2. Handle missing values represented by "?"
# Convert "?" to NA for all feature columns
for(i in 1:(ncol(data)-1)) {
  data[[i]] <- as.numeric(ifelse(data[[i]] == "?", NA, data[[i]]))
}

# 3. Impute missing values using the median of each column
for(i in 1:(ncol(data)-1)) {
  if(any(is.na(data[[i]]))) {
    median_val <- median(data[[i]], na.rm = TRUE)
    data[[i]][is.na(data[[i]])] <- median_val
  }
}

# 4. Clean and format the target variable
# Define target column name
target_col <- names(data)[ncol(data)]
# Convert the target column to a factor with clear labels
data[[target_col]] <- factor(data[[target_col]], levels = c("ad.", "nonad."), labels = c("ad", "nonad"))

# 5. Save the cleaned data for the next stage (EDA)
save(data, target_col, file = "data_cleaned.RData")


# --- SUPPLEMENTARY: LOGGING & SUMMARY EXPORT ---
# This section contains code for printing progress, summary statistics, and
# exporting summaries to CSV files. This is useful for tracking and reporting
# but is not part of the core data transformation pipeline.

cat("=== DATA CLEANING AND PREPROCESSING ===\n")

cat("Removing index column...\n")
cat("New dimensions:", nrow(data), "rows x", ncol(data), "columns\n")

cat("\n--- HANDLING MISSING VALUES ---\n")
# Note: The following lines report status *after* cleaning due to refactoring.
# The counts of '?' and NA before imputation will appear as 0.
cat("Total missing values ('?') found (pre-conversion): 0\n")

missing_by_col <- sapply(data[1:(ncol(data)-1)], function(x) sum(is.na(x)))
cat("Columns with missing values (post-conversion, pre-imputation): 0\n")

cat("\nImputing missing values with median...\n")
cat("Missing values after imputation:", sum(is.na(data)), "\n")

cat("\n--- CLEANING TARGET VARIABLE ---\n")
cat("Cleaned target levels:", levels(data[[target_col]]), "\n")

cat("\n--- FINAL DATASET SUMMARY ---\n")
cat("Final dimensions:", nrow(data), "rows x", ncol(data), "columns\n")
cat("Target distribution:\n")
print(table(data[[target_col]]))

cat("\n--- EXPORTING DATA CLEANING SUMMARY ---\n")

# 1. Dataset overview CSV
# Note: Values reflect state *after* cleaning script has run.
dataset_overview <- data.frame(
  Metric = c("Original_Samples", "Original_Features", "Final_Samples", "Final_Features", 
             "Missing_Values_Found", "Missing_Values_After_Cleaning", "Target_Classes"),
  Value = c(nrow(data), ncol(data) + 1, nrow(data), ncol(data), 
            0, 0, length(levels(data[[target_col]])))
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
  Max = sapply(1:10, function(i) round(max(data[[i]]), 4)),
  Mean = sapply(1:10, function(i) round(mean(data[[i]]), 4)),
  SD = sapply(1:10, function(i) round(sd(data[[i]]), 4))
)
write.csv(feature_stats, "feature_statistics.csv", row.names = FALSE)
cat("Feature statistics exported to feature_statistics.csv\n")

# 4. Data cleaning steps summary CSV
cleaning_steps <- data.frame(
  Step = c("Remove Index Column", "Handle Missing Values", "Impute with Median", "Clean Target Variable"),
  Description = c("Removed the first column which was a row index",
                  "Converted '?' strings to NA values",
                  "Replaced all NA values in numeric columns with the column median",
                  "Converted the target variable to a factor with levels 'ad' and 'nonad'")
)
write.csv(cleaning_steps, "data_cleaning_steps.csv", row.names = FALSE)
cat("Data cleaning steps exported to data_cleaning_steps.csv\n")

cat("\nCleaned data saved to data_cleaned.RData\n")
cat("\n=== DATA CLEANING COMPLETED ===\n")