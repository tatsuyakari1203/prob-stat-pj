# =============================================================================
# 01_data_loading.R
# Data Loading and Initial Exploration
# =============================================================================

# --- SETUP ---
# Clear environment to ensure a clean state
rm(list = ls())


# --- CORE LOGIC: DATA LOADING ---
# This section contains the essential code for loading the dataset and saving it.
# This is the primary logic you would feature in a report.

# Load the dataset from the CSV file
data <- read.csv("../add.csv", header = TRUE, stringsAsFactors = FALSE)

# Identify the target column name
target_col <- names(data)[ncol(data)]

# Save the loaded data and target column name for faster access in subsequent scripts
save(data, target_col, file = "data_loaded.RData")


# --- SUPPLEMENTARY: LOGGING & INITIAL EXPLORATION ---
# This section contains code for printing progress and summary statistics to the
# console. This is useful for development and debugging but is not part of the
# core data processing pipeline.

cat("=== DATA LOADING AND INITIAL EXPLORATION ===\n")

cat("Loading dataset...\n")

# Basic information about the dataset
cat("\n--- DATASET OVERVIEW ---\n")
cat("Dataset dimensions:", nrow(data), "rows x", ncol(data), "columns\n")
cat("Column names (first 10):", paste(names(data)[1:10], collapse = ", "), "...\n")
cat("Target variable:", target_col, "\n")

# Check the target variable distribution
cat("\n--- TARGET VARIABLE DISTRIBUTION ---\n")
target_counts <- table(data[[target_col]])
print(target_counts)
cat("Proportions:\n")
print(round(prop.table(target_counts), 4))

# Check for missing values
cat("\n--- MISSING VALUES CHECK ---\n")
missing_count <- sum(data == "?", na.rm = TRUE)
cat("Total '?' values in dataset:", missing_count, "\n")

# Check missing values by column (first 20 columns)
cat("Missing values in first 20 columns:\n")
for(i in 1:min(20, ncol(data))) {
  missing_in_col <- sum(data[[i]] == "?", na.rm = TRUE)
  if(missing_in_col > 0) {
    cat("Column", i, ":", missing_in_col, "missing values\n")
  }
}

# Basic statistics for first few numeric columns
cat("\n--- BASIC STATISTICS (First 5 numeric columns) ---\n")
for(i in 1:5) {
  col_data <- as.numeric(ifelse(data[[i]] == "?", NA, data[[i]]))
  if(sum(!is.na(col_data)) > 0) {
    cat("Column", i, "- Mean:", round(mean(col_data, na.rm = TRUE), 2),
        "| Median:", round(median(col_data, na.rm = TRUE), 2),
        "| SD:", round(sd(col_data, na.rm = TRUE), 2), "\n")
  }
}

cat("\n=== DATA LOADING COMPLETED ===\n")
cat("Data saved to data_loaded.RData\n")