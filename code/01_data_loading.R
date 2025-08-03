# =============================================================================
# 01_data_loading.R
# Data Loading and Initial Exploration
# =============================================================================

# Clear environment
rm(list = ls())

cat("=== DATA LOADING AND INITIAL EXPLORATION ===\n")

# Load the dataset using base R
cat("Loading dataset...\n")
data <- read.csv("../add.csv", header = TRUE, stringsAsFactors = FALSE)

# Basic information about the dataset
cat("\n--- DATASET OVERVIEW ---\n")
cat("Dataset dimensions:", nrow(data), "rows x", ncol(data), "columns\n")
cat("Column names (first 10):", paste(names(data)[1:10], collapse = ", "), "...\n")
cat("Target variable:", names(data)[ncol(data)], "\n")

# Check the target variable distribution
cat("\n--- TARGET VARIABLE DISTRIBUTION ---\n")
target_col <- names(data)[ncol(data)]
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
numeric_cols <- sapply(data[1:5], function(x) {
  # Convert to numeric, treating "?" as NA
  as.numeric(ifelse(x == "?", NA, x))
})

for(i in 1:5) {
  col_data <- as.numeric(ifelse(data[[i]] == "?", NA, data[[i]]))
  if(sum(!is.na(col_data)) > 0) {
    cat("Column", i, "- Mean:", round(mean(col_data, na.rm = TRUE), 2),
        "| Median:", round(median(col_data, na.rm = TRUE), 2),
        "| SD:", round(sd(col_data, na.rm = TRUE), 2), "\n")
  }
}

cat("\n=== DATA LOADING COMPLETED ===\n")

# Save basic info for next steps
save(data, target_col, file = "data_loaded.RData")
cat("Data saved to data_loaded.RData\n")