# =============================================================================
# PHASE 1: DATA DESCRIPTION AND CLEANING
# =============================================================================
# This script handles data loading, description, and cleaning processes
# according to the project requirements

# Load necessary libraries
library(readr)
library(dplyr)
library(knitr)

# =============================================================================
# 1. DATA LOADING
# =============================================================================

# Load the 3D printer dataset
cat("Loading 3D printer dataset...\n")
data <- read_csv("/home/kari/prob-stat-pj/3dprintdata/data.csv")

cat("Dataset loaded successfully!\n")
cat("Dataset dimensions:", nrow(data), "rows x", ncol(data), "columns\n\n")

# =============================================================================
# 2. DATA DESCRIPTION
# =============================================================================

cat("=== DATA DESCRIPTION ===\n")

# Display basic information about the dataset
cat("\n--- Dataset Structure ---\n")
str(data)

cat("\n--- First 10 rows of the dataset ---\n")
print(head(data, 10))

cat("\n--- Last 5 rows of the dataset ---\n")
print(tail(data, 5))

cat("\n--- Column names ---\n")
print(colnames(data))

cat("\n--- Data types ---\n")
print(sapply(data, class))

cat("\n--- Summary statistics ---\n")
print(summary(data))

# =============================================================================
# 3. DATA QUALITY ASSESSMENT
# =============================================================================

cat("\n=== DATA QUALITY ASSESSMENT ===\n")

# Check for missing values
cat("\n--- Missing values check ---\n")
missing_values <- sapply(data, function(x) sum(is.na(x)))
print(missing_values)
cat("Total missing values:", sum(missing_values), "\n")

# Check for duplicate rows
cat("\n--- Duplicate rows check ---\n")
duplicate_count <- sum(duplicated(data))
cat("Number of duplicate rows:", duplicate_count, "\n")

# Check data ranges for continuous variables
cat("\n--- Data ranges for continuous variables ---\n")
continuous_vars <- c("layer_height", "nozzle_temperature", "bed_temperature", 
                    "print_speed", "infill_density", "wall_thickness", 
                    "fan_speed", "roughness", "tension_strenght", "elongation")

for(var in continuous_vars) {
  if(var %in% colnames(data)) {
    cat(sprintf("%s: Min=%.3f, Max=%.3f, Range=%.3f\n", 
                var, min(data[[var]], na.rm=TRUE), 
                max(data[[var]], na.rm=TRUE),
                max(data[[var]], na.rm=TRUE) - min(data[[var]], na.rm=TRUE)))
  }
}

# Check categorical variables
cat("\n--- Categorical variables ---\n")
categorical_vars <- c("infill_pattern", "material")

for(var in categorical_vars) {
  if(var %in% colnames(data)) {
    cat(sprintf("\n%s levels:\n", var))
    print(table(data[[var]]))
  }
}

# =============================================================================
# 4. DATA CLEANING
# =============================================================================

cat("\n=== DATA CLEANING ===\n")

# Create a copy of original data for cleaning
data_cleaned <- data

# Remove duplicate rows if any
if(duplicate_count > 0) {
  cat("Removing", duplicate_count, "duplicate rows...\n")
  data_cleaned <- data_cleaned[!duplicated(data_cleaned), ]
}

# Handle missing values (if any)
if(sum(missing_values) > 0) {
  cat("Handling missing values...\n")
  # For this dataset, we'll remove rows with missing values
  # In practice, you might want to use imputation methods
  data_cleaned <- na.omit(data_cleaned)
  cat("Rows after removing missing values:", nrow(data_cleaned), "\n")
}

# Check for outliers using IQR method for continuous variables
cat("\n--- Outlier detection using IQR method ---\n")
outlier_summary <- data.frame(
  Variable = character(),
  Q1 = numeric(),
  Q3 = numeric(),
  IQR = numeric(),
  Lower_Bound = numeric(),
  Upper_Bound = numeric(),
  Outliers_Count = numeric(),
  stringsAsFactors = FALSE
)

for(var in continuous_vars) {
  if(var %in% colnames(data_cleaned)) {
    Q1 <- quantile(data_cleaned[[var]], 0.25, na.rm = TRUE)
    Q3 <- quantile(data_cleaned[[var]], 0.75, na.rm = TRUE)
    IQR_val <- Q3 - Q1
    lower_bound <- Q1 - 1.5 * IQR_val
    upper_bound <- Q3 + 1.5 * IQR_val
    
    outliers <- data_cleaned[[var]] < lower_bound | data_cleaned[[var]] > upper_bound
    outlier_count <- sum(outliers, na.rm = TRUE)
    
    outlier_summary <- rbind(outlier_summary, data.frame(
      Variable = var,
      Q1 = Q1,
      Q3 = Q3,
      IQR = IQR_val,
      Lower_Bound = lower_bound,
      Upper_Bound = upper_bound,
      Outliers_Count = outlier_count
    ))
  }
}

print(outlier_summary)

# Data type conversions
cat("\n--- Data type conversions ---\n")

# Convert categorical variables to factors
if("infill_pattern" %in% colnames(data_cleaned)) {
  data_cleaned$infill_pattern <- as.factor(data_cleaned$infill_pattern)
  cat("Converted infill_pattern to factor\n")
}

if("material" %in% colnames(data_cleaned)) {
  data_cleaned$material <- as.factor(data_cleaned$material)
  cat("Converted material to factor\n")
}

# =============================================================================
# 5. FINAL CLEANED DATASET SUMMARY
# =============================================================================

cat("\n=== FINAL CLEANED DATASET SUMMARY ===\n")
cat("Original dataset:", nrow(data), "rows x", ncol(data), "columns\n")
cat("Cleaned dataset:", nrow(data_cleaned), "rows x", ncol(data_cleaned), "columns\n")
cat("Rows removed:", nrow(data) - nrow(data_cleaned), "\n")

cat("\n--- Final data structure ---\n")
str(data_cleaned)

cat("\n--- Final summary statistics ---\n")
print(summary(data_cleaned))

# Save cleaned dataset
write_csv(data_cleaned, "/home/kari/prob-stat-pj/3dprintdata/data_cleaned.csv")
cat("\nCleaned dataset saved to: /home/kari/prob-stat-pj/3dprintdata/data_cleaned.csv\n")

# =============================================================================
# EXPORT LOG TO TXT FILE
# =============================================================================
# Create logs directory if it doesn't exist
if (!dir.exists("/home/kari/prob-stat-pj/logs")) {
  dir.create("/home/kari/prob-stat-pj/logs", recursive = TRUE)
}

# Capture all output and save to log file
log_file <- "/home/kari/prob-stat-pj/logs/01_data_description_cleaning_log.txt"
sink(log_file)

cat("=== DATA DESCRIPTION & CLEANING LOG ===")
cat("\nGenerated on:", as.character(Sys.time()))
cat("\n\n")

cat("1. DATASET OVERVIEW:\n")
cat("   - Dimensions:", nrow(data), "rows x", ncol(data), "columns\n")
cat("   - Variables:", paste(names(data), collapse = ", "), "\n\n")

cat("2. DATA QUALITY ASSESSMENT:\n")
cat("   - Missing values:", sum(is.na(data)), "\n")
cat("   - Duplicate rows:", sum(duplicated(data)), "\n")
cat("   - Outliers detected using IQR method\n\n")

cat("3. DATA CLEANING RESULTS:\n")
cat("   - Original dataset:", nrow(data), "rows x", ncol(data), "columns\n")
cat("   - Cleaned dataset:", nrow(data_cleaned), "rows x", ncol(data_cleaned), "columns\n")
cat("   - Factor variables: infill_pattern, material\n\n")

cat("4. OUTPUT FILES:\n")
cat("   - Cleaned dataset: /home/kari/prob-stat-pj/3dprintdata/data_cleaned.csv\n")
cat("   - Log file: /home/kari/prob-stat-pj/logs/01_data_description_cleaning_log.txt\n\n")

cat("5. SUMMARY STATISTICS (CLEANED DATA):\n")
print(summary(data_cleaned))

sink()

cat("\nLog exported to:", log_file, "\n")

cat("\n=== PHASE 1 COMPLETED SUCCESSFULLY ===\n")
cat("Data description and cleaning process finished.\n")
cat("Ready for Phase 2: Descriptive Statistics\n")