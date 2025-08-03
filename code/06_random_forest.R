# =============================================================================
# 06_random_forest.R
# Random Forest Implementation
# =============================================================================

# Clear environment and load cleaned data
rm(list = ls())
load("data_cleaned.RData")

cat("=== RANDOM FOREST MODEL ===\n")

# Prepare data for modeling
cat("\n--- DATA PREPARATION ---\n")
X <- data[, -ncol(data)]  # All features except target
y <- data[[target_col]]   # Target variable

cat("Feature matrix dimensions:", nrow(X), "x", ncol(X), "\n")
cat("Target variable levels:", levels(y), "\n")
cat("Class distribution:\n")
print(table(y))

# Combine features and target for modeling
model_data <- cbind(X, target = y)

# Train-test split
cat("\n--- TRAIN-TEST SPLIT ---\n")
set.seed(123)  # For reproducibility
train_size <- floor(0.7 * nrow(model_data))
train_indices <- sample(seq_len(nrow(model_data)), size = train_size)

train_data <- model_data[train_indices, ]
test_data <- model_data[-train_indices, ]

cat("Training set size:", nrow(train_data), "\n")
cat("Test set size:", nrow(test_data), "\n")
cat("Training set class distribution:\n")
print(table(train_data$target))
cat("Test set class distribution:\n")
print(table(test_data$target))

# Simple Random Forest implementation
cat("\n--- RANDOM FOREST TRAINING ---\n")

# Function to calculate Gini impurity
gini_impurity <- function(labels) {
  if(length(labels) == 0) return(0)
  proportions <- table(labels) / length(labels)
  return(1 - sum(proportions^2))
}

# Function to bootstrap sample
bootstrap_sample <- function(data) {
  n <- nrow(data)
  indices <- sample(1:n, n, replace = TRUE)
  return(data[indices, ])
}

# Function to select random features
select_random_features <- function(feature_names, m) {
  return(sample(feature_names, min(m, length(feature_names))))
}

# Function to find best split with random features
find_best_split_rf <- function(data, target_col, feature_subset) {
  best_gini <- Inf
  best_feature <- NULL
  best_threshold <- NULL
  
  for(feature in feature_subset) {
    if(feature != target_col) {
      values <- unique(data[[feature]])
      if(length(values) > 1) {
        # Sample some thresholds for efficiency
        thresholds <- sample(values, min(10, length(values)))
        
        for(threshold in thresholds) {
          left_indices <- data[[feature]] <= threshold
          right_indices <- !left_indices
          
          if(sum(left_indices) > 0 && sum(right_indices) > 0) {
            left_gini <- gini_impurity(data[[target_col]][left_indices])
            right_gini <- gini_impurity(data[[target_col]][right_indices])
            
            weighted_gini <- (sum(left_indices) * left_gini + sum(right_indices) * right_gini) / nrow(data)
            
            if(weighted_gini < best_gini) {
              best_gini <- weighted_gini
              best_feature <- feature
              best_threshold <- threshold
            }
          }
        }
      }
    }
  }
  
  return(list(feature = best_feature, threshold = best_threshold, gini = best_gini))
}

# Build single decision tree for Random Forest
build_rf_tree <- function(data, target_col, max_depth = 5, current_depth = 0, m_features = NULL) {
  # Set default m_features (sqrt of total features)
  if(is.null(m_features)) {
    m_features <- floor(sqrt(ncol(data) - 1))
  }
  
  # Base cases
  if(current_depth >= max_depth || nrow(data) < 5 || length(unique(data[[target_col]])) == 1) {
    majority_class <- names(sort(table(data[[target_col]]), decreasing = TRUE))[1]
    return(list(type = "leaf", prediction = majority_class, samples = nrow(data)))
  }
  
  # Select random subset of features
  feature_names <- names(data)[names(data) != target_col]
  feature_subset <- select_random_features(feature_names, m_features)
  
  # Find best split among selected features
  split_info <- find_best_split_rf(data, target_col, feature_subset)
  
  if(is.null(split_info$feature)) {
    majority_class <- names(sort(table(data[[target_col]]), decreasing = TRUE))[1]
    return(list(type = "leaf", prediction = majority_class, samples = nrow(data)))
  }
  
  # Split data
  left_indices <- data[[split_info$feature]] <= split_info$threshold
  right_indices <- !left_indices
  
  left_data <- data[left_indices, ]
  right_data <- data[right_indices, ]
  
  # Recursively build subtrees
  left_tree <- build_rf_tree(left_data, target_col, max_depth, current_depth + 1, m_features)
  right_tree <- build_rf_tree(right_data, target_col, max_depth, current_depth + 1, m_features)
  
  return(list(
    type = "node",
    feature = split_info$feature,
    threshold = split_info$threshold,
    left = left_tree,
    right = right_tree,
    samples = nrow(data)
  ))
}

# Function to make predictions with single tree
predict_single_tree <- function(tree, data) {
  predictions <- character(nrow(data))
  
  for(i in 1:nrow(data)) {
    current_node <- tree
    
    while(current_node$type == "node") {
      if(data[i, current_node$feature] <= current_node$threshold) {
        current_node <- current_node$left
      } else {
        current_node <- current_node$right
      }
    }
    
    predictions[i] <- current_node$prediction
  }
  
  return(predictions)
}

# Build Random Forest
n_trees <- 100  # Number of trees in forest
m_features <- floor(sqrt(ncol(train_data) - 1))  # Number of features to consider at each split

cat("Building Random Forest with", n_trees, "trees...\n")
cat("Features per split:", m_features, "\n")

forest <- list()
for(i in 1:n_trees) {
  if(i %% 10 == 0) cat("Building tree", i, "/", n_trees, "\n")
  
  # Bootstrap sample
  boot_data <- bootstrap_sample(train_data)
  
  # Build tree
  tree <- build_rf_tree(boot_data, "target", max_depth = 5, m_features = m_features)
  forest[[i]] <- tree
}

cat("Random Forest training completed!\n")

# Function to make Random Forest predictions
predict_random_forest <- function(forest, data) {
  n_trees <- length(forest)
  n_samples <- nrow(data)
  
  # Get predictions from all trees
  all_predictions <- matrix(NA, nrow = n_samples, ncol = n_trees)
  
  for(i in 1:n_trees) {
    tree_predictions <- predict_single_tree(forest[[i]], data)
    all_predictions[, i] <- tree_predictions
  }
  
  # Majority voting
  final_predictions <- character(n_samples)
  for(i in 1:n_samples) {
    votes <- table(all_predictions[i, ])
    final_predictions[i] <- names(sort(votes, decreasing = TRUE))[1]
  }
  
  return(factor(final_predictions, levels = levels(train_data$target)))
}

# Make predictions
cat("\n--- RANDOM FOREST PREDICTIONS ---\n")
train_predictions <- predict_random_forest(forest, train_data)
test_predictions <- predict_random_forest(forest, test_data)

# Training accuracy
train_accuracy <- mean(train_predictions == train_data$target)
cat("Training accuracy:", round(train_accuracy, 4), "\n")

# Test accuracy
test_accuracy <- mean(test_predictions == test_data$target)
cat("Test accuracy:", round(test_accuracy, 4), "\n")

# Confusion matrix
confusion_matrix <- table(Predicted = test_predictions, Actual = test_data$target)
cat("\nConfusion Matrix:\n")
print(confusion_matrix)

# Calculate metrics
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
if(confusion_matrix[1,1] + confusion_matrix[1,2] > 0) {
  precision_ad <- confusion_matrix[1,1] / (confusion_matrix[1,1] + confusion_matrix[1,2])
} else {
  precision_ad <- 0
}
if(confusion_matrix[1,1] + confusion_matrix[2,1] > 0) {
  recall_ad <- confusion_matrix[1,1] / (confusion_matrix[1,1] + confusion_matrix[2,1])
} else {
  recall_ad <- 0
}
if(precision_ad + recall_ad > 0) {
  f1_ad <- 2 * (precision_ad * recall_ad) / (precision_ad + recall_ad)
} else {
  f1_ad <- 0
}

cat("\nPerformance Metrics:\n")
cat("Accuracy:", round(accuracy, 4), "\n")
cat("Precision (ad):", round(precision_ad, 4), "\n")
cat("Recall (ad):", round(recall_ad, 4), "\n")
cat("F1-score (ad):", round(f1_ad, 4), "\n")

# Generate plots
cat("\n--- GENERATING RANDOM FOREST PLOTS ---\n")

# Plot confusion matrix
png("../graphics/06-rf-confusion_matrix.png", width = 600, height = 600)
par(mar = c(5, 5, 4, 2))
image(1:2, 1:2, as.matrix(confusion_matrix), 
      col = c("lightblue", "lightcoral", "orange", "lightgreen"),
      xlab = "Predicted", ylab = "Actual", 
      main = "Random Forest Confusion Matrix",
      axes = FALSE)
axis(1, at = 1:2, labels = colnames(confusion_matrix))
axis(2, at = 1:2, labels = rownames(confusion_matrix))
# Add text with counts
for(i in 1:2) {
  for(j in 1:2) {
    text(j, i, confusion_matrix[i,j], cex = 2, font = 2)
  }
}
dev.off()
cat("Confusion matrix plot saved to ../graphics/06-rf-confusion_matrix.png\n")

# Feature importance (simplified)
cat("\n--- FEATURE IMPORTANCE ---\n")
feature_usage <- character()
for(tree in forest) {
  # Extract features used in each tree (simplified)
  extract_features <- function(node) {
    if(node$type == "node") {
      features <- c(node$feature)
      if(!is.null(node$left)) features <- c(features, extract_features(node$left))
      if(!is.null(node$right)) features <- c(features, extract_features(node$right))
      return(features)
    }
    return(character(0))
  }
  
  tree_features <- extract_features(tree)
  feature_usage <- c(feature_usage, tree_features)
}

# Count feature usage
feature_counts <- table(feature_usage)
feature_importance <- sort(feature_counts, decreasing = TRUE)

cat("Top 10 most important features:\n")
print(head(feature_importance, 10))

# Plot feature importance
png("../graphics/06-rf-feature_importance.png", width = 800, height = 600)
par(mar = c(8, 5, 4, 2))
top_features <- head(feature_importance, 10)
barplot(top_features, 
        main = "Random Forest - Top 10 Feature Importance",
        ylab = "Usage Count",
        col = "darkgreen",
        las = 2)
dev.off()
cat("Feature importance plot saved to ../graphics/06-rf-feature_importance.png\n")

cat("\n=== RANDOM FOREST MODEL COMPLETED ===\n")

# Save results
rf_results <- list(
  forest = forest,
  train_accuracy = train_accuracy,
  test_accuracy = test_accuracy,
  confusion_matrix = confusion_matrix,
  accuracy = accuracy,
  precision_ad = precision_ad,
  recall_ad = recall_ad,
  f1_ad = f1_ad,
  feature_importance = feature_importance,
  n_trees = n_trees,
  m_features = m_features
)
save(rf_results, file = "random_forest_results.RData")
cat("Random Forest results saved to random_forest_results.RData\n")