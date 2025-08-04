# =============================================================================
# 05_decision_tree.R
# Decision Tree Implementation
# =============================================================================

# Clear environment and load cleaned data
rm(list = ls())
load("data_cleaned.RData")

cat("=== DECISION TREE MODEL ===\n")

# Prepare data for modeling
cat("\n--- DATA PREPARATION ---\n")
# Create a smaller feature set for interpretability (first 20 features)
feature_subset <- 1:20
X <- data[, feature_subset]
y <- data[[target_col]]

cat("Using", ncol(X), "features for Decision Tree\n")
cat("Feature matrix dimensions:", nrow(X), "x", ncol(X), "\n")
cat("Target variable levels:", levels(y), "\n")

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

# Simple Decision Tree implementation using recursive partitioning
cat("\n--- DECISION TREE TRAINING ---\n")

# Function to calculate Gini impurity
gini_impurity <- function(labels) {
  if(length(labels) == 0) return(0)
  proportions <- table(labels) / length(labels)
  return(1 - sum(proportions^2))
}

# Function to find best split
find_best_split <- function(data, target_col) {
  best_gini <- Inf
  best_feature <- NULL
  best_threshold <- NULL
  
  for(feature in names(data)[names(data) != target_col]) {
    values <- unique(data[[feature]])
    if(length(values) > 1) {
      for(threshold in values) {
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
  
  return(list(feature = best_feature, threshold = best_threshold, gini = best_gini))
}

# Simple decision tree (depth = 3 for interpretability)
build_simple_tree <- function(data, target_col, max_depth = 3, current_depth = 0) {
  # Base cases
  if(current_depth >= max_depth || nrow(data) < 10 || length(unique(data[[target_col]])) == 1) {
    majority_class <- names(sort(table(data[[target_col]]), decreasing = TRUE))[1]
    return(list(type = "leaf", prediction = majority_class, samples = nrow(data)))
  }
  
  # Find best split
  split_info <- find_best_split(data, target_col)
  
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
  left_tree <- build_simple_tree(left_data, target_col, max_depth, current_depth + 1)
  right_tree <- build_simple_tree(right_data, target_col, max_depth, current_depth + 1)
  
  return(list(
    type = "node",
    feature = split_info$feature,
    threshold = split_info$threshold,
    left = left_tree,
    right = right_tree,
    samples = nrow(data)
  ))
}

# Build the tree
cat("Building decision tree (max depth = 3)...\n")
tree <- build_simple_tree(train_data, "target")

# Function to make predictions
predict_tree <- function(tree, data) {
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
  
  return(factor(predictions, levels = levels(train_data$target)))
}

# Make predictions
cat("\n--- DECISION TREE PREDICTIONS ---\n")
train_predictions <- predict_tree(tree, train_data)
test_predictions <- predict_tree(tree, test_data)

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
cat("\n--- GENERATING DECISION TREE PLOTS ---\n")

# Plot confusion matrix
png("../graphics/05-dt-confusion_matrix.png", width = 600, height = 600)
par(mar = c(5, 5, 4, 2))
image(1:2, 1:2, as.matrix(confusion_matrix), 
      col = c("lightblue", "lightcoral", "orange", "lightgreen"),
      xlab = "Predicted", ylab = "Actual", 
      main = "Decision Tree Confusion Matrix",
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
cat("Confusion matrix plot saved to ../graphics/05-dt-confusion_matrix.png\n")

# --- PLOTTING FEATURE IMPORTANCE ---
# Calculate Gini importance: the total reduction in impurity brought by a feature.
# We will create this by traversing the tree after it's built.

feature_importance <- setNames(rep(0, ncol(X)), colnames(X))

get_feature_importance <- function(tree_node, parent_gini, parent_samples) {
  if (tree_node$type == "leaf") {
    return()
  }
  
  # Calculate Gini for children
  left_samples <- tree_node$left$samples
  right_samples <- tree_node$right$samples
  
  if (left_samples > 0) {
    left_gini <- gini_impurity(train_data[1:left_samples, "target"]) # A proxy for actual labels
  } else {
    left_gini <- 0
  }
  
  if (right_samples > 0) {
    right_gini <- gini_impurity(train_data[1:right_samples, "target"]) # A proxy for actual labels
  } else {
    right_gini <- 0
  }

  weighted_child_gini <- (left_samples * left_gini + right_samples * right_gini) / parent_samples
  
  # Importance is the Gini reduction
  importance_gain <- parent_gini - weighted_child_gini
  
  # Update the feature's importance score using global assignment
  feature_importance[tree_node$feature] <<- feature_importance[tree_node$feature] + (parent_samples / nrow(train_data)) * importance_gain
  
  # Recurse
  get_feature_importance(tree_node$left, left_gini, left_samples)
  get_feature_importance(tree_node$right, right_gini, right_samples)
}

# Initial call to calculate importance
initial_gini <- gini_impurity(train_data$target)
get_feature_importance(tree, initial_gini, nrow(train_data))

# Plot top 10 features
top_features <- sort(feature_importance[feature_importance > 0], decreasing = TRUE)
# Ensure we only plot if there are important features
if (length(top_features) > 0) {
    png("../graphics/05-dt-feature_importance.png", width = 800, height = 600)
    par(mar = c(8, 5, 4, 2))
    barplot(head(top_features, 10), 
            main = "Decision Tree - Top 10 Feature Importance (Gini Decrease)",
            ylab = "Importance Score",
            col = "steelblue",
            las = 2)
    dev.off()
    cat("Feature importance plot saved to ../graphics/05-dt-feature_importance.png\n")
} else {
    cat("No features were found to be important.\n")
}

# Print tree structure (simplified)
cat("\n--- DECISION TREE STRUCTURE ---\n")
print_tree <- function(tree, depth = 0) {
  indent <- paste(rep("  ", depth), collapse = "")
  
  if(tree$type == "leaf") {
    cat(indent, "Leaf: Predict", tree$prediction, "(samples:", tree$samples, ")\n")
  } else {
    cat(indent, "Node:", tree$feature, "<=", round(tree$threshold, 3), "(samples:", tree$samples, ")\n")
    cat(indent, "├─ Left:\n")
    print_tree(tree$left, depth + 1)
    cat(indent, "└─ Right:\n")
    print_tree(tree$right, depth + 1)
  }
}

print_tree(tree)

cat("\n=== DECISION TREE MODEL COMPLETED ===\n")

# Save results
dt_results <- list(
  tree = tree,
  train_accuracy = train_accuracy,
  test_accuracy = test_accuracy,
  confusion_matrix = confusion_matrix,
  accuracy = accuracy,
  precision_ad = precision_ad,
  recall_ad = recall_ad,
  f1_ad = f1_ad
)
save(dt_results, file = "decision_tree_results.RData")
cat("Decision Tree results saved to decision_tree_results.RData\n")