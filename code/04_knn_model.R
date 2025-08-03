# =============================================================================
# 04_knn_model.R
# k-Nearest Neighbors Implementation
# =============================================================================

# Clear environment and load cleaned data
rm(list = ls())
load("data_cleaned.RData")

cat("=== k-NEAREST NEIGHBORS MODEL ===\n")

# Prepare data for modeling
cat("\n--- DATA PREPARATION ---\n")
# Separate features and target
X <- data[, 1:(ncol(data)-1)]  # All features except target
y <- data[[target_col]]        # Target variable

cat("Feature matrix dimensions:", nrow(X), "x", ncol(X), "\n")
cat("Target variable levels:", levels(y), "\n")
cat("Class distribution:\n")
print(table(y))

# Normalize features (important for k-NN)
cat("\n--- FEATURE NORMALIZATION ---\n")
X_normalized <- scale(X)
cat("Features normalized using z-score standardization\n")
cat("Sample means after normalization (should be ~0):", round(colMeans(X_normalized)[1:5], 4), "\n")
cat("Sample SDs after normalization (should be ~1):", round(apply(X_normalized, 2, sd)[1:5], 4), "\n")

# Train-test split
cat("\n--- TRAIN-TEST SPLIT ---\n")
set.seed(123)  # For reproducibility
train_size <- floor(0.7 * nrow(X_normalized))
train_indices <- sample(seq_len(nrow(X_normalized)), size = train_size)

X_train <- X_normalized[train_indices, ]
y_train <- y[train_indices]
X_test <- X_normalized[-train_indices, ]
y_test <- y[-train_indices]

cat("Training set size:", nrow(X_train), "\n")
cat("Test set size:", nrow(X_test), "\n")
cat("Training set class distribution:\n")
print(table(y_train))
cat("Test set class distribution:\n")
print(table(y_test))

# Simple k-NN implementation using base R
cat("\n--- k-NN MODEL TRAINING AND PREDICTION ---\n")

# Function to calculate Euclidean distance
euclidean_distance <- function(x1, x2) {
  sqrt(sum((x1 - x2)^2))
}

# Simple k-NN prediction function
knn_predict <- function(X_train, y_train, X_test, k = 5) {
  predictions <- character(nrow(X_test))
  
  for(i in 1:nrow(X_test)) {
    # Calculate distances to all training points
    distances <- numeric(nrow(X_train))
    for(j in 1:nrow(X_train)) {
      distances[j] <- euclidean_distance(X_test[i, ], X_train[j, ])
    }
    
    # Find k nearest neighbors
    k_nearest_indices <- order(distances)[1:k]
    k_nearest_labels <- y_train[k_nearest_indices]
    
    # Majority vote
    vote_counts <- table(k_nearest_labels)
    predictions[i] <- names(vote_counts)[which.max(vote_counts)]
  }
  
  return(factor(predictions, levels = levels(y_train)))
}

# Test different k values
k_values <- c(3, 5, 7, 9, 11)
cat("Testing k values:", k_values, "\n")

results <- data.frame(
  k = k_values,
  accuracy = numeric(length(k_values)),
  stringsAsFactors = FALSE
)

for(i in 1:length(k_values)) {
  k <- k_values[i]
  cat("\nTesting k =", k, "...\n")
  
  # Make predictions (using subset for speed)
  test_subset <- 1:min(100, nrow(X_test))  # Use first 100 test samples for speed
  predictions <- knn_predict(X_train, y_train, X_test[test_subset, ], k = k)
  
  # Calculate accuracy
  accuracy <- mean(predictions == y_test[test_subset])
  results$accuracy[i] <- accuracy
  
  cat("k =", k, "- Accuracy:", round(accuracy, 4), "\n")
}

# Find best k
best_k_index <- which.max(results$accuracy)
best_k <- results$k[best_k_index]
best_accuracy <- results$accuracy[best_k_index]

cat("\n--- k-NN RESULTS SUMMARY ---\n")
cat("Results for different k values:\n")
print(results)
cat("\nBest k value:", best_k, "\n")
cat("Best accuracy:", round(best_accuracy, 4), "\n")

# Plot k-value tuning results
cat("\n--- GENERATING k-NN PLOTS ---\n")
png("../graphics/knn_k_tuning.png", width = 800, height = 600)
plot(results$k, results$accuracy, 
     type = "b", pch = 19, col = "blue", lwd = 2,
     xlab = "k Value", ylab = "Accuracy", 
     main = "k-NN: Accuracy vs k Value",
     ylim = c(min(results$accuracy) - 0.01, max(results$accuracy) + 0.01))
grid()
# Highlight best k
points(best_k, best_accuracy, col = "red", pch = 19, cex = 2)
text(best_k, best_accuracy + 0.005, paste("Best k =", best_k), 
     col = "red", pos = 3, font = 2)
dev.off()
cat("k-value tuning plot saved to ../graphics/knn_k_tuning.png\n")

# Final model with best k on larger test set
cat("\n--- FINAL MODEL EVALUATION ---\n")
test_subset_final <- 1:min(200, nrow(X_test))  # Use more samples for final evaluation
final_predictions <- knn_predict(X_train, y_train, X_test[test_subset_final, ], k = best_k)

# Confusion matrix
confusion_matrix <- table(Predicted = final_predictions, Actual = y_test[test_subset_final])
cat("Confusion Matrix (k =", best_k, "):\n")
print(confusion_matrix)

# Calculate metrics
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
precision_ad <- confusion_matrix[1,1] / sum(confusion_matrix[1,])
recall_ad <- confusion_matrix[1,1] / sum(confusion_matrix[,1])
f1_ad <- 2 * (precision_ad * recall_ad) / (precision_ad + recall_ad)

cat("\nPerformance Metrics:\n")
cat("Accuracy:", round(accuracy, 4), "\n")
cat("Precision (ad):", round(precision_ad, 4), "\n")
cat("Recall (ad):", round(recall_ad, 4), "\n")
cat("F1-score (ad):", round(f1_ad, 4), "\n")

# Plot confusion matrix
png("../graphics/knn_confusion_matrix.png", width = 600, height = 600)
par(mar = c(5, 5, 4, 2))
image(1:2, 1:2, as.matrix(confusion_matrix), 
      col = c("lightblue", "lightcoral", "orange", "lightgreen"),
      xlab = "Predicted", ylab = "Actual", 
      main = paste("k-NN Confusion Matrix (k =", best_k, ")"),
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
cat("Confusion matrix plot saved to ../graphics/knn_confusion_matrix.png\n")

cat("\n=== k-NN MODEL COMPLETED ===\n")

# Save results
knn_results <- list(
  best_k = best_k,
  accuracy = accuracy,
  confusion_matrix = confusion_matrix,
  k_comparison = results
)
save(knn_results, file = "knn_results.RData")
cat("k-NN results saved to knn_results.RData\n")