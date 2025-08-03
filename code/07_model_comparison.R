# =============================================================================
# 07_model_comparison.R
# Model Comparison and Final Results
# =============================================================================

# Clear environment
rm(list = ls())

cat("=== MODEL COMPARISON AND ANALYSIS ===\n")

# Load all model results
cat("\n--- LOADING MODEL RESULTS ---\n")
load("knn_results.RData")
load("decision_tree_results.RData")
load("random_forest_results.RData")

cat("All model results loaded successfully\n")

# Extract performance metrics
cat("\n--- PERFORMANCE COMPARISON ---\n")

# Calculate k-NN metrics from confusion matrix
knn_cm <- knn_results$confusion_matrix
knn_precision_ad <- knn_cm[1,1] / (knn_cm[1,1] + knn_cm[1,2])
knn_recall_ad <- knn_cm[1,1] / (knn_cm[1,1] + knn_cm[2,1])
knn_f1_ad <- 2 * (knn_precision_ad * knn_recall_ad) / (knn_precision_ad + knn_recall_ad)

# Create comparison table
model_names <- c("k-NN (k=3)", "Decision Tree", "Random Forest")
accuracies <- c(
  knn_results$accuracy,
  dt_results$accuracy,
  rf_results$accuracy
)
precisions <- c(
  knn_precision_ad,
  dt_results$precision_ad,
  rf_results$precision_ad
)
recalls <- c(
  knn_recall_ad,
  dt_results$recall_ad,
  rf_results$recall_ad
)
f1_scores <- c(
  knn_f1_ad,
  dt_results$f1_ad,
  rf_results$f1_ad
)

# Create comparison data frame
comparison_df <- data.frame(
  Model = model_names,
  Accuracy = round(accuracies, 4),
  Precision = round(precisions, 4),
  Recall = round(recalls, 4),
  F1_Score = round(f1_scores, 4)
)

cat("Model Performance Comparison:\n")
print(comparison_df)

# Find best model for each metric
cat("\n--- BEST MODELS BY METRIC ---\n")
best_accuracy <- model_names[which.max(accuracies)]
best_precision <- model_names[which.max(precisions)]
best_recall <- model_names[which.max(recalls)]
best_f1 <- model_names[which.max(f1_scores)]

cat("Best Accuracy:", best_accuracy, "(", round(max(accuracies), 4), ")\n")
cat("Best Precision:", best_precision, "(", round(max(precisions), 4), ")\n")
cat("Best Recall:", best_recall, "(", round(max(recalls), 4), ")\n")
cat("Best F1-Score:", best_f1, "(", round(max(f1_scores), 4), ")\n")

# Generate comparison plots
cat("\n--- GENERATING COMPARISON PLOTS ---\n")

# Performance metrics comparison
png("../graphics/07-comparison-model_performance_comparison.png", width = 1000, height = 600)
par(mfrow = c(2, 2), mar = c(8, 5, 4, 2))

# Accuracy comparison
barplot(comparison_df$Accuracy, names.arg = model_names, 
        main = "Model Accuracy Comparison", ylab = "Accuracy",
        col = c("lightblue", "lightcoral", "lightgreen"), las = 2)

# Precision comparison
barplot(comparison_df$Precision, names.arg = model_names,
        main = "Model Precision Comparison", ylab = "Precision",
        col = c("lightblue", "lightcoral", "lightgreen"), las = 2)

# Recall comparison
barplot(comparison_df$Recall, names.arg = model_names,
        main = "Model Recall Comparison", ylab = "Recall",
        col = c("lightblue", "lightcoral", "lightgreen"), las = 2)

# F1-Score comparison
barplot(comparison_df$F1_Score, names.arg = model_names,
        main = "Model F1-Score Comparison", ylab = "F1-Score",
        col = c("lightblue", "lightcoral", "lightgreen"), las = 2)

dev.off()
cat("Performance comparison plot saved to ../graphics/07-comparison-model_performance_comparison.png\n")

# Combined confusion matrices visualization
png("../graphics/07-comparison-all_confusion_matrices.png", width = 1200, height = 400)
par(mfrow = c(1, 3), mar = c(5, 5, 4, 2))

# k-NN confusion matrix
knn_cm <- knn_results$confusion_matrix
image(1:2, 1:2, as.matrix(knn_cm), 
      col = c("lightblue", "lightcoral", "orange", "lightgreen"),
      xlab = "Predicted", ylab = "Actual", 
      main = "k-NN Confusion Matrix", axes = FALSE)
axis(1, at = 1:2, labels = colnames(knn_cm))
axis(2, at = 1:2, labels = rownames(knn_cm))
for(i in 1:2) {
  for(j in 1:2) {
    text(j, i, knn_cm[i,j], cex = 1.5, font = 2)
  }
}

# Decision Tree confusion matrix
dt_cm <- dt_results$confusion_matrix
image(1:2, 1:2, as.matrix(dt_cm), 
      col = c("lightblue", "lightcoral", "orange", "lightgreen"),
      xlab = "Predicted", ylab = "Actual", 
      main = "Decision Tree Confusion Matrix", axes = FALSE)
axis(1, at = 1:2, labels = colnames(dt_cm))
axis(2, at = 1:2, labels = rownames(dt_cm))
for(i in 1:2) {
  for(j in 1:2) {
    text(j, i, dt_cm[i,j], cex = 1.5, font = 2)
  }
}

# Random Forest confusion matrix
rf_cm <- rf_results$confusion_matrix
image(1:2, 1:2, as.matrix(rf_cm), 
      col = c("lightblue", "lightcoral", "orange", "lightgreen"),
      xlab = "Predicted", ylab = "Actual", 
      main = "Random Forest Confusion Matrix", axes = FALSE)
axis(1, at = 1:2, labels = colnames(rf_cm))
axis(2, at = 1:2, labels = rownames(rf_cm))
for(i in 1:2) {
  for(j in 1:2) {
    text(j, i, rf_cm[i,j], cex = 1.5, font = 2)
  }
}

dev.off()
cat("Combined confusion matrices plot saved to ../graphics/all_confusion_matrices.png\n")

# Overall best model (based on F1-score)
overall_best <- model_names[which.max(f1_scores)]
cat("\nOverall Best Model (by F1-score):", overall_best, "\n")

# Detailed confusion matrices
cat("\n--- DETAILED CONFUSION MATRICES ---\n")

cat("\nk-NN Confusion Matrix:\n")
print(knn_results$confusion_matrix)

cat("\nDecision Tree Confusion Matrix:\n")
print(dt_results$confusion_matrix)

cat("\nRandom Forest Confusion Matrix:\n")
print(rf_results$confusion_matrix)

# Model characteristics analysis
cat("\n--- MODEL CHARACTERISTICS ANALYSIS ---\n")

cat("\nk-NN Model:\n")
cat("- Best k value: 3\n")
cat("- Uses distance-based classification\n")
cat("- Requires feature normalization\n")
cat("- Non-parametric method\n")

cat("\nDecision Tree Model:\n")
cat("- Max depth: 3 (for interpretability)\n")
cat("- Uses 20 features\n")
cat("- Highly interpretable\n")
cat("- Prone to overfitting\n")

cat("\nRandom Forest Model:\n")
cat("- Number of trees:", rf_results$n_trees, "\n")
cat("- Features per split:", rf_results$m_features, "\n")
cat("- Uses all", length(rf_results$feature_importance), "features\n")
cat("- Ensemble method - reduces overfitting\n")
cat("- Good balance of accuracy and generalization\n")

# Feature importance from Random Forest
cat("\n--- RANDOM FOREST FEATURE IMPORTANCE ---\n")
cat("Top 10 most important features:\n")
print(head(rf_results$feature_importance, 10))

# Class imbalance analysis
cat("\n--- CLASS IMBALANCE ANALYSIS ---\n")
cat("Dataset has class imbalance:\n")
cat("- 'ad' class: 459 samples (14.0%)\n")
cat("- 'nonad' class: 2820 samples (86.0%)\n")
cat("\nImpact on models:\n")
cat("- All models show high precision but lower recall for 'ad' class\n")
cat("- This indicates models are conservative in predicting 'ad'\n")
cat("- Random Forest shows best balance with perfect precision\n")

# Recommendations
cat("\n--- RECOMMENDATIONS ---\n")
cat("\nBased on the analysis:\n")
cat("\n1. BEST OVERALL MODEL: Random Forest\n")
cat("   - Highest accuracy (", round(rf_results$accuracy, 4), ")\n")
cat("   - Perfect precision (1.0000)\n")
cat("   - Good generalization due to ensemble approach\n")
cat("   - Handles high-dimensional data well\n")

cat("\n2. MOST INTERPRETABLE: Decision Tree\n")
cat("   - Simple tree structure\n")
cat("   - Easy to understand decision rules\n")
cat("   - Good accuracy (", round(dt_results$accuracy, 4), ")\n")

cat("\n3. SIMPLEST APPROACH: k-NN\n")
cat("   - Non-parametric\n")
cat("   - Good performance with k=3\n")
cat("   - Requires careful feature scaling\n")

# Final summary
cat("\n--- FINAL SUMMARY ---\n")
cat("\nFor the Internet Advertisement Classification task:\n")
cat("- Random Forest is recommended for production use\n")
cat("- Decision Tree is recommended for explanatory analysis\n")
cat("- All models handle the classification task well\n")
cat("- Class imbalance affects recall but not precision\n")
cat("- Feature engineering could further improve performance\n")

cat("\n=== MODEL COMPARISON COMPLETED ===\n")

# Save comparison results
final_results <- list(
  comparison_table = comparison_df,
  best_models = list(
    accuracy = best_accuracy,
    precision = best_precision,
    recall = best_recall,
    f1_score = best_f1,
    overall = overall_best
  ),
  knn_results = list(
    accuracy = knn_results$accuracy,
    precision_ad = knn_precision_ad,
    recall_ad = knn_recall_ad,
    f1_ad = knn_f1_ad,
    confusion_matrix = knn_results$confusion_matrix,
    best_k = knn_results$best_k
  ),
  dt_results = dt_results,
  rf_results = rf_results
)

save(final_results, file = "final_comparison_results.RData")
cat("Final comparison results saved to final_comparison_results.RData\n")

# Export comprehensive CSV summary
cat("\n--- EXPORTING CSV SUMMARY ---\n")

# 1. Model performance comparison CSV
write.csv(comparison_df, "model_performance_comparison.csv", row.names = FALSE)
cat("Model performance comparison exported to model_performance_comparison.csv\n")

# 2. Detailed confusion matrices CSV
confusion_matrices <- data.frame(
  Model = rep(c("kNN", "Decision_Tree", "Random_Forest"), each = 4),
  Actual = rep(c("ad", "ad", "nonad", "nonad"), 3),
  Predicted = rep(c("ad", "nonad", "ad", "nonad"), 3),
  Count = c(
    as.vector(knn_results$confusion_matrix),
    as.vector(dt_results$confusion_matrix),
    as.vector(rf_results$confusion_matrix)
  )
)
write.csv(confusion_matrices, "confusion_matrices.csv", row.names = FALSE)
cat("Confusion matrices exported to confusion_matrices.csv\n")

# 3. Model characteristics CSV
model_characteristics <- data.frame(
  Model = c("kNN", "Decision_Tree", "Random_Forest"),
  Best_Parameter = c(
    paste("k =", knn_results$best_k),
    "max_depth = 3",
    paste("n_trees =", rf_results$n_trees)
  ),
  Features_Used = c(
    "All (normalized)",
    "First 20",
    "All"
  ),
  Method_Type = c("Instance-based", "Tree-based", "Ensemble"),
  Interpretability = c("Medium", "High", "Low")
)
write.csv(model_characteristics, "model_characteristics.csv", row.names = FALSE)
cat("Model characteristics exported to model_characteristics.csv\n")

# 4. Feature importance from Random Forest CSV
if(exists("rf_results") && !is.null(rf_results$feature_importance) && length(rf_results$feature_importance) > 0) {
  # Check if feature_importance is a named vector or list
  if(is.vector(rf_results$feature_importance) && !is.null(names(rf_results$feature_importance))) {
    feature_importance_df <- data.frame(
      Feature = names(rf_results$feature_importance),
      Importance = as.numeric(rf_results$feature_importance),
      stringsAsFactors = FALSE
    )
    feature_importance_df <- feature_importance_df[order(feature_importance_df$Importance, decreasing = TRUE), ]
    write.csv(feature_importance_df, "feature_importance.csv", row.names = FALSE)
    cat("Feature importance exported to feature_importance.csv\n")
  } else {
    cat("Feature importance data format not suitable for CSV export\n")
  }
} else {
  cat("No feature importance data available for export\n")
}

# 5. Best models summary CSV
best_models_df <- data.frame(
  Metric = c("Accuracy", "Precision", "Recall", "F1_Score", "Overall"),
  Best_Model = c(best_accuracy, best_precision, best_recall, best_f1, overall_best),
  Best_Value = c(
    round(max(accuracies), 4),
    round(max(precisions), 4),
    round(max(recalls), 4),
    round(max(f1_scores), 4),
    round(max(f1_scores), 4)
  )
)
write.csv(best_models_df, "best_models_summary.csv", row.names = FALSE)
cat("Best models summary exported to best_models_summary.csv\n")