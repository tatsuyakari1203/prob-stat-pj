# Load necessary libraries
library(readr)
library(ggplot2)
library(GGally)

# Load the dataset
data <- read_csv("/home/kari/prob-stat-pj/3dprintdata/data.csv")

# --- Data Preprocessing ---

# Display the first few rows of the dataset
head(data)

# Get a summary of the dataset
summary(data)

# Check for missing values
print(paste("Number of missing values:", sum(is.na(data))))

# --- Descriptive Statistics ---

# Histograms for continuous variables

# Roughness
p_roughness_hist <- ggplot(data, aes(x = roughness)) +
  geom_histogram(binwidth = 10, fill = "blue", color = "black") +
  labs(title = "Distribution of Roughness", x = "Roughness (µm)", y = "Frequency")
ggsave(filename = "/home/kari/prob-stat-pj/graphics/04-roughness_histogram.png", plot = p_roughness_hist)

# Tensile Strength
p_tension_hist <- ggplot(data, aes(x = tension_strenght)) +
  geom_histogram(binwidth = 2, fill = "green", color = "black") +
  labs(title = "Distribution of Tensile Strength", x = "Tensile Strength (Pa)", y = "Frequency")
ggsave(filename = "/home/kari/prob-stat-pj/graphics/04-tensile_strength_histogram.png", plot = p_tension_hist)

# Elongation
p_elongation_hist <- ggplot(data, aes(x = elongation)) +
  geom_histogram(binwidth = 0.2, fill = "red", color = "black") +
  labs(title = "Distribution of Elongation", x = "Elongation (mm)", y = "Frequency")
ggsave(filename = "/home/kari/prob-stat-pj/graphics/04-elongation_histogram.png", plot = p_elongation_hist)

# Boxplots

# Roughness vs. infill_pattern
p_roughness_box <- ggplot(data, aes(x = infill_pattern, y = roughness)) +
    geom_boxplot() +
    labs(title = "Roughness by Infill Pattern", x = "Infill Pattern", y = "Roughness (µm)")
ggsave(filename = "/home/kari/prob-stat-pj/graphics/04-roughness_vs_infill_pattern_boxplot.png", plot = p_roughness_box)

# tension_strength vs. infill_pattern
p_tension_box <- ggplot(data, aes(x = infill_pattern, y = tension_strenght)) +
    geom_boxplot() +
    labs(title = "Tensile Strength by Infill Pattern", x = "Infill Pattern", y = "Tensile Strength (Pa)")
ggsave(filename = "/home/kari/prob-stat-pj/graphics/04-tension_strength_vs_infill_pattern_boxplot.png", plot = p_tension_box)

# elongation vs. infill_pattern
p_elongation_box <- ggplot(data, aes(x = infill_pattern, y = elongation)) +
    geom_boxplot() +
    labs(title = "Elongation by Infill Pattern", x = "Infill Pattern", y = "Elongation (mm)")
ggsave(filename = "/home/kari/prob-stat-pj/graphics/04-elongation_vs_infill_pattern_boxplot.png", plot = p_elongation_box)

# Scatter plots

# Select continuous variables
continuous_vars <- data[, c("layer_height", "nozzle_temperature", "bed_temperature", "print_speed", "infill_density", "wall_thickness", "fan_speed")]

# Scatter plot for tension_strength
png("/home/kari/prob-stat-pj/graphics/04-tension_strength_scatter.png", width=800, height=600)
plot(continuous_vars, pch = 16, col = "blue", main = "Tensile Strength vs. Continuous Variables")
dev.off()

# Scatter plot for roughness
png("/home/kari/prob-stat-pj/graphics/04-roughness_scatter.png", width=800, height=600)
plot(continuous_vars, pch = 16, col = "red", main = "Roughness vs. Continuous Variables")
dev.off()

# Scatter plot for elongation
png("/home/kari/prob-stat-pj/graphics/04-elongation_scatter.png", width=800, height=600)
plot(continuous_vars, pch = 16, col = "green", main = "Elongation vs. Continuous Variables")
dev.off()

# Correlation matrix

# Calculate correlation matrix
correlation_matrix <- cor(data[, unlist(lapply(data, is.numeric))])

# --- Inferential Statistics: Multiple Linear Regression ---

# Model for Roughness
model_roughness <- lm(roughness ~ layer_height + nozzle_temperature + bed_temperature + print_speed + infill_density + wall_thickness + fan_speed + as.factor(infill_pattern) + as.factor(material), data = data)
summary(model_roughness)

# Model for Tensile Strength
model_tension <- lm(tension_strenght ~ layer_height + nozzle_temperature + bed_temperature + print_speed + infill_density + wall_thickness + fan_speed + as.factor(infill_pattern) + as.factor(material), data = data)
summary(model_tension)

# Model for Elongation
model_elongation <- lm(elongation ~ layer_height + nozzle_temperature + bed_temperature + print_speed + infill_density + wall_thickness + fan_speed + as.factor(infill_pattern) + as.factor(material), data = data)
summary(model_elongation)

# --- Model Diagnostics ---

# Residuals vs. Fitted plot for roughness model
png("/home/kari/prob-stat-pj/graphics/05-residuals_vs_fitted_roughness.png", width=800, height=600)
plot(model_roughness, 1)
dev.off()

# Normal Q-Q plot for roughness model
png("/home/kari/prob-stat-pj/graphics/05-qq_plot_roughness.png", width=800, height=600)
plot(model_roughness, 2)
dev.off()

# Residuals vs. Fitted plot for tensile strength model
png("/home/kari/prob-stat-pj/graphics/05-residuals_vs_fitted_tension.png", width=800, height=600)
plot(model_tension, 1)
dev.off()

# Normal Q-Q plot for tensile strength model
png("/home/kari/prob-stat-pj/graphics/05-qq_plot_tension.png", width=800, height=600)
plot(model_tension, 2)
dev.off()

# Residuals vs. Fitted plot for elongation model
png("/home/kari/prob-stat-pj/graphics/05-residuals_vs_fitted_elongation.png", width=800, height=600)
plot(model_elongation, 1)
dev.off()

# Normal Q-Q plot for elongation model
png("/home/kari/prob-stat-pj/graphics/05-qq_plot_elongation.png", width=800, height=600)
plot(model_elongation, 2)
dev.off()