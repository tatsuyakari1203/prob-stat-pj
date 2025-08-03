# =============================================================================
# install_packages.R
# Install required R packages
# =============================================================================

cat("Installing required R packages...\n")

# List of required packages
packages <- c("readr", "dplyr", "ggplot2", "randomForest", "class", "rpart", "rpart.plot")

# Install packages if not already installed
for(pkg in packages) {
  if(!require(pkg, character.only = TRUE)) {
    cat("Installing", pkg, "...\n")
    install.packages(pkg, repos = "https://cran.rstudio.com/")
    library(pkg, character.only = TRUE)
  } else {
    cat(pkg, "is already installed\n")
  }
}

cat("All packages installed successfully!\n")