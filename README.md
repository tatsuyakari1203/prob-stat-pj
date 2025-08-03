# Internet Advertisement Dataset Analysis

This project analyzes the "Internet Advertisements Data Set" from the UCI Machine Learning Repository to build a classification model that predicts whether an image is an advertisement ("ad") or not ("nonad").

## Project Description

### About the Dataset
- **Task**: Predict whether an image is an advertisement ("ad") or not ("nonad")
- **Size**: 1559 data columns
- **Structure**: Each row represents an image labeled "ad" or "nonad" in the last column
- **Attributes**: Columns 0-1557 represent numerical attributes of the image
- **Source**: UCI Machine Learning Repository

### Project Requirements
- Use statistical methods not covered in class to analyze the dataset
- Describe and clean the data
- Descriptive statistics: histogram, boxplot, scatter plot, mean, median, etc.
- Define objectives and statistical methods to achieve them
- Analyze data with R code and explain results

## Directory Structure

- `/code`: Contains all R scripts for data analysis, from loading and cleaning to building and comparing models
- `01_data_loading.R`: Data loading
- `02_data_cleaning.R`: Data cleaning
- `03_eda.R`: Exploratory data analysis
- `04_knn_model.R`: K-Nearest Neighbors model
- `05_decision_tree.R`: Decision tree model
- `06_random_forest.R`: Random Forest model
- `07_model_comparison.R`: Model comparison
- `08_final_summary.R`: Final summary
- `09_plot_summary.R`: Summary plot generation
- `*.csv`: Data files generated from the analysis process
- `*_log.txt`: Log files for each analysis step
- `/graphics`: Contains all charts and images generated during the analysis process
- `/chapters`: Contains LaTeX files for report chapters
- `introduction.tex`: Introduction
- `data_description.tex`: Data description
- `descriptive_statistics.tex`: Descriptive statistics
- `objective.tex`: Objectives and methodology
- `/refs`: Contains BibTeX file for references
- `report.tex`: Main LaTeX file for compiling the report

- `hcmut-report.cls`: Class file for HCMUT report

## Environment Setup

### Installing R on Linux/WSL

#### Ubuntu/Debian:
```bash
# Update package list
sudo apt update

# Install R
sudo apt install r-base r-base-dev

# Install dependencies for R packages
sudo apt install libcurl4-openssl-dev libssl-dev libxml2-dev
sudo apt install libcairo2-dev libxt-dev
```

#### CentOS/RHEL/Fedora:
```bash
# Fedora
sudo dnf install R R-devel

# CentOS/RHEL (requires EPEL repository)
sudo yum install epel-release
sudo yum install R R-devel
```

### Installing TeXLive on Linux/WSL

#### Ubuntu/Debian:
```bash
# Install TeXLive full
sudo apt install texlive-full

# Or install minimal and add required packages
sudo apt install texlive-latex-base texlive-latex-recommended
sudo apt install texlive-latex-extra texlive-fonts-recommended
sudo apt install texlive-bibtex-extra biber
```

#### CentOS/RHEL/Fedora:
```bash
# Fedora
sudo dnf install texlive-scheme-full

# CentOS/RHEL
sudo yum install texlive texlive-latex
```

### Verify Installation
```bash
# Check R
R --version

# Check LaTeX
pdflatex --version
biber --version
```

## How to Run the Project

### 1. Install Required R Packages
```bash
Rscript code/install_packages.R
```

### 2. Run Analysis Pipeline
```bash
# Run all scripts in order
bash code/run_all_scripts.sh

# Or run individual scripts
Rscript code/01_data_loading.R
Rscript code/02_data_cleaning.R
Rscript code/03_eda.R
Rscript code/04_knn_model.R
Rscript code/05_decision_tree.R
Rscript code/06_random_forest.R
Rscript code/07_model_comparison.R
Rscript code/08_final_summary.R
Rscript code/09_plot_summary.R
```

### 3. Compile LaTeX Report
```bash
# Compile report
pdflatex report.tex
biber report
pdflatex report.tex
pdflatex report.tex
```

## Results

After running the project, you will have:
- `report.pdf` file: Complete report
- CSV files in `/code`: Data analysis results
- Log files: Detailed execution process
- Charts in `/graphics`: Illustrative images

## References

Lichman, M. (2013). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

```bibtex
@misc{Lichman:2013,
author = "M. Lichman",
year = "2013",
title = "{UCI} Machine Learning Repository",
url = "http://archive.ics.uci.edu/ml",
institution = "University of California, Irvine, School of Information and Computer Sciences"
}
```
