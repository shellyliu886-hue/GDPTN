# GDPTN: \underline{\textbf{G}}enerative \underline{\textbf{D}}ynamic \underline{\textbf{P}}anel \underline{\textbf{T}}obit \underline{\textbf{N}}etwork

## Project Overview
**GDPTN** is a specialized toolkit designed for handling **Censored Data**. This project integrates advanced data imputation (padding) algorithms with **Tobit Regression** analysis to address data truncation or censoring issues caused by observational limits in practical research.

## Key Features
* **CDP-GAN**: Utilizes generative models and other imputation techniques to scientifically complete restricted or missing censored data.
* **PTRN**: Provides a regression model implementation specifically for censored dependent variables. It estimates causal relationships while overcoming the biases inherent in Ordinary Least Squares (OLS) when dealing with censored data.

## File Descriptions
* `Main(Synthetic Experiments).py`: Main execution script for synthetic data experiments to test model performance in controlled environments.
* `Main(Empirical Experiment).py`: Main execution script for empirical analysis using real-world observational data.
* `CDP_GAN.py`: Core model file, featuring the Generative Adversarial Network (GAN) implementation designed for censored data imputation.
* `PTRN.py`: Core algorithm and network architecture definitions specific to the project.
* `data_handler.py`: Data preprocessing module responsible for loading, cleaning, and formatting censored datasets.
* `utils.py`: Utility function library including evaluation metrics calculation and visualization helpers.
