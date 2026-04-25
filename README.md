# 🏠 Addis Ababa House Price Predictor

A machine learning-powered web application for predicting house prices in Addis Ababa, Ethiopia with bulk purchase discounts and complexity analysis.

## Features

### 1. Single House Prediction
- Manual input of property details
- Instant price prediction
- Comparison with market averages

### 2. Mass Prediction (Bulk)
- Upload CSV file with multiple properties
- Bulk price predictions
- **10% discount for purchases of more than 2 houses**
- Download results and summary reports

### 3. Complexity Analysis
- Detect overfitting and underfitting
- Analyze tree depth effects
- Understand bias-variance tradeoff

### 4. Model Training
- Train Random Forest models with complexity control
- Adjustable parameters (tree depth, min samples)
- Cross-validation for robust performance

## Installation

```bash
# Make scripts executable
chmod +x install.sh run.sh

# Install dependencies
./install.sh

# Run the app
./run.sh