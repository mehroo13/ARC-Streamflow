# Hydrological Streamflow Prediction Web App

A Streamlit web application for hydrological streamflow prediction using Physics-Informed Neural Networks (PINN) with GRU architecture. This application allows users to predict streamflow in both gauged and ungauged catchments with comprehensive evaluation metrics and visualizations.

## Features

- **Interactive UI**: Adjust model parameters, training settings, and visualization options
- **Physics-Informed Neural Network**: GRU architecture with water balance constraints
- **Comprehensive Evaluation**: NSE, KGE, PBIAS, and flow duration biases
- **Advanced Visualizations**: Hydrographs, scatter plots, flow duration curves, and event analysis
- **Model Management**: Save and load trained models for future use
- **Ungauged Prediction**: Predict streamflow in ungauged catchments using trained models

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/hydro_streamflow_app.git
cd hydro_streamflow_app

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run the Streamlit app
streamlit run app.py
```

## Data Format

The application expects Excel files with the following columns:

**Gauged Catchment Data:**
- Date: Date column in datetime format
- Discharge (m³/S): Observed streamflow
- Rainfall (mm): Precipitation data
- Maximum temperature (°C): Maximum daily temperature
- Minimum temperature (°C): Minimum daily temperature
- Daily global solar exposure (MJ/m*m): Solar radiation data
- Land Use Classes, Soil Classes: Categorical variables
- Land Use Percentage, Soil Percentage, Slope (Degree): Numeric variables

**Ungauged Catchment Data:**
- Date: Date column in datetime format
- Rainfall (mm): Precipitation data
- Maximum temperature (°C): Maximum daily temperature
- Minimum temperature (°C): Minimum daily temperature
- Daily global solar exposure (MJ/m*m): Solar radiation data
- Land Use Classes, Soil Classes: Categorical variables
- Land Use Percentage, Soil Percentage, Slope (Degree): Numeric variables

## Model Architecture

The PINN-GRU model architecture includes:

1. Bidirectional GRU layer with attention mechanism
2. Multiple dense layers with batch normalization and dropout
3. Physics-informed loss function incorporating water balance principles
4. Custom training process with masked lagged discharge features

## Adjustable Parameters

- **Model Parameters**: GRU units, dense layer units, dropout rate
- **Training Parameters**: Learning rate, batch size, epochs, physics loss weight
- **Data Parameters**: Number of lagged features, train-test split ratio
- **Early Stopping**: Patience, minimum delta
- **Learning Rate Scheduler**: Reduction factor, patience

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- The PINN-GRU model architecture is based on research in hydrological modeling using deep learning
- Visualization components are inspired by best practices in hydrological model evaluation
