import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import joblib
from tqdm import tqdm
import streamlit as st

def predict_ungauged(model, gauged_data_path, ungauged_data_path, scalers, model_params):
    """
    Generate predictions for ungauged catchment
    
    Parameters:
    -----------
    model : PINNModel
        Trained model
    gauged_data_path : str
        Path to gauged catchment data
    ungauged_data_path : str
        Path to ungauged catchment data
    scalers : dict
        Dictionary of scalers
    model_params : dict
        Dictionary of model parameters
        
    Returns:
    --------
    predictions : ndarray
        Predicted streamflow values
    dates : ndarray
        Dates corresponding to predictions
    visualizations : dict
        Dictionary of visualization figures
    """
    try:
        from utils.data_processing import preprocess_ungauged_data
        from utils.evaluation import calculate_flow_duration_curve
        
        # Load feature information from gauged data
        gauged_data = pd.read_excel(gauged_data_path)
        
        # Define dynamic feature columns
        dynamic_feature_cols = ['Rainfall (mm)', 'Maximum temperature (°C)', 'Minimum temperature (°C)', 
                               'Daily global solar exposure (MJ/m*m)']
        
        num_lagged_features = model_params.get('NUM_LAGGED_FEATURES', 12)
        
        lagged_discharge_cols = [f'Lag_Discharge_{i}' for i in range(1, num_lagged_features + 1)]
        
        lagged_weather_cols = [f'Lag_Rainfall_{i}' for i in range(1, num_lagged_features + 1)] + \
                             [f'Lag_TempMax_{i}' for i in range(1, num_lagged_features + 1)] + \
                             [f'Lag_TempMin_{i}' for i in range(1, num_lagged_features + 1)] + \
                             [f'Lag_Solar_{i}' for i in range(1, num_lagged_features + 1)]
        
        seasonality_cols = ['Month_sin', 'Month_cos']
        
        # Define the full list of dynamic features
        all_dynamic_cols = dynamic_feature_cols + lagged_discharge_cols + lagged_weather_cols + seasonality_cols
        
        # Calculate the indices of lagged discharge features
        lagged_discharge_indices = [all_dynamic_cols.index(f'Lag_Discharge_{i}') for i in range(1, num_lagged_features + 1)]
        
        # Create feature info dictionary
        feature_info = {
            'dynamic_cols': dynamic_feature_cols,
            'lagged_discharge_cols': lagged_discharge_cols,
            'lagged_weather_cols': lagged_weather_cols,
            'seasonality_cols': seasonality_cols,
            'all_dynamic_cols': all_dynamic_cols,
            'static_numeric_cols': ['Land Use Percentage', 'Soil Percentage', 'Slope (Degree)'],
            'lagged_discharge_indices': lagged_discharge_indices
        }
        
        if 'Drainage Density (km/km²)' in gauged_data.columns:
            feature_info['static_numeric_cols'].append('Drainage Density (km/km²)')
        
        # Preprocess ungauged data
        ungauged_data, message = preprocess_ungauged_data(
            gauged_data_path, 
            ungauged_data_path, 
            scalers, 
            feature_info, 
            num_lagged_features
        )
        
        if ungauged_data is None:
            st.error(message)
            return None, None, None
        
        # Extract data
        X_ungauged = ungauged_data['X_ungauged']
        dates = ungauged_data['dates']
        
        # Iterative prediction
        n_steps = X_ungauged.shape[0]
        y_pred_scaled_ungauged = np.zeros((n_steps, 1))
        X_ungauged_dynamic = X_ungauged.copy()
        
        prediction_batch_size = model_params.get('PREDICTION_BATCH_SIZE', 100)
        
        st.info(f"Starting iterative prediction for {n_steps} steps with batch size {prediction_batch_size}...")
        
        progress_bar = st.progress(0)
        
        for start_idx in range(0, n_steps, prediction_batch_size):
            end_idx = min(start_idx + prediction_batch_size, n_steps)
            batch_X = X_ungauged_dynamic[start_idx:end_idx]
            batch_preds = model.predict(batch_X, verbose=0)
            y_pred_scaled_ungauged[start_idx:end_idx] = batch_preds
            
            # Update progress bar
            progress = (end_idx) / n_steps
            progress_bar.progress(progress)
            
            # Update lagged discharge for the next batch
            for t in range(start_idx, end_idx):
                if t < n_steps - 1:
                    for lag in range(1, min(num_lagged_features + 1, t + 2 - start_idx)):
                        lag_idx = t - lag
                        if lag_idx >= start_idx:
                            X_ungauged_dynamic[t + 1, 0, lagged_discharge_indices[lag-1]] = y_pred_scaled_ungauged[lag_idx]
        
        # Inverse transform predictions
        y_pred_ungauged = scalers['y'].inverse_transform(y_pred_scaled_ungauged)
        
        # Ensure non-negative predictions
        y_pred_ungauged = np.abs(y_pred_ungauged)
        
        # Generate visualizations
        visualizations = {}
        
        # Predicted streamflow
        fig_pred = plt.figure(figsize=(16, 8), dpi=300)
        plt.plot(dates, y_pred_ungauged.flatten(), label='Predicted Streamflow', color='red', linewidth=1.5)
        plt.title('Predicted Streamflow - Ungauged Catchment', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Streamflow (m³/s)', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        visualizations['ungauged_predicted'] = fig_pred
        
        # Flow duration curve
        pred_fdc, exceedance_prob = calculate_flow_duration_curve(y_pred_ungauged.flatten())
        fig_fdc = plt.figure(figsize=(12, 8), dpi=300)
        plt.plot(exceedance_prob, pred_fdc, label='Predicted', color='red', linewidth=2)
        plt.title('Flow Duration Curve - Predicted Streamflow (Ungauged Catchment)', fontsize=16, fontweight='bold')
        plt.xlabel('Exceedance Probability (%)', fontsize=14)
        plt.ylabel('Streamflow (m³/s)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(0.1, 100)
        plt.tight_layout()
        visualizations['ungauged_predicted_fdc'] = fig_fdc
        
        # Create a summary statistics table
        stats = {
            'Min': np.min(y_pred_ungauged),
            'Max': np.max(y_pred_ungauged),
            'Mean': np.mean(y_pred_ungauged),
            'Median': np.median(y_pred_ungauged),
            'Std Dev': np.std(y_pred_ungauged),
            'Q10 (90% exceedance)': np.percentile(y_pred_ungauged, 10),
            'Q50 (50% exceedance)': np.percentile(y_pred_ungauged, 50),
            'Q90 (10% exceedance)': np.percentile(y_pred_ungauged, 90)
        }
        
        # Create a monthly statistics figure
        monthly_data = pd.DataFrame({
            'Date': dates,
            'Streamflow': y_pred_ungauged.flatten()
        })
        monthly_data['Month'] = monthly_data['Date'].dt.month
        monthly_data['Year'] = monthly_data['Date'].dt.year
        
        monthly_stats = monthly_data.groupby('Month')['Streamflow'].agg(['mean', 'min', 'max', 'std'])
        
        fig_monthly = plt.figure(figsize=(12, 8))
        plt.plot(monthly_stats.index, monthly_stats['mean'], 'o-', color='blue', label='Mean')
        plt.fill_between(monthly_stats.index, 
                        monthly_stats['mean'] - monthly_stats['std'],
                        monthly_stats['mean'] + monthly_stats['std'],
                        alpha=0.2, color='blue')
        plt.plot(monthly_stats.index, monthly_stats['min'], 's--', color='green', label='Min')
        plt.plot(monthly_stats.index, monthly_stats['max'], '^--', color='red', label='Max')
        plt.title('Monthly Streamflow Statistics - Ungauged Catchment', fontsize=16, fontweight='bold')
        plt.xlabel('Month', fontsize=14)
        plt.ylabel('Streamflow (m³/s)', fontsize=14)
        plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        visualizations['monthly_stats'] = fig_monthly
        
        return y_pred_ungauged.flatten(), dates, visualizations
    
    except Exception as e:
        st.error(f"Error predicting ungauged catchment: {str(e)}")
        return None, None, None

def save_predictions(predictions, dates, output_path):
    """
    Save predictions to Excel file
    
    Parameters:
    -----------
    predictions : ndarray
        Predicted streamflow values
    dates : ndarray
        Dates corresponding to predictions
    output_path : str
        Path to save predictions to
        
    Returns:
    --------
    success : bool
        Whether the predictions were saved successfully
    message : str
        Success or error message
    """
    try:
        # Create DataFrame with predictions
        predictions_df = pd.DataFrame({
            'Date': dates,
            'Predicted_Streamflow': predictions
        })
        
        # Save to Excel
        predictions_df.to_excel(output_path, index=False)
        
        return True, f"Predictions saved successfully to {output_path}"
    
    except Exception as e:
        return False, f"Error saving predictions: {str(e)}"

def batch_predict(model, X_data, batch_size=100):
    """
    Make predictions in batches to avoid memory issues
    
    Parameters:
    -----------
    model : PINNModel
        Trained model
    X_data : ndarray
        Input data
    batch_size : int
        Batch size for predictions
        
    Returns:
    --------
    predictions : ndarray
        Predicted values
    """
    n_samples = X_data.shape[0]
    predictions = np.zeros((n_samples, 1))
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_X = X_data[start_idx:end_idx]
        predictions[start_idx:end_idx] = model.predict(batch_X, verbose=0)
    
    return predictions
