import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import streamlit as st

def calculate_nse(observed, predicted):
    """
    Calculate Nash-Sutcliffe Efficiency
    """
    observed_mean = np.mean(observed)
    numerator = np.sum((observed - predicted) ** 2)
    denominator = np.sum((observed - observed_mean) ** 2)
    nse = 1 - (numerator / denominator)
    return nse

def calculate_kge(observed, predicted):
    """
    Calculate Kling-Gupta Efficiency
    """
    r, _ = pearsonr(observed, predicted)
    beta = np.mean(predicted) / np.mean(observed)
    gamma = (np.std(predicted) / np.mean(predicted)) / (np.std(observed) / np.mean(observed))
    kge = 1 - np.sqrt((r - 1)**2 + (beta - 1)**2 + (gamma - 1)**2 + 1e-8)
    return kge

def calculate_pbias(observed, predicted):
    """
    Calculate Percent Bias
    """
    p_bias = (np.sum(predicted - observed) / np.sum(observed)) * 100
    return p_bias

def calculate_fdc_bias(observed, predicted, high_flow_percentile=90, low_flow_percentile=10):
    """
    Calculate Flow Duration Curve Bias for high and low flows
    """
    observed_sorted = np.sort(observed)
    predicted_sorted = np.sort(predicted)
    n = len(observed)
    
    high_flow_threshold_index = int(n * (1 - (high_flow_percentile / 100)))
    low_flow_threshold_index = int(n * (1 - (low_flow_percentile / 100)))
    
    observed_high_flow = observed_sorted[:high_flow_threshold_index]
    predicted_high_flow = predicted_sorted[:high_flow_threshold_index]
    
    observed_low_flow = observed_sorted[low_flow_threshold_index:]
    predicted_low_flow = predicted_sorted[low_flow_threshold_index:]
    
    hf_bias = (np.sum(predicted_high_flow - observed_high_flow) / np.sum(observed_high_flow)) * 100 if np.sum(observed_high_flow) != 0 else np.nan
    lf_bias = (np.sum(predicted_low_flow - observed_low_flow) / np.sum(observed_low_flow)) * 100 if np.sum(observed_low_flow) != 0 else np.nan
    
    return hf_bias, lf_bias

def calculate_flow_duration_curve(flows):
    """
    Calculate Flow Duration Curve
    """
    sorted_flows = np.sort(flows)[::-1]
    exceedance_prob = np.arange(1, len(flows) + 1) / (len(flows) + 1) * 100
    return sorted_flows, exceedance_prob

def calculate_event_metrics(observed, predicted, event_threshold=0.7, min_duration=3):
    """
    Calculate metrics for high flow events
    """
    observed_events = []
    event_start = None
    
    # Normalize observed flow for event detection
    obs_norm = (observed - np.min(observed)) / (np.max(observed) - np.min(observed))
    
    # Identify events
    for i in range(len(obs_norm)):
        if obs_norm[i] > event_threshold and event_start is None:
            event_start = i
        elif obs_norm[i] <= event_threshold and event_start is not None:
            event_end = i - 1
            if event_end - event_start + 1 >= min_duration:
                observed_events.append((event_start, event_end))
            event_start = None
    
    # Add last event if it extends to the end of the time series
    if event_start is not None:
        event_end = len(obs_norm) - 1
        if event_end - event_start + 1 >= min_duration:
            observed_events.append((event_start, event_end))
    
    # Calculate metrics for each event
    event_metrics = []
    for start, end in observed_events:
        obs_event = observed[start:end+1]
        pred_event = predicted[start:end+1]
        
        event_nse = calculate_nse(obs_event, pred_event)
        event_peak_error = ((np.max(pred_event) - np.max(obs_event)) / np.max(obs_event)) * 100
        event_volume_error = ((np.sum(pred_event) - np.sum(obs_event)) / np.sum(obs_event)) * 100
        
        obs_peak_idx = np.argmax(obs_event)
        pred_peak_idx = np.argmax(pred_event)
        peak_timing_error = abs(obs_peak_idx - pred_peak_idx)
        
        event_metrics.append({
            'start_idx': start,
            'end_idx': end,
            'duration': end - start + 1,
            'observed_peak': np.max(obs_event),
            'predicted_peak': np.max(pred_event),
            'observed_volume': np.sum(obs_event),
            'predicted_volume': np.sum(pred_event),
            'nse': event_nse,
            'peak_error_percent': event_peak_error,
            'volume_error_percent': event_volume_error,
            'peak_timing_error_days': peak_timing_error
        })
    
    return event_metrics

def evaluate_model(model, X_train, X_test, y_train, y_test, dates_train, dates_test, scalers):
    """
    Evaluate model performance and generate visualizations
    
    Parameters:
    -----------
    model : PINNModel
        Trained model
    X_train : ndarray
        Training features
    X_test : ndarray
        Testing features
    y_train : ndarray
        Training targets
    y_test : ndarray
        Testing targets
    dates_train : ndarray
        Training dates
    dates_test : ndarray
        Testing dates
    scalers : dict
        Dictionary of scalers
        
    Returns:
    --------
    metrics : dict
        Dictionary of performance metrics
    visualizations : dict
        Dictionary of visualization figures
    """
    try:
        # Generate predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Inverse transform predictions
        y_train_pred = scalers['y'].inverse_transform(y_train_pred)
        y_test_pred = scalers['y'].inverse_transform(y_test_pred)
        y_train_actual = scalers['y'].inverse_transform(y_train)
        y_test_actual = scalers['y'].inverse_transform(y_test)
        
        # Ensure non-negative predictions
        y_train_pred = np.abs(y_train_pred)
        y_test_pred = np.abs(y_test_pred)
        
        # Calculate basic metrics
        train_rmse = np.sqrt(mean_squared_error(y_train_actual, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred))
        train_mae = mean_absolute_error(y_train_actual, y_train_pred)
        test_mae = mean_absolute_error(y_test_actual, y_test_pred)
        train_r2 = r2_score(y_train_actual, y_train_pred)
        test_r2 = r2_score(y_test_actual, y_test_pred)
        
        # Calculate hydrological metrics
        train_nse = calculate_nse(y_train_actual.flatten(), y_train_pred.flatten())
        test_nse = calculate_nse(y_test_actual.flatten(), y_test_pred.flatten())
        train_kge = calculate_kge(y_train_actual.flatten(), y_train_pred.flatten())
        test_kge = calculate_kge(y_test_actual.flatten(), y_test_pred.flatten())
        train_pbias = calculate_pbias(y_train_actual.flatten(), y_train_pred.flatten())
        test_pbias = calculate_pbias(y_test_actual.flatten(), y_test_pred.flatten())
        
        # Calculate FDC bias
        train_hf_bias, train_lf_bias = calculate_fdc_bias(y_train_actual.flatten(), y_train_pred.flatten())
        test_hf_bias, test_lf_bias = calculate_fdc_bias(y_test_actual.flatten(), y_test_pred.flatten())
        
        # Calculate flow duration curves
        train_obs_fdc, train_exceedance_prob = calculate_flow_duration_curve(y_train_actual.flatten())
        train_pred_fdc, _ = calculate_flow_duration_curve(y_train_pred.flatten())
        test_obs_fdc, test_exceedance_prob = calculate_flow_duration_curve(y_test_actual.flatten())
        test_pred_fdc, _ = calculate_flow_duration_curve(y_test_pred.flatten())
        
        # Calculate event metrics
        train_event_metrics = calculate_event_metrics(y_train_actual.flatten(), y_train_pred.flatten())
        test_event_metrics = calculate_event_metrics(y_test_actual.flatten(), y_test_pred.flatten())
        
        # Create metrics dictionary
        metrics = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_nse': train_nse,
            'test_nse': test_nse,
            'train_kge': train_kge,
            'test_kge': test_kge,
            'train_pbias': train_pbias,
            'test_pbias': test_pbias,
            'train_hf_bias': train_hf_bias,
            'train_lf_bias': train_lf_bias,
            'test_hf_bias': test_hf_bias,
            'test_lf_bias': test_lf_bias,
            'train_event_metrics': train_event_metrics,
            'test_event_metrics': test_event_metrics
        }
        
        # Generate visualizations
        visualizations = {}
        
        # Training hydrograph
        fig_train = plt.figure(figsize=(16, 8))
        plt.plot(dates_train, y_train_actual, label='Observed', color='blue', linewidth=1.5)
        plt.plot(dates_train, y_train_pred, label='Predicted', color='red', linewidth=1.5, alpha=0.8)
        plt.title('Training Set Hydrograph - PINN-GRU Model', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Streamflow (m³/s)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        visualizations['train_hydrograph'] = fig_train
        
        # Testing hydrograph
        fig_test = plt.figure(figsize=(16, 8))
        plt.plot(dates_test, y_test_actual, label='Observed', color='blue', linewidth=1.5)
        plt.plot(dates_test, y_test_pred, label='Predicted', color='red', linewidth=1.5, alpha=0.8)
        plt.title('Testing Set Hydrograph - PINN-GRU Model', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Streamflow (m³/s)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        visualizations['test_hydrograph'] = fig_test
        
        # Log scale hydrographs
        fig_train_log = plt.figure(figsize=(16, 8))
        plt.plot(dates_train, y_train_actual, label='Observed', color='blue', linewidth=1.5)
        plt.plot(dates_train, y_train_pred, label='Predicted', color='red', linewidth=1.5, alpha=0.8)
        plt.title('Training Set Hydrograph (Log Scale) - PINN-GRU Model', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Streamflow (m³/s)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.yscale('log')
        plt.tight_layout()
        visualizations['train_hydrograph_log'] = fig_train_log
        
        fig_test_log = plt.figure(figsize=(16, 8))
        plt.plot(dates_test, y_test_actual, label='Observed', color='blue', linewidth=1.5)
        plt.plot(dates_test, y_test_pred, label='Predicted', color='red', linewidth=1.5, alpha=0.8)
        plt.title('Testing Set Hydrograph (Log Scale) - PINN-GRU Model', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Streamflow (m³/s)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.yscale('log')
        plt.tight_layout()
        visualizations['test_hydrograph_log'] = fig_test_log
        
        # Scatter plots
        fig_train_scatter = plt.figure(figsize=(10, 10))
        plt.scatter(y_train_actual, y_train_pred, alpha=0.6, edgecolor='k', s=20)
        max_val = max(np.max(y_train_actual), np.max(y_train_pred))
        min_val = min(np.min(y_train_actual), np.min(y_train_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5)
        plt.annotate(f'R² = {train_r2:.3f}\nNSE = {train_nse:.3f}\nKGE = {train_kge:.3f}', 
                    xy=(0.05, 0.95), xycoords='axes fraction', 
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8), 
                    fontsize=12, ha='left', va='top')
        plt.title('Training Set Scatter Plot - PINN-GRU Model', fontsize=16, fontweight='bold')
        plt.xlabel('Observed Streamflow (m³/s)', fontsize=14)
        plt.ylabel('Predicted Streamflow (m³/s)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        visualizations['train_scatter'] = fig_train_scatter
        
        fig_test_scatter = plt.figure(figsize=(10, 10))
        plt.scatter(y_test_actual, y_test_pred, alpha=0.6, edgecolor='k', s=20)
        max_val = max(np.max(y_test_actual), np.max(y_test_pred))
        min_val = min(np.min(y_test_actual), np.min(y_test_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5)
        plt.annotate(f'R² = {test_r2:.3f}\nNSE = {test_nse:.3f}\nKGE = {test_kge:.3f}', 
                    xy=(0.05, 0.95), xycoords='axes fraction', 
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8), 
                    fontsize=12, ha='left', va='top')
        plt.title('Testing Set Scatter Plot - PINN-GRU Model', fontsize=16, fontweight='bold')
        plt.xlabel('Observed Streamflow (m³/s)', fontsize=14)
        plt.ylabel('Predicted Streamflow (m³/s)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        visualizations['test_scatter'] = fig_test_scatter
        
        # Flow duration curves
        fig_train_fdc = plt.figure(figsize=(12, 8))
        plt.plot(train_exceedance_prob, train_obs_fdc, label='Observed', color='blue', linewidth=2)
        plt.plot(train_exceedance_prob, train_pred_fdc, label='Predicted', color='red', linewidth=2, alpha=0.8)
        plt.title('Flow Duration Curve - Training Set', fontsize=16, fontweight='bold')
        plt.xlabel('Exceedance Probability (%)', fontsize=14)
        plt.ylabel('Streamflow (m³/s)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(0.1, 100)
        plt.tight_layout()
        visualizations['train_fdc'] = fig_train_fdc
        
        fig_test_fdc = plt.figure(figsize=(12, 8))
        plt.plot(test_exceedance_prob, test_obs_fdc, label='Observed', color='blue', linewidth=2)
        plt.plot(test_exceedance_prob, test_pred_fdc, label='Predicted', color='red', linewidth=2, alpha=0.8)
        plt.title('Flow Duration Curve - Testing Set', fontsize=16, fontweight='bold')
        plt.xlabel('Exceedance Probability (%)', fontsize=14)
        plt.ylabel('Streamflow (m³/s)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(0.1, 100)
        plt.tight_layout()
        visualizations['test_fdc'] = fig_test_fdc
        
        # Residual plots
        train_residuals = y_train_actual.flatten() - y_train_pred.flatten()
        fig_train_residuals = plt.figure(figsize=(16, 8))
        plt.subplot(2, 1, 1)
        plt.plot(dates_train, train_residuals, color='purple', linewidth=1.5)
        plt.axhline(y=0, color='k', linestyle='--', linewidth=1)
        plt.title('Training Set - Residuals Over Time', fontsize=16, fontweight='bold')
        plt.ylabel('Residuals (m³/s)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.subplot(2, 1, 2)
        plt.hist(train_residuals, bins=30, color='purple', alpha=0.7)
        plt.axvline(x=0, color='k', linestyle='--', linewidth=1)
        plt.title('Residuals Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Residuals (m³/s)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        visualizations['train_residuals'] = fig_train_residuals
        
        test_residuals = y_test_actual.flatten() - y_test_pred.flatten()
        fig_test_residuals = plt.figure(figsize=(16, 8))
        plt.subplot(2, 1, 1)
        plt.plot(dates_test, test_residuals, color='purple', linewidth=1.5)
        plt.axhline(y=0, color='k', linestyle='--', linewidth=1)
        plt.title('Testing Set - Residuals Over Time', fontsize=16, fontweight='bold')
        plt.ylab<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>