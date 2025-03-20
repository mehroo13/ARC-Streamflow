import os
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

def validate_data(data):
    """
    Validate that the uploaded data has the required columns
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data to validate
        
    Returns:
    --------
    valid : bool
        Whether the data is valid
    message : str
        Error message if data is invalid, empty string otherwise
    """
    required_columns = ['Date', 'Rainfall (mm)', 'Maximum temperature (°C)', 
                        'Minimum temperature (°C)', 'Daily global solar exposure (MJ/m*m)']
    
    # Check if required columns exist
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"
    
    # Check if Date column is valid
    try:
        pd.to_datetime(data['Date'])
    except:
        return False, "Date column could not be converted to datetime format"
    
    # For gauged data, check if Discharge column exists
    if 'Discharge (m³/S)' in data.columns:
        # Check if Discharge column has valid values
        if data['Discharge (m³/S)'].isnull().all():
            return False, "Discharge column contains only null values"
    
    # Check if static variables exist
    static_vars_exist = any(col in data.columns for col in 
                           ['Land Use Classes', 'Soil Classes', 'Land Use Percentage', 
                            'Soil Percentage', 'Slope (Degree)'])
    
    if not static_vars_exist:
        return False, "No static catchment variables found (Land Use, Soil, Slope)"
    
    return True, ""

def load_data(data_path):
    """
    Load data from Excel file with validation
    
    Parameters:
    -----------
    data_path : str
        Path to Excel file
        
    Returns:
    --------
    data : pandas.DataFrame or None
        Loaded data if valid, None otherwise
    message : str
        Error message if data is invalid, success message otherwise
    """
    try:
        data = pd.read_excel(data_path)
        
        # Validate data
        valid, message = validate_data(data)
        
        if not valid:
            return None, message
        
        # Convert Date to datetime
        data['Date'] = pd.to_datetime(data['Date'])
        
        return data, "Data loaded successfully!"
    except Exception as e:
        return None, f"Error loading data: {str(e)}"

def preprocess_data(data, num_lagged_features=12, train_test_split=0.8):
    """
    Preprocess data for model training with error handling
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Raw data
    num_lagged_features : int
        Number of lagged features to create
    train_test_split : float
        Proportion of data to use for training
        
    Returns:
    --------
    processed_data : dict
        Dictionary containing processed data and scalers
    """
    try:
        # Ensure date column is datetime
        data['Date'] = pd.to_datetime(data['Date'])
        dates = data['Date']
        
        # Extract dynamic features
        X_dynamic_base = data[['Rainfall (mm)', 'Maximum temperature (°C)', 'Minimum temperature (°C)', 
                              'Daily global solar exposure (MJ/m*m)']]
        
        # Extract static features
        static_numeric_cols = ['Land Use Percentage', 'Soil Percentage', 'Slope (Degree)']
        if 'Drainage Density (km/km²)' in data.columns:
            static_numeric_cols.append('Drainage Density (km/km²)')
        
        static_categorical_cols = ['Land Use Classes', 'Soil Classes']
        
        # Fill missing values
        data[['Discharge (m³/S)', 'Rainfall (mm)', 'Maximum temperature (°C)', 
              'Minimum temperature (°C)', 'Daily global solar exposure (MJ/m*m)']] = \
        data[['Discharge (m³/S)', 'Rainfall (mm)', 'Maximum temperature (°C)', 
              'Minimum temperature (°C)', 'Daily global solar exposure (MJ/m*m)']].fillna(0)
        
        data[static_numeric_cols] = data[static_numeric_cols].fillna(0)
        
        # Create lagged features
        for lag in range(1, num_lagged_features + 1):
            data[f'Lag_Discharge_{lag}'] = data['Discharge (m³/S)'].shift(lag).fillna(0)
            data[f'Lag_Rainfall_{lag}'] = data['Rainfall (mm)'].shift(lag).fillna(0)
            data[f'Lag_TempMax_{lag}'] = data['Maximum temperature (°C)'].shift(lag).fillna(0)
            data[f'Lag_TempMin_{lag}'] = data['Minimum temperature (°C)'].shift(lag).fillna(0)
            data[f'Lag_Solar_{lag}'] = data['Daily global solar exposure (MJ/m*m)'].shift(lag).fillna(0)
        
        # Add seasonality features
        data['Month'] = pd.to_datetime(data['Date']).dt.month
        data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
        data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)
        
        # Define feature columns
        dynamic_feature_cols = ['Rainfall (mm)', 'Maximum temperature (°C)', 'Minimum temperature (°C)', 
                               'Daily global solar exposure (MJ/m*m)']
        
        lagged_discharge_cols = [f'Lag_Discharge_{i}' for i in range(1, num_lagged_features + 1)]
        
        lagged_weather_cols = [f'Lag_Rainfall_{i}' for i in range(1, num_lagged_features + 1)] + \
                             [f'Lag_TempMax_{i}' for i in range(1, num_lagged_features + 1)] + \
                             [f'Lag_TempMin_{i}' for i in range(1, num_lagged_features + 1)] + \
                             [f'Lag_Solar_{i}' for i in range(1, num_lagged_features + 1)]
        
        seasonality_cols = ['Month_sin', 'Month_cos']
        
        # Define the full list of dynamic features
        all_dynamic_cols = dynamic_feature_cols + lagged_discharge_cols + lagged_weather_cols + seasonality_cols
        
        # Process categorical features
        X_static_categorical = pd.get_dummies(data[static_categorical_cols], columns=static_categorical_cols, drop_first=True)
        
        # Process numeric features
        X_static_numeric = data[static_numeric_cols].values
        
        # Process dynamic features
        X_dynamic = data[all_dynamic_cols].values
        
        # Process target
        y = data['Discharge (m³/S)'].values
        
        # Save dates
        dates_processed = data['Date']
        
        # Scale features
        scaler_dynamic = MinMaxScaler()
        X_dynamic_scaled = scaler_dynamic.fit_transform(X_dynamic)
        
        scaler_static_numeric = MinMaxScaler()
        X_static_numeric_scaled = scaler_static_numeric.fit_transform(X_static_numeric)
        
        scaler_y = MinMaxScaler()
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))
        
        # Combine features
        X_combined_scaled = np.concatenate([X_dynamic_scaled, X_static_categorical.values, X_static_numeric_scaled], axis=1)
        
        # Reshape for GRU input
        X_combined_scaled = X_combined_scaled.reshape((X_combined_scaled.shape[0], 1, X_combined_scaled.shape[1]))
        
        # Train-test split
        train_size = int(train_test_split * len(X_combined_scaled))
        X_train = X_combined_scaled[:train_size]
        X_test = X_combined_scaled[train_size:]
        y_train = y_scaled[:train_size]
        y_test = y_scaled[train_size:]
        dates_train = dates_processed[:train_size]
        dates_test = dates_processed[train_size:]
        
        # Calculate the indices of lagged discharge features
        lagged_discharge_indices = [all_dynamic_cols.index(f'Lag_Discharge_{i}') for i in range(1, num_lagged_features + 1)]
        
        # Create a masked version of X_train for training
        X_train_masked = X_train.copy()
        mask = np.random.random(X_train.shape[0]) < 0.5  # 50% chance of masking
        for i in range(X_train.shape[0]):
            if mask[i]:
                X_train_masked[i, :, lagged_discharge_indices] = 0
        
        # Create a dictionary with all processed data
        processed_data = {
            'X_train': X_train_masked,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'dates_train': dates_train,
            'dates_test': dates_test,
            'scalers': {
                'dynamic': scaler_dynamic,
                'static_numeric': scaler_static_numeric,
                'y': scaler_y
            },
            'feature_info': {
                'dynamic_cols': dynamic_feature_cols,
                'lagged_discharge_cols': lagged_discharge_cols,
                'lagged_weather_cols': lagged_weather_cols,
                'seasonality_cols': seasonality_cols,
                'all_dynamic_cols': all_dynamic_cols,
                'static_categorical_cols': X_static_categorical.columns.tolist(),
                'static_numeric_cols': static_numeric_cols,
                'lagged_discharge_indices': lagged_discharge_indices
            }
        }
        
        return processed_data, "Data preprocessed successfully!"
    except Exception as e:
        st.error(f"Error preprocessing data: {str(e)}")
        return None, f"Error preprocessing data: {str(e)}"

def preprocess_ungauged_data(gauged_data_path, ungauged_data_path, scalers, feature_info, num_lagged_features=12):
    """
    Preprocess ungauged catchment data for prediction with error handling
    
    Parameters:
    -----------
    gauged_data_path : str
        Path to gauged catchment data
    ungauged_data_path : str
        Path to ungauged catchment data
    scalers : dict
        Dictionary of scalers from gauged data preprocessing
    feature_info : dict
        Dictionary of feature information from gauged data preprocessing
    num_lagged_features : int
        Number of lagged features to create
        
    Returns:
    --------
    processed_data : dict
        Dictionary containing processed ungauged data
    message : str
        Success or error message
    """
    try:
        # Load gauged data for reference
        gauged_data = pd.read_excel(gauged_data_path)
        
        # Load ungauged data
        ungauged_data = pd.read_excel(ungauged_data_path)
        
        # Validate ungauged data
        valid, message = validate_data(ungauged_data)
        if not valid:
            return None, message
        
        # Ensure date column is datetime
        ungauged_data['Date'] = pd.to_datetime(ungauged_data['Date'])
        
        # Fill missing values
        dynamic_cols = ['Rainfall (mm)', 'Maximum temperature (°C)', 'Minimum temperature (°C)', 
                       'Daily global solar exposure (MJ/m*m)']
        
        ungauged_data[dynamic_cols] = ungauged_data[dynamic_cols].fillna(0)
        ungauged_data[feature_info['static_numeric_cols']] = ungauged_data[feature_info['static_numeric_cols']].fillna(0)
        
        # Train a simple linear regression model to estimate discharge
        X_simple = gauged_data[['Rainfall (mm)']].values
        y_simple = gauged_data['Discharge (m³/S)'].values
        simple_model = LinearRegression().fit(X_simple, y_simple)
        
        # Generate synthetic discharge for ungauged data
        ungauged_data['Synthetic_Discharge'] = simple_model.predict(ungauged_data[['Rainfall (mm)']].values)
        
        # Create lagged features
        for lag in range(1, num_lagged_features + 1):
            ungauged_data[f'Lag_Discharge_{lag}'] = ungauged_data['Synthetic_Discharge'].shift(lag).fillna(0)
            ungauged_data[f'Lag_Rainfall_{lag}'] = ungauged_data['Rainfall (mm)'].shift(lag).fillna(0)
            ungauged_data[f'Lag_TempMax_{lag}'] = ungauged_data['Maximum temperature (°C)'].shift(lag).fillna(0)
            ungauged_data[f'Lag_TempMin_{lag}'] = ungauged_data['Minimum temperature (°C)'].shift(lag).fillna(0)
            ungauged_data[f'Lag_Solar_{lag}'] = ungauged_data['Daily global solar exposure (MJ/m*m)'].shift(lag).fillna(0)
        
        # Add seasonality features
        ungauged_data['Month'] = pd.to_datetime(ungauged_data['Date']).dt.month
        ungauged_data['Month_sin'] = np.sin(2 * np.pi * ungauged_data['Month'] / 12)
        ungauged_data['Month_cos'] = np.cos(2 * np.pi * ungauged_data['Month'] / 12)
        
        # Process categorical features
        static_categorical_cols = ['Land Use Classes', 'Soil Classes']
        X_static_categorical_ungauged = pd.get_dummies(ungauged_data[static_categorical_cols], 
                                                     columns=static_categorical_cols, drop_first=True)
        X_static_categorical_ungauged = X_static_categorical_ungauged.reindex(
            columns=feature_info['static_categorical_cols'], fill_value=0)
        
        # Process numeric features
        X_static_numeric_ungauged = ungauged_data[feature_info['static_numeric_cols']].values
        
        # Process dynamic features
        X_dynamic_ungauged = ungauged_data[feature_info['all_dynamic_cols']].values
        
        # Scale features
        X_dynamic_scaled_ungauged = scalers['dynamic'].transform(X_dynamic_ungauged)
        X_static_numeric_scaled_ungauged = scalers['static_numeric'].transform(X_static_numeric_ungauged)
        
        # Combine features
        X_combined_scaled_ungauged = np.concatenate([
            X_dynamic_scaled_ungauged, 
            X_static_categorical_ungauged.values, 
            X_static_numeric_scaled_ungauged
        ], axis=1)
        
        # Reshape for GRU input
        X_combined_scaled_ungauged = X_combined_scaled_ungauged.reshape(
            (X_combined_scaled_ungauged.shape[0], 1, X_combined_scaled_ungauged.shape[1]))
        
        return {
            'X_ungauged': X_combined_scaled_ungauged,
            'dates': ungauged_data['Date'],
            'feature_info': feature_info
        }, "Ungauged data preprocessed successfully!"
    except Exception as e:
        return None, f"Error preprocessing ungauged data: {str(e)}"

def save_uploaded_file(uploaded_file, directory="data"):
    """
    Save an uploaded file to the specified directory
    
    Parameters:
    -----------
    uploaded_file : UploadedFile
        Streamlit uploaded file object
    directory : str
        Directory to save the file to
        
    Returns:
    --------
    file_path : str
        Path to the saved file
    """
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path

def get_data_summary(data):
    """
    Generate a summary of the data
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data to summarize
        
    Returns:
    --------
    summary : dict
        Dictionary containing data summary
    """
    summary = {
        "shape": data.shape,
        "columns": data.columns.tolist(),
        "date_range": (data['Date'].min(), data['Date'].max()),
        "missing_values": data.isnull().sum().to_dict(),
        "has_discharge": 'Discharge (m³/S)' in data.columns,
    }
    
    if 'Discharge (m³/S)' in data.columns:
        summary["discharge_stats"] = {
            "min": data['Discharge (m³/S)'].min(),
            "max": data['Discharge (m³/S)'].max(),
            "mean": data['Discharge (m³/S)'].mean(),
            "median": data['Discharge (m³/S)'].median(),
            "missing": data['Discharge (m³/S)'].isnull().sum()
        }
    
    return summary
