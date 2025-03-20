import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

def load_data(data_path):
    """
    Load data from Excel file
    
    Parameters:
    -----------
    data_path : str
        Path to Excel file
    
    Returns:
    --------
    data : pandas.DataFrame
        Loaded data
    message : str
        Success or error message
    """
    try:
        if not os.path.exists(data_path):
            return None, f"Error: File not found at {data_path}"
        
        data = pd.read_excel(data_path)
        
        if data is None or data.empty:
            return None, "Error: Data is empty or could not be loaded"
        
        return data, "Data loaded successfully"
    except Exception as e:
        return None, f"Error loading data: {str(e)}"

def preprocess_data(data_path, num_lagged_features=12, train_test_split=0.8, 
                   output_variable='Discharge (m³/S)', 
                   dynamic_variables=None, 
                   categorical_variables=None, 
                   numeric_static_variables=None):
    """
    Preprocess data for model training
    
    Parameters:
    -----------
    data_path : str
        Path to Excel file
    num_lagged_features : int
        Number of lagged features to create
    train_test_split : float
        Proportion of data to use for training
    output_variable : str
        Name of the output variable column
    dynamic_variables : list
        List of dynamic variable column names
    categorical_variables : list
        List of categorical static variable column names
    numeric_static_variables : list
        List of numeric static variable column names
    
    Returns:
    --------
    X_train : numpy.ndarray
        Training features
    X_test : numpy.ndarray
        Testing features
    y_train : numpy.ndarray
        Training targets
    y_test : numpy.ndarray
        Testing targets
    dates_train : numpy.ndarray
        Training dates
    dates_test : numpy.ndarray
        Testing dates
    scalers : dict
        Dictionary of fitted scalers
    """
    try:
        # Check if data file exists
        if not os.path.exists(data_path):
            print(f"Error: Data file not found at {data_path}")
            return None, None, None, None, None, None, None
        
        # Load data
        data = pd.read_excel(data_path)
        
        # Check if data is empty
        if data is None or data.empty:
            print("Error: Data is empty or could not be loaded")
            return None, None, None, None, None, None, None
        
        # Set default values if not provided
        if dynamic_variables is None or len(dynamic_variables) == 0:
            dynamic_variables = ['Rainfall (mm)', 'Maximum temperature (°C)', 
                                'Minimum temperature (°C)', 'Daily global solar exposure (MJ/m*m)']
            # Check if these columns exist, otherwise use first 4 non-date columns
            available_cols = [col for col in data.columns if col != 'Date' and col != output_variable]
            if not all(col in data.columns for col in dynamic_variables):
                dynamic_variables = available_cols[:min(4, len(available_cols))]
        
        if categorical_variables is None:
            categorical_variables = ['Land Use Classes', 'Soil Classes']
            # Filter to only include columns that exist
            categorical_variables = [col for col in categorical_variables if col in data.columns]
        
        if numeric_static_variables is None:
            numeric_static_variables = ['Land Use Percentage', 'Soil Percentage', 'Slope (Degree)']
            if 'Drainage Density (km/km²)' in data.columns:
                numeric_static_variables.append('Drainage Density (km/km²)')
            # Filter to only include columns that exist
            numeric_static_variables = [col for col in numeric_static_variables if col in data.columns]
        
        # Check if required columns exist
        required_columns = ['Date'] + [output_variable] + dynamic_variables
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return None, None, None, None, None, None, None
        
        # Extract dates
        dates = data['Date']
        
        # Fill missing values
        data[[output_variable] + dynamic_variables] = data[[output_variable] + dynamic_variables].fillna(0)
        if numeric_static_variables and len(numeric_static_variables) > 0:
            data[numeric_static_variables] = data[numeric_static_variables].fillna(0)
        
        # Create lagged features
        for lag in range(1, num_lagged_features + 1):
            data[f'Lag_{output_variable}_{lag}'] = data[output_variable].shift(lag).fillna(0)
            for var in dynamic_variables:
                var_name = var.replace(' ', '_').replace('(', '').replace(')', '').replace('°', '').replace('/', '_')
                data[f'Lag_{var_name}_{lag}'] = data[var].shift(lag).fillna(0)
        
        # Add seasonality features
        data['Month'] = pd.to_datetime(data['Date']).dt.month
        data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
        data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)
        
        # Define feature groups
        lagged_output_cols = [f'Lag_{output_variable}_{i}' for i in range(1, num_lagged_features + 1)]
        lagged_dynamic_cols = []
        for var in dynamic_variables:
            var_name = var.replace(' ', '_').replace('(', '').replace(')', '').replace('°', '').replace('/', '_')
            lagged_dynamic_cols.extend([f'Lag_{var_name}_{i}' for i in range(1, num_lagged_features + 1)])
        
        seasonality_cols = ['Month_sin', 'Month_cos']
        
        # Define the full list of dynamic features
        all_dynamic_cols = dynamic_variables + lagged_output_cols + lagged_dynamic_cols + seasonality_cols
        
        # Process categorical variables
        X_static_categorical = pd.DataFrame(index=range(len(data)))  # Empty DataFrame with correct number of rows
        if categorical_variables and len(categorical_variables) > 0:
            try:
                X_static_categorical = pd.get_dummies(data[categorical_variables], columns=categorical_variables, drop_first=True)
            except Exception as e:
                print(f"Warning: Error processing categorical variables: {str(e)}")
        
        # Process numeric static variables
        X_static_numeric = np.zeros((data.shape[0], 1))  # Default to single column of zeros
        if numeric_static_variables and len(numeric_static_variables) > 0:
            try:
                X_static_numeric = data[numeric_static_variables].values
            except Exception as e:
                print(f"Warning: Error processing numeric static variables: {str(e)}")
        
        # Extract dynamic features and target
        X_dynamic = data[all_dynamic_cols].values
        y = data[output_variable].values
        dates_processed = data['Date']
        
        # Scale features
        scaler_dynamic = MinMaxScaler()
        X_dynamic_scaled = scaler_dynamic.fit_transform(X_dynamic)
        
        scaler_static_numeric = MinMaxScaler()
        if X_static_numeric.shape[1] > 0:
            X_static_numeric_scaled = scaler_static_numeric.fit_transform(X_static_numeric)
        else:
            X_static_numeric_scaled = X_static_numeric
        
        scaler_y = MinMaxScaler()
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))
        
        # Combine features
        if X_static_categorical.shape[1] > 0:
            X_combined_scaled = np.concatenate([X_dynamic_scaled, X_static_categorical.values, X_static_numeric_scaled], axis=1)
        else:
            X_combined_scaled = np.concatenate([X_dynamic_scaled, X_static_numeric_scaled], axis=1)
        
        X_combined_scaled = X_combined_scaled.reshape((X_combined_scaled.shape[0], 1, X_combined_scaled.shape[1]))
        
        # Train-test split
        train_size = int(train_test_split * len(X_combined_scaled))
        X_train = X_combined_scaled[:train_size]
        X_test = X_combined_scaled[train_size:]
        y_train = y_scaled[:train_size]
        y_test = y_scaled[train_size:]
        dates_train = dates_processed[:train_size]
        dates_test = dates_processed[train_size:]
        
        # Create scalers dictionary
        scalers = {
            'dynamic': scaler_dynamic,
            'static_numeric': scaler_static_numeric,
            'y': scaler_y,
            'static_categorical_columns': X_static_categorical.columns.tolist() if X_static_categorical.shape[1] > 0 else []
        }
        
        return X_train, X_test, y_train, y_test, dates_train, dates_test, scalers
    
    except Exception as e:
        print(f"Error in data preprocessing: {str(e)}")
        return None, None, None, None, None, None, None
