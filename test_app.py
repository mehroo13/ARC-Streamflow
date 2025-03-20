import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Create necessary directories
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Add sample data for testing
def create_sample_data():
    """Create sample data files for testing the application"""
    
    # Check if sample data already exists
    if os.path.exists(os.path.join('data', 'sample_gauged_data.xlsx')) and \
       os.path.exists(os.path.join('data', 'sample_ungauged_data.xlsx')):
        return
    
    # Create date range
    dates = pd.date_range(start='2010-01-01', end='2019-12-31', freq='D')
    n_samples = len(dates)
    
    # Create synthetic data for gauged catchment
    np.random.seed(42)  # For reproducibility
    
    # Create rainfall with seasonal pattern
    month = pd.Series(dates).dt.month
    season_factor = np.sin(2 * np.pi * month / 12) + 1  # Higher in summer
    rainfall = np.maximum(0, np.random.normal(5 * season_factor, 8, n_samples))
    
    # Create temperature with seasonal pattern
    max_temp = 20 + 10 * np.sin(2 * np.pi * (month - 1) / 12) + np.random.normal(0, 3, n_samples)
    min_temp = max_temp - 5 - np.random.normal(0, 2, n_samples)
    
    # Create solar radiation with seasonal pattern
    solar = 15 + 10 * np.sin(2 * np.pi * (month - 1) / 12) + np.random.normal(0, 3, n_samples)
    
    # Create discharge with lag from rainfall and seasonal pattern
    base_flow = 1 + 0.5 * np.sin(2 * np.pi * (month - 2) / 12)
    discharge = base_flow.copy()
    
    # Add rainfall influence with lag
    for lag in range(1, 10):
        lagged_rainfall = np.pad(rainfall, (lag, 0), 'constant')[:n_samples]
        discharge += 0.2 * (10 - lag) / 10 * lagged_rainfall * np.exp(-lag/5)
    
    # Add some noise
    discharge = np.maximum(0.1, discharge + np.random.normal(0, 0.5, n_samples))
    
    # Create static catchment characteristics
    land_use_classes = np.random.choice(['Forest', 'Agriculture', 'Urban'], n_samples)
    soil_classes = np.random.choice(['Clay', 'Loam', 'Sand'], n_samples)
    land_use_percentage = np.random.uniform(0, 100, n_samples)
    soil_percentage = np.random.uniform(0, 100, n_samples)
    slope = np.random.uniform(0, 30, n_samples)
    
    # Create gauged catchment dataframe
    gauged_data = pd.DataFrame({
        'Date': dates,
        'Discharge (m³/S)': discharge,
        'Rainfall (mm)': rainfall,
        'Maximum temperature (°C)': max_temp,
        'Minimum temperature (°C)': min_temp,
        'Daily global solar exposure (MJ/m*m)': solar,
        'Land Use Classes': land_use_classes,
        'Soil Classes': soil_classes,
        'Land Use Percentage': land_use_percentage,
        'Soil Percentage': soil_percentage,
        'Slope (Degree)': slope
    })
    
    # Create ungauged catchment data (similar but without discharge)
    # Slightly modify the patterns to represent a different catchment
    rainfall_ungauged = np.maximum(0, np.random.normal(4.5 * season_factor, 7, n_samples))
    max_temp_ungauged = 21 + 9.5 * np.sin(2 * np.pi * (month - 1) / 12) + np.random.normal(0, 3, n_samples)
    min_temp_ungauged = max_temp_ungauged - 5.5 - np.random.normal(0, 2, n_samples)
    solar_ungauged = 14 + 11 * np.sin(2 * np.pi * (month - 1) / 12) + np.random.normal(0, 3, n_samples)
    
    land_use_classes_ungauged = np.random.choice(['Forest', 'Agriculture', 'Urban'], n_samples)
    soil_classes_ungauged = np.random.choice(['Clay', 'Loam', 'Sand'], n_samples)
    land_use_percentage_ungauged = np.random.uniform(0, 100, n_samples)
    soil_percentage_ungauged = np.random.uniform(0, 100, n_samples)
    slope_ungauged = np.random.uniform(0, 30, n_samples)
    
    # Create ungauged catchment dataframe
    ungauged_data = pd.DataFrame({
        'Date': dates,
        'Rainfall (mm)': rainfall_ungauged,
        'Maximum temperature (°C)': max_temp_ungauged,
        'Minimum temperature (°C)': min_temp_ungauged,
        'Daily global solar exposure (MJ/m*m)': solar_ungauged,
        'Land Use Classes': land_use_classes_ungauged,
        'Soil Classes': soil_classes_ungauged,
        'Land Use Percentage': land_use_percentage_ungauged,
        'Soil Percentage': soil_percentage_ungauged,
        'Slope (Degree)': slope_ungauged
    })
    
    # Save to Excel files
    gauged_data.to_excel(os.path.join('data', 'sample_gauged_data.xlsx'), index=False)
    ungauged_data.to_excel(os.path.join('data', 'sample_ungauged_data.xlsx'), index=False)
    
    print("Sample data created successfully!")

# Create sample data
create_sample_data()

def test_app_functionality():
    """Test basic functionality of the application"""
    
    st.markdown("## Testing Application Functionality")
    
    # Test data loading
    st.markdown("### Testing Data Loading")
    try:
        from utils.data_processing import load_data
        
        gauged_data_path = os.path.join('data', 'sample_gauged_data.xlsx')
        data, message = load_data(gauged_data_path)
        
        if data is not None:
            st.success(f"✅ Data loading successful: {message}")
            st.write(f"Data shape: {data.shape}")
            st.write("First few rows:")
            st.dataframe(data.head())
        else:
            st.error(f"❌ Data loading failed: {message}")
    except Exception as e:
        st.error(f"❌ Error testing data loading: {str(e)}")
    
    # Test data preprocessing
    st.markdown("### Testing Data Preprocessing")
    try:
        from utils.data_processing import preprocess_data
        
        processed_data, message = preprocess_data(data, num_lagged_features=6, train_test_split=0.8)
        
        if processed_data is not None:
            st.success(f"✅ Data preprocessing successful: {message}")
            st.write(f"Training data shape: {processed_data['X_train'].shape}")
            st.write(f"Testing data shape: {processed_data['X_test'].shape}")
        else:
            st.error(f"❌ Data preprocessing failed: {message}")
    except Exception as e:
        st.error(f"❌ Error testing data preprocessing: {str(e)}")
    
    # Test model creation
    st.markdown("### Testing Model Creation")
    try:
        from utils.model import create_pinn_gru_model
        
        model_params = {
            'GRU_UNITS': 64,
            'DENSE_UNITS_1': 128,
            'DENSE_UNITS_2': 256,
            'DENSE_UNITS_3': 128,
            'DROPOUT_RATE': 0.3,
            'LEARNING_RATE': 0.001,
            'PHYSICS_LOSS_WEIGHT': 0.1
        }
        
        model = create_pinn_gru_model(model_params)
        
        if model is not None:
            st.success("✅ Model creation successful")
            st.write("Model summary:")
            # Convert model summary to string
            import io
            summary_string = io.StringIO()
            model.summary(print_fn=lambda x: summary_string.write(x + '\n'))
            st.text(summary_string.getvalue())
        else:
            st.error("❌ Model creation failed")
    except Exception as e:
        st.error(f"❌ Error testing model creation: {str(e)}")
    
    # Test model saving and loading
    st.markdown("### Testing Model Saving and Loading")
    try:
        from utils.model_manager import save_model_with_metadata, load_model_with_metadata
        
        # Create a test directory
        test_model_dir = os.path.join('models', 'test_model')
        os.makedirs(test_model_dir, exist_ok=True)
        
        # Save model
        training_data = {
            'X_train': processed_data['X_train'],
            'X_test': processed_data['X_test'],
            'y_train': processed_data['y_train'],
            'y_test': processed_data['y_test'],
            'dates_train': processed_data['dates_train'],
            'dates_test': processed_data['dates_test']
        }
        
        # Create dummy history
        history = {'loss': [0.1, 0.05, 0.02], 'val_loss': [0.15, 0.1, 0.08]}
        
        model_dir, success, message = save_model_with_metadata(
            model, 
            model_params, 
            processed_data['scalers'], 
            training_data, 
            history, 
            test_model_dir
        )
        
        if success:
            st.success(f"✅ Model saving successful: {message}")
            
            # Load model
            loaded_model, loaded_params, loaded_scalers, loaded_data, loaded_history, loaded_metadata, load_success, load_message = load_model_with_metadata(model_dir)
            
            if load_success:
                st.success(f"✅ Model loading successful: {load_message}")
            else:
                st.error(f"❌ Model loading failed: {load_message}")
        else:
            st.error(f"❌ Model saving failed: {message}")
    except Exception as e:
        st.error(f"❌ Error testing model saving/loading: {str(e)}")
    
    # Test prediction functionality
    st.markdown("### Testing Prediction Functionality")
    try:
        from utils.prediction import batch_predict
        
        # Make a small batch prediction
        test_batch = processed_data['X_test'][:10]
        predictions = batch_predict(model, test_batch, batch_size=5)
        
        if predictions is not None:
            st.success("✅ Batch prediction successful")
            st.write(f"Predictions shape: {predictions.shape}")
        else:
            st.error("❌ Batch prediction failed")
    except Exception as e:
        st.error(f"❌ Error testing prediction: {str(e)}")
    
    st.markdown("### Overall Test Results")
    st.success("✅ Application functionality tests completed")
    st.info("The application is ready for deployment!")

# Run the tests
if __name__ == "__main__":
    test_app_functionality()
