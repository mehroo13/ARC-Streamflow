import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import joblib
from datetime import datetime

# Add the utils directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

# Import utility modules
from utils.data_processing import load_data, preprocess_data
from utils.model import create_pinn_gru_model, train_model
from utils.evaluation import evaluate_model, generate_visualizations
from utils.prediction import predict_ungauged

# Set page configuration
st.set_page_config(
    page_title="Hydrological Streamflow Prediction",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-bottom: 1rem;
    }
    .info-text {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .success-text {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .warning-text {
        background-color: #FFF8E1;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.markdown("<h1 class='main-header'>Hydrological Streamflow Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p class='info-text'>This application uses a Physics-Informed Neural Network (PINN) with GRU architecture to predict streamflow in both gauged and ungauged catchments.</p>", unsafe_allow_html=True)

# Create tabs for different functionalities
tabs = st.tabs(["Data Processing", "Model Training", "Evaluation", "Prediction", "About"])

# Sidebar for configuration
with st.sidebar:
    st.markdown("<h2 class='sub-header'>Configuration</h2>", unsafe_allow_html=True)
    
    # Model parameters section
    st.markdown("### Model Parameters")
    gru_units = st.number_input("GRU Units", min_value=16, max_value=512, value=128, step=16)
    dense_units_1 = st.number_input("Dense Units 1", min_value=32, max_value=1024, value=256, step=32)
    dense_units_2 = st.number_input("Dense Units 2", min_value=32, max_value=1024, value=512, step=32)
    dense_units_3 = st.number_input("Dense Units 3", min_value=32, max_value=1024, value=256, step=32)
    dropout_rate = st.slider("Dropout Rate", min_value=0.1, max_value=0.7, value=0.4, step=0.1)
    
    # Training parameters section
    st.markdown("### Training Parameters")
    learning_rate = st.number_input("Learning Rate", min_value=0.00001, max_value=0.01, value=0.0001, format="%.5f")
    batch_size = st.number_input("Batch Size", min_value=8, max_value=128, value=32, step=8)
    epochs = st.number_input("Epochs", min_value=10, max_value=1000, value=500, step=10)
    physics_loss_weight = st.slider("Physics Loss Weight", min_value=0.0, max_value=1.0, value=0.1, step=0.1)
    
    # Data parameters section
    st.markdown("### Data Parameters")
    num_lagged_features = st.number_input("Number of Lagged Features", min_value=1, max_value=24, value=12, step=1)
    train_test_split = st.slider("Train-Test Split", min_value=0.5, max_value=0.9, value=0.8, step=0.05)
    
    # Prediction parameters
    st.markdown("### Prediction Parameters")
    prediction_batch_size = st.number_input("Prediction Batch Size", min_value=10, max_value=500, value=100, step=10)

# Data Processing Tab
with tabs[0]:
    st.markdown("<h2 class='sub-header'>Data Processing</h2>", unsafe_allow_html=True)
    
    # Upload gauged data
    st.markdown("### Upload Gauged Catchment Data")
    gauged_data_file = st.file_uploader("Upload Excel file with gauged catchment data", type=["xlsx", "xls"])
    
    # Upload ungauged data
    st.markdown("### Upload Ungauged Catchment Data (Optional)")
    ungauged_data_file = st.file_uploader("Upload Excel file with ungauged catchment data", type=["xlsx", "xls"])
    
    if gauged_data_file is not None:
        if st.button("Process Data"):
            with st.spinner("Processing data..."):
                # Save uploaded file
                gauged_data_path = os.path.join("data", "gauged_data.xlsx")
                with open(gauged_data_path, "wb") as f:
                    f.write(gauged_data_file.getvalue())
                
                # Save ungauged data if provided
                ungauged_data_path = None
                if ungauged_data_file is not None:
                    ungauged_data_path = os.path.join("data", "ungauged_data.xlsx")
                    with open(ungauged_data_path, "wb") as f:
                        f.write(ungauged_data_file.getvalue())
                
                # Process data
                data = load_data(gauged_data_path)
                
                # Display data info
                st.markdown("<p class='success-text'>Data loaded successfully!</p>", unsafe_allow_html=True)
                st.write("Data shape:", data.shape)
                st.write("Data columns:", data.columns.tolist())
                
                # Display sample data
                st.markdown("### Sample Data")
                st.dataframe(data.head())
                
                # Save data info to session state
                st.session_state['gauged_data_path'] = gauged_data_path
                st.session_state['ungauged_data_path'] = ungauged_data_path
                st.session_state['data_processed'] = True
    else:
        st.markdown("<p class='warning-text'>Please upload gauged catchment data to proceed.</p>", unsafe_allow_html=True)

# Model Training Tab
with tabs[1]:
    st.markdown("<h2 class='sub-header'>Model Training</h2>", unsafe_allow_html=True)
    
    if st.session_state.get('data_processed', False):
        # Create columns for training options
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Training Options")
            early_stopping = st.checkbox("Use Early Stopping", value=True)
            patience = st.number_input("Patience for Early Stopping", min_value=10, max_value=500, value=300, step=10)
            min_delta = st.number_input("Min Delta for Early Stopping", min_value=0.0001, max_value=0.01, value=0.001, format="%.4f")
            
        with col2:
            st.markdown("### Learning Rate Scheduler")
            use_lr_scheduler = st.checkbox("Use Learning Rate Scheduler", value=True)
            lr_factor = st.number_input("LR Reduction Factor", min_value=0.1, max_value=0.9, value=0.8, step=0.1)
            lr_patience = st.number_input("LR Scheduler Patience", min_value=5, max_value=100, value=20, step=5)
        
        if st.button("Train Model"):
            with st.spinner("Training model... This may take some time."):
                # Get parameters from sidebar
                model_params = {
                    'GRU_UNITS': gru_units,
                    'DENSE_UNITS_1': dense_units_1,
                    'DENSE_UNITS_2': dense_units_2,
                    'DENSE_UNITS_3': dense_units_3,
                    'DROPOUT_RATE': dropout_rate,
                    'LEARNING_RATE': learning_rate,
                    'BATCH_SIZE': batch_size,
                    'EPOCHS': epochs,
                    'PHYSICS_LOSS_WEIGHT': physics_loss_weight,
                    'NUM_LAGGED_FEATURES': num_lagged_features,
                    'TRAIN_TEST_SPLIT': train_test_split,
                    'EARLY_STOPPING': early_stopping,
                    'PATIENCE': patience,
                    'MIN_DELTA': min_delta,
                    'USE_LR_SCHEDULER': use_lr_scheduler,
                    'LR_FACTOR': lr_factor,
                    'LR_PATIENCE': lr_patience
                }
                
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Define callback for updating progress
                def update_progress(epoch, logs):
                    progress = (epoch + 1) / epochs
                    progress_bar.progress(progress)
                    status_text.text(f"Epoch {epoch+1}/{epochs} - Loss: {logs.get('loss', 0):.4f} - Val Loss: {logs.get('val_loss', 0):.4f}")
                
                # Train model
                model, history, X_train, X_test, y_train, y_test, dates_train, dates_test, scalers = train_model(
                    st.session_state['gauged_data_path'], 
                    model_params,
                    update_progress
                )
                
                # Save model and related objects
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_dir = os.path.join("models", f"model_{timestamp}")
                os.makedirs(model_dir, exist_ok=True)
                
                model.save(os.path.join(model_dir, "model.h5"))
                joblib.dump(scalers, os.path.join(model_dir, "scalers.pkl"))
                joblib.dump(model_params, os.path.join(model_dir, "model_params.pkl"))
                
                # Save training data for evaluation
                joblib.dump({
                    'X_train': X_train,
                    'X_test': X_test,
                    'y_train': y_train,
                    'y_test': y_test,
                    'dates_train': dates_train,
                    'dates_test': dates_test
                }, os.path.join(model_dir, "training_data.pkl"))
                
                # Save history
                joblib.dump(history.history, os.path.join(model_dir, "history.pkl"))
                
                # Update session state
                st.session_state['model_dir'] = model_dir
                st.session_state['model_trained'] = True
                
                # Display success message
                st.markdown("<p class='success-text'>Model trained successfully!</p>", unsafe_allow_html=True)
                
                # Plot training history
                st.markdown("### Training History")
                fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                
                # Plot loss
                ax[0].plot(history.history['loss'], label='Training Loss')
                if 'val_loss' in history.history:
                    ax[0].plot(history.history['val_loss'], label='Validation Loss')
                ax[0].set_title('Model Loss')
                ax[0].set_xlabel('Epoch')
                ax[0].set_ylabel('Loss')
                ax[0].legend()
                ax[0].grid(True, alpha=0.3)
                
                # Plot learning rate if available
                if 'lr' in history.history:
                    ax[1].plot(history.history['lr'], color='green')
                    ax[1].set_title('Learning Rate')
                    ax[1].set_xlabel('Epoch')
                    ax[1].set_ylabel('Learning Rate')
                    ax[1].grid(True, alpha=0.3)
                
                st.pyplot(fig)
    else:
        st.markdown("<p class='warning-text'>Please process data in the Data Processing tab first.</p>", unsafe_allow_html=True)

# Evaluation Tab
with tabs[2]:
    st.markdown("<h2 class='sub-header'>Model Evaluation</h2>", unsafe_allow_html=True)
    
    if st.session_state.get('model_trained', False):
        if st.button("Evaluate Model"):
            with st.spinner("Evaluating model..."):
                # Load model and training data
                model_dir = st.session_state['model_dir']
                model = create_pinn_gru_model(joblib.load(os.path.join(model_dir, "model_params.pkl")))
                model.load_weights(os.path.join(model_dir, "model.h5"))
                
                training_data = joblib.load(os.path.join(model_dir, "training_data.pkl"))
                scalers = joblib.load(os.path.join(model_dir, "scalers.pkl"))
                
                # Evaluate model
                metrics, visualizations = evaluate_model(
                    model, 
                    training_data['X_train'], 
                    training_data['X_test'], 
                    training_data['y_train'], 
                    training_data['y_test'],
                    training_data['dates_train'],
                    training_data['dates_test'],
                    scalers
                )
                
                # Display metrics
                st.markdown("### Performance Metrics")
                
                # Create columns for train and test metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Training Set")
                    st.write(f"RMSE: {metrics['train_rmse']:.4f}")
                    st.write(f"MAE: {metrics['train_mae']:.4f}")
                    st.write(f"R²: {metrics['train_r2']:.4f}")
                    st.write(f"NSE: {metrics['train_nse']:.4f}")
                    st.write(f"KGE: {metrics['train_kge']:.4f}")
                    st.write(f"PBIAS: {metrics['train_pbias']:.4f}%")
                    st.write(f"High Flow Bias: {metrics['train_hf_bias']:.4f}%")
                    st.write(f"Low Flow Bias: {metrics['train_lf_bias']:.4f}%")
                
                with col2:
                    st.markdown("#### Testing Set")
                    st.write(f"RMSE: {metrics['test_rmse']:.4f}")
                    st.write(f"MAE: {metrics['test_mae']:.4f}")
                    st.write(f"R²: {metrics['test_r2']:.4f}")
                    st.write(f"NSE: {metrics['test_nse']:.4f}")
                    st.write(f"KGE: {metrics['test_kge']:.4f}")
                    st.write(f"PBIAS: {metrics['test_pbias']:.4f}%")
                    st.write(f"High Flow Bias: {metrics['test_hf_bias']:.4f}%")
                    st.write(f"Low Flow Bias: {metrics['test_lf_bias']:.4f}%")
                
                # Display visualizations
                st.markdown("### Visualizations")
                
                # Hydrographs
                st.markdown("#### Hydrographs")
                st.pyplot(visualizations['train_hydrograph'])
                st.pyplot(visualizations['test_hydrograph'])
                
                # Log scale hydrographs
                st.markdown("#### Log Scale Hydrographs")
                st.pyplot(visualizations['train_hydrograph_log'])
                st.pyplot(visualizations['test_hydrograph_log'])
                
                # Scatter plots
                st.markdown("#### Scatter Plots")
                col1, col2 = st.columns(2)
                with col1:
                    st.pyplot(visualizations['train_scatter'])
                with col2:
                    st.pyplot(visualizations['test_scatter'])
                
                # Flow duration curves
                st.markdown("#### Flow Duration Curves")
                col1, col2 = st.columns(2)
                with col1:
                    st.pyplot(visualizations['train_fdc'])
                with col2:
                    st.pyplot(visualizations['test_fdc'])
                
                # Residual plots
                st.markdown("#### Residual Analysis")
                col1, col2 = st.columns(2)
                with col1:
                    st.pyplot(visualizations['train_residuals'])
                with col2:
                    st.pyplot(visualizations['test_residuals'])
                
                # Event analysis if available
                if 'test_events' in visualizations:
                    st.markdown("#### Event Analysis")
                    st.pyplot(visualizations['test_events'])
                
                # Save evaluation results
                st.markdown("### Download Results")
                
      <response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>
