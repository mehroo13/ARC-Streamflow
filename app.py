import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
from datetime import datetime
from utils.data_processing import load_data, preprocess_data
from utils.model import train_model, PINNModel, Attention
from utils.evaluation import evaluate_model, generate_visualizations
from utils.prediction import predict_ungauged, save_predictions
from utils.model_manager import save_model_with_metadata, load_model_with_metadata, list_available_models, display_model_info

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 36px;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 20px;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scalers' not in st.session_state:
    st.session_state.scalers = None

# Sidebar for model parameters
st.sidebar.header("Model Parameters")
gru_units = st.sidebar.slider("GRU Units", 32, 256, 128, step=32)
dense_units_1 = st.sidebar.slider("Dense Units 1", 64, 512, 256, step=64)
dense_units_2 = st.sidebar.slider("Dense Units 2", 64, 1024, 512, step=64)
dense_units_3 = st.sidebar.slider("Dense Units 3", 64, 512, 256, step=64)
dropout_rate = st.sidebar.slider("Dropout Rate", 0.1, 0.5, 0.4, step=0.1)
learning_rate = st.sidebar.number_input("Learning Rate", 0.00001, 0.01, 0.0001, format="%.5f")
physics_loss_weight = st.sidebar.slider("Physics Loss Weight", 0.0, 1.0, 0.1, step=0.1)
batch_size = st.sidebar.slider("Batch Size", 16, 128, 32, step=16)
epochs = st.sidebar.slider("Epochs", 100, 1000, 500, step=100)
prediction_batch_size = st.sidebar.slider("Prediction Batch Size", 50, 500, 100, step=50)

model_params = {
    "GRU_UNITS": gru_units,
    "DENSE_UNITS_1": dense_units_1,
    "DENSE_UNITS_2": dense_units_2,
    "DENSE_UNITS_3": dense_units_3,
    "DROPOUT_RATE": dropout_rate,
    "LEARNING_RATE": learning_rate,
    "PHYSICS_LOSS_WEIGHT": physics_loss_weight,
    "BATCH_SIZE": batch_size,
    "EPOCHS": epochs,
    "PREDICTION_BATCH_SIZE": prediction_batch_size,
    "NUM_LAGGED_FEATURES": 12,
    "TRAIN_TEST_SPLIT": 0.8,
    "OUTPUT_VARIABLE": "Discharge (m³/S)",
    "USE_LR_SCHEDULER": True,
    "LR_FACTOR": 0.8,
    "LR_PATIENCE": 20,
    "EARLY_STOPPING": True,
    "PATIENCE": 300,
    "MIN_DELTA": 0.001
}

# Main title
st.markdown('<div class="main-header">Hydrological Streamflow Prediction with PINN</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data Processing", "Model Training", "Evaluation", "Prediction", "About"])

# Data Processing Tab
with tab1:
    st.header("Data Processing")
    uploaded_file = st.file_uploader("Upload gauged catchment data (Excel)", type=["xlsx"], key="gauged")
    uploaded_ungauged_file = st.file_uploader("Upload ungauged catchment data (Excel, optional)", type=["xlsx"], key="ungauged")
    
    if uploaded_file:
        try:
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())
            data, message = load_data(uploaded_file.name)
            if data is not None:
                st.success(message)
                
                # Variable selection
                all_columns = data.columns.tolist()
                dynamic_cols = st.multiselect("Select Dynamic Variables", all_columns, 
                                            default=['Rainfall (mm)', 'Maximum temperature (°C)', 
                                                    'Minimum temperature (°C)', 'Daily global solar exposure (MJ/m*m)'])
                output_col = st.selectbox("Select Output Variable", all_columns, index=all_columns.index('Discharge (m³/S)') if 'Discharge (m³/S)' in all_columns else 0)
                categorical_cols = st.multiselect("Select Categorical Static Variables", all_columns, 
                                                default=['Land Use Classes', 'Soil Classes'] if 'Land Use Classes' in all_columns else [])
                numeric_static_cols = st.multiselect("Select Numeric Static Variables", all_columns, 
                                                    default=['Land Use Percentage', 'Soil Percentage', 'Slope (Degree)'] if 'Land Use Percentage' in all_columns else [])
                
                if st.button("Process Data"):
                    X_train, X_test, y_train, y_test, dates_train, dates_test, scalers = preprocess_data(
                        uploaded_file.name, dynamic_variables=dynamic_cols, output_variable=output_col,
                        categorical_variables=categorical_cols, numeric_static_variables=numeric_static_cols
                    )
                    if X_train is not None:
                        st.session_state.processed_data = {
                            'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test,
                            'dates_train': dates_train, 'dates_test': dates_test
                        }
                        st.session_state.scalers = scalers
                        st.success("Data processed successfully!")
                    else:
                        st.error("Error processing data")
            else:
                st.error(message)
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")

# Model Training Tab
with tab2:
    st.header("Model Training")
    
    if st.session_state.processed_data is None:
        st.warning("Please process data first in the Data Processing tab.")
    else:
        def update_progress(epoch, logs):
            progress = (epoch + 1) / model_params["EPOCHS"]
            progress_bar.progress(progress)
            loss_display.text(f"Epoch {epoch + 1}/{model_params['EPOCHS']} - Loss: {logs.get('loss', 0):.4f} - Val Loss: {logs.get('val_loss', 0):.4f}")
        
        if st.button("Train Model"):
            progress_bar = st.progress(0)
            loss_display = st.empty()
            
            try:
                model, history, X_train, X_test, y_train, y_test, dates_train, dates_test, scalers = train_model(
                    uploaded_file.name,
                    model_params,
                    progress_callback=update_progress
                )
                
                # Save model artifacts only if model is not None
                if model is not None:
                    st.session_state.model = model
                    st.session_state.scalers = scalers
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_dir = f"models/model_{timestamp}"
                    os.makedirs(output_dir, exist_ok=True)
                    model.save(os.path.join(output_dir, "model"))
                    model.save(os.path.join(output_dir, "model.h5"))
                    
                    training_data = {
                        'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test,
                        'dates_train': dates_train, 'dates_test': dates_test
                    }
                    joblib.dump(scalers, os.path.join(output_dir, "scalers.pkl"))
                    joblib.dump(model_params, os.path.join(output_dir, "model_params.pkl"))
                    joblib.dump(training_data, os.path.join(output_dir, "training_data.pkl"))
                    joblib.dump(history.history, os.path.join(output_dir, "history.pkl"))
                    
                    st.success(f"Model trained and saved successfully to {output_dir}")
                    
                    # Plot training history
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    ax1.plot(history.history['loss'], label='Training Loss')
                    ax1.plot(history.history['val_loss'], label='Validation Loss')
                    ax1.set_title('Model Loss')
                    ax1.set_xlabel('Epoch')
                    ax1.set_ylabel('Loss')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    
                    if 'lr' in history.history:
                        ax2.plot(history.history['lr'], label='Learning Rate')
                        ax2.set_title('Learning Rate')
                        ax2.set_xlabel('Epoch')
                        ax2.set_ylabel('Learning Rate')
                        ax2.legend()
                        ax2.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"Error training model: {str(e)}")

# Evaluation Tab
with tab3:
    st.header("Model Evaluation")
    
    eval_option = st.radio("Evaluate:", ("Trained Model", "Uploaded Model"))
    
    if eval_option == "Trained Model" and st.session_state.model is None:
        st.warning("Please train a model first in the Model Training tab.")
    elif eval_option == "Uploaded Model" and st.session_state.processed_data is None:
        st.warning("Please process data first in the Data Processing tab.")
    else:
        if eval_option == "Uploaded Model":
            uploaded_model_file = st.file_uploader("Upload trained model", type=["h5", "zip"], key="model_upload")
            if uploaded_model_file:
                model_path = uploaded_model_file.name
                with open(model_path, "wb") as f:
                    f.write(uploaded_model_file.getbuffer())
                
                try:
                    custom_objects = {"PINNModel": PINNModel, "Attention": Attention}
                    try:
                        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
                    except:
                        model = tf.keras.models.load_model(model_path + ".h5", custom_objects=custom_objects)
                    st.session_state.model = model
                    st.success("Model uploaded successfully!")
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
        
        if st.session_state.model is not None and st.button("Evaluate Model"):
            try:
                processed_data = st.session_state.processed_data
                metrics, visualizations = evaluate_model(
                    st.session_state.model,
                    processed_data['X_train'], processed_data['X_test'],
                    processed_data['y_train'], processed_data['y_test'],
                    processed_data['dates_train'], processed_data['dates_test'],
                    st.session_state.scalers
                )
                
                # Display metrics
                st.subheader("Performance Metrics")
                metrics_df = pd.DataFrame(metrics.items(), columns=['Metric', 'Value'])
                st.dataframe(metrics_df)
                
                csv = metrics_df.to_csv(index=False)
                st.download_button("Download Metrics as CSV", csv, "metrics.csv", "text/csv")
                
                # Display visualizations
                st.subheader("Visualizations")
                for vis_name, fig in visualizations.items():
                    st.pyplot(fig)
                
                test_events = st.session_state.get('test_events', None)
                if test_events is not None:
                    st.subheader("Test Events")
                    st.pyplot(test_events)
            except Exception as e:
                st.error(f"Error evaluating model: {str(e)}")

# Prediction Tab
with tab4:
    st.header("Prediction for Ungauged Catchment")
    
    if st.session_state.model is None:
        st.warning("Please train or upload a model first.")
    elif st.session_state.scalers is None or st.session_state.processed_data is None:
        st.warning("Please process gauged catchment data first.")
    elif uploaded_ungauged_file is None:
        st.warning("Please upload ungauged catchment data in the Data Processing tab.")
    else:
        if st.button("Generate Predictions"):
            try:
                with open(uploaded_ungauged_file.name, "wb") as f:
                    f.write(uploaded_ungauged_file.getbuffer())
                
                predictions, dates, visualizations = predict_ungauged(
                    st.session_state.model,
                    uploaded_file.name,
                    uploaded_ungauged_file.name,
                    st.session_state.scalers,
                    model_params
                )
                
                if predictions is not None:
                    st.success("Predictions generated successfully!")
                    
                    # Display predictions
                    st.subheader("Predicted Streamflow")
                    predictions_df = pd.DataFrame({'Date': dates, 'Predicted Streamflow (m³/s)': predictions})
                    st.dataframe(predictions_df)
                    
                    # Download button
                    csv = predictions_df.to_csv(index=False)
                    st.download_button("Download Predictions as CSV", csv, "predictions.csv", "text/csv")
                    
                    # Display visualizations
                    st.subheader("Visualizations")
                    for vis_name, fig in visualizations.items():
                        st.pyplot(fig)
                
                else:
                    st.error("Error generating predictions")
            except Exception as e:
                st.error(f"Error generating predictions: {str(e)}")

# About Tab
with tab5:
    st.header("About")
    st.markdown("""
    <div class="info-box">
        <h3>Streamflow Prediction Application</h3>
        <p>This application uses a Physics-Informed Neural Network (PINN) with GRU architecture to predict streamflow in both gauged and ungauged catchments.</p>
        <h4>Features:</h4>
        <ul>
            <li>Data preprocessing with dynamic and static variables</li>
            <li>Model training with customizable hyperparameters</li>
            <li>Comprehensive model evaluation with hydrological metrics</li>
            <li>Prediction for ungauged catchments</li>
        </ul>
        <h4>Data Requirements:</h4>
        <p>Gauged catchment data should include Date, Discharge, and meteorological variables. Ungauged data requires similar inputs excluding Discharge.</p>
        <h4>References:</h4>
        <ul>
            <li>Rasheed et al., "Physics-informed neural networks" (2020)</li>
            <li>Cho et al., "Learning Phrase Representations using RNN" (2014)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
