import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import joblib
from datetime import datetime

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
        # Variable selection section
        st.markdown("### Variable Selection")
        
        # Save uploaded file temporarily to read columns
        temp_gauged_data_path = os.path.join("data", "temp_gauged_data.xlsx")
        os.makedirs("data", exist_ok=True)  # Ensure data directory exists
        with open(temp_gauged_data_path, "wb") as f:
            f.write(gauged_data_file.getvalue())
        
        # Read the data to get column names
        try:
            temp_data = pd.read_excel(temp_gauged_data_path)
            if temp_data is not None and not temp_data.empty:
                # Get column names excluding 'Date'
                columns = [col for col in temp_data.columns if col != 'Date']
                
                # Select output variable (default to 'Discharge (m³/S)' if available)
                default_output = 'Discharge (m³/S)' if 'Discharge (m³/S)' in columns else columns[0]
                output_variable = st.selectbox("Select Output Variable", columns, index=columns.index(default_output) if default_output in columns else 0)
                
                # Remove output variable from the list of potential input variables
                input_columns = [col for col in columns if col != output_variable]
                
                # Select dynamic input variables
                st.markdown("#### Select Dynamic Input Variables")
                st.markdown("Dynamic variables change over time (e.g., rainfall, temperature)")
                
                # Default dynamic variables if available
                default_dynamic = ['Rainfall (mm)', 'Maximum temperature (°C)', 'Minimum temperature (°C)', 'Daily global solar exposure (MJ/m*m)']
                default_dynamic = [var for var in default_dynamic if var in input_columns]
                
                dynamic_variables = st.multiselect(
                    "Dynamic Variables", 
                    input_columns,
                    default=default_dynamic
                )
                
                # Remaining columns are static by default
                static_variables = [col for col in input_columns if col not in dynamic_variables]
                
                # Select categorical static variables
                st.markdown("#### Select Categorical Static Variables")
                st.markdown("Categorical variables are non-numeric (e.g., land use classes, soil types)")
                
                # Default categorical variables if available
                default_categorical = ['Land Use Classes', 'Soil Classes']
                default_categorical = [var for var in default_categorical if var in static_variables]
                
                categorical_variables = st.multiselect(
                    "Categorical Static Variables", 
                    static_variables,
                    default=default_categorical
                )
                
                # Remaining static variables are numeric
                numeric_static_variables = [col for col in static_variables if col not in categorical_variables]
                
                st.markdown("#### Variable Summary")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Dynamic Variables:**")
                    for var in dynamic_variables:
                        st.write(f"- {var}")
                    
                    st.markdown("**Output Variable:**")
                    st.write(f"- {output_variable}")
                
                with col2:
                    st.markdown("**Categorical Static Variables:**")
                    for var in categorical_variables:
                        st.write(f"- {var}")
                    
                    st.markdown("**Numeric Static Variables:**")
                    for var in numeric_static_variables:
                        st.write(f"- {var}")
                
                # Process data button
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
                        data, message = load_data(gauged_data_path)
                        
                        if data is not None and not data.empty:
                            # Display data info
                            st.markdown("<p class='success-text'>Data loaded successfully!</p>", unsafe_allow_html=True)
                            st.write("Data shape:", data.shape)
                            st.write("Data columns:", data.columns.tolist())
                            
                            # Display sample data
                            st.markdown("### Sample Data")
                            st.dataframe(data.head())
                            
                            # Save variable selections to session state
                            st.session_state['gauged_data_path'] = gauged_data_path
                            st.session_state['ungauged_data_path'] = ungauged_data_path
                            st.session_state['output_variable'] = output_variable
                            st.session_state['dynamic_variables'] = dynamic_variables
                            st.session_state['categorical_variables'] = categorical_variables
                            st.session_state['numeric_static_variables'] = numeric_static_variables
                            st.session_state['data_processed'] = True
                        else:
                            st.error(f"Error loading data: {message}")
            else:
                st.error("Error reading the uploaded file. Please check the file format.")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
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
        
        # Display variable selections
        st.markdown("### Selected Variables")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Dynamic Variables:**")
            for var in st.session_state.get('dynamic_variables', []):
                st.write(f"- {var}")
            
            st.markdown("**Output Variable:**")
            st.write(f"- {st.session_state.get('output_variable', 'Not selected')}")
        
        with col2:
            st.markdown("**Categorical Static Variables:**")
            for var in st.session_state.get('categorical_variables', []):
                st.write(f"- {var}")
            
            st.markdown("**Numeric Static Variables:**")
            for var in st.session_state.get('numeric_static_variables', []):
                st.write(f"- {var}")
        
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
                    'LR_PATIENCE': lr_patience,
                    'OUTPUT_VARIABLE': st.session_state.get('output_variable', 'Discharge (m³/S)'),
                    'DYNAMIC_VARIABLES': st.session_state.get('dynamic_variables', []),
                    'CATEGORICAL_VARIABLES': st.session_state.get('categorical_variables', []),
                    'NUMERIC_STATIC_VARIABLES': st.session_state.get('numeric_static_variables', [])
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
                model_dir = os.path.join("models", "model_" + timestamp)
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
                
                # Save metrics to CSV
                metrics_df = pd.DataFrame({
                    'Metric': ['RMSE', 'MAE', 'R2', 'NSE', 'KGE', 'PBIAS', 'HF_Bias', 'LF_Bias'],
                    'Training': [metrics['train_rmse'], metrics['train_mae'], metrics['train_r2'], 
                                metrics['train_nse'], metrics['train_kge'], metrics['train_pbias'], 
                                metrics['train_hf_bias'], metrics['train_lf_bias']],
                    'Testing': [metrics['test_rmse'], metrics['test_mae'], metrics['test_r2'], 
                               metrics['test_nse'], metrics['test_kge'], metrics['test_pbias'], 
                               metrics['test_hf_bias'], metrics['test_lf_bias']]
                })
                
                metrics_csv = metrics_df.to_csv(index=False)
                st.download_button(
                    label="Download Metrics CSV",
                    data=metrics_csv,
                    file_name="model_metrics.csv",
                    mime="text/csv"
                )
    elif st.session_state.get('data_processed', False):
        st.markdown("<p class='warning-text'>Please train a model in the Model Training tab first.</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p class='warning-text'>Please process data in the Data Processing tab first.</p>", unsafe_allow_html=True)
    
    # Load existing model section
    st.markdown("### Load Existing Model")
    model_path = st.text_input("Enter path to saved model directory")
    
    if model_path and os.path.exists(model_path):
        if st.button("Load Model"):
            with st.spinner("Loading model..."):
                # Update session state
                st.session_state['model_dir'] = model_path
                st.session_state['model_trained'] = True
                
                # Display success message
                st.markdown("<p class='success-text'>Model loaded successfully!</p>", unsafe_allow_html=True)

# Prediction Tab
with tabs[3]:
    st.markdown("<h2 class='sub-header'>Prediction for Ungauged Catchment</h2>", unsafe_allow_html=True)
    
    if st.session_state.get('model_trained', False) and st.session_state.get('ungauged_data_path', None) is not None:
        # Display variable selections
        st.markdown("### Selected Variables")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Dynamic Variables:**")
            for var in st.session_state.get('dynamic_variables', []):
                st.write(f"- {var}")
            
            st.markdown("**Output Variable:**")
            st.write(f"- {st.session_state.get('output_variable', 'Not selected')}")
        
        with col2:
            st.markdown("**Categorical Static Variables:**")
            for var in st.session_state.get('categorical_variables', []):
                st.write(f"- {var}")
            
            st.markdown("**Numeric Static Variables:**")
            for var in st.session_state.get('numeric_static_variables', []):
                st.write(f"- {var}")
        
        if st.button("Generate Predictions"):
            with st.spinner("Generating predictions for ungauged catchment..."):
                # Load model and related objects
                model_dir = st.session_state['model_dir']
                model = create_pinn_gru_model(joblib.load(os.path.join(model_dir, "model_params.pkl")))
                model.load_weights(os.path.join(model_dir, "model.h5"))
                
                scalers = joblib.load(os.path.join(model_dir, "scalers.pkl"))
                model_params = joblib.load(os.path.join(model_dir, "model_params.pkl"))
                
                # Generate predictions
                predictions, dates, visualizations = predict_ungauged(
                    model,
                    st.session_state['gauged_data_path'],
                    st.session_state['ungauged_data_path'],
                    scalers,
                    model_params
                )
                
                # Display predictions
                st.markdown("### Prediction Results")
                
                # Create DataFrame with predictions
                predictions_df = pd.DataFrame({
                    'Date': dates,
                    'Predicted Streamflow (m³/s)': predictions
                })
                
                # Display predictions table
                st.dataframe(predictions_df)
                
                # Display visualizations
                st.markdown("### Visualizations")
                
                # Predicted streamflow
                st.markdown("#### Predicted Streamflow")
                st.pyplot(visualizations['ungauged_predicted'])
                
                # Flow duration curve
                st.markdown("#### Flow Duration Curve")
                st.pyplot(visualizations['ungauged_predicted_fdc'])
                
                # Save predictions to CSV
                predictions_csv = predictions_df.to_csv(index=False)
                st.download_button(
                    label="Download Predictions CSV",
                    data=predictions_csv,
                    file_name="ungauged_predictions.csv",
                    mime="text/csv"
                )
    elif not st.session_state.get('model_trained', False):
        st.markdown("<p class='warning-text'>Please train a model in the Model Training tab first.</p>", unsafe_allow_html=True)
    elif st.session_state.get('ungauged_data_path', None) is None:
        st.markdown("<p class='warning-text'>Please upload ungauged catchment data in the Data Processing tab.</p>", unsafe_allow_html=True)
        
        # Option to upload ungauged data here
        st.markdown("### Upload Ungauged Catchment Data")
        ungauged_data_file = st.file_uploader("Upload Excel file with ungauged catchment data", type=["xlsx", "xls"], key="ungauged_upload_prediction")
        
        if ungauged_data_file is not None:
            # Save uploaded file
            ungauged_data_path = os.path.join("data", "ungauged_data.xlsx")
            with open(ungauged_data_path, "wb") as f:
                f.write(ungauged_data_file.getvalue())
            
            # Update session state
            st.session_state['ungauged_data_path'] = ungauged_data_path
            
            # Display success message
            st.markdown("<p class='success-text'>Ungauged data uploaded successfully!</p>", unsafe_allow_html=True)

# About Tab
with tabs[4]:
    st.markdown("<h2 class='sub-header'>About This Application</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    ### Physics-Informed Neural Network for Hydrological Streamflow Prediction
    
    This application implements a Physics-Informed Neural Network (PINN) with GRU architecture for streamflow prediction in both gauged and ungauged catchments. The model incorporates physical constraints through a custom loss function that enforces water balance principles.
    
    #### Key Features:
    
    - **Data Processing**: Handles both gauged and ungauged catchment data with automatic preprocessing
    - **Variable Selection**: Allows users to specify which variables are static vs. dynamic and which are inputs vs. outputs
    - **Model Architecture**: Bidirectional GRU with attention mechanism and dense layers
    - **Physics-Informed Loss**: Incorporates water balance principles into the loss function
    - **Comprehensive Evaluation**: Calculates hydrological metrics including NSE, KGE, PBIAS, and flow duration biases
    - **Advanced Visualizations**: Generates hydrographs, scatter plots, flow duration curves, and event analysis
    - **Model Saving/Loading**: Allows saving and loading trained models for future use
    
    #### Required Data Format:
    
    The application expects Excel files with the following columns:
    
    **Gauged Catchment Data:**
    - Date: Date column in datetime format
    - Discharge (m³/S): Observed streamflow (or other output variable)
    - Dynamic variables: Time-varying inputs like rainfall, temperature, etc.
    - Static variables: Catchment characteristics that don't change over time
    
    **Ungauged Catchment Data:**
    - Date: Date column in datetime format
    - Dynamic variables: Same dynamic variables as in gauged data
    - Static variables: Same static variables as in gauged data
    
    #### References:
    
    - Kratzert, F., Klotz, D., Brenner, C., Schulz, K., & Herrnegger, M. (2018). Rainfall–runoff modelling using Long Short-Term Memory (LSTM) networks. Hydrology and Earth System Sciences, 22(11), 6005-6022.
    - Rahmani, F., Lawson, K., Ouyang, W., Appling, A., Oliver, S., & Shen, C. (2021). Exploring the exceptional performance of a deep learning stream temperature model and the value of streamflow data. Environmental Research Letters, 16(2), 024025.
    - Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686-707.
    """)
    
    st.markdown("### GitHub Repository")
    st.markdown("The source code for this application is available on GitHub: [Hydrological Streamflow Prediction](https://github.com/yourusername/hydro_streamflow_app) ")
