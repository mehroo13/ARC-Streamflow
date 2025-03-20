import os
import joblib
import tensorflow as tf
import streamlit as st
from datetime import datetime
from utils.model import create_pinn_gru_model, PINNModel, Attention

def save_model_with_metadata(model, model_params, scalers, training_data, history, output_dir=None):
    """
    Save model and related objects with timestamp
    
    Parameters:
    -----------
    model : PINNModel
        Trained model
    model_params : dict
        Dictionary of model parameters
    scalers : dict
        Dictionary of scalers
    training_data : dict
        Dictionary of training data
    history : History
        Training history
    output_dir : str
        Directory to save model to (if None, creates a timestamped directory)
        
    Returns:
    --------
    model_dir : str
        Path to saved model directory
    success : bool
        Whether the model was saved successfully
    message : str
        Success or error message
    """
    try:
        # Create output directory with timestamp if not provided
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join("models", f"model_{timestamp}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        model.save(os.path.join(output_dir, "model.h5"))
        
        # Save scalers
        joblib.dump(scalers, os.path.join(output_dir, "scalers.pkl"))
        
        # Save model parameters
        joblib.dump(model_params, os.path.join(output_dir, "model_params.pkl"))
        
        # Save training data
        joblib.dump(training_data, os.path.join(output_dir, "training_data.pkl"))
        
        # Save history
        if hasattr(history, 'history'):
            joblib.dump(history.history, os.path.join(output_dir, "history.pkl"))
        else:
            joblib.dump(history, os.path.join(output_dir, "history.pkl"))
        
        # Save metadata about the model
        metadata = {
            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model_type': 'PINN-GRU',
            'gru_units': model_params.get('GRU_UNITS', 128),
            'dense_units': [
                model_params.get('DENSE_UNITS_1', 256),
                model_params.get('DENSE_UNITS_2', 512),
                model_params.get('DENSE_UNITS_3', 256)
            ],
            'dropout_rate': model_params.get('DROPOUT_RATE', 0.4),
            'physics_loss_weight': model_params.get('PHYSICS_LOSS_WEIGHT', 0.1),
            'num_lagged_features': model_params.get('NUM_LAGGED_FEATURES', 12),
            'train_test_split': model_params.get('TRAIN_TEST_SPLIT', 0.8)
        }
        
        joblib.dump(metadata, os.path.join(output_dir, "metadata.pkl"))
        
        return output_dir, True, f"Model saved successfully to {output_dir}"
    
    except Exception as e:
        return None, False, f"Error saving model: {str(e)}"

def load_model_with_metadata(model_dir):
    """
    Load model and related objects with validation
    
    Parameters:
    -----------
    model_dir : str
        Directory containing saved model
        
    Returns:
    --------
    model : PINNModel
        Loaded model
    model_params : dict
        Dictionary of model parameters
    scalers : dict
        Dictionary of scalers
    training_data : dict
        Dictionary of training data
    history : dict
        Training history
    metadata : dict
        Model metadata
    success : bool
        Whether the model was loaded successfully
    message : str
        Success or error message
    """
    try:
        # Check if model directory exists
        if not os.path.exists(model_dir):
            return None, None, None, None, None, None, False, f"Model directory {model_dir} does not exist"
        
        # Check if required files exist
        required_files = ["model.h5", "scalers.pkl", "model_params.pkl", "training_data.pkl", "history.pkl"]
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_dir, f))]
        
        if missing_files:
            return None, None, None, None, None, None, False, f"Missing required files: {', '.join(missing_files)}"
        
        # Load model parameters
        model_params = joblib.load(os.path.join(model_dir, "model_params.pkl"))
        
        # Create and load model
        custom_objects = {
            'PINNModel': PINNModel,
            'Attention': Attention
        }
        
        model = tf.keras.models.load_model(
            os.path.join(model_dir, "model.h5"),
            custom_objects=custom_objects
        )
        
        # Load scalers
        scalers = joblib.load(os.path.join(model_dir, "scalers.pkl"))
        
        # Load training data
        training_data = joblib.load(os.path.join(model_dir, "training_data.pkl"))
        
        # Load history
        history = joblib.load(os.path.join(model_dir, "history.pkl"))
        
        # Load metadata if available
        metadata = None
        if os.path.exists(os.path.join(model_dir, "metadata.pkl")):
            metadata = joblib.load(os.path.join(model_dir, "metadata.pkl"))
        else:
            # Create basic metadata from model_params
            metadata = {
                'model_type': 'PINN-GRU',
                'gru_units': model_params.get('GRU_UNITS', 128),
                'dense_units': [
                    model_params.get('DENSE_UNITS_1', 256),
                    model_params.get('DENSE_UNITS_2', 512),
                    model_params.get('DENSE_UNITS_3', 256)
                ],
                'dropout_rate': model_params.get('DROPOUT_RATE', 0.4),
                'physics_loss_weight': model_params.get('PHYSICS_LOSS_WEIGHT', 0.1),
                'num_lagged_features': model_params.get('NUM_LAGGED_FEATURES', 12),
                'train_test_split': model_params.get('TRAIN_TEST_SPLIT', 0.8)
            }
        
        return model, model_params, scalers, training_data, history, metadata, True, "Model loaded successfully"
    
    except Exception as e:
        return None, None, None, None, None, None, False, f"Error loading model: {str(e)}"

def list_available_models(models_dir="models"):
    """
    List all available saved models
    
    Parameters:
    -----------
    models_dir : str
        Directory containing saved models
        
    Returns:
    --------
    models : list
        List of dictionaries with model information
    """
    try:
        if not os.path.exists(models_dir):
            return []
        
        models = []
        
        for model_dir in os.listdir(models_dir):
            model_path = os.path.join(models_dir, model_dir)
            
            if not os.path.isdir(model_path):
                continue
                
            # Check if this is a valid model directory
            if not os.path.exists(os.path.join(model_path, "model.h5")):
                continue
                
            # Get metadata if available
            metadata = None
            if os.path.exists(os.path.join(model_path, "metadata.pkl")):
                metadata = joblib.load(os.path.join(model_path, "metadata.pkl"))
            
            # Get model parameters if available
            model_params = None
            if os.path.exists(os.path.join(model_path, "model_params.pkl")):
                model_params = joblib.load(os.path.join(model_path, "model_params.pkl"))
            
            # Create model info
            model_info = {
                'name': model_dir,
                'path': model_path,
                'created_at': metadata.get('created_at', 'Unknown') if metadata else 'Unknown',
                'model_type': metadata.get('model_type', 'PINN-GRU') if metadata else 'PINN-GRU',
                'gru_units': metadata.get('gru_units', model_params.get('GRU_UNITS', 128)) if metadata else model_params.get('GRU_UNITS', 128) if model_params else 128,
                'physics_loss_weight': metadata.get('physics_loss_weight', model_params.get('PHYSICS_LOSS_WEIGHT', 0.1)) if metadata else model_params.get('PHYSICS_LOSS_WEIGHT', 0.1) if model_params else 0.1
            }
            
            models.append(model_info)
        
        # Sort by creation date if available
        models.sort(key=lambda x: x['created_at'] if x['created_at'] != 'Unknown' else '', reverse=True)
        
        return models
    
    except Exception as e:
        st.error(f"Error listing models: {str(e)}")
        return []

def display_model_info(model_info, metadata=None, model_params=None):
    """
    Display model information in Streamlit
    
    Parameters:
    -----------
    model_info : dict
        Dictionary with basic model information
    metadata : dict
        Dictionary with model metadata
    model_params : dict
        Dictionary with model parameters
    """
    st.markdown(f"### Model: {model_info['name']}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Basic Information")
        st.write(f"**Model Type:** {model_info['model_type']}")
        st.write(f"**Created At:** {model_info['created_at']}")
        st.write(f"**Path:** {model_info['path']}")
    
    with col2:
        st.markdown("#### Model Architecture")
        st.write(f"**GRU Units:** {model_info['gru_units']}")
        
        if metadata:
            dense_units = metadata.get('dense_units', [256, 512, 256])
            st.write(f"**Dense Units:** {dense_units}")
            st.write(f"**Dropout Rate:** {metadata.get('dropout_rate', 0.4)}")
        
        st.write(f"**Physics Loss Weight:** {model_info['physics_loss_weight']}")
    
    if model_params:
        st.markdown("#### Training Parameters")
        st.write(f"**Learning Rate:** {model_params.get('LEARNING_RATE', 0.0001)}")
        st.write(f"**Batch Size:** {model_params.get('BATCH_SIZE', 32)}")
        st.write(f"**Epochs:** {model_params.get('EPOCHS', 500)}")
        st.write(f"**Train-Test Split:** {model_params.get('TRAIN_TEST_SPLIT', 0.8)}")
        st.write(f"**Number of Lagged Features:** {model_params.get('NUM_LAGGED_FEATURES', 12)}")
