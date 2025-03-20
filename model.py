import tensorflow as tf
import numpy as np
import os
import joblib
from tensorflow.keras.callbacks import Callback
import streamlit as st

class Attention(tf.keras.layers.Layer):
    """
    Attention layer for GRU model
    """
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", 
                                 shape=(input_shape[-1], input_shape[-1]),
                                 initializer="glorot_uniform",
                                 trainable=True)
        self.b = self.add_weight(name="att_bias", 
                                 shape=(input_shape[-1],),
                                 initializer="zeros",
                                 trainable=True)
        
    def call(self, inputs):
        score = tf.nn.tanh(tf.matmul(inputs, self.W) + self.b)
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector
    
    def get_config(self):
        config = super(Attention, self).get_config()
        return config

def water_balance_loss(y_true, y_pred, inputs):
    """
    Physics-informed loss function based on water balance
    """
    pcp, temp_max, temp_min, solar = inputs[:, 0, 0], inputs[:, 0, 1], inputs[:, 0, 2], inputs[:, 0, 3]
    et = 0.0023 * (temp_max - temp_min) * (temp_max + temp_min) * solar
    predicted_Q = y_pred
    balance_term = pcp - (et + predicted_Q)
    return tf.reduce_mean(tf.square(balance_term))

def custom_loss(inputs, y_true, y_pred, physics_loss_weight=0.1):
    """
    Custom loss function combining MSE and physics-informed loss
    """
    # Penalize overpredictions more heavily to address high PBIAS
    weights = tf.where(y_pred > y_true, 5.0, 1.0)  # 5x penalty for overpredictions
    weighted_mse_loss = tf.reduce_mean(weights * tf.square(y_true - y_pred))
    
    # Physics-informed loss
    physics_loss = water_balance_loss(y_true, y_pred, inputs)
    
    return weighted_mse_loss + physics_loss_weight * physics_loss

class PINNModel(tf.keras.Model):
    """
    Physics-Informed Neural Network Model
    """
    def __init__(self, physics_loss_weight=0.1, *args, **kwargs):
        super(PINNModel, self).__init__(*args, **kwargs)
        self.physics_loss_weight = physics_loss_weight
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.val_loss_tracker = tf.keras.metrics.Mean(name='val_loss')
        
    def train_step(self, data):
        X, y = data
        
        with tf.GradientTape() as tape:
            y_pred = self(X, training=True)
            loss = custom_loss(X, y, y_pred, self.physics_loss_weight)
            
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
    
    def test_step(self, data):
        X, y = data
        
        y_pred = self(X, training=False)
        loss = custom_loss(X, y, y_pred, self.physics_loss_weight)
        
        self.val_loss_tracker.update_state(loss)
        return {"val_loss": self.val_loss_tracker.result()}
    
    @property
    def metrics(self):
        return [self.loss_tracker, self.val_loss_tracker]
    
    def call(self, inputs):
        return super().call(inputs)

def create_pinn_gru_model(params):
    """
    Create a PINN-GRU model with the specified parameters
    
    Parameters:
    -----------
    params : dict
        Dictionary of model parameters
        
    Returns:
    --------
    model : PINNModel
        Compiled PINN-GRU model
    """
    # Extract parameters
    gru_units = params.get('GRU_UNITS', 128)
    dense_units_1 = params.get('DENSE_UNITS_1', 256)
    dense_units_2 = params.get('DENSE_UNITS_2', 512)
    dense_units_3 = params.get('DENSE_UNITS_3', 256)
    dropout_rate = params.get('DROPOUT_RATE', 0.4)
    learning_rate = params.get('LEARNING_RATE', 0.0001)
    physics_loss_weight = params.get('PHYSICS_LOSS_WEIGHT', 0.1)
    
    # Create model architecture
    inputs = tf.keras.Input(shape=(1, None))  # Dynamic input shape
    
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(gru_units, return_sequences=True))(inputs)
    x = Attention()(x)
    
    x = tf.keras.layers.Dense(dense_units_1, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    x = tf.keras.layers.Dense(dense_units_2, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    x = tf.keras.layers.Dense(dense_units_3, activation='relu')(x)
    output = tf.keras.layers.Dense(1)(x)
    
    # Create and compile model
    model = PINNModel(physics_loss_weight=physics_loss_weight, inputs=inputs, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                 loss='mae', 
                 run_eagerly=True)
    
    return model

class ProgressCallback(Callback):
    """
    Custom callback for updating Streamlit progress bar
    """
    def __init__(self, update_function):
        super(ProgressCallback, self).__init__()
        self.update_function = update_function
        
    def on_epoch_end(self, epoch, logs=None):
        self.update_function(epoch, logs)

def train_model(data_path, model_params, update_progress=None):
    """
    Train a PINN-GRU model on the provided data
    
    Parameters:
    -----------
    data_path : str
        Path to data file
    model_params : dict
        Dictionary of model parameters
    update_progress : function
        Function to update progress in Streamlit
        
    Returns:
    --------
    model : PINNModel
        Trained model
    history : History
        Training history
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
    """
    try:
        from utils.data_processing import load_data, preprocess_data
        
        # Load data
        data, message = load_data(data_path)
        if data is None:
            st.error(message)
            return None, None, None, None, None, None, None, None, None
        
        # Preprocess data
        processed_data, message = preprocess_data(
            data, 
            num_lagged_features=model_params.get('NUM_LAGGED_FEATURES', 12),
            train_test_split=model_params.get('TRAIN_TEST_SPLIT', 0.8)
        )
        
        if processed_data is None:
            st.error(message)
            return None, None, None, None, None, None, None, None, None
        
        # Extract data
        X_train = processed_data['X_train']
        X_test = processed_data['X_test']
        y_train = processed_data['y_train']
        y_test = processed_data['y_test']
        dates_train = processed_data['dates_train']
        dates_test = processed_data['dates_test']
        scalers = processed_data['scalers']
        
        # Create model
        model = create_pinn_gru_model(model_params)
        
        # Create callbacks
        callbacks = []
        
        # Add early stopping if specified
        if model_params.get('EARLY_STOPPING', True):
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=model_params.get('PATIENCE', 300),
                min_delta=model_params.get('MIN_DELTA', 0.001),
                restore_best_weights=True
            )
            callbacks.append(early_stopping)
        
        # Add learning rate scheduler if specified
        if model_params.get('USE_LR_SCHEDULER', True):
            lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=model_params.get('LR_FACTOR', 0.8),
                patience=model_params.get('LR_PATIENCE', 20),
                verbose=1
            )
            callbacks.append(lr_scheduler)
        
        # Add progress callback if provided
        if update_progress is not None:
            progress_callback = ProgressCallback(update_progress)
            callbacks.append(progress_callback)
        
        # Train model
        history = model.fit(
            X_train, 
            y_train,
            epochs=model_params.get('EPOCHS', 500),
            batch_size=model_params.get('BATCH_SIZE', 32),
            validation_data=(X_test, y_test),
            verbose=1,
            callbacks=callbacks
        )
        
        return model, history, X_train, X_test, y_train, y_test, dates_train, dates_test, scalers
    
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None, None, None, None, None, None, None, None, None

def save_model(model, model_params, scalers, training_data, history, output_dir):
    """
    Save model and related objects
    
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
        Directory to save model to
        
    Returns:
    --------
    success : bool
        Whether the model was saved successfully
    message : str
        Success or error message
    """
    try:
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
        joblib.dump(history.history, os.path.join(output_dir, "history.pkl"))
        
        return True, f"Model saved successfully to {output_dir}"
    
    except Exception as e:
        return False, f"Error saving model: {str(e)}"

def load_model(model_dir):
    """
    Load model and related objects
    
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
    success : bool
        Whether the model was loaded successfully
    message : str
        Success or error message
    """
    try:
        # Check if model directory exists
        if not os.path.exists(model_dir):
            return None, None, None, None, None, False, f"Model directory {model_dir} does not exist"
        
        # Check if required files exist
        required_files = ["model.h5", "scalers.pkl", "model_params.pkl", "training_data.pkl", "history.pkl"]
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_dir, f))]
        
        if missing_files:
            return None, None, None, None, None, False, f"Missing required files: {', '.join(missing_files)}"
        
        # Load model parameters
        model_params = joblib.load(os.path.join(model_dir, "model_params.pkl"))
        
        # Create and load model
        model = create_pinn_gru_model(model_params)
        model.load_weights(os.path.join(model_dir, "model.h5"))
        
        # Load scalers
        scalers = joblib.load(os.path.join(model_dir, "scalers.pkl"))
        
        # Load training data
        training_data = joblib.load(os.path.join(model_dir, "training_data.pkl"))
        
        # Load history
        history = joblib.load(os.path.join(model_dir, "history.pkl"))
        
        return model, model_params, scalers, training_data, history, True, "Model loaded successfully"
    
    except Exception as e:
        return None, None, None, None, None, False, f"Error loading model: {str(e)}"
