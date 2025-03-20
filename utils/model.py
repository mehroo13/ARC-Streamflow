import tensorflow as tf
import numpy as np
import os  # Added to resolve NameError
from tensorflow.keras.callbacks import Callback

class Attention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], input_shape[-1]), 
                                 initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[-1],), 
                                 initializer="zeros", trainable=True)
        super(Attention, self).build(input_shape)
    
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
    try:
        # Extract dynamic inputs (assuming first 4 features are rainfall, temp_max, temp_min, solar)
        pcp, temp_max, temp_min, solar = inputs[:, 0, 0], inputs[:, 0, 1], inputs[:, 0, 2], inputs[:, 0, 3]
        
        # Simple ET calculation based on temperature and solar radiation
        et = 0.0023 * (temp_max - temp_min) * (temp_max + temp_min) * solar
        
        # Water balance: precipitation - (evapotranspiration + runoff)
        predicted_Q = y_pred
        balance_term = pcp - (et + predicted_Q)
        
        return tf.reduce_mean(tf.square(balance_term))
    except Exception as e:
        print(f"Warning: Error in water_balance_loss: {str(e)}")
        return 0.0  # Return zero loss if there's an error

def custom_loss(inputs, y_true, y_pred, physics_loss_weight=0.1):
    try:
        # Mean squared error loss
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        
        # Penalize overpredictions more heavily to address high PBIAS
        weights = tf.where(y_pred > y_true, 5.0, 1.0)  # 5x penalty for overpredictions
        weighted_mse_loss = tf.reduce_mean(weights * tf.square(y_true - y_pred))
        
        # Physics-based loss
        physics_loss = water_balance_loss(y_true, y_pred, inputs)
        
        # Combined loss
        return weighted_mse_loss + physics_loss_weight * physics_loss
    except Exception as e:
        print(f"Warning: Error in custom_loss: {str(e)}")
        return tf.reduce_mean(tf.square(y_true - y_pred))  # Fall back to MSE

class PINNModel(tf.keras.Model):
    def __init__(self, inputs, outputs, physics_loss_weight=0.1):
        super(PINNModel, self).__init__(inputs=inputs, outputs=outputs)
        self.physics_loss_weight = physics_loss_weight
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.val_loss_tracker = tf.keras.metrics.Mean(name='val_loss')
    
    def train_step(self, data):
        X, y = data
        
        with tf.GradientTape() as tape:
            y_pred = self(X, training=True)
            
            # Use a simpler loss function if the input structure doesn't match what's expected
            try:
                loss = custom_loss(X, y, y_pred, self.physics_loss_weight)
            except Exception as e:
                print(f"Warning: Using MSE loss instead of custom loss due to: {str(e)}")
                loss = tf.reduce_mean(tf.square(y - y_pred))
        
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
    
    def test_step(self, data):
        X, y = data
        y_pred = self(X, training=False)
        
        # Use a simpler loss function if the input structure doesn't match what's expected
        try:
            loss = custom_loss(X, y, y_pred, self.physics_loss_weight)
        except Exception as e:
            print(f"Warning: Using MSE loss instead of custom loss due to: {str(e)}")
            loss = tf.reduce_mean(tf.square(y - y_pred))
        
        self.val_loss_tracker.update_state(loss)
        return {"val_loss": self.val_loss_tracker.result()}
    
    @property
    def metrics(self):
        return [self.loss_tracker, self.val_loss_tracker]
    
    def get_config(self):
        config = super(PINNModel, self).get_config()
        config.update({"physics_loss_weight": self.physics_loss_weight})
        return config

def create_pinn_gru_model(params):
    # Get parameters
    gru_units = params.get('GRU_UNITS', 128)
    dense_units_1 = params.get('DENSE_UNITS_1', 256)
    dense_units_2 = params.get('DENSE_UNITS_2', 512)
    dense_units_3 = params.get('DENSE_UNITS_3', 256)
    dropout_rate = params.get('DROPOUT_RATE', 0.4)
    physics_loss_weight = params.get('PHYSICS_LOSS_WEIGHT', 0.1)
    
    # Define input shape explicitly - this is the key fix
    input_shape = (1, params.get('INPUT_SHAPE', 384))  # Default to 384 if not specified
    
    # Create model with explicit input shape
    inputs = tf.keras.Input(shape=input_shape)
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
    
    model = PINNModel(inputs, output, physics_loss_weight)
    return model

def train_model(data_path, model_params, progress_callback=None):
    from utils.data_processing import preprocess_data
    
    # Check if data path exists
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return None, None, None, None, None, None, None, None, None
    
    # Preprocess data
    X_train, X_test, y_train, y_test, dates_train, dates_test, scalers = preprocess_data(
        data_path, 
        model_params.get('NUM_LAGGED_FEATURES', 12), 
        model_params.get('TRAIN_TEST_SPLIT', 0.8),
        model_params.get('OUTPUT_VARIABLE', 'Discharge (m³/S)'),
        model_params.get('DYNAMIC_VARIABLES', None),
        model_params.get('CATEGORICAL_VARIABLES', None),
        model_params.get('NUMERIC_STATIC_VARIABLES', None)
    )
    
    # Check if preprocessing was successful
    if X_train is None or X_test is None or y_train is None or y_test is None or dates_train is None or dates_test is None or scalers is None:
        print("Error: Data preprocessing failed")
        return None, None, None, None, None, None, None, None, None
    
    # Store input shape in model parameters
    model_params['INPUT_SHAPE'] = X_train.shape[2]
    
    # Create model
    model = create_pinn_gru_model(model_params)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=model_params.get('LEARNING_RATE', 0.0001)),
        loss='mae',  # This is just a placeholder, the actual loss is defined in the model
        run_eagerly=True  # Required for custom loss function
    )
    
    # Define callbacks
    callbacks = []
    
    # Add learning rate scheduler if specified
    if model_params.get('USE_LR_SCHEDULER', True):
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=model_params.get('LR_FACTOR', 0.8),
            patience=model_params.get('LR_PATIENCE', 20),
            verbose=1,
            min_lr=1e-6
        )
        callbacks.append(lr_scheduler)
    
    # Add early stopping if specified
    if model_params.get('EARLY_STOPPING', True):
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=model_params.get('PATIENCE', 300),
            min_delta=model_params.get('MIN_DELTA', 0.001),
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
    
    # Add progress callback if provided
    if progress_callback:
        class ProgressCallback(Callback):
            def on_epoch_end(self, epoch, logs=None):
                if logs is None:
                    logs = {}
                progress_callback(epoch, logs)
        callbacks.append(ProgressCallback())
    
    # Train model
    try:
        history = model.fit(
            X_train, y_train,
            epochs=model_params.get('EPOCHS', 500),
            batch_size=model_params.get('BATCH_SIZE', 32),
            validation_data=(X_test, y_test),
            verbose=1,
            callbacks=callbacks
        )
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        return None, None, None, None, None, None, None, None, None
    
    return model, history, X_train, X_test, y_train, y_test, dates_train, dates_test, scalers
