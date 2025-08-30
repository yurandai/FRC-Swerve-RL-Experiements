import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')

# Configuration
BATCH_SIZE = 64
EPOCHS = 200
VALIDATION_SPLIT = 0.2
RANDOM_STATE = 42

# Target column indices (0-indexed) and names - in the correct order
TARGET_INDICES = [21, 22, 23]  # vx, vy, omega
TARGET_NAMES = ['omega (rad/s)','vx (m/s)', 'vy (m/s)']

def load_and_preprocess_data(folder_path):
    """Load all CSV files and create input-output pairs using previous timestep"""
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    print(f"Found {len(all_files)} CSV files")
    
    all_inputs = []
    all_targets = []
    file_names = []
    skipped_files = 0
    
    for file in all_files:
        file_path = os.path.join(folder_path, file)
        try:
            # Load data, skipping the first row (header)
            df = pd.read_csv(file_path, skiprows=1)
            
            # Check if we have enough data (at least 2 rows)
            if len(df) < 2:
                print(f"Skipping {file}: insufficient data ({len(df)} rows)")
                skipped_files += 1
                continue
            
            # Check if all target indices exist in this file
            if df.shape[1] <= max(TARGET_INDICES):
                print(f"Skipping {file}: not enough columns ({df.shape[1]} columns)")
                skipped_files += 1
                continue
                
            # Use all columns as input features
            all_data = df.values.astype(np.float32)
            
            # Create input-output pairs
            for i in range(1, len(df)):
                # Get current and previous timesteps
                prev_data = all_data[i-1]
                curr_data = all_data[i]
                
                # Calculate time difference (assuming timestamp is in column 0)
                time_diff = curr_data[0] - prev_data[0]
                
                # Input: all columns from previous timestep except timestamp (column 0)
                # plus the time difference
                input_features = np.concatenate([
                    prev_data[1:],  # All columns except timestamp
                    [time_diff]     # Time difference
                ])
                
                # Output: target columns from current timestep
                target = curr_data[TARGET_INDICES]
                
                all_inputs.append(input_features)
                all_targets.append(target)
                file_names.append(file)
                
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            skipped_files += 1
    
    print(f"Skipped {skipped_files} files due to errors or insufficient data")
    print(f"Successfully processed {len(all_inputs)} samples from {len(set(file_names))} files")
    
    if len(all_inputs) == 0:
        print("No valid samples found. Please check your data.")
        return np.array([]), np.array([]), [], None, None
    
    return np.array(all_inputs), np.array(all_targets), file_names

def create_train_val_split(inputs, targets, file_names):
    """Split data into training and validation sets by file to prevent data leakage"""
    if len(inputs) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    unique_files = list(set(file_names))
    
    if len(unique_files) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    # Split files to ensure no data leakage
    train_files, val_files = train_test_split(
        unique_files, test_size=VALIDATION_SPLIT, random_state=RANDOM_STATE
    )
    
    train_indices = [i for i, f in enumerate(file_names) if f in train_files]
    val_indices = [i for i, f in enumerate(file_names) if f in val_files]
    
    X_train = inputs[train_indices]
    y_train = targets[train_indices]
    X_val = inputs[val_indices]
    y_val = targets[val_indices]
    
    return X_train, X_val, y_train, y_val

def build_ffn_model(input_dim, output_dim):
    """Build a feedforward neural network model"""
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim, 
              kernel_initializer='he_normal', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu', 
              kernel_initializer='he_normal', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(32, activation='relu', 
              kernel_initializer='he_normal', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(output_dim)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='mse', 
                  metrics=['mae'])
    
    return model

def plot_training_history(history, save_path):
    """Plot training history and save to file"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_model(model, X_val, y_val, target_scaler, save_path):
    """Evaluate model and create detailed visualizations for each target"""
    if len(X_val) == 0:
        print("No validation data to evaluate")
        return
    
    # Make predictions
    y_pred = model.predict(X_val)
    
    # Inverse transform predictions and targets if scaler is provided
    if target_scaler is not None:
        y_pred_rescaled = target_scaler.inverse_transform(y_pred)
        y_val_rescaled = target_scaler.inverse_transform(y_val)
    else:
        y_pred_rescaled = y_pred
        y_val_rescaled = y_val
    
    # Calculate overall metrics
    mse = mean_squared_error(y_val_rescaled, y_pred_rescaled)
    mae = mean_absolute_error(y_val_rescaled, y_pred_rescaled)
    r2 = r2_score(y_val_rescaled, y_pred_rescaled)
    
    print(f"Overall Validation MSE: {mse:.6f}")
    print(f"Overall Validation MAE: {mae:.6f}")
    print(f"Overall Validation R²: {r2:.6f}")
    
    # Create evaluation for each target
    for i, target_name in enumerate(TARGET_NAMES):
        print(f"\n--- Evaluation for {target_name} ---")
        target_mse = mean_squared_error(y_val_rescaled[:, i], y_pred_rescaled[:, i])
        target_mae = mean_absolute_error(y_val_rescaled[:, i], y_pred_rescaled[:, i])
        target_r2 = r2_score(y_val_rescaled[:, i], y_pred_rescaled[:, i])
        
        print(f"MSE: {target_mse:.6f}")
        print(f"MAE: {target_mae:.6f}")
        print(f"R²: {target_r2:.6f}")
        
        # Create comprehensive visualization for this target
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Model Evaluation for {target_name}', fontsize=16)
        
        # 1. Scatter plot of true vs predicted values
        axes[0, 0].scatter(y_val_rescaled[:, i], y_pred_rescaled[:, i], alpha=0.5)
        min_val = min(y_val_rescaled[:, i].min(), y_pred_rescaled[:, i].min())
        max_val = max(y_val_rescaled[:, i].max(), y_pred_rescaled[:, i].max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        axes[0, 0].set_xlabel('True Values')
        axes[0, 0].set_ylabel('Predictions')
        axes[0, 0].set_title(f'True vs Predicted\nMSE: {target_mse:.6f}, MAE: {target_mae:.6f}')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Error distribution
        errors = y_pred_rescaled[:, i] - y_val_rescaled[:, i]
        axes[0, 1].hist(errors, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(x=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Prediction Error')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Error Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Time series comparison (first 100 samples)
        n_samples = min(100, len(y_val_rescaled))
        axes[1, 0].plot(range(n_samples), y_val_rescaled[:n_samples, i], label='True', linewidth=2)
        axes[1, 0].plot(range(n_samples), y_pred_rescaled[:n_samples, i], label='Predicted', linewidth=2, alpha=0.8)
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].set_title('Time Series Comparison (First 100 Samples)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Residual plot
        axes[1, 1].scatter(y_pred_rescaled[:, i], errors, alpha=0.5)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        
        # Add trend line to identify systematic bias
        z = np.polyfit(y_pred_rescaled[:, i], errors, 1)
        p = np.poly1d(z)
        axes[1, 1].plot(y_pred_rescaled[:, i], p(y_pred_rescaled[:, i]), "b--", alpha=0.8, 
                       label=f'Trend: {z[0]:.4f}x + {z[1]:.4f}')
        
        axes[1, 1].set_xlabel('Predicted Values')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Residual Plot')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'evaluation_{target_name.replace(" ", "_").replace("/", "_")}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create a combined metrics table
    metrics_data = []
    for i, target_name in enumerate(TARGET_NAMES):
        target_mse = mean_squared_error(y_val_rescaled[:, i], y_pred_rescaled[:, i])
        target_mae = mean_absolute_error(y_val_rescaled[:, i], y_pred_rescaled[:, i])
        target_r2 = r2_score(y_val_rescaled[:, i], y_pred_rescaled[:, i])
        metrics_data.append([target_name, target_mse, target_mae, target_r2])
    
    metrics_df = pd.DataFrame(metrics_data, columns=['Target', 'MSE', 'MAE', 'R²'])
    metrics_df.to_csv(os.path.join(save_path, 'metrics_summary.csv'), index=False)
    print("\nMetrics summary saved to metrics_summary.csv")
    
    # Save detailed error analysis
    error_analysis = pd.DataFrame({
        'True_vx': y_val_rescaled[:, 0],
        'Pred_vx': y_pred_rescaled[:, 0],
        'Error_vx': y_pred_rescaled[:, 0] - y_val_rescaled[:, 0],
        'True_vy': y_val_rescaled[:, 1],
        'Pred_vy': y_pred_rescaled[:, 1],
        'Error_vy': y_pred_rescaled[:, 1] - y_val_rescaled[:, 1],
        'True_omega': y_val_rescaled[:, 2],
        'Pred_omega': y_pred_rescaled[:, 2],
        'Error_omega': y_pred_rescaled[:, 2] - y_val_rescaled[:, 2],
    })
    error_analysis.to_csv(os.path.join(save_path, 'error_analysis.csv'), index=False)
    print("Detailed error analysis saved to error_analysis.csv")

def main():
    # Set random seeds for reproducibility
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)
    
    # Create output directory
    output_dir = "ffn_training_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    folder_path = os.path.expanduser("~/Desktop/CLEANED")
    inputs, targets, file_names = load_and_preprocess_data(folder_path)
    
    if len(inputs) == 0:
        print("No data to train on. Exiting.")
        return
    
    print(f"Created {len(inputs)} samples")
    print(f"Input shape: {inputs.shape}, Target shape: {targets.shape}")
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = create_train_val_split(inputs, targets, file_names)
    
    if len(X_train) == 0:
        print("No training data after split. Exiting.")
        return
        
    print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")
    
    # Scale targets
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train)
    y_val_scaled = target_scaler.transform(y_val)
    
    # Build and compile model
    print("Building feedforward neural network model...")
    model = build_ffn_model(X_train.shape[1], y_train.shape[1])
    model.summary()
    
    # Define callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.CSVLogger(os.path.join(output_dir, 'training_log.csv'))
    ]
    
    # Train model
    print("Training model...")
    history = model.fit(
        X_train, y_train_scaled,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val_scaled),
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history, output_dir)
    
    # Evaluate model
    print("Evaluating model...")
    evaluate_model(model, X_val, y_val_scaled, target_scaler, output_dir)
    
    # Save model and data
    model.save(os.path.join(output_dir, 'trained_model.keras'))
    
    with open(os.path.join(output_dir, 'training_data.pkl'), 'wb') as f:
        pickle.dump({
            'X_train': X_train,
            'X_val': X_val,
            'y_train': y_train,
            'y_val': y_val,
            'y_train_scaled': y_train_scaled,
            'y_val_scaled': y_val_scaled,
            'file_names': file_names,
            'target_indices': TARGET_INDICES,
            'target_names': TARGET_NAMES,
            'target_scaler': target_scaler
        }, f)
    
    # Generate a summary report
    with open(os.path.join(output_dir, 'training_summary.txt'), 'w') as f:
        f.write("Feedforward Neural Network Training Summary\n")
        f.write("==========================================\n\n")
        f.write(f"Input Features: {X_train.shape[1]}\n")
        f.write(f"Output Features: {y_train.shape[1]}\n")
        f.write(f"Training Samples: {X_train.shape[0]}\n")
        f.write(f"Validation Samples: {X_val.shape[0]}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Final Training Loss: {history.history['loss'][-1]:.6f}\n")
        f.write(f"Final Validation Loss: {history.history['val_loss'][-1]:.6f}\n")
        f.write(f"Best Validation Loss: {min(history.history['val_loss']):.6f}\n")
        f.write(f"Epochs Trained: {len(history.history['loss'])}\n")
    
    print(f"Training completed. Results saved to {output_dir}")

if __name__ == "__main__":
    main()