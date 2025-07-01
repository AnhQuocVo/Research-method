import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("=== NEURAL ECONOMETRIC ENSEMBLE FOR ICT-IC IMPACT ANALYSIS ===")
print("Loading and preprocessing data...")

# Load data
df = pd.read_csv("rawdata_1.csv")
print(f"Original data shape: {df.shape}")

# Filter years between 2000-2022
df = df[df["Time"].between(2000, 2022)]
print(f"Data shape after time filtering: {df.shape}")

# Fill with meanmean
df = df.fillna(df.mean(numeric_only=True))

# Display basic info
print("\n=== DATA OVERVIEW ===")
print(df.head())
print(f"\nColumns: {df.columns.tolist()}")
print(f"Countries: {df['Country Name'].nunique()}")
print(f"Years: {df['Time'].min()} - {df['Time'].max()}")

# ===============================
# 1. CREATE COMPOSITE INDICES
# ===============================

print("\n=== CREATING COMPOSITE INDICES ===")

# ICT Index Components (normalize to 0-1 scale)
scaler_ict = MinMaxScaler()
ict_components = ["sec_srv", "mob_sub", "ter_enr", "inet_usr"]
df[ict_components] = scaler_ict.fit_transform(df[ict_components])

# Create ICT Index using PCA
pca_ict = PCA(n_components=1)
df["ICT_Index"] = pca_ict.fit_transform(df[ict_components]).flatten()

# Normalize ICT Index to 0-1
df["ICT_Index"] = MinMaxScaler().fit_transform(df[["ICT_Index"]])

print(f"ICT Index explained variance: {pca_ict.explained_variance_ratio_[0]:.3f}")

# IC Index Components
scaler_ic = MinMaxScaler()
ic_components = ["edu_exp", "rnd_exp", "sci_art", "hci"]
df[ic_components] = scaler_ic.fit_transform(df[ic_components])

# Create IC Index using PCA
pca_ic = PCA(n_components=1)
df["IC_Index"] = pca_ic.fit_transform(df[ic_components]).flatten()

# Normalize IC Index to 0-1
df["IC_Index"] = MinMaxScaler().fit_transform(df[["IC_Index"]])

print(f"IC Index explained variance: {pca_ic.explained_variance_ratio_[0]:.3f}")

# Development Level Index (for clustering)
development_components = ["gdp", "urb_area", "trade"]
scaler_dev = MinMaxScaler()
df[development_components] = scaler_dev.fit_transform(df[development_components])

pca_dev = PCA(n_components=1)
df["Development_Index"] = pca_dev.fit_transform(df[development_components]).flatten()
df["Development_Index"] = MinMaxScaler().fit_transform(df[["Development_Index"]])

print(
    f"Development Index explained variance: {pca_dev.explained_variance_ratio_[0]:.3f}"
)

# ===============================
# 2. COUNTRY CLUSTERING
# ===============================

print("\n=== COUNTRY CLUSTERING ===")

# Prepare clustering features
clustering_features = ["ICT_Index", "IC_Index", "Development_Index"]

# Get average values per country for clustering
country_avg = df.groupby("Country Name")[clustering_features].mean().reset_index()

# Perform K-means clustering
n_clusters = 33
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
country_avg["Cluster"] = kmeans.fit_predict(country_avg[clustering_features])

# Map clusters back to original dataframe
cluster_mapping = dict(zip(country_avg["Country Name"], country_avg["Cluster"]))
df["Cluster"] = df["Country Name"].map(cluster_mapping)

# Display cluster characteristics
print("\n=== CLUSTER CHARACTERISTICS ===")
cluster_stats = df.groupby("Cluster")[clustering_features + ["gdp"]].mean()
print(cluster_stats)

# Visualize clusters
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
scatter = plt.scatter(
    df["ICT_Index"], df["IC_Index"], c=df["Cluster"], cmap="viridis", alpha=0.6
)
plt.xlabel("ICT Index")
plt.ylabel("IC Index")
plt.title("Countries Clustered by ICT-IC Patterns")
plt.colorbar(scatter)

plt.subplot(1, 3, 2)
scatter = plt.scatter(
    df["ICT_Index"], df["Development_Index"], c=df["Cluster"], cmap="viridis", alpha=0.6
)
plt.xlabel("ICT Index")
plt.ylabel("Development Index")
plt.title("ICT vs Development by Cluster")
plt.colorbar(scatter)

plt.subplot(1, 3, 3)
scatter = plt.scatter(
    df["IC_Index"], df["Development_Index"], c=df["Cluster"], cmap="viridis", alpha=0.6
)
plt.xlabel("IC Index")
plt.ylabel("Development Index")
plt.title("IC vs Development by Cluster")
plt.colorbar(scatter)

plt.tight_layout()
plt.show()

# ===============================
# 3. NEURAL NETWORK ARCHITECTURE
# ===============================

print("\n=== BUILDING NEURAL NETWORK MODELS ===")

# Define features and targets
feature_columns = ["ICT_Index", "IC_Index", "pop", "infl", "urb_area", "trade"]
target_columns = ["gdp", "hte", "ict_exp", "fdi"]

# Standardize features
scaler_features = StandardScaler()
df[feature_columns] = scaler_features.fit_transform(df[feature_columns])

# Standardize targets (for better neural network training)
scaler_targets = {}
for target in target_columns:
    scaler_targets[target] = StandardScaler()
    df[f"{target}_scaled"] = scaler_targets[target].fit_transform(df[[target]])


def create_cluster_neural_model(input_dim=6, target_name="gdp"):
    """Create neural network for specific cluster"""
    model = keras.Sequential(
        [
            layers.Dense(
                128,
                activation="relu",
                input_shape=(input_dim,),
                kernel_regularizer=keras.regularizers.l2(0.001),
            ),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            layers.Dense(
                64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)
            ),
            layers.Dropout(0.2),
            layers.BatchNormalization(),
            layers.Dense(
                32, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)
            ),
            layers.Dropout(0.1),
            layers.Dense(1, activation="linear", name=f"{target_name}_prediction"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="huber",  # Robust to outliers
        metrics=["mae", "mse"],
    )

    return model


def create_economics_informed_nn(input_dim=6, target_name="gdp"):
    """Create economics-informed neural network"""

    # Input layer
    inputs = keras.Input(shape=(input_dim,))

    # ICT pathway
    ict_features = inputs[:, :2]  # ICT_Index, IC_Index
    ict_branch = layers.Dense(32, activation="relu", name="ict_processing")(
        ict_features
    )
    ict_branch = layers.Dropout(0.2)(ict_branch)

    # Economic controls pathway
    control_features = inputs[:, 2:]  # pop, infl, urb_area, trade
    control_branch = layers.Dense(32, activation="relu", name="controls_processing")(
        control_features
    )
    control_branch = layers.Dropout(0.2)(control_branch)

    # Interaction layer (ICT × Controls effects)
    ict_expanded = layers.RepeatVector(4)(
        ict_branch[:, :1]
    )  # Repeat ICT for interaction
    ict_expanded = layers.Flatten()(ict_expanded)
    interaction = layers.Multiply()([ict_expanded, control_features])
    interaction_processed = layers.Dense(
        16, activation="relu", name="interaction_layer"
    )(interaction)

    # Combine all pathways
    combined = layers.Concatenate()([ict_branch, control_branch, interaction_processed])
    combined = layers.Dense(64, activation="relu")(combined)
    combined = layers.Dropout(0.3)(combined)
    combined = layers.Dense(32, activation="relu")(combined)
    combined = layers.Dropout(0.2)(combined)

    # Output layer
    outputs = layers.Dense(1, activation="linear", name=f"{target_name}_prediction")(
        combined
    )

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="huber",
        metrics=["mae", "mse"],
    )

    return model


# ===============================
# 4. NEURAL ENSEMBLE TRAINING
# ===============================


class NeuralEconometricEnsemble:
    def __init__(self, n_clusters=5, n_models_per_cluster=3):
        self.n_clusters = n_clusters
        self.n_models_per_cluster = n_models_per_cluster
        self.cluster_models = {}
        self.model_weights = {}
        self.scalers = {}

    def fit(self, X, y, clusters, target_name="gdp", epochs=100, batch_size=32):
        """Train ensemble of neural networks"""

        print(f"\nTraining Neural Ensemble for {target_name}...")

        for cluster_id in range(self.n_clusters):
            print(f"Training models for Cluster {cluster_id}...")

            # Get cluster data
            cluster_mask = clusters == cluster_id
            X_cluster = X[cluster_mask]
            y_cluster = y[cluster_mask]

            if len(X_cluster) < 10:  # Skip clusters with too few samples
                print(
                    f"Cluster {cluster_id} has only {len(X_cluster)} samples, skipping..."
                )
                continue

            # Train multiple models for this cluster
            cluster_models = []
            cluster_scores = []

            for model_idx in range(self.n_models_per_cluster):
                # Create model
                if model_idx == 0:
                    model = create_economics_informed_nn(X.shape[1], target_name)
                else:
                    model = create_cluster_neural_model(X.shape[1], target_name)

                # Split cluster data for training
                X_train, X_val, y_train, y_val = train_test_split(
                    X_cluster, y_cluster, test_size=0.2, random_state=42 + model_idx
                )

                # Early stopping
                early_stopping = keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=20, restore_best_weights=True
                )

                # Train model
                model.fit(
                    X_train,
                    y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stopping],
                    verbose=0,
                )

                # Evaluate model
                val_pred = model.predict(X_val, verbose=0)
                val_score = r2_score(y_val, val_pred)

                cluster_models.append(model)
                cluster_scores.append(val_score)

                print(f"  Model {model_idx}: R² = {val_score:.4f}")

            self.cluster_models[cluster_id] = cluster_models
            self.model_weights[cluster_id] = np.array(cluster_scores) / np.sum(
                cluster_scores
            )

    def predict(self, X, clusters):
        """Make ensemble predictions"""
        predictions = np.zeros(len(X))

        for cluster_id in range(self.n_clusters):
            if cluster_id not in self.cluster_models:
                continue

            cluster_mask = clusters == cluster_id
            if not np.any(cluster_mask):
                continue

            X_cluster = X[cluster_mask]

            # Get predictions from all models in cluster
            cluster_preds = []
            for model in self.cluster_models[cluster_id]:
                pred = model.predict(X_cluster, verbose=0).flatten()
                cluster_preds.append(pred)

            # Weighted ensemble prediction
            cluster_preds = np.array(cluster_preds)
            weights = self.model_weights[cluster_id].reshape(-1, 1)
            ensemble_pred = np.sum(cluster_preds * weights, axis=0)

            predictions[cluster_mask] = ensemble_pred

        return predictions


# ===============================
# 5. TRAIN ENSEMBLE FOR EACH TARGET
# ===============================

print("\n=== TRAINING NEURAL ENSEMBLE MODELS ===")

# Prepare data
X = df[feature_columns].values
clusters = df["Cluster"].values

# Store results
ensemble_results = {}
ensemble_models = {}

for target in target_columns:
    print(f"\n{'=' * 50}")
    print(f"Training ensemble for: {target.upper()}")
    print(f"{'=' * 50}")

    # Get target values (scaled)
    y = df[f"{target}_scaled"].values

    # Split data
    X_train, X_test, y_train, y_test, clusters_train, clusters_test = train_test_split(
        X, y, clusters, test_size=0.2, random_state=42, stratify=clusters
    )

    # Create and train ensemble
    ensemble = NeuralEconometricEnsemble(n_clusters=n_clusters, n_models_per_cluster=3)
    ensemble.fit(X_train, y_train, clusters_train, target_name=target, epochs=150)

    # Make predictions
    y_pred_train = ensemble.predict(X_train, clusters_train)
    y_pred_test = ensemble.predict(X_test, clusters_test)

    # Convert back to original scale
    y_train_orig = (
        scaler_targets[target].inverse_transform(y_train.reshape(-1, 1)).flatten()
    )
    y_test_orig = (
        scaler_targets[target].inverse_transform(y_test.reshape(-1, 1)).flatten()
    )
    y_pred_train_orig = (
        scaler_targets[target].inverse_transform(y_pred_train.reshape(-1, 1)).flatten()
    )
    y_pred_test_orig = (
        scaler_targets[target].inverse_transform(y_pred_test.reshape(-1, 1)).flatten()
    )

    # Calculate metrics
    train_r2 = r2_score(y_train_orig, y_pred_train_orig)
    test_r2 = r2_score(y_test_orig, y_pred_test_orig)
    train_mae = mean_absolute_error(y_train_orig, y_pred_train_orig)
    test_mae = mean_absolute_error(y_test_orig, y_pred_test_orig)

    ensemble_results[target] = {
        "train_r2": train_r2,
        "test_r2": test_r2,
        "train_mae": train_mae,
        "test_mae": test_mae,
        "y_test": y_test_orig,
        "y_pred": y_pred_test_orig,
    }

    ensemble_models[target] = ensemble

    print(f"\nResults for {target.upper()}:")
    print(f"Train R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Train MAE: {train_mae:.4f}")
    print(f"Test MAE: {test_mae:.4f}")

# ===============================
# 6. VISUALIZATION AND ANALYSIS
# ===============================

print("\n=== GENERATING VISUALIZATIONS ===")

# Plot results for all targets
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

for idx, target in enumerate(target_columns):
    ax = axes[idx]

    y_test = ensemble_results[target]["y_test"]
    y_pred = ensemble_results[target]["y_pred"]
    r2 = ensemble_results[target]["test_r2"]

    ax.scatter(y_test, y_pred, alpha=0.6, color="blue")
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    ax.set_xlabel(f"Actual {target.upper()}")
    ax.set_ylabel(f"Predicted {target.upper()}")
    ax.set_title(f"{target.upper()} Prediction (R² = {r2:.4f})")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Performance summary
print("\n=== NEURAL ENSEMBLE PERFORMANCE SUMMARY ===")
performance_df = pd.DataFrame(ensemble_results).T
performance_df = performance_df[["train_r2", "test_r2", "train_mae", "test_mae"]]
performance_df.columns = ["Train R²", "Test R²", "Train MAE", "Test MAE"]
print(performance_df.round(4))

# Feature importance analysis (using one model from each target)
print("\n=== FEATURE IMPORTANCE ANALYSIS ===")


def calculate_feature_importance(model, X_sample, feature_names):
    """Calculate permutation-based feature importance"""
    baseline_pred = model.predict(X_sample, verbose=0)
    baseline_mse = np.mean(baseline_pred**2)

    importances = []
    for i, feature_name in enumerate(feature_names):
        X_permuted = X_sample.copy()
        X_permuted[:, i] = np.random.permutation(X_permuted[:, i])

        permuted_pred = model.predict(X_permuted, verbose=0)
        permuted_mse = np.mean(permuted_pred**2)

        importance = (permuted_mse - baseline_mse) / baseline_mse
        importances.append(importance)

    return np.array(importances)


# Calculate average feature importance across all models and targets
all_importances = []

for target in target_columns:
    ensemble = ensemble_models[target]
    target_importances = []

    for cluster_id in ensemble.cluster_models.keys():
        for model in ensemble.cluster_models[cluster_id]:
            # Sample data for importance calculation
            sample_size = min(1000, len(X))
            sample_indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[sample_indices]

            importance = calculate_feature_importance(model, X_sample, feature_columns)
            target_importances.append(importance)

    if target_importances:
        avg_importance = np.mean(target_importances, axis=0)
        all_importances.append(avg_importance)

if all_importances:
    # Average importance across all targets
    overall_importance = np.mean(all_importances, axis=0)

    # Create feature importance plot
    plt.figure(figsize=(10, 6))
    feature_importance_df = pd.DataFrame(
        {"Feature": feature_columns, "Importance": overall_importance}
    ).sort_values("Importance", ascending=True)

    plt.barh(feature_importance_df["Feature"], feature_importance_df["Importance"])
    plt.xlabel("Feature Importance (Permutation-based)")
    plt.title("Average Feature Importance Across All Models and Targets")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("Feature Importance Rankings:")
    for i, (feature, importance) in enumerate(
        zip(feature_importance_df["Feature"], feature_importance_df["Importance"])
    ):
        print(f"{i + 1}. {feature}: {importance:.4f}")

print("\n=== ANALYSIS COMPLETE ===")
print("Neural Econometric Ensemble successfully trained!")
print(
    f"Total models trained: {sum(len(models) for models in ensemble_models['gdp'].cluster_models.values()) * len(target_columns)}"
)
print("The ensemble approach provides:")
print("1. Country-specific modeling through clustering")
print("2. Multiple neural architectures for robustness")
print("3. Economics-informed network design")
print("4. Ensemble predictions for better accuracy")
