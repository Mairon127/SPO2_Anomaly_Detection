import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, silhouette_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.src.callbacks import EarlyStopping
from keras.src.saving import load_model
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.losses import MeanSquaredError

processed_data_path = 'Processed_Saturation.csv'
history_file = 'history.csv'
model_file = 'autoencoder_model_main.h5'
kmeans_model_path = 'kmeans_model.pkl'
dbscan_model_path = 'dbscan_model.pkl'
isolation_forest_model_path = 'isolation_forest_model.pkl'
one_class_svm_model_path = 'one_class_svm_model.pkl'
scaler_path = 'scaler.pkl'
classified_data_path = 'Clustering_Anomaly_Results.csv'
combined_classified_data_path = 'Combined_Anomaly_Results.csv'

def evaluate_model_performance(true_labels, predictions, model_name):
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)
    print(f"\n--- {model_name} ---")
    print(f"Precyzja: {precision:.4f}")
    print(f"Czułość (Recall): {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("\nRaport klasyfikacji:")
    print(classification_report(true_labels, predictions, target_names=['Normalne', 'Anomalie'], zero_division=0))

def display_model_parameters(model, model_name):
    print(f"\n--- Parametry modelu {model_name} ---")
    if hasattr(model, 'get_params'):
        params = model.get_params()
        for param, value in params.items():
            print(f"{param}: {value}")
    elif model_name == 'Standard Deviation':
        print(f"Mnożnik odchyleń standardowych (std_dev_multiplier): {model}")
    else:
        print("Brak dostępnych parametrów do wyświetlenia.")

print("=== Autoenkoder Keras ===")
processed_data = pd.read_csv(processed_data_path).dropna(subset=['HR', 'SpO2'])

train_data, val_data = train_test_split(processed_data, test_size=0.2, random_state=42)

scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(train_data[['HR', 'SpO2']])
val_data_scaled = scaler.transform(val_data[['HR', 'SpO2']])

joblib.dump(scaler, scaler_path)
print(f"Scaler zapisany do pliku: {scaler_path}")

if os.path.exists(model_file):
    print("Wczytywanie wytrenowanego modelu autoenkodera...")
    autoencoder = load_model(model_file)
    print("\n--- Parametry modelu Autoencoder ---")
    autoencoder.summary()

else:
    print("Trenowanie nowego modelu autoenkodera...")
    autoencoder = Sequential([
        Input(shape=(train_data_scaled.shape[1],)),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dense(train_data_scaled.shape[1], activation='sigmoid')
    ])

    autoencoder.compile(optimizer='adam', loss=MeanSquaredError())

    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    history = autoencoder.fit(
        train_data_scaled, train_data_scaled,
        epochs=300,
        batch_size=32,
        shuffle=True,
        validation_data=(val_data_scaled, val_data_scaled),
        callbacks=[early_stopping],
        verbose=1
    )

    autoencoder.save(model_file)
    print(f"Model autoenkodera zapisano do pliku: {model_file}")
    print("\n--- Parametry modelu Autoencoder ---")
    autoencoder.summary()

print("\n=== Testowanie modeli na history.csv ===")
test_data = pd.read_csv(history_file).dropna(subset=['HR', 'SpO2'])

if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
    print("Scaler został załadowany.")
else:
    raise FileNotFoundError(f"Scaler nie został znaleziony pod ścieżką: {scaler_path}")

test_data_scaled = scaler.transform(test_data[['HR', 'SpO2']])

print("\n=== Wyliczanie fizycznych anomalii w danych testowych ===")
physical_anomalies_test = (
    (test_data['HR'] > 100) |
    (test_data['HR'] < 68) |
    (test_data['SpO2'] < 90)
).astype(int)
num_physical_anomalies_test = physical_anomalies_test.sum()
print(f"Liczba fizycznych anomalii (HR > 100 lub HR < 68 lub SpO2 < 90): {num_physical_anomalies_test}")
test_data['Physical_Anomaly'] = physical_anomalies_test

print("\n=== Wykrywanie anomalii przy użyciu Autoenkodera ===")
reconstructed_test = autoencoder.predict(test_data_scaled)
loss_test = np.mean(np.square(test_data_scaled - reconstructed_test), axis=1)

percentile = 80
threshold = np.percentile(loss_test, percentile)
print(f"Próg anomalii (percentyl {percentile}): {threshold}")

anomalies_autoencoder = loss_test > threshold
test_data['Autoencoder_Anomaly'] = anomalies_autoencoder.astype(int)

plt.figure(figsize=(12, 6))
plt.hist(loss_test[~anomalies_autoencoder], bins=50, alpha=0.6, label='Normalne dane')
plt.hist(loss_test[anomalies_autoencoder], bins=50, alpha=0.6, label='Anomalie')
plt.axvline(threshold, color='red', linestyle='--', label=f'Próg anomalii ({percentile} percentyl)')
plt.title('Błąd rekonstrukcji autoenkodera na danych testowych')
plt.xlabel('Błąd rekonstrukcji')
plt.ylabel('Liczba próbek')
plt.legend()
plt.show()

anomalous_data_autoencoder_test = test_data.iloc[anomalies_autoencoder]
print("Anomalie wykryte przez autoenkoder:\n", anomalous_data_autoencoder_test)

print("\n=== Metody scikit-learn ===")

print("\n--- K-Means Clustering ---")
if os.path.exists(kmeans_model_path):
    kmeans = joblib.load(kmeans_model_path)
    print("Model K-Means został załadowany.")
    display_model_parameters(kmeans, "K-Means")

else:
    print("Trenowanie modelu K-Means...")
    processed_X_scaled = train_data_scaled

    silhouette_scores = []
    cluster_range = range(2, 10)
    for n_clusters in cluster_range:
        kmeans_tmp = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans_tmp.fit_predict(processed_X_scaled)
        score = silhouette_score(processed_X_scaled, labels)
        silhouette_scores.append(score)
        print(f"Silhouette Score dla {n_clusters} klastrów: {score:.4f}")

    optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
    print(f"Optymalna liczba klastrów dla K-Means: {optimal_clusters}")

    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    kmeans.fit(processed_X_scaled)
    joblib.dump(kmeans, kmeans_model_path)
    print("Model K-Means został wytrenowany i zapisany.")
    display_model_parameters(kmeans, "K-Means")

test_data_scaled_for_kmeans = test_data_scaled

kmeans_predictions = kmeans.predict(test_data_scaled_for_kmeans)

distances = kmeans.transform(test_data_scaled_for_kmeans)
min_distances = distances.min(axis=1)

training_distances = kmeans.transform(train_data_scaled)
min_train_distances = training_distances.min(axis=1)
threshold_kmeans = np.percentile(min_train_distances, 95)
print(f"Próg K-Means Anomaly Distance (95 percentyl z treningu): {threshold_kmeans}")

kmeans_anomalies = (min_distances > threshold_kmeans).astype(int)
test_data['KMeans_Anomaly'] = kmeans_anomalies
num_kmeans_anomalies = kmeans_anomalies.sum()
print(f"Liczba anomalii wykrytych przez K-Means: {num_kmeans_anomalies}")

print("\n--- DBSCAN Clustering ---")
if os.path.exists(dbscan_model_path):
    dbscan = joblib.load(dbscan_model_path)
    print("Model DBSCAN został załadowany.")
    display_model_parameters(dbscan, "DBSCAN")

else:
    print("Trenowanie modelu DBSCAN...")
    best_eps = 0.5
    best_min_samples = 5
    best_f1 = 0

    eps_values = [0.3, 0.4, 0.5, 0.6, 0.7]
    min_samples_values = [3, 5, 7, 9]

    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan_tmp = DBSCAN(eps=eps, min_samples=min_samples)
            dbscan_tmp.fit(train_data_scaled)
            labels = dbscan_tmp.labels_

            val_labels = (
                (val_data['HR'] > 100) |
                (val_data['HR'] < 68) |
                (val_data['SpO2'] < 90)
            ).astype(int)

            val_labels_scaled = scaler.transform(val_data[['HR', 'SpO2']])
            dbscan_val_labels = dbscan_tmp.fit_predict(val_labels_scaled)

            dbscan_val_predictions = (dbscan_val_labels == -1).astype(int)

            f1 = f1_score(val_labels, dbscan_val_predictions, zero_division=0)
            print(f"DBSCAN eps={eps}, min_samples={min_samples}, F1-score={f1:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                best_eps = eps
                best_min_samples = min_samples

    print(f"Optymalne parametry DBSCAN: eps={best_eps}, min_samples={best_min_samples}, F1-score={best_f1:.4f}")

    dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
    dbscan.fit(train_data_scaled)
    joblib.dump(dbscan, dbscan_model_path)
    print("Model DBSCAN został wytrenowany i zapisany.")
    display_model_parameters(dbscan, "DBSCAN")

train_clusters = set(dbscan.labels_)
train_clusters.discard(-1)

if not train_clusters:
    dbscan_predictions = np.ones(len(test_data), dtype=int)
    print("DBSCAN nie znalazł żadnych klastrów w danych treningowych. Wszystkie punkty testowe są oznaczone jako anomalie.")
else:
    from sklearn.metrics import pairwise_distances_argmin_min

    cluster_centers = []
    for cluster in train_clusters:
        cluster_points = train_data_scaled[dbscan.labels_ == cluster]
        if len(cluster_points) > 0:
            cluster_center = cluster_points.mean(axis=0)
            cluster_centers.append(cluster_center)
    cluster_centers = np.array(cluster_centers)

    closest_clusters, distances_to_closest = pairwise_distances_argmin_min(test_data_scaled_for_kmeans, cluster_centers)

    val_data_scaled = scaler.transform(val_data[['HR', 'SpO2']])
    distances_val = pairwise_distances_argmin_min(val_data_scaled, cluster_centers)[1]
    threshold_dbscan = np.percentile(distances_val, 95)
    print(f"Próg DBSCAN Anomaly Distance (95 percentyl z walidacji): {threshold_dbscan}")

    dbscan_predictions = (distances_to_closest > threshold_dbscan).astype(int)
    test_data['DBSCAN_Outlier'] = dbscan_predictions
    num_dbscan_anomalies = dbscan_predictions.sum()
    print(f"Liczba anomalii wykrytych przez DBSCAN: {num_dbscan_anomalies}")

print("\n--- Local Outlier Factor (LOF) ---")
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True)
lof.fit(train_data_scaled)
display_model_parameters(lof, "Local Outlier Factor (LOF)")

lof_predictions = lof.predict(test_data_scaled)
lof_predictions = (lof_predictions == -1).astype(int)
test_data['LOF_Outlier'] = lof_predictions
num_lof_anomalies = lof_predictions.sum()
print(f"Liczba anomalii wykrytych przez LOF: {num_lof_anomalies}")

print("\n--- Isolation Forest ---")
if os.path.exists(isolation_forest_model_path):
    isolation_forest = joblib.load(isolation_forest_model_path)
    print("Model Isolation Forest został załadowany.")
    display_model_parameters(isolation_forest, "Isolation Forest")

else:
    print("Trenowanie modelu Isolation Forest...")
    isolation_forest = IsolationForest(contamination=0.1, random_state=42)
    isolation_forest.fit(train_data_scaled)
    joblib.dump(isolation_forest, isolation_forest_model_path)
    print("Model Isolation Forest został wytrenowany i zapisany.")
    display_model_parameters(isolation_forest, "Isolation Forest")

if_outliers = isolation_forest.predict(test_data_scaled_for_kmeans)
isolation_forest_predictions = (if_outliers == -1).astype(int)
test_data['IsolationForest_Outlier'] = isolation_forest_predictions
num_if_anomalies = isolation_forest_predictions.sum()
print(f"Liczba anomalii wykrytych przez Isolation Forest: {num_if_anomalies}")

print("\n--- One-Class SVM ---")
if os.path.exists(one_class_svm_model_path):
    one_class_svm = joblib.load(one_class_svm_model_path)
    print("Model One-Class SVM został załadowany.")
    display_model_parameters(one_class_svm, "One-Class SVM")

else:
    print("Trenowanie modelu One-Class SVM...")
    one_class_svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
    one_class_svm.fit(train_data_scaled)
    joblib.dump(one_class_svm, one_class_svm_model_path)
    print("Model One-Class SVM został wytrenowany i zapisany.")
    display_model_parameters(one_class_svm, "One-Class SVM")

svm_outliers = one_class_svm.predict(test_data_scaled_for_kmeans)
one_class_svm_predictions = (svm_outliers == -1).astype(int)
test_data['OneClassSVM_Outlier'] = one_class_svm_predictions
num_svm_anomalies = one_class_svm_predictions.sum()
print(f"Liczba anomalii wykrytych przez One-Class SVM: {num_svm_anomalies}")

print("\n--- Anomaly Detection using Standard Deviation ---")
mean_hr_train = train_data['HR'].mean()
std_hr_train = train_data['HR'].std()
mean_spo2_train = train_data['SpO2'].mean()
std_spo2_train = train_data['SpO2'].std()

print("Dostosowywanie mnożnika odchyleń standardowych na podstawie zbioru walidacyjnego...")
best_multiplier = 2
best_f1 = 0

for multiplier in [1.5, 2, 2.5, 3]:
    std_dev_anomalies_val = (
        (val_data['HR'] > mean_hr_train + multiplier * std_hr_train) |
        (val_data['HR'] < mean_hr_train - multiplier * std_hr_train) |
        (val_data['SpO2'] > mean_spo2_train + multiplier * std_spo2_train) |
        (val_data['SpO2'] < mean_spo2_train - multiplier * std_spo2_train)
    ).astype(int)
    val_labels = (
        (val_data['HR'] > 100) |
        (val_data['HR'] < 68) |
        (val_data['SpO2'] < 90)
    ).astype(int)
    f1 = f1_score(val_labels, std_dev_anomalies_val, zero_division=0)
    print(f"F1-score dla mnożnika {multiplier}: {f1:.4f}")
    if f1 > best_f1:
        best_f1 = f1
        best_multiplier = multiplier

print(f"Optymalny mnożnik odchyleń standardowych: {best_multiplier} z F1-score={best_f1:.4f}")

std_dev_multiplier = best_multiplier

std_dev_anomalies = (
    (test_data['HR'] > mean_hr_train + std_dev_multiplier * std_hr_train) |
    (test_data['HR'] < mean_hr_train - std_dev_multiplier * std_hr_train) |
    (test_data['SpO2'] > mean_spo2_train + std_dev_multiplier * std_hr_train) |
    (test_data['SpO2'] < mean_spo2_train - std_dev_multiplier * std_hr_train)
)
std_dev_predictions = std_dev_anomalies.astype(int)
test_data['StdDev_Outlier'] = std_dev_predictions
num_stddev_anomalies = std_dev_predictions.sum()
print(f"Liczba anomalii wykrytych przez Standard Deviation (multiplier={std_dev_multiplier}): {num_stddev_anomalies}")

print("\n--- Dodawanie wyników do DataFrame ---")

def plot_anomalies(X, predictions, title, cmap='viridis'):
    plt.figure(figsize=(10, 6))
    colors = {0: 'blue', 1: 'red'}
    for label, color in colors.items():
        subset = X[predictions == label]
        label_name = 'Normalne' if label == 0 else 'Anomalia'
        plt.scatter(subset['HR'], subset['SpO2'], c=color, label=label_name, alpha=0.7)

    plt.title(title)
    plt.xlabel('HR')
    plt.ylabel('SpO2')
    plt.legend()
    plt.show()

print("\n--- Wizualizacja K-Means Clustering ---")
plt.figure(figsize=(10, 6))
plt.scatter(test_data['HR'], test_data['SpO2'], c=kmeans_predictions, cmap='viridis', alpha=0.7)
plt.title('K-Means Clustering (HR vs. SpO2)')
plt.xlabel('HR')
plt.ylabel('SpO2')
plt.colorbar(label='Cluster')
plt.show()

print("\n--- Wizualizacja DBSCAN Anomalies ---")
plot_anomalies(test_data, dbscan_predictions, 'DBSCAN Outliers (HR vs. SpO2)', cmap='coolwarm')

print("\n--- Wizualizacja Local Outlier Factor (LOF) ---")
plot_anomalies(test_data, lof_predictions, 'Local Outlier Factor (HR vs. SpO2)', cmap='coolwarm')

print("\n--- Wizualizacja Isolation Forest Anomalies ---")
plot_anomalies(test_data, isolation_forest_predictions, 'Isolation Forest (HR vs. SpO2)', cmap='autumn')

print("\n--- Wizualizacja One-Class SVM Anomalies ---")
plot_anomalies(test_data, one_class_svm_predictions, 'One-Class SVM Anomalies (HR vs. SpO2)', cmap='plasma')

print("\n--- Wizualizacja Anomalies based on Standard Deviation ---")
plot_anomalies(test_data, std_dev_predictions, 'Standard Deviation Anomalies (HR vs. SpO2)', cmap='binary')

print("\n--- Zapisywanie wyników do pliku CSV ---")
test_data.to_csv(classified_data_path, index=False)
print(f"Wyniki metod scikit-learn oraz autoenkodera zostały zapisane do pliku: {classified_data_path}")

print("\n=== Porównanie jakości modeli ===")

models = {
    'Autoencoder': test_data['Autoencoder_Anomaly'].fillna(0).astype(int),
    'KMeans': test_data['KMeans_Anomaly'].astype(int),
    'DBSCAN': test_data['DBSCAN_Outlier'].astype(int),
    'LOF': test_data['LOF_Outlier'].astype(int),
    'IsolationForest': test_data['IsolationForest_Outlier'].astype(int),
    'OneClassSVM': test_data['OneClassSVM_Outlier'].astype(int),
    'StdDev': test_data['StdDev_Outlier'].astype(int)
}

true_labels = test_data['Physical_Anomaly']

metrics_list = []

for model_name, predictions in models.items():
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)
    auc_roc = roc_auc_score(true_labels, predictions)
    metrics_list.append({
        'Model': model_name,
        'Precyzja': precision,
        'Czułość': recall,
        'F1-score': f1,
        'AUC-ROC': auc_roc
    })
    print(f"\n--- {model_name} ---")
    print(classification_report(true_labels, predictions, target_names=['Normalne', 'Anomalie'], zero_division=0))

metrics_df = pd.DataFrame(metrics_list)

print("\n=== Tabela porównująca metryki modeli ===")
print(metrics_df)

metrics_df_melted = metrics_df.melt(id_vars='Model', var_name='Metryka', value_name='Wartość')

plt.figure(figsize=(12, 6))
sns.barplot(x='Model', y='Wartość', hue='Metryka', data=metrics_df_melted)
plt.title('Porównanie jakości modeli wykrywania anomalii')
plt.ylabel('Wartość metryki')
plt.xlabel('Model')
plt.legend(title='Metryka')
plt.show()

plt.figure(figsize=(10, 6))

for model_name, predictions in models.items():
    fpr, tpr, _ = roc_curve(true_labels, predictions)
    roc_auc = roc_auc_score(true_labels, predictions)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('Krzywa ROC dla różnych modeli')
plt.xlabel('Odsetek wyników fałszywie dodatnich')
plt.ylabel('Odsetek wyników prawdziwie dodatnich')
plt.legend()
plt.show()

for model_name, predictions in models.items():
    false_positives = test_data[(predictions == 1) & (true_labels == 0)]
    false_negatives = test_data[(predictions == 0) & (true_labels == 1)]
    print(f"\n=== {model_name} - False Positives (FP): ===")
    print(false_positives[['HR', 'SpO2', 'Physical_Anomaly']])
    print(f"\n=== {model_name} - False Negatives (FN): ===")
    print(false_negatives[['HR', 'SpO2', 'Physical_Anomaly']])

print("\n--- Zapisywanie wszystkich wyników do osobnego pliku CSV ---")
test_data.to_csv(combined_classified_data_path, index=False)
print(f"Połączone wyniki anomalii zostały zapisane do pliku: {combined_classified_data_path}")
