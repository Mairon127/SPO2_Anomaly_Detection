import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
from hurst import compute_Hc
import matplotlib.dates as mdates

sns.set(style="whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

DEFAULT_HISTORY_FILE = 'history2.csv'
DEFAULT_EVALUATION_LOG_FILE = 'evaluation_log.csv'
DEFAULT_PREDICTIONS_FILE = 'predictions.csv'
MODELS = {
    'RidgeRegression': 'ridge_model.pkl',
    'RandomForest': 'random_forest_model.pkl',
    'SVR': 'svr_model.pkl'
}
SCALER_FILE = 'scaler_2.pkl'


def preprocess_data(file_path,
                    method='IQR',
                    threshold=1.5,
                    scale=False,
                    #
                    filters=None,
                    #
                    apply_filters=False,
                    apply_outlier_removal=True,
                    scaler=None):

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Plik {file_path} nie istnieje.")
        return None, None
    except pd.errors.EmptyDataError:
        print("Brak danych w pliku.")
        return None, None

    df = df.dropna(subset=['HR', 'SpO2'])

    df['HR_diff'] = df['HR'].diff().fillna(0)
    df['HR_rolling_mean'] = df['HR'].rolling(window=3).mean().fillna(df['HR'])

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if apply_outlier_removal and method:
        if method == 'Z-score':
            z_scores = np.abs(stats.zscore(df[numeric_cols]))
            df = df[(z_scores < threshold).all(axis=1)]
            print(f"Usunięto wartości odstające metodą Z-score. Pozostało {len(df)} rekordów.")
        elif method == 'IQR':
            Q1 = df[numeric_cols].quantile(0.25)
            Q3 = df[numeric_cols].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            condition = ~((df[numeric_cols] < lower_bound) | (df[numeric_cols] > upper_bound)).any(axis=1)
            df = df[condition]
            print(f"Usunięto wartości odstające metodą IQR. Pozostało {len(df)} rekordów.")
        elif method == 'IsolationForest':
            iso = IsolationForest(contamination=0.01, random_state=42)
            iso.fit(df[numeric_cols])
            y_pred = iso.predict(df[numeric_cols])
            df = df[y_pred == 1]
            print(f"Usunięto wartości odstające metodą IsolationForest. Pozostało {len(df)} rekordów.")
        elif method is None:
            print("Nie zastosowano usuwania wartości odstających.")
        else:
            print(f"Nieznana metoda usuwania wartości odstających: {method}.")

    scaler_fitted = scaler
    if scale:
        if scaler is None:
            scaler_fitted = StandardScaler()
            df[numeric_cols] = scaler_fitted.fit_transform(df[numeric_cols])
            joblib.dump(scaler_fitted, SCALER_FILE)
            print("Dane zostały skalowane i scaler został wytrenowany oraz zapisany.")
        else:
            df[numeric_cols] = scaler.transform(df[numeric_cols])
            print("Dane zostały skalowane przy użyciu załadowanego scalera.")

    return df, scaler_fitted


def train_models(training_file=DEFAULT_HISTORY_FILE,
                 preprocessing_method='IQR',
                 threshold=1.5,
                 scale=False,
                 #
                 filters=None):

    try:
        data, scaler = preprocess_data(
            file_path=training_file,
            method=preprocessing_method,
            threshold=threshold,
            scale=scale,
            filters=None,
            apply_filters=False,
            apply_outlier_removal=True
        )
        if data is None:
            print("Brak danych do trenowania modeli.")
            return None
    except Exception as e:
        print(f"Błąd podczas preprocessingu danych: {e}")
        return None

    if len(data) < 50:
        print("Za mało danych do trenowania modeli. Czekam na więcej danych...")
        return None

    data['HR_diff'] = data['HR'].diff().fillna(0)
    data['HR_rolling_mean'] = data['HR'].rolling(window=3).mean().fillna(data['HR'])
    data = data.dropna()

    X = data[['HR', 'HR_diff', 'HR_rolling_mean']]
    y = data['SpO2']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    y_baseline = [y_train.mean()] * len(y_val)
    baseline_mae = mean_absolute_error(y_val, y_baseline)
    print(f"Model bazowy (średnia wartość SpO2): MAE: {baseline_mae:.2f}")

    trained_models = {}

    ridge = Ridge()
    ridge_params = {'alpha': [0.1, 1.0, 10.0, 100.0, 200.0, 500.0]}
    ridge_grid = GridSearchCV(ridge, ridge_params, cv=5, scoring='r2', n_jobs=-1)
    ridge_grid.fit(X_train, y_train)
    best_ridge = ridge_grid.best_estimator_
    trained_models['RidgeRegression'] = best_ridge
    joblib.dump(best_ridge, MODELS['RidgeRegression'])
    print(f"RidgeRegression wytrenowany. Najlepsze alpha: {ridge_grid.best_params_['alpha']}")

    rf = RandomForestRegressor(random_state=42)
    rf_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='r2', n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_
    trained_models['RandomForest'] = best_rf
    joblib.dump(best_rf, MODELS['RandomForest'])
    print(f"RandomForest wytrenowany. Najlepsze parametry: {rf_grid.best_params_}")

    svr = SVR()
    svr_params = {
        'C': [0.1, 1, 10, 100, 200],
        'epsilon': [0.01, 0.1, 0.5, 1],
        'kernel': ['rbf']
    }
    svr_grid = GridSearchCV(svr, svr_params, cv=5, scoring='r2', n_jobs=-1)
    svr_grid.fit(X_train, y_train)
    best_svr = svr_grid.best_estimator_
    trained_models['SVR'] = best_svr
    joblib.dump(best_svr, MODELS['SVR'])
    print(f"SVR wytrenowany. Najlepsze parametry: {svr_grid.best_params_}")

    print("\nOcena modeli na zbiorze walidacyjnym:")
    evaluation_results = []
    for name, model in trained_models.items():
        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, y_pred)
        ev = explained_variance_score(y_val, y_pred)
        evaluation_results.append({
            'Model': name,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'Explained Variance': ev
        })
        print(f"{name}: MAE: {mae:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}, "
              f"R²: {r2:.2f}, Explained Variance: {ev:.2f}")

    evaluation_df = pd.DataFrame(evaluation_results)
    return trained_models, evaluation_df


def load_models():
    loaded_models = {}
    all_models_exist = True
    for name, file in MODELS.items():
        if os.path.exists(file):
            loaded_models[name] = joblib.load(file)
            print(f"Model {name} został załadowany z pliku {file}.")
        else:
            all_models_exist = False
            print(f"Model {name} nie został znaleziony ({file}).")
    if all_models_exist:
        print("Wszystkie modele zostały załadowane pomyślnie.")
        return loaded_models
    else:
        print("Nie wszystkie modele są dostępne. Trzeba je wytrenować.")
        return None


def make_predictions(prediction_file,
                     loaded_models,
                     prediction_data_file=DEFAULT_HISTORY_FILE,
                     scale=False):
    try:
        data, _ = preprocess_data(
            file_path=prediction_data_file,
            method=None,
            threshold=None,
            scale=scale,
            filters=None,
            apply_filters=False,
            apply_outlier_removal=False
        )
        if data is None:
            return None
    except Exception as e:
        print(f"Błąd podczas preprocessingu danych: {e}")
        return None

    if 'HR' in data.columns:
        data['HR_diff'] = data['HR'].diff().fillna(0)
        data['HR_rolling_mean'] = data['HR'].rolling(window=3).mean().fillna(data['HR'])
    else:
        print("Kolumna 'HR' nie została znaleziona. Nie można obliczyć cech.")
        return None

    X = data[['HR', 'HR_diff', 'HR_rolling_mean']]

    if scale:
        if os.path.exists(SCALER_FILE):
            scaler = joblib.load(SCALER_FILE)
            X = scaler.transform(X)
            print("Dane zostały skalowane przy użyciu załadowanego scalera.")
        else:
            print(f"Plik scalera {SCALER_FILE} nie istnieje. Skalowanie danych nie zostanie zastosowane.")

    predictions = {}
    for name, model in loaded_models.items():
        predictions[name] = model.predict(X)

    return pd.DataFrame(predictions)


def save_predictions(predictions_df, timestamps, predictions_file=DEFAULT_PREDICTIONS_FILE):
    if predictions_df is None or predictions_df.empty:
        print("Brak predykcji do zapisania.")
        return
    predictions_df_with_timestamp = predictions_df.copy()
    predictions_df_with_timestamp['Timestamp'] = timestamps
    cols = ['Timestamp'] + [col for col in predictions_df.columns]
    predictions_df_with_timestamp = predictions_df_with_timestamp[cols]
    predictions_df_with_timestamp.to_csv(
        predictions_file,
        mode='a',
        header=not os.path.exists(predictions_file),
        index=False
    )
    print(f"Predykcje zostały zapisane do pliku {predictions_file}.")


def draw_plot(predictions_df, actual_spo2, timestamps):
    if predictions_df is None or predictions_df.empty:
        print("Brak danych do narysowania wykresu.")
        return
    timestamps = pd.to_datetime(timestamps, errors='coerce')
    predictions_df = predictions_df.tail(50)
    actual_spo2 = actual_spo2.tail(50)
    timestamps = timestamps.tail(50)

    plt.figure(figsize=(14, 8))
    colors = sns.color_palette("tab10", n_colors=predictions_df.shape[1] + 2)

    plt.plot(
        timestamps,
        actual_spo2,
        label='Rzeczywiste SpO$_2$',
        color=colors[0],
        linestyle='-',
        linewidth=2
    )

    for idx, column in enumerate(predictions_df.columns, start=1):
        plt.plot(
            timestamps,
            predictions_df[column],
            label=f'Przewidywane SpO$_2$ ({column})',
            linestyle='--',
            linewidth=1.5,
            color=colors[idx]
        )

    plt.axhline(
        y=90,
        color=colors[-2],
        linestyle='--',
        label='Zagrożenie hipoksji',
        linewidth=1.5
    )

    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))
    plt.xticks(rotation=45)
    plt.title('Porównanie rzeczywistych i przewidywanych wartości SpO$_2$ ')
    plt.xlabel('Czas')
    plt.ylabel('SpO$_2$ (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def draw_individual_plots(predictions_df, actual_spo2, timestamps):
    if predictions_df is None or predictions_df.empty:
        print("Brak danych do narysowania indywidualnych wykresów.")
        return

    timestamps = pd.to_datetime(timestamps, errors='coerce')
    predictions_df = predictions_df.tail(50)
    actual_spo2 = actual_spo2.tail(50)
    timestamps = timestamps.tail(50)

    colors = sns.color_palette("tab10", n_colors=predictions_df.shape[1] + 2)

    for idx, column in enumerate(predictions_df.columns, start=1):
        plt.figure(figsize=(14, 6))
        plt.plot(
            timestamps,
            actual_spo2,
            label='Rzeczywiste SpO$_2$',
            color=colors[0],
            linestyle='-',
            linewidth=2
        )
        plt.plot(
            timestamps,
            predictions_df[column],
            label=f'Przewidywane SpO$_2$ ({column})',
            linestyle='--',
            linewidth=1.5,
            color=colors[idx]
        )
        plt.axhline(
            y=90,
            color=colors[-2],
            linestyle='--',
            label='Zagrożenie hipoksji',
            linewidth=1.5
        )
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))
        plt.xticks(rotation=45)
        plt.xlabel('Czas')
        plt.ylabel('SpO$_2$ (%)')
        plt.title(f'Model: {column} ')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()


def plot_model_parameters(results_df):
    if results_df is None or results_df.empty:
        print("Brak danych do narysowania wykresów parametrów modeli.")
        return

    sns.set(style="whitegrid")

    metrics_1 = ['MAE', 'MSE', 'RMSE']
    metrics_2 = ['R2', 'Explained Variance']

    melted_df_1 = results_df.melt(
        id_vars=['Model'],
        value_vars=metrics_1,
        var_name='Metryka',
        value_name='Wartość'
    )

    g1 = sns.catplot(
        data=melted_df_1,
        kind="bar",
        x="Model",
        y="Wartość",
        hue="Model",
        col="Metryka",
        palette="tab10",
        height=5,
        aspect=1
    )
    g1.fig.subplots_adjust(top=0.88)
    g1.fig.suptitle('Porównanie parametrów modeli — MAE, MSE, RMSE')

    melted_df_2 = results_df.melt(
        id_vars=['Model'],
        value_vars=metrics_2,
        var_name='Metryka',
        value_name='Wartość'
    )

    g2 = sns.catplot(
        data=melted_df_2,
        kind="bar",
        x="Model",
        y="Wartość",
        hue="Model",
        col="Metryka",
        palette="tab10",
        height=5,
        aspect=1
    )
    g2.fig.subplots_adjust(top=0.88)
    g2.fig.suptitle('Porównanie parametrów modeli — R2, Explained Variance')

    plt.show()


def analyze_hurst(actual_spo2, predictions_df):
    H_actual = compute_Hc(actual_spo2, kind='price', simplified=True)[0]
    print(f"Współczynnik Hurst dla rzeczywistych SpO2: {H_actual:.4f}")
    for column in predictions_df.columns:
        H_pred = compute_Hc(predictions_df[column], kind='price', simplified=True)[0]
        print(f"Współczynnik Hurst dla predykcji {column}: {H_pred:.4f}")


def evaluate_predictions(actual_spo2, predictions_df):
    results = []
    for column in predictions_df.columns:
        y_pred = predictions_df[column]
        mae = mean_absolute_error(actual_spo2, y_pred)
        mse = mean_squared_error(actual_spo2, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual_spo2, y_pred)
        ev = explained_variance_score(actual_spo2, y_pred)
        results.append({
            'Model': column,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'Explained Variance': ev
        })
    return pd.DataFrame(results)


def save_evaluation_log(results_df, evaluation_log_file=DEFAULT_EVALUATION_LOG_FILE):
    if results_df is None or results_df.empty:
        print("Brak wyników do zapisania w logu ewaluacji.")
        return
    if os.path.exists(evaluation_log_file):
        results_df.to_csv(evaluation_log_file, mode='a', header=False, index=False)
    else:
        results_df.to_csv(evaluation_log_file, mode='w', header=True, index=False)
    print(f"Wyniki ewaluacji zostały zapisane do pliku {evaluation_log_file}.")


def main(training_file=DEFAULT_HISTORY_FILE, prediction_data_file=DEFAULT_HISTORY_FILE):
    loaded_models = load_models()
    if loaded_models is None:
        print("Rozpoczynanie trenowania modeli (bez filtrowania)...")
        trained_models, evaluation_df = train_models(
            training_file=training_file,
            preprocessing_method='IQR',
            threshold=1.5,
            scale=False,
            filters=None
        )
        if trained_models is None:
            print("Trenowanie modeli nie powiodło się.")
            return
        loaded_models = trained_models
    else:
        print("Modele zostały załadowane. Ewaluacja na nowych danych...")

    predictions_df = make_predictions(
        prediction_file=DEFAULT_PREDICTIONS_FILE,
        loaded_models=loaded_models,
        prediction_data_file=prediction_data_file,
        scale=False
    )
    if predictions_df is None:
        print("Brak predykcji do kontynuowania.")
        return

    try:
        data = pd.read_csv(prediction_data_file)
    except Exception as e:
        print(f"Błąd podczas odczytu pliku {prediction_data_file}: {e}")
        return

    if 'Timestamp' not in data.columns or 'SpO2' not in data.columns:
        print("Brak wymaganych kolumn 'Timestamp' lub 'SpO2' w danych.")
        return

    timestamps = data['Timestamp']
    actual_spo2 = data['SpO2']

    save_predictions(predictions_df, timestamps)

    draw_plot(predictions_df, actual_spo2, timestamps)
    draw_individual_plots(predictions_df, actual_spo2, timestamps)

    analyze_hurst(actual_spo2, predictions_df)

    results_df = evaluate_predictions(actual_spo2, predictions_df)
    print("\nWyniki ewaluacji modeli:")
    print(results_df)

    plot_model_parameters(results_df)

    save_evaluation_log(results_df)


if __name__ == "__main__":
    main(
        training_file='SPO2_train.csv',
        prediction_data_file='SPO2.csv'
    )
