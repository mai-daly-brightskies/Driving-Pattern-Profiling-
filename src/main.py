import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
import mlflow
import mlflow.sklearn
import optuna

def load_data():
    """
    Loads and preprocesses the training and test datasets.

    Returns:
        X_train (np.ndarray): Scaled features for the training set.
        y_train (np.ndarray): Encoded labels for the training set.
        X_test (np.ndarray): Scaled features for the test set.
        y_test (np.ndarray): Encoded labels for the test set.
    """
    df_train = pd.read_csv('df_train_preprocessed.csv')
    df_test = pd.read_csv('df_test_preprocessed.csv')
    X_train = df_train.iloc[:, :-1]
    y_train = df_train.iloc[:, -1]    
    X_test = df_test.iloc[:, :-1]
    y_test = df_test.iloc[:, -1]
    exclude_column = 'PathOrder'
    feature_columns = [col for col in df_train.columns if col not in ['Class', exclude_column]]
    preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), feature_columns),
        ('passthrough', 'passthrough', [exclude_column])
    ],
    remainder='passthrough'
)
    X_train = preprocessor.fit_transform(df_train.drop(columns=['Class']))
    X_test = preprocessor.transform(df_test.drop(columns=['Class']))

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    return X_train, y_train, X_test, y_test

def optimize_hyperparameters(model, search_space, X_train, y_train, X_test, y_test, n_trials=200):
    """
    Optimizes the hyperparameters of a given model using Optuna.

    Args:
        model (estimator): The machine learning model to optimize.
        search_space (dict): The hyperparameter search space.
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test labels.
        n_trials (int): Number of optimization trials.

    Returns:
        study (optuna.Study): The study object containing the optimization results.
        best_model (estimator): The model with the best found hyperparameters.
    """
    def objective(trial):
        params = {}
        for key, value in search_space.items():
            if isinstance(value, list):
                params[key] = trial.suggest_categorical(key, value)
            elif isinstance(value, tuple) and len(value) == 2:
                if isinstance(value[0], int) and isinstance(value[1], int):
                    params[key] = trial.suggest_int(key, value[0], value[1])
                else:
                    params[key] = trial.suggest_float(key, value[0], value[1])
            else:
                raise ValueError("Invalid search space format for key: {}".format(key))
        
        trial_model = clone(model).set_params(**params)
        trial_model.fit(X_train, y_train)
        y_pred = trial_model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='macro')
        accuracy = accuracy_score(y_test, y_pred)
        trial.set_user_attr('accuracy', accuracy)
        trial.set_user_attr('f1', f1)
        
        return f1
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    best_model = clone(model).set_params(**best_params)
    best_model.fit(X_train, y_train)
    
    return study, best_model

def get_best_model(X_train, y_train, X_test, y_test, model, search_space):
    """
    Gets the best model by optimizing hyperparameters using Optuna.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test labels.
        model (estimator): The machine learning model to optimize.
        search_space (dict): The hyperparameter search space.

    Returns:
        study (optuna.Study): The study object containing the optimization results.
        best_model (estimator): The model with the best found hyperparameters.
    """
    study, best_model = optimize_hyperparameters(model, search_space, X_train, y_train, X_test, y_test)
    return study, best_model

def random_forest_classifier(X_train, y_train, X_test, y_test):
    """
    Optimizes a Random Forest classifier and returns the study and best model.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test labels.

    Returns:
        study (optuna.Study): The study object containing the optimization results.
        best_model (RandomForestClassifier): The best Random Forest model.
    """
    model = RandomForestClassifier()
    search_space = {
        'n_estimators': (10, 300),
        'max_depth': (1, 30),
        'min_samples_split': (2, 15),
        'min_samples_leaf': (1, 15),
    }
    study, best_model = get_best_model(X_train, y_train, X_test, y_test, model, search_space)
    return study, best_model

def multiLayerPerceptron(X_train, y_train, X_test, y_test):
    """
    Optimizes a Multi-Layer Perceptron classifier and returns the study and best model.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test labels.

    Returns:
        study (optuna.Study): The study object containing the optimization results.
        best_model (MLPClassifier): The best MLP model.
    """
    model = MLPClassifier()
    search_space = {
        'hidden_layer_sizes': [(10,), (20,), (30,), (40,), (50,), (60,), (70,), (80,), (90,), (100,), (110,), (120,), (130,), (140,), (150,), (160,), (170,), (180,), (190,), (200,)],
        'activation': ['logistic', 'tanh', 'relu'],
        'solver': ['adam'],
        'alpha': (0.0001, 0.1),
        'learning_rate': ['constant', 'invscaling', 'adaptive']
    }
    study, best_model = get_best_model(X_train, y_train, X_test, y_test, model, search_space)
    return study, best_model

def kmeans(X_train, y_train, X_test, y_test):
    """
    Fits a KMeans model and evaluates its performance.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test labels.

    Returns:
        model (KMeans): The fitted KMeans model.
        f1 (float): The F1 score of the model.
        accuracy (float): The accuracy of the model.
    """
    model = KMeans()
    search_space = {
        'n_clusters': 5,
        'init': ['k-means++', 'random'],
        'n_init': (1, 100),
        'max_iter': (1, 1000),
        'tol': (1e-5, 1e-1),
        'precompute_distances': [True, False],
        'verbose': [0],
        'random_state': [None]
    }
    model.fit(X_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    accuracy = accuracy_score(y_test, y_pred)
    return model, f1, accuracy

def log_study(study, run_name):
    """
    Logs the Optuna study results to MLflow.

    Args:
        study (optuna.Study): The study object containing the optimization results.
        run_name (str): The name of the MLflow run.
    """
    logged_params = set()
    for trial in study.trials:
        new_params = {k: v for k, v in trial.params.items() if k not in logged_params}
        if new_params:
            mlflow.log_params(new_params)
            logged_params.update(new_params.keys())
        mlflow.log_metric('f1', trial.user_attrs['f1'])
        mlflow.log_metric('accuracy', trial.user_attrs['accuracy'])
    
    mlflow.log_params({k: v for k, v in study.best_params.items() if k not in logged_params})
    mlflow.log_metric('best_f1', study.best_value)

def log_model(model, name):
    """
    Logs the model to MLflow.

    Args:
        model (estimator): The machine learning model to log.
        name (str): The name of the model.

    Returns:
        model (estimator): The logged model.
    """
    mlflow.sklearn.log_model(model, name)
    return model

def save_model(model, name):
    """
    Saves the model to a file using MLflow.

    Args:
        model (estimator): The machine learning model to save.
        name (str): The name of the saved model file.

    Returns:
        model (estimator): The saved model.
    """
    mlflow.sklearn.save_model(model, name)
    return model

def main():
    """
    Main function to run the hyperparameter optimization and logging.

    Executes the following steps:
    1. Sets up MLflow experiment.
    2. Loads data.
    3. Runs Random Forest optimization and logging.
    4. Runs MLP optimization and logging.
    5. Runs KMeans fitting and logging.
    """
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
    experiment_name = 'Hyperparameter Optimization'
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if (experiment is None):
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    X_train, y_train, X_test, y_test = load_data()

    with mlflow.start_run(experiment_id=experiment_id, run_name='RandomForest'):
        random_forest_study, random_forest_model = random_forest_classifier(X_train, y_train, X_test, y_test)
        save_model(random_forest_model, 'random_forest_model')
        log_study(random_forest_study, 'random_forest_study')
        log_model(random_forest_model, 'random_forest_model')
        mlflow.end_run()
    
    with mlflow.start_run(experiment_id=experiment_id, run_name='MLP'):
        mlp_study, mlp_model = multiLayerPerceptron(X_train, y_train, X_test, y_test)
        save_model(mlp_model, 'mlp_model')
        log_study(mlp_study, 'mlp_study')
        log_model(mlp_model, 'mlp_model')
        mlflow.end_run()
    
    with mlflow.start_run(experiment_id=experiment_id, run_name='KMeans'):
        model, f1, accuracy = kmeans(X_train, y_train, X_test, y_test)
        save_model(model, 'kmeans_model')
        mlflow.log_metric('f1', f1)
        mlflow.log_metric('accuracy', accuracy)
        mlflow.end_run()

if __name__ == "__main__":
    main()
# mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
