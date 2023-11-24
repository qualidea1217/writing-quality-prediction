import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import optuna
from tqdm import tqdm


def preprocess_data(data):
    # Using pd.factorize for categorical features as low memory alternative instead of one hot encoder
    cat_cols = ['activity', 'down_event', 'up_event', 'text_change']
    for col in tqdm(cat_cols):
        data[col], _ = pd.factorize(data[col])
    # Handling missing values
    data.fillna(0, inplace=True)
    return data


def kfold_cv(X, y, model, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmse_scores = []
    for train_index, val_index in kf.split(X):
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_val_fold)
        rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
        rmse_scores.append(rmse)
    return np.mean(rmse_scores)


def objective_lgbm(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
    }
    model = LGBMRegressor(**params, random_state=42, device="gpu")
    rmse = kfold_cv(X_train, y_train, model)
    return rmse


def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
    }
    model = XGBRegressor(**params, random_state=42, tree_method='hist', device="cuda")
    rmse = kfold_cv(X_train, y_train, model)
    return rmse


if __name__ == "__main__":
    train_logs = pd.read_csv('data/train_logs.csv', low_memory=True)
    test_logs = pd.read_csv('data/test_logs.csv', low_memory=True)
    train_scores = pd.read_csv('data/train_scores.csv', low_memory=True)

    train_data = pd.merge(train_logs, train_scores, on='id')
    features = train_data.drop(['id', "event_id", 'score'], axis=1)
    target = train_data['score']

    features = preprocess_data(features)
    X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, random_state=42)

    study_lgbm = optuna.create_study(direction='minimize')
    study_lgbm.optimize(objective_lgbm, n_trials=10)

    study_xgb = optuna.create_study(direction='minimize')
    study_xgb.optimize(objective_xgb, n_trials=10)

    best_params_lgbm = study_lgbm.best_params
    best_model_lgbm = LGBMRegressor(**best_params_lgbm, random_state=42, device="gpu")

    best_params_xgb = study_xgb.best_params
    best_model_xgb = XGBRegressor(**best_params_xgb, random_state=42, tree_method='gpu_hist')

    ensemble = VotingRegressor(estimators=[('lgbm', best_model_lgbm), ('xgb', best_model_xgb)], n_jobs=-1)
    # ensemble = VotingRegressor(estimators=[('xgb', best_model_xgb)], n_jobs=-1)
    ensemble.fit(X_train, y_train)

    val_pred = ensemble.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    print(f'Validation RMSE with Voting Ensemble: {val_rmse}')

    model1 = LGBMRegressor(random_state=42, device="gpu")
    model1.fit(X_train, y_train)
    feature_importance_lgbm = model1.feature_importances_
    feature_names = X_train.columns
    feature_importance_lgbm_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance_lgbm})
    feature_importance_lgbm_df = feature_importance_lgbm_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_lgbm_df)
    plt.title('Feature Importance for LightGBM (in the ensemble)')
    plt.show()

    test_features = preprocess_data(test_logs.drop('id', axis=1))
    test_predictions = ensemble.predict(test_features)
    print(test_predictions)

    submission = pd.DataFrame({'id': test_logs['id'], 'score': test_predictions})
    submission_grouped = submission.groupby('id')['score'].mean().reset_index()
    submission_grouped.to_csv('/kaggle/working/submission.csv', index=False)
    print(submission_grouped)