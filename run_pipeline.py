
from datasets import load_dataset
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from huggingface_hub import HfApi
import os

def run_pipeline():
    dataset = load_dataset("Vishnu-J-S/super-kart")
    train_df = dataset['train'].to_pandas()
    test_df = dataset['test'].to_pandas()

    X_train = train_df.drop('Product_Store_Sales_Total', axis=1)
    y_train = train_df['Product_Store_Sales_Total']
    X_test = test_df.drop('Product_Store_Sales_Total', axis=1)
    y_test = test_df['Product_Store_Sales_Total']

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }

    grid_search = GridSearchCV(
        estimator=XGBRegressor(objective='reg:squarederror', random_state=42),
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    best_xgb = grid_search.best_estimator_
    y_pred = best_xgb.predict(X_test)
    
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    print("Best XGB Params:", grid_search.best_params_)
    print("Test RMSE:", rmse)
    print("R² Score:", r2)

    joblib.dump(best_xgb, "estimator_model.joblib")
    
    print("Model and encoders saved locally")

    HF_TOKEN = os.getenv("HF_TOKEN")
    api = HfApi(token=HF_TOKEN)
    repo_id = "Vishnu-J-S/estimator-model"

    #api.create_repo(repo_id, repo_type="model", exist_ok=True)

    api.upload_file(
        path_or_fileobj="estimator_model.joblib",
        path_in_repo="estimator_model.joblib",
        repo_id=repo_id,
        repo_type="model"
    )


if __name__ == "__main__":
    run_pipeline()
