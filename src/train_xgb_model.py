import os
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def train_xgb_model(file_path, output_path):
    # Get the directory of the current script
    script_dir = os.getcwd()

    # Construct the file path dynamically for input dataset
    data_folder = os.path.join(script_dir, ".", "data")
    model_folder = os.path.join(script_dir, ".", "models")
    file_path = os.path.join(data_folder, "preprocessed_data.npz")

    # Ensure the data folder exists
    os.makedirs(data_folder, exist_ok=True)
    os.makedirs(model_folder, exist_ok=True)

    # 📌 Load preprocessed data
    data = np.load(file_path, allow_pickle=True)
    X_train, X_test, y_train, y_test = data["X_train"], data["X_test"], data["y_train"], data["y_test"]

    # 🚀 Define the XGBoost Regressor (Fixed for all versions)
    xgb_regressor = xgb.XGBRegressor(
        objective="reg:squarederror",  # Standard for regression
        n_estimators=300,              # Number of boosting rounds
        learning_rate=0.05,            # Step size shrinkage
        max_depth=6,                   # Depth of each tree
        subsample=0.8,                 # Sample ratio per tree
        colsample_bytree=0.8,          # Feature selection ratio
        reg_lambda=1.0,                # L2 regularization
        reg_alpha=0.1,                 # L1 regularization
        random_state=42
    )

    # 🏋️ Train the Model (Handle different XGBoost versions)
    if hasattr(xgb_regressor, "fit"):
        xgb_regressor.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)
    else:
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        params = xgb_regressor.get_params()
        params["eval_metric"] = "rmse"

        xgb_regressor = xgb.train(params, dtrain, num_boost_round=500, evals=[(dtest, "test")])

    # 🎯 Make Predictions
    y_pred = xgb_regressor.predict(X_test)

    # 📊 Evaluate Performance
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"✅ Model Training Complete!")
    print(f"📉 RMSE: {rmse:.4f}")
    print(f"📈 R² Score: {r2:.4f}")

    # 💾 Save the trained model
    output_path = os.path.join(model_folder, "xgb_energy_model.json")
    joblib.dump(xgb_regressor, output_path)

    print(f"✅ Model saved to {output_path}.")
    
if __name__ == "__main__":
    train_xgb_model("data/preprocessed_data.npz", "data/xgb_energy_model.json")
