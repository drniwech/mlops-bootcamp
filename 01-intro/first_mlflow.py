import mlflow
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mlflow.set_experiment("iris-rf-quickstart")

with mlflow.start_run(run_name="rf-nice"):
    X, y = load_iris(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_tr, y_tr)
    
    preds = model.predict(X_te)
    acc = accuracy_score(y_te, preds)
    
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    mlflow.log_metric("accuracy", acc)
    
    mlflow.sklearn.log_model(model, name="model")
    
    print(f"-> accuracy= {acc:.3f}")