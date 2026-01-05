from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import random

intervalos_regressao_logistica = {
    "C": lambda: random.uniform(0.001, 100),
    "solver": lambda: random.choice(["lbfgs", "liblinear"]),
    "max_iter": lambda: random.randint(100, 1000)
}

intervalos_arvore_de_descisao = {
    "max_depth": lambda: random.randint(2, 30),
    "min_samples_split": lambda: random.randint(2, 20),
    "min_samples_leaf": lambda: random.randint(1, 20),
    "criterion": lambda: random.choice(["gini", "entropy"])
}

intervalos_arvore_aleatoria = {
    "n_estimators": lambda: random.randint(50, 300),
    "max_depth": lambda: random.randint(3, 30),
    "min_samples_split": lambda: random.randint(2, 20),
    "min_samples_leaf": lambda: random.randint(1, 20),
    "max_features": lambda: random.uniform(0.1, 1.0)
}

intervalos_xgb = {
    "n_estimators": lambda: random.randint(50, 300),
    "max_depth": lambda: random.randint(3, 10),
    "learning_rate": lambda: random.uniform(0.01, 0.3),
    "subsample": lambda: random.uniform(0.5, 1.0),
    "colsample_bytree": lambda: random.uniform(0.5, 1.0)
}

modelos = {
    # "LogisticRegression": (LogisticRegression, intervalos_regressao_logistica),
    "DecisionTree": (DecisionTreeClassifier, intervalos_arvore_de_descisao)#,
    # "RandomForest": (RandomForestClassifier, intervalos_arvore_aleatoria),
    # "XGBoost": (xgb.XGBClassifier, intervalos_xgb)
}