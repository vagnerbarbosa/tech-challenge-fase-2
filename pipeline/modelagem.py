from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

def obter_modelos():
    """
    Cria e retorna um dicionário com modelos de classificação configurados para melhor recall.
    """
    return {
        'Regressão Logística': LogisticRegression(class_weight='balanced', random_state=42),
        'Árvore de Decisão': DecisionTreeClassifier(class_weight='balanced', random_state=42),
        'Floresta Aleatória': RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(scale_pos_weight=1, eval_metric='logloss', random_state=42)
    }

def treinar_modelos(modelos, X_treino, y_treino):
    """
    Treina N modelos e os retorna treinados.
    """
    for modelo in modelos.values():
        modelo.fit(X_treino, y_treino)
    return modelos