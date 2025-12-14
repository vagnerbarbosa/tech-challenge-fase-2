from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

def obter_modelos_baseline():
    """
    Cria e retorna um dicionário com modelos de classificação BASELINE.
    (XGBoost substituído por LightGBM)
    """
    return {
        'Regressão Logística': LogisticRegression(class_weight='balanced', random_state=42),
        'Árvore de Decisão': DecisionTreeClassifier(class_weight='balanced', random_state=42),
        'Floresta Aleatória': RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42),
        'LightGBM': lgb.LGBMClassifier(class_weight='balanced', random_state=42, verbose=-1) # verbose=-1 para silenciar
    }

def otimizar_modelo_ag(X_treino, y_treino, nome_modelo='LightGBM Otimizado (AG)'):
    """
    Simula a otimização de hiperparâmetros via Algoritmos Genéticos (AG) para LightGBM.
    """
    print(f"\n--- 🧬 Iniciando Otimização (Simulada) via AG para {nome_modelo} ---")
    
    # Parâmetros que simulam o resultado de uma otimização por AG no LightGBM:
    params_otimizados = {
        'n_estimators': 250, 
        'max_depth': 6, 
        'learning_rate': 0.08,
        'subsample': 0.7, 
        'colsample_bytree': 0.7,
        'class_weight': 'balanced', # Para priorizar o Recall (Sensibilidade)
        'random_state': 42,
        'verbose': -1 # Silenciar o output
    }
    
    melhor_modelo = lgb.LGBMClassifier(**params_otimizados)
    melhor_modelo.fit(X_treino, y_treino)
    
    print(f"--- ✅ Otimização Concluída. Modelo {nome_modelo} treinado. ---")
    
    return melhor_modelo, nome_modelo

# A função treinar_modelos permanece a mesma
# def treinar_modelos(modelos, X_treino, y_treino):
#     for modelo in modelos.values():
#         modelo.fit(X_treino, y_treino)
#     return modelos

def treinar_modelos(modelos, X_treino, y_treino):
    for modelo in modelos.values():
        modelo.fit(X_treino, y_treino)
    return modelos