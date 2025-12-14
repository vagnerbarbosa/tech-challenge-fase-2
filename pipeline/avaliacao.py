from sklearn.metrics import recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pandas as pd

def avaliar_modelos(modelos, X_teste, y_teste, nomes_features, limiar=0.3):
    """
    Avalia múltiplos modelos de classificação, exibe métricas e matriz de confusão,
    e retorna o modelo com o melhor Recall.
    """
    melhor_recall = -1
    melhor_f1 = -1
    nome_melhor_modelo = None
    melhor_modelo = None

    for nome, modelo in modelos.items():

        if hasattr(modelo, "predict_proba"):
            y_proba = modelo.predict_proba(X_teste)[:, 1]
            y_pred = (y_proba >= limiar).astype(int)
        else:
            y_pred = modelo.predict(X_teste)
            
        recall_atual = recall_score(y_teste, y_pred)
        f1_atual = f1_score(y_teste, y_pred)
        
        # Lógica para rastrear o melhor modelo (baseado no Recall)
        if recall_atual > melhor_recall:
            melhor_recall = recall_atual
            melhor_f1 = f1_atual 
            nome_melhor_modelo = nome
            melhor_modelo = modelo

        print(f"\n--- {nome} ---")
        print(f"Recall (Sensibilidade): {recall_atual:.2f}")
        print(f"F1-score: {f1_atual:.2f}")
        print("\nRelatório de classificação detalhado:")
        print(classification_report(y_teste, y_pred))
        print("Matriz de Confusão (linhas = real, colunas = predito):")
        sns.heatmap(confusion_matrix(y_teste, y_pred), annot=True, fmt='d', cmap='Blues')
        plt.title(f'Matriz de Confusão - {nome} (Limiar={limiar})')
        plt.xlabel('Predito')
        plt.ylabel('Real')
        plt.show()

    print("==================================================")
    print(f"🏆 Melhor Modelo Encontrado ({nome_melhor_modelo})")
    print(f"   Critério de Seleção: Maior Recall")
    print(f"   Recall (Sensibilidade): {melhor_recall:.2f}")
    print(f"   F1-score: {melhor_f1:.2f}")
    print("==================================================")
    
    return nome_melhor_modelo, melhor_modelo

def interpretar_melhor_modelo(melhor_modelo, X_teste_df, nomes_features):
    """
    Realiza a interpretação do melhor modelo utilizando Importância das Features e SHAP.
    """
    
    print("\n\n=== 🔎 INTERPRETAÇÃO DO MELHOR MODELO ===")

    # 1. IMPORTÂNCIA DAS FEATURES (para modelos baseados em árvore)
    if hasattr(melhor_modelo, 'feature_importances_'):
        importancias = melhor_modelo.feature_importances_
        importancia_features = pd.Series(importancias, index=nomes_features).sort_values(ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importancia_features.values, y=importancia_features.index, palette="viridis")
        plt.title(f'Importância das Features - {type(melhor_modelo).__name__}')
        plt.show()
        
        print("\n--- Top 5 Features (Importância Clássica) ---")
        print(importancia_features.head())
    
    # 2. ANÁLISE SHAP
    try:
        if type(melhor_modelo).__name__ in ['RandomForestClassifier', 'XGBClassifier', 'DecisionTreeClassifier']:
            explainer = shap.TreeExplainer(melhor_modelo)
        else:
            explainer = shap.Explainer(melhor_modelo, X_teste_df) 
        
        shap_values = explainer.shap_values(X_teste_df)
        
        if isinstance(shap_values, list):
            # Focamos na classe positiva (diabetes=1)
            shap_values = shap_values[1] 

        # SHAP Summary Plot (Importância e Impacto Global)
        print("\n--- SHAP Summary Plot (Importância e Impacto Global) ---")
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_teste_df, feature_names=nomes_features, show=False)
        plt.title('SHAP Summary Plot (Impacto Global)')
        plt.tight_layout()
        plt.show()

        # SHAP Bar Plot (Média da Magnitude do Impacto)
        print("\n--- SHAP Bar Plot (Importância Média Absoluta) ---")
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_teste_df, feature_names=nomes_features, plot_type="bar", show=False)
        plt.title('SHAP Bar Plot (Importância Média Absoluta)')
        plt.tight_layout()
        plt.show()
        
        print("\nInterpretação SHAP: Pontos vermelhos (alto valor da feature) à direita do zero aumentam a chance de diabetes. Pontos azuis (baixo valor) à direita do zero também aumentam a chance.")
        
    except Exception as e:
        print(f"\nNão foi possível rodar a análise SHAP para este modelo ({type(melhor_modelo).__name__}): {e}")