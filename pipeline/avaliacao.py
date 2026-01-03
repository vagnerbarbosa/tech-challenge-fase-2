from sklearn.metrics import recall_score, classification_report

def avaliar_modelos(modelos, X_teste, y_teste, limiar=0.3):
    """
    Avalia múltiplos modelos de classificação, exibe métricas e matriz de confusão, e retorna o modelo com o melhor Recall.
    """
    melhor_recall = -1
    nome_melhor_modelo = None
    melhor_modelo = None

    for nome, modelo in modelos.items():

        if hasattr(modelo, "predict_proba"):
            print(f'O modelo {nome} tem o atributo predict_proba')
            y_proba = modelo.predict_proba(X_teste)[:, 1]
            y_pred = (y_proba >= limiar).astype(int)
        else:
            y_pred = modelo.predict(X_teste)
            
        recall_atual = recall_score(y_teste, y_pred)
        
        # Lógica para rastrear o melhor modelo (baseado no Recall)
        if recall_atual > melhor_recall:
            melhor_recall = recall_atual
            nome_melhor_modelo = nome
            melhor_modelo = modelo

        print(f"\n--- {nome} ---")
        print(f"Recall (Sensibilidade): {recall_atual:.2f}")
        print("\nRelatório de classificação detalhado:")
        print(classification_report(y_teste, y_pred))
        print("Matriz de Confusão (linhas = real, colunas = predito):")

    print("==================================================")
    print(f"🏆 Melhor Modelo Encontrado: {nome_melhor_modelo}")
    print(f"   Critério de Seleção: Maior Recall")
    print(f"   Recall (Sensibilidade): {melhor_recall:.2f}")
    print("==================================================")
    
    return nome_melhor_modelo, melhor_modelo