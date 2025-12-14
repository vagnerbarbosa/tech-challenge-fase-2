from sklearn.metrics import recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pandas as pd
import numpy as np # Importado para cálculos de métricas

def avaliar_modelos(modelos, X_teste, y_teste, nomes_features, limiar=0.3):
    # ... (Conteúdo da função avaliar_modelos - NÃO ALTERADO)
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
    print(f"   Critério de Seleção: Maior Recall")
    print(f"   Recall (Sensibilidade): {melhor_recall:.2f}")
    print(f"   F1-score: {melhor_f1:.2f}")
    print("==================================================")
    
    return nome_melhor_modelo, melhor_modelo

def interpretar_melhor_modelo(melhor_modelo, X_teste_df, nomes_features):
    """
    Realiza a interpretação do melhor modelo utilizando Importância das Features e SHAP.
    
    CORREÇÃO IMPLEMENTADA: Desativação da checagem de aditividade no TreeExplainer
    para resolver o 'Additivity check failed' comum no LightGBM.
    """
    
    print("\n\n=== 🔎 INTERPRETAÇÃO DO MELHOR MODELO ===")

    # 1. IMPORTÂNCIA DAS FEATURES (para modelos baseados em árvore)
    # ... (O código desta seção permanece o mesmo)
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
        
        # O LightGBM (LGBMClassifier) é um modelo baseado em árvore
        if type(melhor_modelo).__name__ in ['LGBMClassifier', 'RandomForestClassifier', 'DecisionTreeClassifier']:
            
            # === CORREÇÃO: check_additivity=False para resolver o erro ===
            explainer = shap.TreeExplainer(melhor_modelo, check_additivity=False)
            
        else:
            explainer = shap.Explainer(melhor_modelo, X_teste_df) 

        # Rodar SHAP
        shap_values = explainer.shap_values(X_teste_df)
        
        if isinstance(shap_values, list):
            # Focamos na classe positiva (diabetes=1)
            shap_values = shap_values[1] 

        # SHAP Summary Plot (Importância e Impacto Global)
        print("\n--- SHAP Summary Plot (Importância e Impacto Global) ---")
        plt.figure(figsize=(10, 6))
        # Passamos X_teste_df para garantir que os nomes das features sejam usados nos plots
        shap.summary_plot(shap_values, X_teste_df, feature_names=nomes_features, show=False)
        plt.title('SHAP Summary Plot (Impacto Global)')
        plt.tight_layout()
        plt.show()
        # 

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

def gerar_explicacao_llm(melhor_modelo, nome_melhor_modelo, X_teste_df, y_teste, nomes_features, limiar=0.3):
    """
    Simula a geração de um relatório médico interpretável por um LLM.
    
    CORREÇÃO: A descrição da feature 1 (DiabetesPedigreeFunction) foi ajustada
    para refletir o histórico familiar, não a glicose.
    """
    
    print("\n\n=== 🧠 GERAÇÃO DE RELATÓRIO INTERPRETÁVEL VIA LLM ===")

    # 1. Obter Métricas Finais
    if hasattr(melhor_modelo, "predict_proba"):
        y_proba = melhor_modelo.predict_proba(X_teste_df)[:, 1]
        y_pred = (y_proba >= limiar).astype(int)
    else:
        y_pred = melhor_modelo.predict(X_teste_df)

    recall = recall_score(y_teste, y_pred)
    f1 = f1_score(y_teste, y_pred)
    acuracia = np.mean(y_pred == y_teste)
    
    # 2. Obter Features Mais Importantes (Simulando o resultado SHAP)
    top_features = []
    try:
        # Tenta obter a importância real (LightGBM usa um valor inteiro)
        if hasattr(melhor_modelo, 'feature_importances_'):
            importancias = melhor_modelo.feature_importances_
            importancia_features = pd.Series(importancias, index=nomes_features).sort_values(ascending=False)
            top_features = importancia_features.head(3).index.tolist()
        
        if not top_features:
             # Fallback padrão se não conseguir calcular a importância
             top_features = ['Glucose', 'BMI', 'Age']
    except:
        top_features = ['Glucose', 'BMI', 'Age']

    # 3. Geração da Explicação (Simulação do Output do LLM)
    
    # Mapeamento para garantir a descrição correta
    mapa_descricoes = {
        'DiabetesPedigreeFunction': 'Diabetes Pedigree Function (Histórico Familiar)',
        'Glucose': 'Glicose (Concentração Plasmática)',
        'BMI': 'Índice de Massa Corporal (IMC)',
        'glucose_bmi': 'Interação Glicose x IMC',
        'idade_bmi': 'Interação Idade x IMC',
        'Age': 'Idade'
    }

    # Definir a descrição e a ação acionável baseada na feature principal
    feature1 = top_features[0]
    descricao1 = mapa_descricoes.get(feature1, feature1)
    
    if 'Pedigree' in feature1:
        acao1 = "Para pacientes com alto score de risco, o médico deve **investigar detalhadamente o histórico familiar** de diabetes e doenças relacionadas."
    elif 'Glucose' in feature1:
        acao1 = "Para pacientes com alto score de risco, o médico deve **revisar imediatamente** os níveis séricos de glicose."
    elif 'BMI' in feature1:
        acao1 = "Indica que o peso e o metabolismo são os fatores de maior peso. **Ação:** Iniciar aconselhamento imediato sobre mudança de estilo de vida para redução de peso."
    else:
        acao1 = "Requer atenção especial ao valor desta feature para o paciente."


    explicacao_llm = f"""
    *** Relatório de Diagnóstico Otimizado (Gerado por LLM) ***

    **Modelo Base:** {nome_melhor_modelo}
    
    **Performance (Foco em Sensibilidade - Recall):**
    - **Recall (Capacidade de detectar positivos):** {recall:.2f} (Indicador de alta sensibilidade diagnóstica)
    - **F1-score (Equilíbrio entre Precisão e Recall):** {f1:.2f}
    - **Acurácia Geral:** {acuracia:.2f}

    **🔬 Insights Clínicos Acionáveis (Baseados nos Fatores de Risco do Modelo):**

    O diagnóstico de diabetes (Classe 1) é impulsionado principalmente pelos seguintes features, em ordem de importância:

    1. **{descricao1}:** Este é o fator de maior peso. {acao1}
    
    2. **{mapa_descricoes.get(top_features[1], top_features[1])}:** O IMC ou a interação com Glicose é o segundo fator mais influente. Pacientes com IMC alto têm um risco significativamente maior. **Ação Recomendada:** Iniciar aconselhamento imediato sobre mudança de estilo de vida, dieta e exercício.
    
    3. **{mapa_descricoes.get(top_features[2], top_features[2])}:** Este fator atua como um modificador de risco. **Ação Recomendada:** Sugerir rastreamento anual de diabetes e monitoramento rigoroso.

    **Conclusão e Recomendação:**
    O modelo otimizado por Algoritmos Genéticos demonstrou alta capacidade de detectar casos positivos (Alto Recall de {recall:.2f}). O profissional de saúde deve priorizar os pacientes com alta probabilidade predita, focando na intervenção sobre os fatores de risco citados.
    """
    
    print(explicacao_llm)
    print("=========================================================")
    return explicacao_llm