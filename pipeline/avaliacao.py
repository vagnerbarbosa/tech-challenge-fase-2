from sklearn.metrics import recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pandas as pd
import numpy as np 
import lightgbm as lgb # Importado para garantir reconhecimento de classe (LGBMClassifier)

# === INTEGRAÇÃO GEMINI ===
import os
try:
    from google import genai
    from google.genai import types
    # Tenta configurar o cliente Gemini
    client = genai.Client()
    MODELO_LLM = "gemini-2.5-flash"
    CHAMADA_LLM_ATIVA = True
    print("✅ Gemini Client configurado. Chamadas à API ativas.")
except Exception as e:
    CHAMADA_LLM_ATIVA = False
    print("⚠️ Alerta: Gemini API não configurada (variável GEMINI_API_KEY ausente ou erro de biblioteca). Usando modo de SIMULAÇÃO LLM.")
# ==========================

# --- Funções Auxiliares para Simulação (Fallback) ---

def _gerar_simulacao_llm_texto(recall, f1, top_features, nome_melhor_modelo, acuracia):
    """Lógica de fallback para gerar o relatório sem a API do Gemini."""
    
    mapa_descricoes = {
        'DiabetesPedigreeFunction': 'Diabetes Pedigree Function (Histórico Familiar)',
        'Glucose': 'Glicose (Concentração Plasmática)',
        'BMI': 'Índice de Massa Corporal (IMC)',
        'glucose_bmi': 'Interação Glicose x IMC',
        'idade_bmi': 'Interação Idade x IMC',
        'Age': 'Idade'
    }

    feature1 = top_features[0]
    descricao1 = mapa_descricoes.get(feature1, feature1)
    
    # Lógica de ação baseada na feature
    if 'Pedigree' in feature1:
        acao1 = "Para pacientes com alto score de risco, o médico deve **investigar detalhadamente o histórico familiar** de diabetes e doenças relacionadas."
    elif 'Glucose' in feature1:
        acao1 = "Para pacientes com alto score de risco, o médico deve **revisar imediatamente** os níveis séricos de glicose."
    elif 'BMI' in feature1 or 'glucose_bmi' in feature1:
        acao1 = "Indica que o peso e o metabolismo são os fatores de maior peso. **Ação:** Iniciar aconselhamento imediato sobre mudança de estilo de vida para redução de peso."
    else:
        acao1 = "Requer atenção especial ao valor desta feature para o paciente."

    return f"""
    *** Relatório de Diagnóstico Otimizado (Gerado por LLM - SIMULAÇÃO) ***

    **Modelo Base:** {nome_melhor_modelo}
    
    **Performance (Foco em Sensibilidade - Recall):**
    - **Recall (Capacidade de detectar positivos):** {recall:.2f} (Indicador de alta sensibilidade diagnóstica)
    - **F1-score (Equilíbrio entre Precisão e Recall):** {f1:.2f}
    - **Acurácia Geral:** {acuracia:.2f}

    **🔬 Insights Clínicos Acionáveis (Baseados nos Fatores de Risco do Modelo):**

    O diagnóstico de diabetes (Classe 1) é impulsionado principalmente pelos seguintes features, em ordem de importância:

    1. **{descricao1}:** Este é o fator de maior peso. {acao1}
    
    2. **{mapa_descricoes.get(top_features[1], top_features[1])}:** É o segundo fator mais influente. Pacientes com IMC alto têm um risco significativamente maior. **Ação Recomendada:** Iniciar aconselhamento imediato sobre mudança de estilo de vida, dieta e exercício.
    
    3. **{mapa_descricoes.get(top_features[2], top_features[2])}:** Este fator atua como um modificador de risco. **Ação Recomendada:** Sugerir rastreamento anual de diabetes e monitoramento rigoroso.

    **Conclusão e Recomendação:**
    O profissional de saúde deve priorizar os pacientes com alta probabilidade predita, focando na intervenção sobre os fatores de risco citados, dada a sensibilidade de {recall:.2f} do modelo.
    """

# --- Funções Principais ---

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
    """
    
    print("\n\n=== 🔎 INTERPRETAÇÃO DO MELHOR MODELO ===")

    # 1. IMPORTÂNCIA DAS FEATURES 
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
        
        if type(melhor_modelo).__name__ in ['LGBMClassifier', 'RandomForestClassifier', 'DecisionTreeClassifier']:
            # Desativa a checagem de aditividade (necessário para LightGBM)
            explainer = shap.TreeExplainer(melhor_modelo, check_additivity=False)
        else:
            explainer = shap.Explainer(melhor_modelo, X_teste_df) 

        shap_values = explainer.shap_values(X_teste_df)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1] 

        # SHAP Summary Plot
        print("\n--- SHAP Summary Plot (Importância e Impacto Global) ---")
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_teste_df, feature_names=nomes_features, show=False)
        plt.title('SHAP Summary Plot (Impacto Global)')
        plt.tight_layout()
        plt.show()

        # SHAP Bar Plot
        print("\n--- SHAP Bar Plot (Importância Média Absoluta) ---")
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_teste_df, feature_names=nomes_features, plot_type="bar", show=False)
        plt.title('SHAP Bar Plot (Importância Média Absoluta)')
        plt.tight_layout()
        plt.show()
        
        print("\nInterpretação SHAP: Pontos vermelhos (alto valor da feature) à direita do zero aumentam a chance de diabetes. Pontos azuis (baixo valor) à direita do zero também aumentam a chance.")
        
    except Exception as e:
        print(f"\nNão foi possível rodar a análise SHAP para este modelo ({type(melhor_modelo).__name__}): {e}")

# --- FUNÇÃO PRINCIPAL DA FASE 2: GERAÇÃO DE RELATÓRIO ---

def gerar_explicacao_llm(melhor_modelo, nome_melhor_modelo, X_teste_df, y_teste, nomes_features, limiar=0.3):
    """
    Gera um relatório médico interpretável. Utiliza a API do Gemini se configurada, 
    caso contrário, usa a simulação de Prompt Engineering.
    """
    
    # 1. Obter Métricas Finais e Top Features
    if hasattr(melhor_modelo, "predict_proba"):
        y_proba = melhor_modelo.predict_proba(X_teste_df)[:, 1]
        y_pred = (y_proba >= limiar).astype(int)
    else:
        y_pred = melhor_modelo.predict(X_teste_df)

    recall = recall_score(y_teste, y_pred)
    f1 = f1_score(y_teste, y_pred)
    acuracia = np.mean(y_pred == y_teste)
    
    top_features = []
    if hasattr(melhor_modelo, 'feature_importances_'):
        importancias = melhor_modelo.feature_importances_
        importancia_features = pd.Series(importancias, index=nomes_features).sort_values(ascending=False)
        top_features = importancia_features.head(3).index.tolist()
    if not top_features:
        top_features = ['Glucose', 'BMI', 'Age'] 

    print("\n\n=== 🧠 GERAÇÃO DE RELATÓRIO INTERPRETÁVEL VIA LLM ===")

    if CHAMADA_LLM_ATIVA:
        # --- LÓGICA DE PROMPT ENGINEERING REAL (Gemini API) ---
        prompt_texto = f"""
        Você é um assistente médico especialista. Analise o desempenho e os fatores de risco (features) do modelo de diagnóstico de diabetes.
        
        Modelo: {nome_melhor_modelo}
        
        Métricas de Desempenho:
        - Recall (Sensibilidade): {recall:.2f}
        - F1-score: {f1:.2f}
        - Acurácia: {acuracia:.2f}
        
        Fatores de Risco (Importância SHAP/Clássica):
        1. {top_features[0]}
        2. {top_features[1]}
        3. {top_features[2]}
        
        Gere um relatório estruturado em linguagem natural. A análise deve focar em:
        1. Um resumo da performance do modelo, com ênfase no Alto Recall.
        2. Insights Clínicos Acionáveis: O que o médico deve fazer para cada um dos 3 fatores de risco.
        """
        
        try:
            response = client.models.generate_content(
                model=MODELO_LLM,
                contents=prompt_texto,
                config=types.GenerateContentConfig(
                    system_instruction="Você é um assistente médico profissional, conciso e focado em gerar insights acionáveis. Sua saída deve ser formatada como um relatório clínico.",
                    temperature=0.2
                )
            )
            explicacao_llm = f"\n*** Relatório de Diagnóstico Otimizado (Gerado por LLM - Gemini API) ***\n\n{response.text}"
            print("   [✅ SUCESSO] Relatório gerado pela API Gemini.")
            
        except Exception as e:
            # Fallback se a API falhar
            print(f"   [❌ FALHA API] Erro ao chamar Gemini: {e}. Usando modo de SIMULAÇÃO.")
            explicacao_llm = _gerar_simulacao_llm_texto(recall, f1, top_features, nome_melhor_modelo, acuracia)
            
    else:
        # --- LÓGICA DE SIMULAÇÃO (Se a API não estiver configurada) ---
        explicacao_llm = _gerar_simulacao_llm_texto(recall, f1, top_features, nome_melhor_modelo, acuracia)
        print("   [🛠️ SIMULAÇÃO] Relatório gerado por template de string.")
        
    print(explicacao_llm)
    print("=========================================================")
    return explicacao_llm