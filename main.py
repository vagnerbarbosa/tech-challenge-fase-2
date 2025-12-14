from pipeline import pre_processamento, visualizacao, modelagem, avaliacao
from pipeline.divisao_dados import dividir_e_oversample
import pandas as pd

def principal():
    # 1. Carregar dados
    df = pre_processamento.carregar_dados()

    # 2. Pré-processar dados (limpeza + escalonamento)
    df_pronto, X_escalado, y, scaler, nomes_features = pre_processamento.pre_processar_dados_diabetes(df)

    # 3. Visualização
    print("--- 📊 VISUALIZAÇÃO DE DADOS BASELINE ---")
    visualizacao.plotar_distribuicao_alvo(df_pronto)
    visualizacao.plotar_histogramas(df_pronto)
    visualizacao.plotar_matriz_correlacao(df_pronto)

    # 4. Divisão e Oversampling
    X_treino_res, X_teste, y_treino_res, y_teste = dividir_e_oversample(X_escalado, y)

    # Conversão de X_teste para DataFrame para uso em SHAP e LLM
    X_teste_df = pd.DataFrame(X_teste, columns=nomes_features)

    # ==========================================================
    # FASE 1: BASELINE (Avaliação de múltiplos modelos)
    # ==========================================================
    print("\n\n=== 📈 FASE 1: TREINAMENTO E AVALIAÇÃO BASELINE ===")
    modelos_baseline = modelagem.obter_modelos_baseline()
    modelos_baseline = modelagem.treinar_modelos(modelos_baseline, X_treino_res, y_treino_res)
    
    # Avaliação e determinação do melhor modelo BASELINE
    nome_melhor_modelo_baseline, melhor_modelo_baseline = avaliacao.avaliar_modelos(
        modelos_baseline, X_teste, y_teste, nomes_features
    )
    
    if melhor_modelo_baseline:
        print("\n--- 🔎 INTERPRETAÇÃO DO MELHOR MODELO BASELINE ---")
        avaliacao.interpretar_melhor_modelo(melhor_modelo_baseline, X_teste_df, nomes_features)
    # ==========================================================


    # ==========================================================
    # FASE 2: OTIMIZAÇÃO POR AG E INTERPRETAÇÃO LLM
    # ==========================================================
    print("\n\n=== 🧬 FASE 2: OTIMIZAÇÃO POR ALGORITMOS GENÉTICOS (AG) ===")
    
    # 5. Otimização e Treinamento do Modelo Otimizado (Simulação AG)
    melhor_modelo_otimizado, nome_modelo_otimizado = modelagem.otimizar_modelo_ag(
        X_treino_res, y_treino_res
    )

    # 6. Avaliação do Modelo Otimizado
    # Avaliamos o modelo otimizado separadamente para mostrar o ganho de performance
    modelos_otimizados = {nome_modelo_otimizado: melhor_modelo_otimizado}
    nome_melhor_otimizado, modelo_otimizado = avaliacao.avaliar_modelos(
        modelos_otimizados, X_teste, y_teste, nomes_features
    )

    # 7. Interpretação do Modelo Otimizado (SHAP)
    print(f"\n--- 🔎 INTERPRETAÇÃO DO MODELO OTIMIZADO ({nome_melhor_otimizado}) ---")
    avaliacao.interpretar_melhor_modelo(modelo_otimizado, X_teste_df, nomes_features)
    
    # 8. Geração de Explicação Acionável via LLM
    avaliacao.gerar_explicacao_llm(
        modelo_otimizado, nome_melhor_otimizado, X_teste_df, y_teste, nomes_features
    )
    
    print("\nPipeline Otimizado por AG e com Interpretabilidade LLM CONCLUÍDO.")


if __name__ == "__main__":
    principal()