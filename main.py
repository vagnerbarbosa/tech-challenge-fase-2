from pipeline import pre_processamento, visualizacao, modelagem, avaliacao
from pipeline.divisao_dados import dividir_e_oversample
import pandas as pd

def principal():
    # 1. Carregar dados
    df = pre_processamento.carregar_dados()

    # 2. Pré-processar dados (limpeza + escalonamento)
    df_pronto, X_escalado, y, scaler, nomes_features = pre_processamento.pre_processar_dados_diabetes(df)

    # 3. Visualização
    visualizacao.plotar_distribuicao_alvo(df_pronto)
    visualizacao.plotar_histogramas(df_pronto)
    visualizacao.plotar_matriz_correlacao(df_pronto)

    # 4. Divisão e Oversampling
    X_treino_res, X_teste, y_treino_res, y_teste = dividir_e_oversample(X_escalado, y)

    # 5. Modelagem e treinamento
    modelos = modelagem.obter_modelos()
    modelos = modelagem.treinar_modelos(modelos, X_treino_res, y_treino_res)
    
    # 6. Avaliação e determinação do melhor modelo
    nome_melhor_modelo, melhor_modelo = avaliacao.avaliar_modelos(modelos, X_teste, y_teste, nomes_features)

    # 7. Interpretação do Melhor Modelo
    if melhor_modelo:
        X_teste_df = pd.DataFrame(X_teste, columns=nomes_features)
        avaliacao.interpretar_melhor_modelo(melhor_modelo, X_teste_df, nomes_features)
    else:
        print("Nenhum modelo foi avaliado ou encontrado para interpretação.")

if __name__ == "__main__":
    principal()