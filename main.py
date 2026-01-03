from pipeline import pre_processamento, modelagem, avaliacao
from pipeline.divisao_dados import dividir_e_oversample

def principal():

    # 1. Carregar dados
    df = pre_processamento.carregar_dados()

    # 2. Pré-processar dados (limpeza + escalonamento)
    df_pronto, X_escalado, y, scaler, nomes_features = pre_processamento.pre_processar_dados_diabetes(df)

    # 3. Divisão e Oversampling
    X_treino_res, X_teste, y_treino_res, y_teste = dividir_e_oversample(X_escalado, y)

    # 4. Modelagem e treinamento
    modelos = modelagem.obter_modelos()
    modelos = modelagem.treinar_modelos(modelos, X_treino_res, y_treino_res)
    
    # 5. Avaliação e determinação do melhor modelo
    avaliacao.avaliar_modelos(modelos, X_teste, y_teste)

if __name__ == "__main__":
    principal()