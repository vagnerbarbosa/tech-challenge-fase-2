from pipeline.pre_processamento import carregar_dados, pre_processar_dados_diabetes
from pipeline.divisao_dados import dividir_e_oversample
from pipeline.modelagem import modelos
from pipeline.avaliacao import avaliar_modelo, extrair_parametros
from pipeline.otimizacao import aplicar_algoritmo_genetico

def principal():

    # 1. Carregar dados
    df = carregar_dados()

    # 2. Pré-processar dados (limpeza + escalonamento)
    X_escalado, y = pre_processar_dados_diabetes(df)

    # 3. Divisão e Oversampling
    X_treino_res, X_teste, y_treino_res, y_teste = dividir_e_oversample(X_escalado, y)

    # 4. Otimização, modelagem e treinamento
    for nome, (classe_modelo, intervalo) in modelos.items():
        print(f"\n🔬 {nome}")

        # 4.1. Avaliação com modelo padrão
        print("\nAvaliando modelo padrão")
        resultado_padrao = avaliar_modelo(classe_modelo, X_treino_res,y_treino_res)

        # 4.2. Avaliação com mais população
        print("\nAvaliando modelo otimizado e com mais população")

        melhor_individuo_populacao, melhor_pontuacao_populacao = aplicar_algoritmo_genetico(
            classe_modelo,
            intervalo,
            X_treino_res,
            y_treino_res,
            tamanho_populacao=100,
            numero_geracoes=10,
            taxa_mutacao=0.1,
        )

        # 4.3. Avaliação com mais gerações
        print("\nAvaliando modelo otimizado e com mais gerações")

        melhor_individuo_geracao, melhor_pontuacao_geracao = aplicar_algoritmo_genetico(
            classe_modelo,
            intervalo,
            X_treino_res,
            y_treino_res,
            tamanho_populacao=20,
            numero_geracoes=30,
            taxa_mutacao=0.1,
        )

        # 4.4. Avaliação com mais mutações
        print("\nAvaliando modelo otimizado e com mais mutações")

        melhor_individuo_mutacao, melhor_pontuacao_mutacao = aplicar_algoritmo_genetico(
            classe_modelo,
            intervalo,
            X_treino_res,
            y_treino_res,
            tamanho_populacao=20,
            numero_geracoes=10,
            taxa_mutacao=0.5,
        )

        # 4.5. Comparando o desempenho dos modelos
        print(f"\nAvaliação com modelo padrão: {resultado_padrao}")
        print(f"Avaliação com modelo otimizado e mais população: {melhor_pontuacao_populacao}")
        print(f"Avaliação com modelo otimizado e mais gerações: {melhor_pontuacao_geracao}")
        print(f"Avaliação com modelo otimizado e mais mutações: {melhor_pontuacao_mutacao}")

        melhor_pontuacao = max(resultado_padrao, melhor_pontuacao_populacao, melhor_pontuacao_geracao, melhor_pontuacao_mutacao)
        print("")
        if melhor_pontuacao == resultado_padrao:
            print("O modelo padrão teve o melhor resultado.")
        elif melhor_pontuacao in (melhor_pontuacao_populacao, melhor_pontuacao_geracao, melhor_pontuacao_mutacao):
            if melhor_pontuacao == melhor_pontuacao_populacao:
                print("O modelo otimizado e com mais população teve o melhor resultado.")
                melhor_individuo = melhor_individuo_populacao
            elif melhor_pontuacao == melhor_pontuacao_geracao:
                print("O modelo otimizado e com mais gerações teve o melhor resultado.")
                melhor_individuo = melhor_individuo_geracao
            elif melhor_pontuacao == melhor_pontuacao_mutacao:
                print("O modelo otimizado e com mais mutações teve o melhor resultado.")
                melhor_individuo = melhor_individuo_mutacao

            melhores_parametros = extrair_parametros(melhor_individuo, intervalo)

            print("Melhor pontuação:", melhor_pontuacao)
            print("Melhores hiperparâmetros:", melhores_parametros)

if __name__ == "__main__":
    principal()