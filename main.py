from pipeline.pre_processamento import carregar_dados, pre_processar_dados_diabetes
from pipeline.divisao_dados import dividir_e_oversample
from pipeline.modelagem import modelos
from pipeline.otimizacao import aplicar_algoritmo_genetico
from pipeline.avaliacao import extrair_parametros, avaliar_modelo

def principal():

    # 1. Carregar dados
    df = carregar_dados()

    # 2. Pré-processar dados (limpeza + escalonamento)
    X_escalado, y = pre_processar_dados_diabetes(df)

    # 3. Divisão e Oversampling
    X_treino_res, X_teste, y_treino_res, y_teste = dividir_e_oversample(X_escalado, y)

    # 4. Otimização, modelagem e treinamento
    melhor_pontuacao_geral = 0

    for nome, (classe_modelo, intervalo) in modelos.items():
        print(f"\n🔬 Modelo: {nome}")

        # 4.1. Avaliação com modelo padrão
        print("\nAvaliando com hiperparâmetros padrão")
        resultado_padrao = avaliar_modelo(classe_modelo, X_treino_res, y_treino_res)

        # 4.2. Avaliação com mais população
        print("\nAvaliando otimização dos hiperparâmetros com mais população")

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
        print("\nAvaliando otimização dos hiperparâmetros com mais mais gerações")

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
        print("\nAvaliando otimização dos hiperparâmetros com mais mutações")

        melhor_individuo_mutacao, melhor_pontuacao_mutacao = aplicar_algoritmo_genetico(
            classe_modelo,
            intervalo,
            X_treino_res,
            y_treino_res,
            tamanho_populacao=20,
            numero_geracoes=10,
            taxa_mutacao=0.5,
        )

        # 4.5. Comparando o desempenho
        print(f"\nPontuação do modelo padrão: {resultado_padrao}")
        print(f"Pontuação do modelo otimizado com mais população: {melhor_pontuacao_populacao}")
        print(f"Pontuação do modelo otimizado com mais gerações: {melhor_pontuacao_geracao}")
        print(f"Pontuação do modelo otimizado com mais mutações: {melhor_pontuacao_mutacao}\n")

        melhor_pontuacao_modelo = max(resultado_padrao, melhor_pontuacao_populacao, melhor_pontuacao_geracao, melhor_pontuacao_mutacao)

        if melhor_pontuacao_modelo == resultado_padrao:
            print("O modelo padrão teve o melhor resultado.")
        elif melhor_pontuacao_modelo in (melhor_pontuacao_populacao, melhor_pontuacao_geracao, melhor_pontuacao_mutacao):
            if melhor_pontuacao_modelo == melhor_pontuacao_populacao:
                print("O modelo otimizado com mais população teve o melhor resultado.")
                melhor_individuo = melhor_individuo_populacao
            elif melhor_pontuacao_modelo == melhor_pontuacao_geracao:
                print("O modelo otimizado com mais gerações teve o melhor resultado.")
                melhor_individuo = melhor_individuo_geracao
            elif melhor_pontuacao_modelo == melhor_pontuacao_mutacao:
                print("O modelo otimizado com mais mutações teve o melhor resultado.")
                melhor_individuo = melhor_individuo_mutacao

            melhores_parametros_modelo = extrair_parametros(melhor_individuo, intervalo)

            print("Melhor pontuação:", melhor_pontuacao_modelo)
            print("Melhores hiperparâmetros:", melhores_parametros_modelo)

        # if melhor_pontuacao_modelo > melhor_pontuacao_geral:
        #     melhor_pontuacao_geral = melhor_pontuacao_modelo
        #     nome_melhor_modelo = nome
        #     classe_melhor_modelo = classe_modelo
        #     parametros_melhor_modelo = None if melhor_pontuacao_modelo == resultado_padrao else melhores_parametros_modelo

    # if classe_melhor_modelo:
    #     if parametros_melhor_modelo:
    #         melhor_modelo = classe_melhor_modelo(**parametros_melhor_modelo, random_state=SEMENTE)
    #     else:
    #         melhor_modelo = classe_melhor_modelo(random_state=SEMENTE)

    #     melhor_modelo.fit(X_treino, y_treino)

    #     if hasattr(melhor_modelo, "predict_proba"):
    #         y_probabilistico = melhor_modelo.predict_proba(X_teste)[:, 1]
    #         y_previsto = (y_probabilistico >= 0.3).astype(int)
    #     else:
    #         y_previsto = melhor_modelo.predict(X_teste)        

    #     valor_recall = recall_score(y_teste, y_previsto)

    #     print("==================================================")
    #     print(f"🏆 Melhor Modelo Encontrado ({nome_melhor_modelo})")
    #     print(f"   Critério de Seleção: Maior Recall")
    #     print(f"   Recall (Sensibilidade) com threshold de 30%: {valor_recall:.4f}")
    #     print("==================================================")        

    #     return melhor_modelo
    # else:
    #     print("Não foi possível identificar o melhor modelo")

if __name__ == "__main__":
    principal()