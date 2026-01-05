import random
from pipeline.avaliacao import avaliar_modelo
import numpy as np

def criar_individuo(intervalos):
    """
    Cria um indivíduo a partir do intervalo de seus hiperparâmetros

    Parâmetros:
        intervalos (biblioteca): intervalos dos hiperparâmetros de um algoritimo de IA

    Retorna:
        individuo.
    """
    return [gerador() for gerador in intervalos.values()]

def selecao(populacao, pontuacoes, k=3):
    selecionado = random.sample(list(zip(populacao, pontuacoes)), k)
    selecionado.sort(key=lambda x: x[1], reverse=True)
    return selecionado[0][0]

def cruzamento(pai_1, pai_2):
    ponto = random.randint(1, len(pai_1) - 1)
    return (
        pai_1[:ponto] + pai_2[ponto:],
        pai_2[:ponto] + pai_1[ponto:]
    )

def mutacao(individuo, intervalo, taxa_mutacao=0.1):
    for i, gerador in enumerate(intervalo.values()):
        if random.random() < taxa_mutacao:
            individuo[i] = gerador()
    return individuo

def aplicar_algoritmo_genetico(
    classe_modelo,
    intervalo,
    X,
    y,
    tamanho_populacao=30,
    numero_geracoes=20,
    taxa_mutacao=0.1,
    tamanho_elite=2
):
    # População inicial
    populacao = [criar_individuo(intervalo) for _ in range(tamanho_populacao)]

    for geracao in range(numero_geracoes):
        pontuacoes = [
            avaliar_modelo(classe_modelo, X, y, individuo, intervalo)
            for individuo in populacao
        ]

        # Elitismo simples
        indices_elite = np.argsort(pontuacoes)[-tamanho_elite:]
        elites = [populacao[i] for i in indices_elite]

        nova_populacao = elites.copy()

        while len(nova_populacao) < tamanho_populacao:
            pai = selecao(populacao, pontuacoes)
            mae = selecao(populacao, pontuacoes)

            filho_1, filho_2 = cruzamento(pai, mae)

            filho_1 = mutacao(filho_1, intervalo, taxa_mutacao)
            filho_2 = mutacao(filho_2, intervalo, taxa_mutacao)

            nova_populacao.extend([filho_1, filho_2])

        populacao = nova_populacao[:tamanho_populacao]

        print(f"Geração {geracao + 1} | Melhor score: {max(pontuacoes):.4f}")

    # Melhor indivíduo final
    pontuacoes_finais = [
        avaliar_modelo(classe_modelo, X, y, individuo, intervalo)
        for individuo in populacao
    ]
    id_melhor_individuo = np.argmax(pontuacoes_finais)

    return populacao[id_melhor_individuo], pontuacoes_finais[id_melhor_individuo]