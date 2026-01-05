from sklearn.model_selection import cross_val_score

def extrair_parametros(individuo, intervalos):
    return dict(zip(intervalos.keys(), individuo))

def avaliar_modelo(classe_modelo, X, y, individuo = None, intervalos = None):

    SEMENTE = 42

    if individuo is None:
        modelo = classe_modelo(random_state=SEMENTE)
    else:
        parametros = extrair_parametros(individuo, intervalos)
        modelo = classe_modelo(**parametros, random_state=SEMENTE)

    pontuacao = cross_val_score(modelo, X, y, cv=5, scoring="recall").mean()

    return pontuacao