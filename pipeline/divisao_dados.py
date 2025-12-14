from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def dividir_e_oversample(X, y, proporcao_teste=0.2, semente=42, estratificar=True, smote_semente=42):
    """
    Divide os dados em treino/teste e aplica SMOTE apenas nos dados de treino.

    Parâmetros:
        X (array ou DataFrame): Features já processadas/escaladas.
        y (array ou Series): Labels/variável alvo.
        proporcao_teste (float): Proporção dos dados para teste.
        semente (int): Semente para reprodutibilidade.
        estratificar (bool): Se True, faz estratificação em y.
        smote_semente (int): Semente do SMOTE.

    Retorna:
        X_treino_res, X_teste, y_treino_res, y_teste
    """
    estrato = y if estratificar else None
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=proporcao_teste, random_state=semente, stratify=estrato
    )

    smote = SMOTE(random_state=smote_semente)
    X_treino_res, y_treino_res = smote.fit_resample(X_treino, y_treino)

    return X_treino_res, X_teste, y_treino_res, y_teste