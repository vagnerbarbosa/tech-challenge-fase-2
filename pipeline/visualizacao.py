import matplotlib.pyplot as plt
import seaborn as sns

def plotar_distribuicao_alvo(df):
    """
    Plota a distribuição da variável alvo (Outcome).
    """
    sns.countplot(x='Outcome', data=df)
    plt.title('Distribuição de Diagnóstico (0=Não Diabético, 1=Sim Diabético)')
    plt.show()

def plotar_histogramas(df):
    """
    Plota histogramas para todas as variáveis numéricas do DataFrame.
    """
    df.hist(bins=20, figsize=(15,10))
    plt.show()

def plotar_matriz_correlacao(df):
    """
    Plota a matriz de correlação entre as variáveis do DataFrame.
    """
    plt.figure(figsize=(10,8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Matriz de Correlação')
    plt.show()