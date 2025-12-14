import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from pathlib import Path
import kagglehub

from pathlib import Path
import pandas as pd
import kagglehub

from pathlib import Path
import pandas as pd
import kagglehub

from pathlib import Path
import pandas as pd

def carregar_dados() -> pd.DataFrame:
    '''
    Tenta carregar o Dataset localmente de 'pipeline/dados/diabetes.csv'.
    Se não encontrar, baixa do Kaggle "uciml/pima-indians-diabetes".
    Retorno: DataFrame com o conteúdo do Dataset.
    '''
    local_path = Path('pipeline/dados/diabetes.csv')
    try:
        return pd.read_csv(local_path)
    except Exception as e:
        print(f"Arquivo local não encontrado ou erro ao ler: {e}")
        try:
            import kagglehub
            endereco_de_origem = kagglehub.dataset_download(handle='uciml/pima-indians-diabetes', force_download=True)
            diretorio_de_origem = Path(endereco_de_origem).resolve()
            lista_dados_csv = []
            for item in diretorio_de_origem.iterdir():
                if item.is_file() and item.suffix.lower() == '.csv':
                    lista_dados_csv.append(pd.read_csv(item))
            dados = pd.concat(lista_dados_csv, axis=0, ignore_index=True)
            return dados
        except Exception as e2:
            raise FileNotFoundError(f"Não foi possível carregar os dados localmente nem baixar do Kaggle: {e2}")

def pre_processar_dados_diabetes(df):
    """
    Pré-processamento:
    - Remove SkinThickness e Insulin
    - Trata zeros em Glucose, BloodPressure, BMI (converte para NaN e imputa mediana)
    - Mantém zeros em Pregnancies
    - Engenharia de features: cria variáveis de risco
    - Aplica RobustScaler (exceto na coluna alvo)
    """
    # 1. Remove colunas
    df = df.drop(['SkinThickness', 'Insulin'], axis=1)
    
    # 2. Trata zeros biologicamente impossíveis
    colunas_tratar = ['Glucose', 'BloodPressure', 'BMI']
    for col in colunas_tratar:
        df[col] = df[col].replace(0, np.nan)
    imputador = SimpleImputer(strategy='median')
    df[colunas_tratar] = imputador.fit_transform(df[colunas_tratar])

    # 3. ENGENHARIA DE FEATURES: Variáveis de risco
    df['idade_maior_45'] = (df['Age'] >= 45).astype(int)
    df['imc_obeso'] = (df['BMI'] >= 30).astype(int)
    df['idade_bmi'] = df['Age'] * df['BMI']  # interação
    df['glucose_bmi'] = df['Glucose'] * df['BMI']  # interação

    # 4. Separa features e target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome'].values

    nomes_features = X.columns.tolist()

    # 5. Escalonamento
    scaler = RobustScaler()
    X_escalado = scaler.fit_transform(X)
    return df, X_escalado, y, scaler, nomes_features