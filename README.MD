# 🧬 Diagnóstico de Diabetes Otimizado: Algoritmos Genéticos e Interpretabilidade com LLMs

Este projeto evoluiu do diagnóstico inicial de diabetes (Fase 1) para um sistema otimizado que utiliza **Algoritmos Genéticos (AGs)** para aprimorar os hiperparâmetros dos modelos **e o limiar de classificação**, além de **Large Language Models (LLMs)** para gerar explicações médicas acionáveis.

A principal melhoria da Fase 2 é a otimização dos modelos de Machine Learning (ML) médicos através de um **Pipeline Otimizado por AG**, garantindo a máxima performance de diagnóstico (Recall) e a interpretabilidade dos resultados para os profissionais de saúde.

O Relatório técnico e a análise detalhada se encontram no **Notebook** (`diabetes_analysis.ipynb`) e no pipeline Python (`main.py`).

O objetivo principal do projeto é:
1. **Executar a Otimização via Algoritmos Genéticos (AG)**: Otimizar os hiperparâmetros do Random Forest **e o limiar de classificação** para maximizar o Recall (Sensibilidade).
2. **Executar o Modelo Otimizado**: Aplicar o conjunto de hiperparâmetros e limiar encontrados pelo AG ao modelo final.
3. **Demonstrar a Interpretabilidade via LLM**: Integrar a API Gemini para traduzir o diagnóstico e as métricas de desempenho em linguagem natural, fornecendo *insights* acionáveis para a equipe médica.

---
## 🎞️ Vídeo de Demonstração
https://youtu.be/d62wPNKj1TU?si=My-jKimrJMECClqP

## 🚀 Como Executar Localmente

### ✅ Pré-requisitos

- [Python 3.10 ou 3.11](https://www.python.org/downloads/)
- Chave de API do Google Gemini ([obtenha aqui](https://aistudio.google.com/app/apikey))

### ▶️ Passos

1. Clone este repositório:
   ```bash
   git clone https://github.com/vagnerbarbosa/tech-challenge-fase-2.git
   cd tech-challenge-fase-2
   ```

2. Crie um ambiente virtual e ative-o:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   .venv\Scripts\activate     # Windows
   ```

3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure a chave da API Gemini:
   - Crie um arquivo `.env` na raiz do projeto
   - Adicione sua chave: `GEMINI_API_KEY=sua-chave-aqui`

5. Execute o pipeline:
   ```bash
   jupyter notebook diabetes_analysis.ipynb
   ```

> **Nota**: O pipeline executa automaticamente a Fase 1 (Baseline) e a Fase 2 (Otimização por AG + Interpretabilidade LLM).

> **Nota 2**: Certifique-se de que o Jupyter esteja configurado no PATH do sistema operacional para o comando funcionar corretamente.

---

## 📊 Dataset

- O dataset utilizado é o **Pima Indians Diabetes Dataset**, carregado diretamente do Kaggle via `kagglehub`.
- Link da fonte: [Kaggle Diabetes Dataset](https://www.kaggle.com/datasets/mathchi/diabetes-data-set/data)

---

## 🛠️ Componentes da Fase 2

### 1. Otimização de Hiperparâmetros via Algoritmos Genéticos (AG)

Para a Fase 2, implementamos um Algoritmo Genético (AG) usando a biblioteca **DEAP** (Distributed Evolutionary Algorithms in Python) para otimizar os hiperparâmetros do modelo **Random Forest** e o **limiar de classificação**, priorizando o **Recall** (Sensibilidade).

**DEAP** é uma biblioteca Python especializada em algoritmos evolutivos que fornece:
- ✅ **Algoritmos Genéticos completos** - População, seleção, crossover, mutação
- ✅ **Operadores genéticos prontos** - cxTwoPoint, mutGaussian, selTournament, etc.
- ✅ **Fitness customizável** - Maximização ou minimização de objetivos
- ✅ **Algoritmos evolutivos** - eaSimple, eaMuPlusLambda, eaMuCommaLambda
- ✅ **Estatísticas e logging** - Acompanhamento da evolução ao longo das gerações

* **Codificação e Operadores:** Utilizamos vetores numéricos como representação de genes para os hiperparâmetros relevantes (`n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `bootstrap`) e o **limiar de classificação** (threshold).
* **Função *Fitness***: Baseada no **Recall médio** obtido via validação cruzada estratificada (3-fold), aplicando o limiar otimizado em cada fold.
* **Operadores Genéticos**:
  - **Seleção**: Torneio (tournsize=3)
  - **Cruzamento**: Two-Point Crossover (cxpb=0.7)
  - **Mutação**: Customizada com sigma específico para cada parâmetro (mutpb=0.2, indpb=0.3)
* **Parâmetros do AG**: 30 gerações, população de 50 indivíduos

### 2. Interpretabilidade e *Prompt Engineering* com LLMs

O modelo otimizado (Random Forest) é integrado à **API Gemini** (modelo `gemini-2.5-flash-lite`) para transformar os resultados técnicos em informações úteis para o médico.

* **Geração de Explicações:** A LLM gera explicações em linguagem natural dos diagnósticos produzidos pelos modelos.
* ***Insights* Acionáveis:** A LLM transforma dados numéricos e estatísticos (Recall, F1-score, Acurácia) e fatores de risco (importância SHAP) em *insights* acionáveis para médicos.
* ***Prompt Engineering***: Implementamos técnicas de *prompt engineering* para obter respostas relevantes e adequadas ao contexto médico, com instruções de sistema específicas para gerar relatórios clínicos estruturados.

---

## 📈 Resultados Chave e Comparação Pós-AG

A otimização via Algoritmos Genéticos foi aplicada com sucesso, demonstrando melhorias significativas no Recall.

| Métrica de Desempenho | Pipeline Baseline (Fase 1) | Pipeline Otimizado (AG - Fase 2) |
| :--- | :--- | :--- |
| **Melhor Modelo** | **Random Forest** | **Random Forest Otimizado (AG)** |
| **Recall Máximo** | 0.91 | **0.94** ✅ |
| **F1-score** | **0.72** | 0.66 |
| **Limiar de Classificação** | 0.30 (fixo) | **0.20** (otimizado pelo AG) |
| **Interpretabilidade**| Baixa (Métricas) | **Alta (LLM + Insights Acionáveis)** |

### 🔍 Destaques da Otimização por AG

- **Melhoria no Recall:** O AG aumentou o Recall de **0.91 → 0.94** (+3.3%), detectando **94% dos casos reais de diabetes** no conjunto de teste.
- **Limiar Otimizado:** O AG descobriu que o limiar ideal é **0.20** (em vez de 0.30), priorizando a detecção de casos positivos.
- **Trade-off Recall vs F1:** O F1-score diminuiu de 0.72 para 0.66 devido ao aumento de falsos positivos, mas isso é **esperado e desejável** em diagnósticos médicos, onde **não detectar um diabético é mais grave** que um falso alarme.
- **Interpretabilidade Aumentada:** A principal vitória da Fase 2 é a camada de Interpretabilidade. O relatório LLM, baseado na análise SHAP do modelo otimizado, oferece *insights* acionáveis e específicos para o médico:
    1.  **glucose_bmi (Interação Glicose x IMC)** - Fator de maior peso
    2.  **Glucose (Nível de Glicose Sanguínea)**
    3.  **idade_bmi (Interação Idade x IMC)**
- **Foco na Tradução:** A LLM transforma a saída do modelo em informação imediatamente útil para o profissional de saúde, gerando explicações claras sobre o que está impulsionando o diagnóstico de risco e **ações médicas recomendadas** para cada fator.

### 📊 Evolução do AG

O AG convergiu em 30 gerações, com o Recall médio da população evoluindo de **0.91** (geração 0) para **0.94** (geração 30), demonstrando a eficácia da otimização.

---

## 🏗️ Estrutura do Projeto

```
tech-challenge-fase-2/
├── main.py                      # Pipeline principal (Fase 1 + Fase 2)
├── pipeline/
│   ├── preprocessamento.py      # Carregamento e pré-processamento dos dados
│   ├── modelagem.py             # Modelos baseline + AG otimização
│   └── avaliacao.py             # Avaliação, SHAP e integração LLM
├── requirements.txt             # Dependências (incluindo DEAP e python-dotenv)
├── .env                         # Chave da API Gemini (não versionado)
├── diabetes_analysis.ipynb      # Notebook de análise
└── README.md                    # Este arquivo
```

---

## 👯 Contribuidores

- [Adriel Santos](https://github.com/AdrielCandido)
- [João Marcos](https://github.com/github)
- [Leticia Nepomucena](https://github.com/LeticiaNepomucena)
- [Lucas Silva](https://github.com/lucfsilva)
- [Vagner Barbosa](https://github.com/vagnerbarbosa)

---

## 🔒 Conformidade e Ética dos Dados
Este projeto utiliza apenas dados públicos do Kaggle para fins educacionais e de pesquisa.
