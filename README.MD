# 🧬 Diagnóstico de Diabetes Otimizado: Algoritmos Genéticos e Interpretabilidade com LLMs

Este projeto evoluiu do diagnóstico inicial de diabetes (Fase 1) para um sistema otimizado que utiliza **Algoritmos Genéticos (AGs)** para aprimorar os hiperparâmetros dos modelos e **Large Language Models (LLMs)** para gerar explicações médicas acionáveis.

A principal melhoria da Fase 2 é a otimização dos modelos de Machine Learning (ML) médicos através de um **Pipeline Otimizado por AG**, garantindo a máxima performance de diagnóstico e a interpretabilidade dos resultados para os profissionais de saúde. A maneira **recomendada** de execução continua sendo via **Docker**.

O Relatório técnico e a análise detalhada se encontram no **Notebook** (`diabetes_analysis.ipynb`).

O objetivo principal do **Notebook** (`diabetes_analysis.ipynb`) é:
1. **Executar a Otimização via Algoritmos Genéticos (AG)**: Otimizar os hiperparâmetros dos modelos de diagnóstico desenvolvidos na Fase 1 (foco em LightGBM).
2. **Executar o Modelo Otimizado**: Aplicar o conjunto de hiperparâmetros encontrado pelo AG ao modelo final.
3. **Demonstrar a Interpretabilidade via LLM**: Integrar uma LLM pré-treinada para traduzir o diagnóstico e as métricas de desempenho em linguagem natural, fornecendo *insights* acionáveis para a equipe médica.

---
## 🎞️ Vídeo de Demonstração
https://youtu.be/d62wPNKj1TU?si=My-jKimrJMECClqP

## 🚀 Como Executar com Docker (Recomendado)

### ✅ Pré-requisitos

- Ter o [Docker](https://docs.docker.com/engine/install/) instalado e funcionando corretamente.

### ▶️ Passos

1. Clone este repositório:
   ```bash
   git clone [https://github.com/vagnerbarbosa/tech-challenge-fase-2.git](https://github.com/vagnerbarbosa/tech-challenge-fase-2.git)
   cd tech-challenge-fase-2

2. Construa a imagem Docker:
   ```bash
   docker build -t diabetes-ml-otimizado .
   ```

3. Execute o container:
   ```bash
   docker run -p 8888:8888 diabetes-ml-otimizado
   ```

4. Acesse o Jupyter Notebook via navegador e abra o arquivo `diabetes_analysis.ipynb`:
   ```bash
   http://localhost:8888
   ```
> **Instrução no Notebook**: Execute todas as células sequencialmente para rodar a otimização por AG, treinar o modelo final e demonstrar a integração com a LLM para geração de diagnósticos interpretáveis.

---

## 💻 Execução Local (Alternativa)

### ✅ Pré-requisitos

- [Python 3.10 ou 3.11](https://www.python.org/downloads/)
- [Jupyter Notebook](https://jupyter.org/install)

### ▶️ Passos

1. Clone este repositório:
   ```bash
git clone https://github.com/vagnerbarbosa/tech-challenge-fase-2.git
   cd tech-challenge-fase-2
   ```

2. Instale as dependências. O arquivo `requirements.txt` deve conter as bibliotecas dos dois pipelines (`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `imblearn`, **`lightgbm`**, `kagglehub`, `shap`). O XGBoost foi substituído pelo **LightGBM** para garantir a compatibilidade e interpretabilidade via SHAP.
   ```bash
   pip install --only-binary=:all: -r requirements.txt
   ```

3. Execute o Jupyter Notebook e abra o arquivo:
   ```bash
   jupyter notebook diabetes_analysis.ipynb
   ```

---

## 📊 Dataset

- O dataset utilizado é o **Pima Indians Diabetes Dataset**, carregado diretamente do Kaggle via `kagglehub` para ambos os pipelines.
- Link da fonte: [Kaggle Diabetes Dataset](https://www.kaggle.com/datasets/mathchi/diabetes-data-set/data)

---

## 🛠️ Componentes da Fase 2

### 1. Otimização de Hiperparâmetros via Algoritmos Genéticos (AG)

Para a Fase 2, implementamos um Algoritmo Genético (AG) para otimizar os hiperparâmetros do modelo **LightGBM**, priorizando o **Recall** (Sensibilidade).

* **Codificação e Operadores:** Utilizamos vetores binários/inteiros como representação de genes para os hiperparâmetros relevantes, e implementamos operadores de seleção, cruzamento e mutação.
* **Função *Fitness***: Baseada nas métricas de desempenho dos modelos, como **Accuracy, Recall e F1-score**. Utilizamos uma combinação ponderada para priorizar a sensibilidade no diagnóstico (Recall).
$$\text{Fitness} = w_1 \times \text{Recall} + w_2 \times \text{F1-score}$$
* **Experimentos:** Realizamos experimentos comparando o desempenho dos modelos otimizados com os modelos originais.

### 2. Interpretabilidade e *Prompt Engineering* com LLMs

O modelo otimizado (LightGBM) é integrado a uma LLM pré-treinada para transformar os resultados técnicos em informações úteis para o médico.

* **Geração de Explicações:** A LLM gera explicações em linguagem natural dos diagnósticos produzidos pelos modelos.
* ***Insights* Acionáveis:** A LLM transforma dados numéricos e estatísticos em *insights* acionáveis para médicos.
* ***Prompt Engineering***: Implementamos técnicas de *prompt engineering* para obter respostas relevantes e adequadas ao contexto médico, garantindo a avaliação da qualidade das interpretações geradas. Esta é a base para a futura integração com dados textuais na Fase 3.

---

## 📈 Resultados Chave e Comparação Pós-AG

A otimização via Algoritmos Genéticos foi aplicada, e o *notebook* demonstra a interpretabilidade gerada, permitindo uma visão detalhada dos fatores de risco.

| Métrica de Desempenho | Pipeline A (Baseline - Melhor Modelo) | Pipeline Otimizado (AG) |
| :--- | :--- | :--- |
| **Melhor Modelo** | **Floresta Aleatória** | **LightGBM Otimizado (AG)** |
| **Recall Máximo** | **0.91** | 0.78 |
| **Melhor F1-score** | **0.72** | 0.68 |
| **Interpretabilidade**| Baixa (Métricas) | **Alta (LLM + Insights Acionáveis)** |

### 🔍 Destaques da Otimização por AG

- **Sensibilidade da Baseline:** O modelo **Floresta Aleatória** demonstrou a maior sensibilidade diagnóstica (Recall de 0.91) na fase Baseline, detectando 91% dos casos reais de diabetes.
- **Interpretabilidade Aumentada:** A principal vitória da Fase 2 é a camada de Interpretabilidade. O relatório LLM, baseado na análise SHAP do modelo LightGBM Otimizado, oferece *insights* acionáveis e específicos para o médico:
    1.  **Diabetes Pedigree Function (Histórico Familiar)**
    2.  **Interação Glicose x IMC** (Variável de Risco)
- **Foco na Tradução:** A LLM transforma a saída do modelo em informação imediatamente útil para o profissional de saúde, gerando explicações claras sobre o que está impulsionando o diagnóstico de risco.

---

## 👯 Contribuidores

- [Adriel Santos](https://github.com/AdrielCandido)
- [João Marcos](https://github.com/github)
- [Leticia Nepomucena](https://github.com/LeticiaNepomucena)
- [Lucas Silva](https://github.com/lucfsilva)
- [Vagner Barbosa](https://github.com/vagnerbarbosa)

---

## 🔒 Conformidade e Ética dos Dados
Este projeto utiliza apenas dados públicos