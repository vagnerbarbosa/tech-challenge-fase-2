from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import recall_score, make_scorer
import lightgbm as lgb
import numpy as np
import random
from deap import base, creator, tools, algorithms



def obter_modelos_baseline():
    """
    Cria e retorna um dicionário com modelos de classificação BASELINE.
    """
    return {
        'Regressão Logística': LogisticRegression(class_weight='balanced', random_state=42),
        'Árvore de Decisão': DecisionTreeClassifier(class_weight='balanced', random_state=42),
        'Floresta Aleatória': RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42),
        'LightGBM': lgb.LGBMClassifier(class_weight='balanced', random_state=42, verbose=-1)
    }

def otimizar_random_forest_ag(X_treino, y_treino, nome_modelo='Random Forest Otimizado (AG)', geracoes=30, populacao=50):
    """
    Otimiza hiperparâmetros do Random Forest + LIMIAR usando Algoritmo Genético (DEAP).
    Foco em maximizar o Recall (Sensibilidade).
    
    COMPATIBILIDADE: Mantida para backward compatibility. 
    Use executar_experimentos_ag() para múltiplos experimentos.
    """
    modelo, nome, limiar, recall_cv = otimizar_random_forest_ag_customizado(
        X_treino, y_treino, nome_modelo, geracoes, populacao
    )
    return modelo, nome, limiar

def treinar_modelos(modelos, X_treino, y_treino):
    for modelo in modelos.values():
        modelo.fit(X_treino, y_treino)
    return modelos

def executar_experimentos_ag(X_treino, y_treino, experimentos_configs):
    """
    Executa múltiplos experimentos com diferentes configurações do AG.
    
    Args:
        X_treino: Dados de treino
        y_treino: Labels de treino
        experimentos_configs: Lista de dicionários com configurações de cada experimento
            Exemplo: [
                {'nome': 'Exp1', 'geracoes': 20, 'populacao': 30, 'cxpb': 0.7, 'mutpb': 0.2},
                {'nome': 'Exp2', 'geracoes': 30, 'populacao': 50, 'cxpb': 0.8, 'mutpb': 0.3},
                ...
            ]
    
    Returns:
        Lista de dicionários com resultados de cada experimento
    """
    resultados = []
    
    print("="*80)
    print("🧬 INICIANDO MÚLTIPLOS EXPERIMENTOS COM ALGORITMO GENÉTICO")
    print("="*80)
    
    for i, config in enumerate(experimentos_configs, 1):
        print(f"\n{'='*80}")
        print(f"📊 EXPERIMENTO {i}/{len(experimentos_configs)}: {config['nome']}")
        print(f"{'='*80}")
        print(f"Configurações:")
        print(f"  - Gerações: {config['geracoes']}")
        print(f"  - População: {config['populacao']}")
        print(f"  - Taxa de Cruzamento (cxpb): {config.get('cxpb', 0.7)}")
        print(f"  - Taxa de Mutação (mutpb): {config.get('mutpb', 0.2)}")
        print(f"  - Torneio: {config.get('tournsize', 3)}")
        print(f"  - Probabilidade de Mutação Individual (indpb): {config.get('indpb', 0.3)}")
        
        # Executar otimização com configuração customizada
        modelo, nome, limiar, recall_cv = otimizar_random_forest_ag_customizado(
            X_treino, 
            y_treino,
            nome_modelo=config['nome'],
            geracoes=config['geracoes'],
            populacao=config['populacao'],
            cxpb=config.get('cxpb', 0.7),
            mutpb=config.get('mutpb', 0.2),
            tournsize=config.get('tournsize', 3),
            indpb=config.get('indpb', 0.3)
        )
        
        resultado = {
            'experimento': config['nome'],
            'configuracao': config,
            'modelo': modelo,
            'limiar': limiar,
            'recall_cv': recall_cv
        }
        resultados.append(resultado)
        
        print(f"\n✅ {config['nome']} Concluído - Recall CV: {recall_cv:.4f} | Limiar: {limiar:.4f}")
    
    print(f"\n{'='*80}")
    print("📊 RESUMO DE TODOS OS EXPERIMENTOS")
    print(f"{'='*80}")
    print(f"{'Experimento':<25} {'Recall (CV)':<15} {'Limiar':<10} {'Gerações':<12} {'População':<12}")
    print("-"*80)
    for r in resultados:
        print(f"{r['experimento']:<25} {r['recall_cv']:<15.4f} {r['limiar']:<10.4f} "
              f"{r['configuracao']['geracoes']:<12} {r['configuracao']['populacao']:<12}")
    
    # Identificar o melhor experimento
    melhor_exp = max(resultados, key=lambda x: x['recall_cv'])
    print(f"\n🏆 MELHOR EXPERIMENTO: {melhor_exp['experimento']} com Recall CV = {melhor_exp['recall_cv']:.4f}")
    print(f"{'='*80}\n")
    
    return resultados, melhor_exp

def otimizar_random_forest_ag_customizado(X_treino, y_treino, nome_modelo='Random Forest Otimizado (AG)', 
                                          geracoes=30, populacao=50, cxpb=0.7, mutpb=0.2, 
                                          tournsize=3, indpb=0.3):
    """
    Versão customizável da otimização AG que aceita parâmetros do algoritmo genético.
    
    Args:
        cxpb: Probabilidade de cruzamento
        mutpb: Probabilidade de mutação
        tournsize: Tamanho do torneio para seleção
        indpb: Probabilidade de mutação individual de cada gene
    
    Returns:
        modelo, nome_modelo, limiar_otimizado, recall_cv
    """
    print(f"\n--- 🧬 Iniciando Otimização via AG: {nome_modelo} ---")
    print(f"Parâmetros AG: Gerações={geracoes}, População={populacao}, Cruzamento={cxpb}, Mutação={mutpb}")

    if hasattr(creator, "FitnessMax"):
        del creator.FitnessMax
    if hasattr(creator, "Individual"):
        del creator.Individual

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    toolbox.register("attr_n_estimators", random.randint, 100, 500)
    toolbox.register("attr_max_depth", random.randint, 5, 30)
    toolbox.register("attr_min_samples_split", random.randint, 2, 20)
    toolbox.register("attr_min_samples_leaf", random.randint, 1, 10)
    toolbox.register("attr_max_features", random.uniform, 0.3, 1.0)
    toolbox.register("attr_bootstrap", random.choice, [True, False])
    toolbox.register("attr_threshold", random.uniform, 0.2, 0.5)

    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_n_estimators, toolbox.attr_max_depth,
                      toolbox.attr_min_samples_split, toolbox.attr_min_samples_leaf,
                      toolbox.attr_max_features, toolbox.attr_bootstrap, toolbox.attr_threshold), n=1)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def avaliar_individuo(individual):
        n_estimators = max(50, min(500, int(individual[0])))
        max_depth = max(5, min(30, int(individual[1])))
        min_samples_split = max(2, min(20, int(individual[2])))
        min_samples_leaf = max(1, min(10, int(individual[3])))
        max_features = max(0.3, min(1.0, individual[4]))
        bootstrap = bool(individual[5])
        threshold = max(0.2, min(0.5, individual[6]))

        try:
            modelo = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                bootstrap=bootstrap,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )

            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            recalls = []

            for train_idx, val_idx in skf.split(X_treino, y_treino):
                X_train_fold = X_treino.iloc[train_idx] if hasattr(X_treino, 'iloc') else X_treino[train_idx]
                y_train_fold = y_treino.iloc[train_idx] if hasattr(y_treino, 'iloc') else y_treino[train_idx]
                X_val_fold = X_treino.iloc[val_idx] if hasattr(X_treino, 'iloc') else X_treino[val_idx]
                y_val_fold = y_treino.iloc[val_idx] if hasattr(y_treino, 'iloc') else y_treino[val_idx]

                modelo.fit(X_train_fold, y_train_fold)
                y_proba = modelo.predict_proba(X_val_fold)[:, 1]
                y_pred = (y_proba >= threshold).astype(int)
                recall = recall_score(y_val_fold, y_pred)
                recalls.append(recall)

            recall_medio = np.mean(recalls)
            return (recall_medio,)
        except Exception as e:
            return (0.0,)

    toolbox.register("evaluate", avaliar_individuo)
    toolbox.register("mate", tools.cxTwoPoint)

    def mutacao_customizada(individual, indpb_param):
        for i in range(len(individual)):
            if random.random() < indpb_param:
                if i == 0:
                    individual[i] += random.gauss(0, 50)
                elif i == 1:
                    individual[i] += random.gauss(0, 3)
                elif i == 2:
                    individual[i] += random.gauss(0, 2)
                elif i == 3:
                    individual[i] += random.gauss(0, 1)
                elif i == 4:
                    individual[i] += random.gauss(0, 0.1)
                elif i == 5:
                    individual[i] = random.choice([True, False])
                elif i == 6:
                    individual[i] += random.gauss(0, 0.05)
        return (individual,)

    toolbox.register("mutate", mutacao_customizada, indpb_param=indpb)
    toolbox.register("select", tools.selTournament, tournsize=tournsize)

    random.seed(42)
    np.random.seed(42)

    pop = toolbox.population(n=populacao)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    stats.register("min", np.min)

    print("\n🔬 Evolução do AG:")
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb,
                                     ngen=geracoes, stats=stats,
                                     halloffame=hof, verbose=True)

    melhor_individuo = hof[0]

    params_otimizados = {
        'n_estimators': int(melhor_individuo[0]),
        'max_depth': int(melhor_individuo[1]),
        'min_samples_split': int(melhor_individuo[2]),
        'min_samples_leaf': int(melhor_individuo[3]),
        'max_features': melhor_individuo[4],
        'bootstrap': bool(melhor_individuo[5]),
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    }

    limiar_otimizado = max(0.2, min(0.5, melhor_individuo[6]))
    recall_cv = melhor_individuo.fitness.values[0]

    print(f"\n✅ Melhor Recall (CV): {recall_cv:.4f}")
    print(f"🎯 Limiar Otimizado: {limiar_otimizado:.4f}")
    print(f"🧬 Hiperparâmetros Otimizados:")
    for param, valor in params_otimizados.items():
        if param not in ['class_weight', 'random_state', 'n_jobs']:
            print(f"   {param}: {valor:.4f}" if isinstance(valor, float) else f"   {param}: {valor}")

    melhor_modelo = RandomForestClassifier(**params_otimizados)
    melhor_modelo.fit(X_treino, y_treino)

    print(f"\n--- ✅ Otimização AG Concluída ---")

    return melhor_modelo, nome_modelo, limiar_otimizado, recall_cv