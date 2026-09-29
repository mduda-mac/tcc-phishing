"""
Funções de avaliação de desempenho preditivo, custo computacional e
complexidade estrutural dos modelos.

As três hipóteses do estudo exigem medidas distintas:

    H1  desempenho preditivo   -> avaliar_predicao
    H2  custo e complexidade   -> medir_tempo_inferencia, contar_nos
    H3  generalização externa  -> avaliar_predicao aplicado ao conjunto
                                  independente

Reunir essas medições em um módulo testado evita que cada notebook
reimplemente a mesma conta de forma ligeiramente diferente, o que tornaria
os números não comparáveis entre si.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Métricas reportadas, em ordem canônica.
METRICAS: tuple[str, ...] = (
    "f1", "precisao", "recall", "roc_auc", "pr_auc",
    "vp", "fp", "vn", "fn",
)


# ---------------------------------------------------------------------------
# H1 e H3 - desempenho preditivo
# ---------------------------------------------------------------------------

def avaliar_predicao(modelo: Any, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Calcula as métricas de classificação com phishing como classe positiva.

    A classe positiva é 1 (phishing). F1, precisão e recall são calculados
    sobre essa classe, não em média macro: o interesse do estudo é a detecção
    de phishing, e uma média entre as duas classes mascararia o desempenho
    justamente na classe que importa.

    PR-AUC é incluída porque, sob desbalanceamento, ela é mais informativa que
    ROC-AUC a respeito do comportamento na classe minoritária.
    """
    y_pred = modelo.predict(X)
    y_score = _obter_escores(modelo, X)

    vn, fp, fn, vp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()

    return {
        "f1": f1_score(y, y_pred, pos_label=1),
        "precisao": precision_score(y, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y, y_pred, pos_label=1),
        "roc_auc": roc_auc_score(y, y_score),
        "pr_auc": average_precision_score(y, y_score),
        "vp": int(vp),
        "fp": int(fp),
        "vn": int(vn),
        "fn": int(fn),
    }


def _obter_escores(modelo: Any, X: pd.DataFrame) -> np.ndarray:
    """Escores contínuos da classe positiva, para métricas baseadas em ranking."""
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(X)[:, 1]
    return modelo.decision_function(X)


# ---------------------------------------------------------------------------
# H2 - custo computacional
# ---------------------------------------------------------------------------

def medir_tempo_inferencia(
    modelo: Any,
    X: pd.DataFrame,
    n_repeticoes: int = 7,
    n_aquecimento: int = 1,
) -> dict[str, float]:
    """Mede o tempo de inferência por URL, em microssegundos.

    A medição é repetida e reporta-se a mediana, não a média: execuções
    isoladas sofrem interferência de outros processos do sistema operacional,
    e a mediana é robusta a esses picos. As repetições de aquecimento
    descartam o custo de alocação inicial de estruturas internas.

    O desvio padrão acompanha o resultado para que a comparação entre modelos
    possa ser interpretada à luz da variabilidade da própria medição.
    """
    for _ in range(n_aquecimento):
        modelo.predict(X)

    tempos = []
    for _ in range(n_repeticoes):
        inicio = time.perf_counter()
        modelo.predict(X)
        tempos.append(time.perf_counter() - inicio)

    tempos = np.asarray(tempos)
    por_url_us = tempos / len(X) * 1e6

    return {
        "tempo_mediana_us": float(np.median(por_url_us)),
        "tempo_desvio_us": float(np.std(por_url_us)),
        "tempo_lote_mediana_s": float(np.median(tempos)),
        "n_amostras": len(X),
        "n_repeticoes": n_repeticoes,
    }


def contar_nos(modelo: Any) -> dict[str, Any]:
    """Conta nós e profundidade das estruturas de árvore do modelo.

    Modelos lineares não possuem árvores; para eles reporta-se o número de
    coeficientes, que é a medida análoga de tamanho do modelo. O campo
    'tipo_complexidade' explicita qual grandeza está sendo contada, de modo
    que a tabela final não compare unidades diferentes sem aviso.
    """
    estimador = _desempacotar(modelo)

    # Árvore de decisão isolada
    if hasattr(estimador, "tree_"):
        return {
            "tipo_complexidade": "nos_arvore",
            "n_arvores": 1,
            "n_nos_total": int(estimador.tree_.node_count),
            "n_folhas_total": int(estimador.get_n_leaves()),
            "profundidade_max": int(estimador.get_depth()),
        }

    # Floresta de árvores (Random Forest e similares)
    if hasattr(estimador, "estimators_") and len(getattr(estimador, "estimators_", [])):
        arvores = estimador.estimators_
        if hasattr(arvores[0], "tree_"):
            return {
                "tipo_complexidade": "nos_arvore",
                "n_arvores": len(arvores),
                "n_nos_total": int(sum(a.tree_.node_count for a in arvores)),
                "n_folhas_total": int(sum(a.get_n_leaves() for a in arvores)),
                "profundidade_max": int(max(a.get_depth() for a in arvores)),
            }

    # XGBoost
    if hasattr(estimador, "get_booster"):
        df = estimador.get_booster().trees_to_dataframe()
        folhas = df["Feature"] == "Leaf"
        return {
            "tipo_complexidade": "nos_arvore",
            "n_arvores": int(df["Tree"].nunique()),
            "n_nos_total": int(len(df)),
            "n_folhas_total": int(folhas.sum()),
            "profundidade_max": _profundidade_xgboost(df),
        }

    # Modelo linear
    if hasattr(estimador, "coef_"):
        return {
            "tipo_complexidade": "coeficientes",
            "n_arvores": 0,
            "n_nos_total": int(np.asarray(estimador.coef_).size),
            "n_folhas_total": 0,
            "profundidade_max": 0,
        }

    raise TypeError(
        f"Não foi possível medir a complexidade de {type(estimador).__name__}."
    )


def _profundidade_xgboost(df: pd.DataFrame) -> int:
    """Profundidade máxima entre as árvores de um booster XGBoost.

    A representação tabular do XGBoost não expõe a profundidade diretamente:
    a coluna 'Node' é um identificador, não um nível. A profundidade é obtida
    percorrendo os vínculos pai-filho das colunas 'Yes' e 'No' a partir da
    raiz de cada árvore.
    """
    profundidade_max = 0

    for _, arvore in df.groupby("Tree"):
        filhos: dict[str, list[str]] = {}
        for _, linha in arvore.iterrows():
            if linha["Feature"] != "Leaf":
                filhos[linha["ID"]] = [linha["Yes"], linha["No"]]

        raiz = arvore["ID"].iloc[0]
        pilha = [(raiz, 0)]
        while pilha:
            no, nivel = pilha.pop()
            profundidade_max = max(profundidade_max, nivel)
            for filho in filhos.get(no, []):
                pilha.append((filho, nivel + 1))

    return int(profundidade_max)


def _desempacotar(modelo: Any) -> Any:
    """Retorna o estimador final, atravessando um Pipeline se houver."""
    return modelo[-1] if hasattr(modelo, "steps") else modelo


# ---------------------------------------------------------------------------
# Consolidação
# ---------------------------------------------------------------------------

def avaliar_completo(
    nome: str,
    modelo: Any,
    X: pd.DataFrame,
    y: pd.Series,
    medir_custo: bool = True,
) -> dict[str, Any]:
    """Reúne desempenho, custo e complexidade em um único registro."""
    registro: dict[str, Any] = {"modelo": nome}
    registro.update(avaliar_predicao(modelo, X, y))
    if medir_custo:
        registro.update(medir_tempo_inferencia(modelo, X))
        registro.update(contar_nos(modelo))
    return registro


def ic_bootstrap_f1(
    modelo: Any,
    X: pd.DataFrame,
    y: pd.Series,
    n_reamostras: int = 1000,
    semente: int = 42,
) -> dict[str, float]:
    """IC 95% do F1 da classe positiva por bootstrap sobre as instâncias de avaliação.

    O modelo já deve estar ajustado: apenas as previsões são recalculadas em
    cada reamostragem com reposição de (X, y), nunca o treino. O intervalo é
    o percentil 2,5-97,5 da distribuição de F1 resultante.
    """
    rng = np.random.default_rng(semente)
    y_true = np.asarray(y)
    y_pred = np.asarray(modelo.predict(X))
    n = len(y_true)

    f1s = np.empty(n_reamostras)
    for i in range(n_reamostras):
        indices = rng.integers(0, n, size=n)
        f1s[i] = f1_score(y_true[indices], y_pred[indices], pos_label=1, zero_division=0)

    return {
        "f1_observado": float(f1_score(y_true, y_pred, pos_label=1)),
        "ic95_inferior": float(np.percentile(f1s, 2.5)),
        "ic95_superior": float(np.percentile(f1s, 97.5)),
        "n_reamostras": n_reamostras,
    }


def ordens_de_magnitude(valor_a: float, valor_b: float) -> float:
    """Diferença em ordens de magnitude entre dois valores positivos.

    Usada na verificação da H2, que exige aumento de pelo menos uma ordem de
    magnitude em tempo de inferência e número de nós.
    """
    if valor_a <= 0 or valor_b <= 0:
        return float("nan")
    return float(np.log10(valor_a / valor_b))
