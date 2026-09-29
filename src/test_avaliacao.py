"""
Testes do módulo de avaliação de desempenho preditivo, custo computacional e
complexidade estrutural dos modelos.

Execução:
    pytest src/ -v

Os quatro tipos de modelo usados no estudo (linear, árvore isolada, floresta
e XGBoost) expõem a complexidade estrutural de formas distintas, por isso
`contar_nos` é verificada separadamente para cada um.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from avaliacao import (
    _profundidade_xgboost,
    avaliar_predicao,
    contar_nos,
    ic_bootstrap_f1,
    ordens_de_magnitude,
)

SEMENTE = 42


@pytest.fixture(scope="module")
def dados():
    X, y = make_classification(
        n_samples=300,
        n_features=8,
        n_informative=5,
        weights=[0.6, 0.4],
        random_state=SEMENTE,
    )
    X = pd.DataFrame(X, columns=[f"a{i}" for i in range(X.shape[1])])
    y = pd.Series(y, name="y")
    return X, y


# ---------------------------------------------------------------------------
# avaliar_predicao
# ---------------------------------------------------------------------------

def test_avaliar_predicao_usa_classe_positiva_1(dados):
    X, y = dados
    modelo = LogisticRegression(max_iter=1000, random_state=SEMENTE).fit(X, y)
    resultado = avaliar_predicao(modelo, X, y)
    y_pred = modelo.predict(X)

    assert resultado["f1"] == pytest.approx(f1_score(y, y_pred, pos_label=1))
    assert resultado["precisao"] == pytest.approx(precision_score(y, y_pred, pos_label=1))
    assert resultado["recall"] == pytest.approx(recall_score(y, y_pred, pos_label=1))
    assert resultado["vp"] + resultado["fn"] == int((y == 1).sum())
    assert resultado["vn"] + resultado["fp"] == int((y == 0).sum())


def test_avaliar_predicao_retorna_todas_as_metricas(dados):
    X, y = dados
    modelo = DecisionTreeClassifier(random_state=SEMENTE).fit(X, y)
    resultado = avaliar_predicao(modelo, X, y)
    assert set(resultado) == {
        "f1", "precisao", "recall", "roc_auc", "pr_auc", "vp", "fp", "vn", "fn",
    }


# ---------------------------------------------------------------------------
# contar_nos - um caso por tipo de modelo
# ---------------------------------------------------------------------------

def test_contar_nos_modelo_linear(dados):
    X, y = dados
    modelo = LogisticRegression(max_iter=1000, random_state=SEMENTE).fit(X, y)
    info = contar_nos(modelo)
    assert info["tipo_complexidade"] == "coeficientes"
    assert info["n_arvores"] == 0
    assert info["n_nos_total"] == X.shape[1]
    assert info["n_folhas_total"] == 0
    assert info["profundidade_max"] == 0


def test_contar_nos_arvore_de_decisao(dados):
    X, y = dados
    modelo = DecisionTreeClassifier(random_state=SEMENTE).fit(X, y)
    info = contar_nos(modelo)
    assert info["tipo_complexidade"] == "nos_arvore"
    assert info["n_arvores"] == 1
    assert info["n_nos_total"] == modelo.tree_.node_count
    assert info["n_folhas_total"] == modelo.get_n_leaves()
    assert info["profundidade_max"] == modelo.get_depth()


def test_contar_nos_random_forest(dados):
    X, y = dados
    modelo = RandomForestClassifier(n_estimators=5, random_state=SEMENTE, n_jobs=1).fit(X, y)
    info = contar_nos(modelo)
    assert info["tipo_complexidade"] == "nos_arvore"
    assert info["n_arvores"] == 5
    assert info["n_nos_total"] == sum(a.tree_.node_count for a in modelo.estimators_)
    assert info["n_folhas_total"] == sum(a.get_n_leaves() for a in modelo.estimators_)
    assert info["profundidade_max"] == max(a.get_depth() for a in modelo.estimators_)


def test_contar_nos_xgboost(dados):
    X, y = dados
    modelo = XGBClassifier(
        n_estimators=5, max_depth=3, random_state=SEMENTE, n_jobs=1,
        tree_method="hist", eval_metric="logloss",
    ).fit(X, y)
    info = contar_nos(modelo)
    assert info["tipo_complexidade"] == "nos_arvore"
    assert info["n_arvores"] == 5
    assert info["profundidade_max"] <= 3


# ---------------------------------------------------------------------------
# _profundidade_xgboost
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("max_depth", [1, 2, 4])
def test_profundidade_xgboost_respeita_max_depth_configurado(dados, max_depth):
    X, y = dados
    modelo = XGBClassifier(
        n_estimators=10, max_depth=max_depth, random_state=SEMENTE, n_jobs=1,
        tree_method="hist", eval_metric="logloss",
    ).fit(X, y)
    df = modelo.get_booster().trees_to_dataframe()
    profundidade = _profundidade_xgboost(df)
    assert 0 <= profundidade <= max_depth


def test_profundidade_xgboost_stump_tem_profundidade_um(dados):
    X, y = dados
    modelo = XGBClassifier(
        n_estimators=10, max_depth=1, random_state=SEMENTE, n_jobs=1,
        tree_method="hist", eval_metric="logloss",
    ).fit(X, y)
    df = modelo.get_booster().trees_to_dataframe()
    assert _profundidade_xgboost(df) == 1


# ---------------------------------------------------------------------------
# ordens_de_magnitude
# ---------------------------------------------------------------------------

def test_ordens_de_magnitude_valores_iguais():
    assert ordens_de_magnitude(5.0, 5.0) == pytest.approx(0.0)


def test_ordens_de_magnitude_uma_ordem():
    assert ordens_de_magnitude(1000.0, 100.0) == pytest.approx(1.0)


def test_ordens_de_magnitude_valor_nao_positivo_retorna_nan():
    assert np.isnan(ordens_de_magnitude(0.0, 10.0))
    assert np.isnan(ordens_de_magnitude(10.0, -1.0))


# ---------------------------------------------------------------------------
# ic_bootstrap_f1
# ---------------------------------------------------------------------------

def test_ic_bootstrap_f1_contem_f1_observado(dados):
    X, y = dados
    modelo = LogisticRegression(max_iter=1000, random_state=SEMENTE).fit(X, y)
    ic = ic_bootstrap_f1(modelo, X, y, n_reamostras=200, semente=SEMENTE)
    assert ic["ic95_inferior"] <= ic["f1_observado"] <= ic["ic95_superior"]
    assert 0.0 <= ic["ic95_inferior"] <= ic["ic95_superior"] <= 1.0
    assert ic["n_reamostras"] == 200


def test_ic_bootstrap_f1_e_deterministico_para_mesma_semente(dados):
    X, y = dados
    modelo = LogisticRegression(max_iter=1000, random_state=SEMENTE).fit(X, y)
    ic_a = ic_bootstrap_f1(modelo, X, y, n_reamostras=200, semente=7)
    ic_b = ic_bootstrap_f1(modelo, X, y, n_reamostras=200, semente=7)
    assert ic_a == ic_b


def test_ic_bootstrap_f1_nao_retreina_o_modelo(dados):
    X, y = dados
    modelo = DecisionTreeClassifier(random_state=SEMENTE).fit(X, y)
    n_nos_antes = modelo.tree_.node_count
    ic_bootstrap_f1(modelo, X, y, n_reamostras=50, semente=SEMENTE)
    assert modelo.tree_.node_count == n_nos_antes
