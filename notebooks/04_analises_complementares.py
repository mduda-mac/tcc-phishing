

# %%
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

AQUI = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
RAIZ = AQUI.parent
sys.path.insert(0, str(RAIZ / "src"))
from avaliacao import avaliar_predicao, contar_nos  # noqa: E402
from extracao_lexica import extrair_dataframe  # noqa: E402

BRUTOS = RAIZ / "dados" / "brutos"
PROC = RAIZ / "dados" / "processados"
TABELAS = RAIZ / "resultados" / "tabelas"
TABELAS.mkdir(parents=True, exist_ok=True)
SEMENTE = 42
saida: dict = {}


def carregar(nome: str) -> pd.DataFrame:
    return pd.read_csv(PROC / f"{nome}.csv", index_col=0)


X_tr, X_val, X_te = carregar("X_tr"), carregar("X_val"), carregar("X_teste")
y_tr, y_val, y_te = carregar("y_tr")["y"], carregar("y_val")["y"], carregar("y_teste")["y"]

hann = pd.read_csv(BRUTOS / "dataset_B_05_2020.csv")
phi = pd.read_csv(BRUTOS / "PhiUSIIL_Phishing_URL_Dataset.csv")
urls_te = hann.loc[X_te.index, "url"]
urls_val = phi.loc[X_val.index, "URL"]
print(X_tr.shape, X_val.shape, X_te.shape)

# %% Retreino idêntico ao notebook 02 (mesma semente e hiperparâmetros)
peso = float((y_tr == 0).sum() / (y_tr == 1).sum())
modelos = {
    "Regressão Logística": Pipeline([
        ("escala", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=2000,
                                   random_state=SEMENTE)),
    ]),
    "Árvore de Decisão": DecisionTreeClassifier(class_weight="balanced",
                                                random_state=SEMENTE),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                            random_state=SEMENTE, n_jobs=1),
    "XGBoost": XGBClassifier(n_estimators=100, scale_pos_weight=peso,
                             random_state=SEMENTE, n_jobs=1, tree_method="hist",
                             eval_metric="logloss"),
}
for nome, m in modelos.items():
    m.fit(X_tr, y_tr)

# Conferência: as matrizes de confusão devem ser idênticas às das Tabelas 2 e 4.
ref_val = pd.read_csv(TABELAS / "resultados_validacao.csv", index_col=0)
ref_te = pd.read_csv(TABELAS / "resultados_teste_independente.csv", index_col=0)
for nome, m in modelos.items():
    rv, rt = avaliar_predicao(m, X_val, y_val), avaliar_predicao(m, X_te, y_te)
    for k in ("vp", "fp", "vn", "fn"):
        assert rv[k] == ref_val.loc[nome, k], f"validação diverge: {nome} {k}"
        assert rt[k] == ref_te.loc[nome, k], f"teste diverge: {nome} {k}"
print("Matrizes de confusão idênticas às publicadas.")

pred_val = {n: m.predict(X_val) for n, m in modelos.items()}
pred_te = {n: m.predict(X_te) for n, m in modelos.items()}
prob_te = {n: m.predict_proba(X_te)[:, 1] for n, m in modelos.items()}

# %% Tabela 7 – McNemar (teste binomial exato bilateral)
nomes = list(modelos)
linhas = []
for i, a in enumerate(nomes):
    for b in nomes[i + 1:]:
        ca = pred_val[a] == y_val.values
        cb = pred_val[b] == y_val.values
        so_a, so_b = int((ca & ~cb).sum()), int((~ca & cb).sum())
        p = binomtest(so_a, so_a + so_b, 0.5).pvalue if so_a + so_b else 1.0
        linhas.append({"modelo_a": a, "modelo_b": b, "so_a": so_a, "so_b": so_b, "p": p})
mcnemar = pd.DataFrame(linhas)
mcnemar.to_csv(TABELAS / "mcnemar_validacao.csv", index=False)
print(mcnemar)

# %% Bootstrap pareado da diferença de F1 (XGBoost − Árvore de Decisão)
rng = np.random.default_rng(SEMENTE)
yv = y_val.values
dif = []
for _ in range(1000):
    idx = rng.integers(0, len(yv), len(yv))
    dif.append(f1_score(yv[idx], pred_val["XGBoost"][idx])
               - f1_score(yv[idx], pred_val["Árvore de Decisão"][idx]))
saida["ic95_dif_f1_xgb_menos_arvore"] = [float(np.percentile(dif, 2.5)),
                                         float(np.percentile(dif, 97.5))]
print(saida["ic95_dif_f1_xgb_menos_arvore"])


# %% Categorias de URL (caminho × esquema) e Tabela 8
def categoria(X: pd.DataFrame) -> pd.Series:
    cam = np.where(X["path_length"] > 0, "Com caminho", "Sem caminho")
    esq = np.where(X["scheme_is_https"] == 1, "https", "http")
    return pd.Series([f"{c}, {e}" for c, e in zip(cam, esq)], index=X.index)


cat_te = categoria(X_te)
leg = y_te.values == 0
tab8 = pd.DataFrame({"Total": cat_te[leg].value_counts()})
for n in nomes:
    tab8[n] = cat_te[leg & (pred_te[n] == 0)].value_counts()
tab8 = tab8.fillna(0).astype(int)
tab8.loc["Total"] = tab8.sum()
tab8.to_csv(TABELAS / "tabela8_legitimas_reconhecidas.csv")
print(tab8)

# Tabela 6 (contagens) e Tabela 5 (médias por classe), agora em código.
cat_val = categoria(X_val)
tab6 = pd.concat({
    "PhiUSIIL": pd.crosstab(y_val.map({0: "legítimas", 1: "phishing"}), cat_val),
    "H&Y": pd.crosstab(y_te.map({0: "legítimas", 1: "phishing"}), cat_te),
})
tab6.to_csv(TABELAS / "tabela6_caminho_esquema.csv")
cols5 = ["scheme_is_https", "path_length", "num_tokens_path", "url_length",
         "count_slash", "count_digits"]
tab5 = pd.DataFrame({
    "PhiUSIIL legítimas": X_val[y_val.values == 0][cols5].mean(),
    "PhiUSIIL phishing": X_val[y_val.values == 1][cols5].mean(),
    "H&Y legítimas": X_te[leg][cols5].mean(),
    "H&Y phishing": X_te[~leg][cols5].mean(),
}).round(3)
tab5.to_csv(TABELAS / "tabela5_medias_por_classe.csv")
print(tab6, tab5, sep="\n\n")


# %% Normalização de URLs (sobreposição residual e variações internas)
def normalizar(url: str) -> str:
    """Remove esquema, prefixo www. e barra final; host em minúsculas."""
    u = str(url).strip()
    if "://" not in u:
        u = "http://" + u
    try:
        p = urlsplit(u)
        host = (p.hostname or "").lower().removeprefix("www.")
        porta = f":{p.port}" if p.port else ""
        caminho = p.path.rstrip("/")
        return urlunsplit(("", host + porta, caminho, p.query, "")).lstrip("/")
    except ValueError:
        return u.lower()


norm_phi = set(phi.drop_duplicates("URL")["URL"].map(normalizar))
norm_te = urls_te.map(normalizar)
sobreposta = norm_te.isin(norm_phi)
saida["sobreposicao_normalizada"] = {
    "total": int(sobreposta.sum()),
    "legitimas": int((sobreposta & leg).sum()),
    "phishing": int((sobreposta & ~leg).sum()),
}
saida["variacoes_internas_teste"] = int(norm_te.duplicated(keep="first").sum())
print(saida["sobreposicao_normalizada"], saida["variacoes_internas_teste"])
# A contagem de variações internas e a de sobreposição dependem da
# definição de normalizar(). Versões anteriores deste script produziam
# 786 variações internas. Os valores atuais correspondem à função acima
# e são os reportados no texto.

sem_sobrep = ~sobreposta.values
saida["metricas_sem_sobreposicao"] = {
    n: {"f1": float(f1_score(y_te.values[sem_sobrep], pred_te[n][sem_sobrep])),
        "vn": int(((pred_te[n] == 0) & leg & sem_sobrep).sum())}
    for n in nomes
}
p_sub = y_te.values[sem_sobrep].mean()
saida["f1_trivial_sem_sobreposicao"] = float(2 * p_sub / (1 + p_sub))

# %% E3 – Teste deduplicado por URL normalizada (parecer, comentário sobre a 3.2)
unica = (~norm_te.duplicated(keep="first")).values
mask = unica & sem_sobrep
p_dd = y_te.values[mask].mean()
saida["E3_teste_dedup_normalizado"] = {
    "n": int(mask.sum()), "legitimas": int((leg & mask).sum()),
    "f1_trivial": float(2 * p_dd / (1 + p_dd)),
    "modelos": {n: {"f1": float(f1_score(y_te.values[mask], pred_te[n][mask])),
                    "auc": float(roc_auc_score(y_te.values[mask], prob_te[n][mask])),
                    "vn": int(((pred_te[n] == 0) & leg & mask).sum())}
                for n in nomes},
}


# %% E1 – Barra final isolada normalizada no teste
def sem_barra_isolada(url: str) -> str:
    """Remove a barra quando ela é o único caminho (https://x.com/ -> https://x.com)."""
    u = str(url).strip()
    try:
        p = urlsplit(u if "://" in u else "http://" + u)
    except ValueError:
        return u
    if p.path == "/" and not p.query and not p.fragment:
        return u[:-1] if u.endswith("/") else u
    return u


urls_te_nb = urls_te.map(sem_barra_isolada)
alteradas = int((urls_te_nb != urls_te).sum())
X_te_nb = extrair_dataframe(urls_te_nb)[X_te.columns]
cat_nb = categoria(X_te_nb)
saida["E1_barra_final"] = {
    "urls_alteradas": alteradas,
    "legitimas_alteradas": int(((urls_te_nb != urls_te).values & leg).sum()),
    "legitimas_sem_caminho_https_apos": int(((cat_nb == "Sem caminho, https").values & leg).sum()),
    "modelos": {},
}
for n, m in modelos.items():
    r = avaliar_predicao(m, X_te_nb, y_te)
    saida["E1_barra_final"]["modelos"][n] = {
        "f1": float(r["f1"]), "auc": float(r["roc_auc"]), "vn": r["vn"],
        "vn_original": int(ref_te.loc[n, "vn"]),
    }
print(json.dumps(saida["E1_barra_final"], indent=2, ensure_ascii=False))

# %% E2 – Valores distintos de probabilidade no teste
saida["E2_valores_distintos_prob"] = {
    n: {"distintos": int(np.unique(np.round(p, 6)).size),
        "pct_no_valor_mais_frequente": float(
            pd.Series(np.round(p, 6)).value_counts(normalize=True).iloc[0])}
    for n, p in prob_te.items()
}
print(saida["E2_valores_distintos_prob"])

# %% E4 – Latência de uma URL isolada (inclui o custo fixo de cada chamada)
rng = np.random.default_rng(SEMENTE)
amostra = rng.choice(len(X_val), size=1000, replace=False)
linhas_isoladas = [X_val.iloc[[i]] for i in amostra]
saida["E4_latencia_url_isolada_us"] = {}
for n, m in modelos.items():
    for linha in linhas_isoladas[:20]:  # aquecimento
        m.predict(linha)
    tempos = []
    for linha in linhas_isoladas:
        t0 = time.perf_counter()
        m.predict(linha)
        tempos.append((time.perf_counter() - t0) * 1e6)
    saida["E4_latencia_url_isolada_us"][n] = {
        "mediana": float(np.median(tempos)), "p95": float(np.percentile(tempos, 95))}
lat = saida["E4_latencia_url_isolada_us"]
saida["E4_ordens_grandeza_xgb_vs_arvore"] = float(
    np.log10(lat["XGBoost"]["mediana"] / lat["Árvore de Decisão"]["mediana"]))
print(json.dumps(lat, indent=2, ensure_ascii=False))

# %% E5 – Árvore de Decisão podada
saida["E5_arvore_podada"] = {}
for prof in (5, 6, 7, 8):
    arv = DecisionTreeClassifier(class_weight="balanced", max_depth=prof,
                                 random_state=SEMENTE).fit(X_tr, y_tr)
    rv, rt = avaliar_predicao(arv, X_val, y_val), avaliar_predicao(arv, X_te, y_te)
    nos = contar_nos(arv)
    saida["E5_arvore_podada"][prof] = {
        "f1_validacao": float(rv["f1"]), "f1_teste": float(rt["f1"]),
        "auc_teste": float(rt["roc_auc"]), "vn_teste": rt["vn"],
        "nos": int(nos.get("n_nos_total", 0)), "folhas": int(nos.get("n_folhas_total", 0)),
    }
print(json.dumps(saida["E5_arvore_podada"], indent=2, ensure_ascii=False))

# %% Salvar tudo
with open(TABELAS / "analises_complementares.json", "w", encoding="utf-8") as f:
    json.dump(saida, f, indent=2, ensure_ascii=False)
print("Resultados salvos em resultados/tabelas/analises_complementares.json")