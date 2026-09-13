"""
Testes do módulo de extração de atributos léxicos.

Execução:
    pytest src/ -v

A função de extração é aplicada aos dois conjuntos de dados do estudo. Um erro
silencioso aqui comprometeria simultaneamente o treino e o teste de
generalização, razão pela qual cada grupo da taxonomia é verificado
individualmente.
"""

import math

import pandas as pd
import pytest

from extracao_lexica import (
    COLUNAS,
    entropia_shannon,
    extrair_atributos,
    extrair_dataframe,
)


# ---------------------------------------------------------------------------
# Contrato de saída
# ---------------------------------------------------------------------------

def test_retorna_todos_os_atributos():
    a = extrair_atributos("http://exemplo.com/")
    assert set(a) == set(COLUNAS)
    assert len(COLUNAS) == 37


def test_entrada_invalida_nao_quebra():
    for entrada in ["", "   ", None, float("nan"), 123]:
        a = extrair_atributos(entrada)
        assert set(a) == set(COLUNAS)
        assert all(v == 0.0 for v in a.values())


def test_url_malformada_nao_quebra():
    a = extrair_atributos("http://[::malformado/caminho")
    assert set(a) == set(COLUNAS)


# ---------------------------------------------------------------------------
# Grupo 1 - comprimento
# ---------------------------------------------------------------------------

def test_comprimentos():
    url = "https://sub.exemplo.com.br/pasta/pag?x=1&y=2"
    a = extrair_atributos(url)
    assert a["url_length"] == len(url)
    assert a["hostname_length"] == len("sub.exemplo.com.br")
    assert a["path_length"] == len("/pasta/pag")
    assert a["query_length"] == len("x=1&y=2")


def test_maior_token():
    a = extrair_atributos("http://abc.exemplomuitolongo.com/x")
    assert a["longest_token_length"] == len("exemplomuitolongo")


# ---------------------------------------------------------------------------
# Grupo 2 - contagens
# ---------------------------------------------------------------------------

def test_contagem_caracteres():
    a = extrair_atributos("http://a-b.exemplo.com/p_1?x=2&y=3@z%20")
    assert a["count_dots"] == 2
    assert a["count_hyphens"] == 1
    assert a["count_underscore"] == 1
    assert a["count_slash"] == 3
    assert a["count_question"] == 1
    assert a["count_equals"] == 2
    assert a["count_ampersand"] == 1
    assert a["count_at"] == 1
    assert a["count_percent"] == 1
    assert a["count_digits"] == 5


# ---------------------------------------------------------------------------
# Grupo 3 - composição estatística
# ---------------------------------------------------------------------------

def test_razoes_sao_proporcoes_validas():
    a = extrair_atributos("https://exemplo123.com/a1b2?c=3")
    for chave in ("digit_ratio", "letter_ratio", "special_char_ratio"):
        assert 0.0 <= a[chave] <= 1.0
    soma = a["digit_ratio"] + a["letter_ratio"] + a["special_char_ratio"]
    assert soma == pytest.approx(1.0)


def test_entropia_casos_conhecidos():
    assert entropia_shannon("") == 0.0
    assert entropia_shannon("aaaa") == 0.0
    assert entropia_shannon("ab") == pytest.approx(1.0)
    assert entropia_shannon("abcd") == pytest.approx(2.0)


def test_entropia_distingue_aleatoriedade():
    regular = extrair_atributos("http://banco.com/")
    aleatoria = extrair_atributos("http://x7q2mz9vk4bt.com/")
    assert aleatoria["shannon_entropy_hostname"] > regular["shannon_entropy_hostname"]


# ---------------------------------------------------------------------------
# Grupo 4 - estrutura do host
# ---------------------------------------------------------------------------

def test_dominio_registravel_com_sufixo_composto():
    """O sufixo .com.br tem duas partes; um split ingênuo por ponto erraria."""
    a = extrair_atributos("http://loja.exemplo.com.br/")
    assert a["domain_length"] == len("exemplo")
    assert a["tld_length"] == len("com.br")
    assert a["num_subdomains"] == 1


def test_contagem_subdominios():
    assert extrair_atributos("http://exemplo.com/")["num_subdomains"] == 0
    assert extrair_atributos("http://www.exemplo.com/")["num_subdomains"] == 1
    assert extrair_atributos("http://a.b.c.exemplo.com/")["num_subdomains"] == 3


def test_ip_no_host():
    assert extrair_atributos("http://192.168.0.1/login")["has_ip_in_host"] == 1
    assert extrair_atributos("http://[2001:db8::1]/x")["has_ip_in_host"] == 1
    assert extrair_atributos("http://exemplo.com/")["has_ip_in_host"] == 0


def test_porta_explicita():
    assert extrair_atributos("http://exemplo.com:8080/x")["has_port"] == 1
    assert extrair_atributos("http://exemplo.com/x")["has_port"] == 0


def test_hifen_no_dominio_ignora_hifen_no_caminho():
    assert extrair_atributos("http://ex-emplo.com/a")["hyphen_in_domain"] == 1
    assert extrair_atributos("http://exemplo.com/a-b")["hyphen_in_domain"] == 0


def test_punycode():
    assert extrair_atributos("http://xn--80ak6aa92e.com/")["is_punycode"] == 1
    assert extrair_atributos("http://exemplo.com/")["is_punycode"] == 0


# ---------------------------------------------------------------------------
# Grupo 5 - ofuscação
# ---------------------------------------------------------------------------

def test_arroba_e_barra_dupla():
    a = extrair_atributos("http://legitimo.com@malicioso.com/x")
    assert a["has_at_symbol"] == 1
    b = extrair_atributos("http://exemplo.com/a//b")
    assert b["double_slash_in_path"] == 1


def test_codificacao_hexadecimal():
    assert extrair_atributos("http://exemplo.com/a%2Fb")["has_hex_encoding"] == 1
    assert extrair_atributos("http://exemplo.com/ab")["has_hex_encoding"] == 0


def test_token_https_dentro_do_host():
    """Host contendo 'https' como texto, distinto do esquema real."""
    a = extrair_atributos("http://https-banco.com/login")
    assert a["https_token_in_host"] == 1
    assert a["scheme_is_https"] == 0


def test_esquema_https():
    assert extrair_atributos("https://exemplo.com/")["scheme_is_https"] == 1
    assert extrair_atributos("http://exemplo.com/")["scheme_is_https"] == 0


def test_url_sem_esquema_nao_vira_https():
    """A normalização interna não pode inventar informação de esquema."""
    a = extrair_atributos("exemplo.com/pagina")
    assert a["scheme_is_https"] == 0
    assert a["hostname_length"] == len("exemplo.com")


# ---------------------------------------------------------------------------
# Grupo 6 - léxico
# ---------------------------------------------------------------------------

def test_termos_suspeitos():
    a = extrair_atributos("http://x.com/login/secure/verify")
    assert a["count_suspicious_words"] == 3
    assert extrair_atributos("http://x.com/pagina")["count_suspicious_words"] == 0


def test_encurtador():
    assert extrair_atributos("http://bit.ly/abc")["has_shortener"] == 1
    assert extrair_atributos("http://www.bit.ly/abc")["has_shortener"] == 1
    assert extrair_atributos("http://exemplo.com/abc")["has_shortener"] == 0


def test_tokens_do_caminho():
    a = extrair_atributos("http://exemplo.com/a/b/c")
    assert a["num_tokens_path"] == 3
    assert extrair_atributos("http://exemplo.com/")["num_tokens_path"] == 0


# ---------------------------------------------------------------------------
# Aplicação em lote
# ---------------------------------------------------------------------------

def test_dataframe_preserva_indice_e_colunas():
    s = pd.Series(
        ["http://a.com/", "https://b.org/x", "http://192.168.1.1/"],
        index=[10, 20, 30],
    )
    X = extrair_dataframe(s)
    assert list(X.columns) == list(COLUNAS)
    assert list(X.index) == [10, 20, 30]
    assert len(X) == 3
    assert not X.isna().any().any()


def test_extracao_e_deterministica():
    url = "https://sub.exemplo.com.br/login?id=123"
    assert extrair_atributos(url) == extrair_atributos(url)


def test_mesma_url_mesmo_vetor_entre_chamadas_em_lote():
    """Garante que treino e teste independente passam pelo caminho idêntico."""
    url = "http://secure-login.exemplo.com/verify"
    individual = extrair_atributos(url)
    em_lote = extrair_dataframe(pd.Series([url])).iloc[0].to_dict()
    for coluna in COLUNAS:
        assert individual[coluna] == pytest.approx(em_lote[coluna])
