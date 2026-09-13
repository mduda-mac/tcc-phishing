"""
Extração de atributos léxicos a partir da string bruta da URL.

Este módulo implementa a taxonomia de 37 atributos definida na metodologia.
Todos os atributos são derivados exclusivamente da string da URL: nenhuma
consulta a DNS, WHOIS, conteúdo HTML ou serviço externo é realizada.

A mesma função é aplicada ao conjunto de treino (PhiUSIIL) e ao conjunto de
teste independente (Hannousse e Yahiouche), garantindo que a comparação
entre eles não seja contaminada por diferenças de definição de atributo.

Uso:
    from extracao_lexica import extrair_atributos, extrair_dataframe

    atributos = extrair_atributos("http://exemplo.com/login")
    X = extrair_dataframe(df["URL"])
"""

from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter
from urllib.parse import urlsplit

import pandas as pd
import tldextract

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

# Extrator offline: usa o snapshot da Public Suffix List embarcado no pacote.
# suffix_list_urls=() desativa qualquer requisição de rede, o que torna a
# extração determinística e reprodutível independentemente da data de execução.
_EXTRATOR_TLD = tldextract.TLDExtract(suffix_list_urls=())

# Listas fechadas, congeladas antes da observação de qualquer resultado
# experimental. Não devem ser alteradas após o início dos experimentos.
TERMOS_SUSPEITOS: tuple[str, ...] = (
    "login", "signin", "verify", "account", "update", "secure",
    "banking", "confirm", "password", "webscr", "ebayisapi", "paypal",
)

ENCURTADORES: frozenset[str] = frozenset({
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly",
    "is.gd", "buff.ly", "adf.ly", "bit.do", "cutt.ly",
})

_DELIMITADORES = re.compile(r"[./\-_?=&]+")
_HEX = re.compile(r"%[0-9A-Fa-f]{2}")
_ESQUEMA = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*://")

# Ordem canônica das colunas de saída. Fixada para que a matriz de atributos
# tenha sempre o mesmo layout nos dois conjuntos de dados.
COLUNAS: tuple[str, ...] = (
    # Grupo 1 - comprimento
    "url_length", "hostname_length", "path_length", "query_length",
    "longest_token_length",
    # Grupo 2 - contagem de caracteres especiais
    "count_dots", "count_hyphens", "count_underscore", "count_slash",
    "count_question", "count_equals", "count_at", "count_ampersand",
    "count_percent", "count_digits",
    # Grupo 3 - composição estatística
    "digit_ratio", "letter_ratio", "special_char_ratio",
    "shannon_entropy_url", "shannon_entropy_hostname",
    # Grupo 4 - estrutura do host
    "num_subdomains", "has_ip_in_host", "has_port", "hyphen_in_domain",
    "domain_length", "tld_length", "is_punycode", "digits_in_hostname",
    # Grupo 5 - ofuscação e codificação
    "has_at_symbol", "double_slash_in_path", "has_hex_encoding",
    "https_token_in_host", "scheme_is_https",
    # Grupo 6 - léxico e tokenização
    "count_suspicious_words", "has_shortener", "num_tokens_path",
    "avg_token_length",
)


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def entropia_shannon(texto: str) -> float:
    """Entropia de Shannon da distribuição de caracteres, em bits.

    Retorna 0.0 para string vazia. Valores altos indicam cadeias com aparência
    aleatória, padrão comum em domínios gerados automaticamente.
    """
    if not texto:
        return 0.0
    n = len(texto)
    return -sum(
        (c / n) * math.log2(c / n) for c in Counter(texto).values()
    )


def _e_endereco_ip(host: str) -> bool:
    """Verifica se o host é um endereço IP literal (v4 ou v6)."""
    if not host:
        return False
    candidato = host.strip("[]")
    try:
        ipaddress.ip_address(candidato)
        return True
    except ValueError:
        return False


def _tokenizar(texto: str) -> list[str]:
    """Divide a string nos delimitadores estruturais da URL."""
    return [t for t in _DELIMITADORES.split(texto) if t]


# ---------------------------------------------------------------------------
# Extração
# ---------------------------------------------------------------------------

def extrair_atributos(url: str) -> dict[str, float]:
    """Extrai os 37 atributos léxicos de uma única URL.

    URLs sem esquema explícito recebem 'http://' apenas para viabilizar o
    parsing estrutural; o atributo scheme_is_https é determinado a partir da
    string original, de modo que essa normalização não introduz informação.

    Entradas inválidas ou ausentes produzem um vetor de zeros, evitando que
    uma falha de parsing interrompa o processamento do conjunto.
    """
    if not isinstance(url, str) or not url.strip():
        return dict.fromkeys(COLUNAS, 0.0)

    bruta = url.strip()
    minuscula = bruta.lower()

    tem_esquema = bool(_ESQUEMA.match(bruta))
    scheme_is_https = int(minuscula.startswith("https://"))

    try:
        partes = urlsplit(bruta if tem_esquema else f"http://{bruta}")
        host = (partes.hostname or "").lower()
        caminho = partes.path or ""
        query = partes.query or ""
        tem_porta = int(partes.port is not None)
    except ValueError:
        # URL malformada a ponto de impedir o parsing estrutural.
        host, caminho, query, tem_porta = "", "", "", 0

    # Corpo da URL sem o esquema, usado na tokenização.
    corpo = _ESQUEMA.sub("", bruta)
    tokens = _tokenizar(corpo)

    # Decomposição do host em subdomínio, domínio registrável e sufixo.
    ext = _EXTRATOR_TLD(host)
    subdominio, dominio, sufixo = ext.subdomain, ext.domain, ext.suffix

    n = len(bruta)
    digitos = sum(c.isdigit() for c in bruta)
    letras = sum(c.isalpha() for c in bruta)
    especiais = sum(not c.isalnum() for c in bruta)

    return {
        # Grupo 1
        "url_length": n,
        "hostname_length": len(host),
        "path_length": len(caminho),
        "query_length": len(query),
        "longest_token_length": max((len(t) for t in tokens), default=0),
        # Grupo 2
        "count_dots": bruta.count("."),
        "count_hyphens": bruta.count("-"),
        "count_underscore": bruta.count("_"),
        "count_slash": bruta.count("/"),
        "count_question": bruta.count("?"),
        "count_equals": bruta.count("="),
        "count_at": bruta.count("@"),
        "count_ampersand": bruta.count("&"),
        "count_percent": bruta.count("%"),
        "count_digits": digitos,
        # Grupo 3
        "digit_ratio": digitos / n,
        "letter_ratio": letras / n,
        "special_char_ratio": especiais / n,
        "shannon_entropy_url": entropia_shannon(bruta),
        "shannon_entropy_hostname": entropia_shannon(host),
        # Grupo 4
        "num_subdomains": len(subdominio.split(".")) if subdominio else 0,
        "has_ip_in_host": int(_e_endereco_ip(host)),
        "has_port": tem_porta,
        "hyphen_in_domain": int("-" in dominio),
        "domain_length": len(dominio),
        "tld_length": len(sufixo),
        "is_punycode": int("xn--" in host),
        "digits_in_hostname": sum(c.isdigit() for c in host),
        # Grupo 5
        "has_at_symbol": int("@" in bruta),
        "double_slash_in_path": int("//" in caminho),
        "has_hex_encoding": int(bool(_HEX.search(bruta))),
        "https_token_in_host": int("https" in host),
        "scheme_is_https": scheme_is_https,
        # Grupo 6
        "count_suspicious_words": sum(
            minuscula.count(termo) for termo in TERMOS_SUSPEITOS
        ),
        "has_shortener": int(
            host.removeprefix("www.") in ENCURTADORES
        ),
        "num_tokens_path": len([s for s in caminho.split("/") if s]),
        "avg_token_length": (
            sum(len(t) for t in tokens) / len(tokens) if tokens else 0.0
        ),
    }


def extrair_dataframe(urls: pd.Series) -> pd.DataFrame:
    """Aplica a extração a uma série de URLs.

    Retorna um DataFrame com as 37 colunas em ordem canônica, preservando o
    índice da série de entrada para permitir alinhamento com os rótulos.
    """
    registros = [extrair_atributos(u) for u in urls]
    return pd.DataFrame(registros, columns=list(COLUNAS), index=urls.index)
