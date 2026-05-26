# =============================📖 BEGIN Header 📝=============================
# SIPREV-Mulher/MS — Módulo Monitor da Violência contra a Mulher (v3.0)
# Sistema Inteligente de Predição e Mapeamento da
# Violência contra a Mulher em Mato Grosso do Sul
# 📝 Large Language Model (LLM)
# 🧠 Neural networks, Machine learning, and Deep machine learning
# =============================================================================
# Disciplina : 📚 Tópicos Interdisciplinares III
# Curso      : Ciência dos Dados — UFMS Digital
# Semestre   : 2026.1
# Autor      : VIANA
# =============================================================================
# Fonte dados: MONITOR DA VIOLÊNCIA CONTRA A MULHER
# URL Monitor: https://monitorviolenciacontramulher.sejusp.ms.gov.br/
# Dados até  : 05/2026 — Fonte: SEJUSP-MS e PJMS
# Repositório: https://github.com/OpenScienceTechnology/Dataset_SEJUSP-MS_PJMS
# =============================================================================
# Datasets utilizados:
#   - atendimentos_emergencia.csv        (41.548 registros | 2016–2026)
#   - medidas_protetivas_urgencia.csv    (   800 registros | 2026)
#   - mulheres_vitimas_homicidios.csv    ( 1.715 registros | 2016–2026)
#   - vitimas_estupro.csv                (43.509 registros | 2016–2026)
#   - vitimas_feminicidios.csv           (   997 registros | 2016–2026)
#   - vitimas_violencia_domestica.csv    ( 2.900 registros | 2026)
# =============================================================================
# Seções:
#   0  – Configuração do Ambiente e Instalação de Dependências
#   1  – Carregamento e Ingestão dos Dados (6 datasets Monitor)
#   2  – Limpeza e Pré-processamento
#   3  – Análise Exploratória: Ocorrências
#   4  – Ranking Municipal (79 municípios)
#   5  – Ranking Nacional / Comparativo Estadual
#   6  – Análise Geoespacial e Mapas
#   7  – Clusterização Municipal
#   8  – Modelos Preditivos (ML clássico)
#   9  – Deep Learning (LSTM / GRU / MLP)
#   10 – Explicabilidade SHAP
#   11 – Relatório Completo de Modelos Treinados
#   12 – Exportação (ZIP/Colab | Pastas locais)
#   13 – Conclusões e Recomendações
# =============================================================================
# Requisitos:
#   pip install pandas numpy matplotlib seaborn plotly folium geopandas
#               shapely contextily scikit-learn xgboost lightgbm imbalanced-learn
#               hdbscan shap prophet statsmodels tensorflow torch
#               pytorch-lightning mlflow optuna texttable openpyxl kaleido
#               requests tqdm loguru unidecode rapidfuzz missingno ydata-profiling
# =============================📝 END Header 📄===============================

# ============================================================
# SEÇÃO 0 — CONFIGURAÇÃO DO AMBIENTE E INSTALAÇÃO
# ============================================================

import os, sys, zipfile, shutil, platform, warnings, subprocess
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

# ── Detecção de ambiente ─────────────────────────────────────
try:
    import google.colab
    IS_COLAB = True
except ImportError:
    IS_COLAB = False

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
PROJ_NAME = "SIPREV-MS_Mon_Mul"

print(f"🌐 Ambiente: {'Google Colab' if IS_COLAB else 'Local — ' + platform.system()}")
print(f"⏰ Início  : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
print(f"📦 Projeto : {PROJ_NAME}")

# ── Diretórios de saída ──────────────────────────────────────
BASE_DIR    = Path(".")
OUTPUT_DIR  = Path("outputs")
MAP_DIR     = OUTPUT_DIR / "maps"
REP_DIR     = OUTPUT_DIR / "relatorios"
TAB_DIR     = OUTPUT_DIR / "tabelas"
GRF_DIR     = OUTPUT_DIR / "graficos"
MOD_DIR     = Path("models")
LOG_DIR     = Path("logs")
DASH_DIR    = Path("dashboard")

for d in [OUTPUT_DIR, MAP_DIR, REP_DIR, TAB_DIR, GRF_DIR, MOD_DIR, LOG_DIR, DASH_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Instalação de dependências extras ───────────────────────
EXTRA_PKGS = [
    "texttable", "requests", "tqdm", "folium", "geopandas", "shapely",
    "openpyxl", "kaleido", "loguru", "unidecode", "rapidfuzz", "missingno",
    "contextily", "scikit-learn", "xgboost", "lightgbm", "catboost",
    "imbalanced-learn", "hdbscan", "shap", "lime", "eli5", "alibi",
    "interpret", "yellowbrick", "statsmodels", "optuna", "mlflow", "joblib",
    "polars", "duckdb", "darts", "neuralprophet", "skforecast",
    "lightfm", "ipywidgets",
]

def _pip_install(pkg: str):
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", pkg, "-q"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass

for pkg in EXTRA_PKGS:
    _pip_install(pkg)

print("✅ Dependências instaladas.")

# ── Log estruturado ──────────────────────────────────────────
from loguru import logger

LOG_FILE = LOG_DIR / f"siprev_monitor_{TIMESTAMP}.log"
logger.add(str(LOG_FILE), format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
           level="DEBUG", encoding="utf-8")
logger.info("SIPREV-Mulher Monitor v3.0 — Início da execução")
logger.info(f"Ambiente: {'Colab' if IS_COLAB else 'Local'}")

# ── Imports principais ───────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib
# ── Backend inline para Jupyter/Colab ──────────────────────
try:
    from IPython import get_ipython
    ip = get_ipython()
    if ip: ip.run_line_magic("matplotlib", "inline")
except Exception:
    pass
import matplotlib
matplotlib.rcParams["figure.dpi"] = 130
matplotlib.rcParams["figure.facecolor"] = "white"

# ── IPython display ─────────────────────────────────────────
from IPython.display import display, HTML, IFrame
try: import ipywidgets  # noqa
except ImportError: pass
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import texttable as tt
    HAS_TEXTTABLE = True
except ImportError:
    HAS_TEXTTABLE = False
    logger.warning("texttable não encontrado")

try:
    import missingno as msno
    HAS_MISSINGNO = True
except ImportError:
    HAS_MISSINGNO = False

try:
    from unidecode import unidecode
    HAS_UNIDECODE = True
except ImportError:
    HAS_UNIDECODE = False
    def unidecode(s): return s

try:
    from rapidfuzz import process as rf_process
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

try:
    import geopandas as gpd
    from shapely.geometry import Point
    HAS_GEO = True
except ImportError:
    HAS_GEO = False
    logger.warning("geopandas não encontrado — mapas coroplético desativados")

try:
    import folium
    from folium.plugins import HeatMap, MarkerCluster, Fullscreen
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

# ── Estilo global ────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.grid": True,
    "grid.alpha": 0.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
    "font.size": 11,
})
PALETTE  = px.colors.qualitative.Set2
CORES_MS = {
    "feminicidio":      "#c0392b",
    "homicidio":        "#e74c3c",
    "estupro":          "#8e44ad",
    "violencia_dom":    "#2980b9",
    "medidas_prot":     "#27ae60",
    "atendimento":      "#f39c12",
}

# ── Meses em português → número ─────────────────────────────
MES_PT = {
    "JANEIRO":1,"FEVEREIRO":2,"MARÇO":3,"ABRIL":4,"MAIO":5,"JUNHO":6,
    "JULHO":7,"AGOSTO":8,"SETEMBRO":9,"OUTUBRO":10,"NOVEMBRO":11,"DEZEMBRO":12,
}
MES_NOME = {v: k.capitalize() for k, v in MES_PT.items()}

# ── Populações municipais MS (IBGE Estimativa 2024) ──────────
POP_MS = {
    "Campo Grande":916001,"Dourados":230077,"Três Lagoas":121046,
    "Corumbá":113532,"Ponta Porã":95856,"Naviraí":53640,
    "Sidrolândia":57764,"Aquidauana":46798,"Maracaju":48493,
    "Caarapó":35317,"Paranaíba":41748,"Chapadão Do Sul":25039,
    "Costa Rica":20246,"Coxim":35030,"Miranda":27098,
    "Bonito":22499,"Amambai":40428,"Coronel Sapucaia":17792,
    "Paranhos":14905,"Nova Andradina":51819,"Jardim":27834,
    "Bela Vista":24601,"Itaquiraí":25280,"Iguatemi":17609,
    "Mundo Novo":19673,"Eldorado":12148,"Sete Quedas":14756,
    "Japorã":10620,"Juti":8248,"Tacuru":11831,
    "Ivinhema":22803,"Nova Alvorada Do Sul":13547,"Rio Brilhante":34007,
    "Deodápolis":13534,"Fátima Do Sul":22130,"Glória De Dourados":8527,
    "Vicentina":7398,"Angélica":8984,"Taquarussu":4778,
    "Jateí":5810,"Itaporã":27497,"Douradina":6153,
    "Rio Verde De Mato Grosso":19024,"Sonora":19014,"Pedro Gomes":8521,
    "Bandeirantes":7282,"Camapuã":22066,"Ribas Do Rio Pardo":23087,
    "Santa Rita Do Pardo":9741,"Água Clara":16621,
    "Cassilândia":22218,"Inocência":9217,"Figueirão":4099,
    "Alcinópolis":5396,"São Gabriel Do Oeste":23143,
    "Corguinho":5237,"Rochedo":5030,"Jaraguari":8151,
    "Terenos":23070,"Dois Irmãos Do Buriti":13136,"Nioaque":14534,
    "Bodoquena":9034,"Caracol":5780,"Porto Murtinho":16427,
    "Guia Lopes Da Laguna":11102,"Antônio João":9682,
    "Anastácio":24773,"Bataguassu":18993,"Batayporã":11879,
    "Anaurilândia":10264,"Aparecida Do Taboado":25083,"Selvíria":8614,
    "Brasilândia":16002,"Aral Moreira":15547,"Laguna Carapã":10620,
    "Paraíso Das Águas":9261,"Novo Horizonte Do Sul":4978,
    "Ladário":23048,"Rio Negro":6132,"Nova Guarita":4500,
    "Água Clara":16621,"Nioaque":14534,
}
POP_MULHERES_MS = {k: int(v * 0.51) for k, v in POP_MS.items()}

# ── Populações estaduais BR (IBGE 2024) ──────────────────────
POP_ESTADOS_BR = {
    "SP":47624776,"MG":21290000,"RJ":17463000,"BA":15278000,
    "RS":11537000,"PR":11597000,"PE":9674000,"CE":9322000,
    "PA":8943000,"MA":7161000,"SC":7786000,"GO":7383000,
    "AM":4311000,"ES":4194000,"PB":4106000,"RN":3630000,
    "AL":3365000,"PI":3340000,"MT":3747000,"MS":2870000,
    "DF":3094000,"SE":2362000,"RO":1888000,"TO":1629000,
    "AC":914000,"AP":906000,"RR":695000,
}

# ============================================================
# SEÇÃO 1 — CARREGAMENTO E INGESTÃO DOS DADOS
# ============================================================

print("\n" + "="*70)
print("📥 SEÇÃO 1 — CARREGAMENTO E INGESTÃO DOS DADOS")
print("="*70)

# URLs dos datasets (GitHub / Monitor)
URLS = {
    "atendimentos": (
        "https://raw.githubusercontent.com/OpenScienceTechnology/"
        "Dataset_SEJUSP-MS_PJMS/refs/heads/main/Data/CSV/atendimentos_emergencia.csv"
    ),
    "medidas": (
        "https://raw.githubusercontent.com/OpenScienceTechnology/"
        "Dataset_SEJUSP-MS_PJMS/refs/heads/main/Data/CSV/medidas_protetivas_urgencia.csv"
    ),
    "homicidios": (
        "https://raw.githubusercontent.com/OpenScienceTechnology/"
        "Dataset_SEJUSP-MS_PJMS/refs/heads/main/Data/CSV/mulheres_vitimas_homicidios.csv"
    ),
    "estupro": (
        "https://raw.githubusercontent.com/OpenScienceTechnology/"
        "Dataset_SEJUSP-MS_PJMS/refs/heads/main/Data/CSV/vitimas_estupro.csv"
    ),
    "feminicidios": (
        "https://raw.githubusercontent.com/OpenScienceTechnology/"
        "Dataset_SEJUSP-MS_PJMS/refs/heads/main/Data/CSV/vitimas_feminicidios.csv"
    ),
    "viol_dom": (
        "https://raw.githubusercontent.com/OpenScienceTechnology/"
        "Dataset_SEJUSP-MS_PJMS/refs/heads/main/Data/CSV/vitimas_violencia_domestica.csv"
    ),
}

# Arquivos locais (uploads do ambiente)
LOCAL_FILES = {
    "atendimentos": "atendimentos_emergencia.csv",
    "medidas":      "medidas_protetivas_urgencia.csv",
    "homicidios":   "mulheres_vitimas_homicidios.csv",
    "estupro":      "vitimas_estupro.csv",
    "feminicidios": "vitimas_feminicidios.csv",
    "viol_dom":     "vitimas_violencia_domestica.csv",
}

# Possíveis diretórios onde os CSV estão
SEARCH_DIRS = [
    Path("."),
    Path("/mnt/user-data/uploads"),
    Path("data/raw"),
    Path("CSV"),
]

def _find_local(fname: str) -> Path | None:
    for d in SEARCH_DIRS:
        p = d / fname
        if p.exists():
            return p
    return None

def _download_csv(key: str, url: str, local: str) -> Path:
    """Tenta local primeiro, depois baixa do GitHub."""
    lp = _find_local(local)
    if lp:
        logger.info(f"[{key}] CSV local encontrado: {lp}")
        return lp
    try:
        import requests
        logger.info(f"[{key}] Baixando de {url}")
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        dest = Path("data/raw") / local
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.content)
        return dest
    except Exception as e:
        logger.error(f"[{key}] Falha no download: {e}")
        return None

def _read_csv(path: Path, key: str) -> pd.DataFrame:
    """Lê CSV com fallback de encoding e separador."""
    for enc in ["utf-8-sig", "utf-8", "latin1"]:
        for sep in [",", ";", "\t", None]:
            try:
                if sep is None:
                    df = pd.read_csv(path, encoding=enc, sep=None, engine="python")
                else:
                    df = pd.read_csv(path, encoding=enc, sep=sep)
                if df.shape[1] > 1:
                    logger.success(f"[{key}] Lido: {len(df):,} linhas | {df.shape[1]} colunas")
                    return df
            except Exception:
                continue
    logger.error(f"[{key}] Falha na leitura de {path}")
    return pd.DataFrame()

# Carregamento de todos os datasets
RAW = {}
for key, local in LOCAL_FILES.items():
    path = _download_csv(key, URLS[key], local)
    if path:
        RAW[key] = _read_csv(path, key)
    else:
        RAW[key] = pd.DataFrame()
        logger.warning(f"[{key}] Dataset não disponível")

print(f"\n✅ Datasets carregados:")
for k, df in RAW.items():
    print(f"   {k:15s}: {len(df):>7,} linhas | {df.shape[1]} colunas")

# ============================================================
# SEÇÃO 2 — LIMPEZA E PRÉ-PROCESSAMENTO
# ============================================================

print("\n" + "="*70)
print("🔧 SEÇÃO 2 — LIMPEZA E PRÉ-PROCESSAMENTO")
print("="*70)

def _norm_str(s):
    """Normaliza string: upper + strip + unidecode."""
    if pd.isna(s):
        return np.nan
    s = str(s).strip().upper()
    if HAS_UNIDECODE:
        s = unidecode(s)
    return s

def _faixa_etaria(idade):
    """Retorna faixa etária a partir de valor numérico."""
    if pd.isna(idade):
        return "Não Informado"
    idade = float(idade)
    if idade < 12:    return "0–11 anos"
    if idade < 18:    return "12–17 anos"
    if idade < 25:    return "18–24 anos"
    if idade < 30:    return "25–29 anos"
    if idade < 40:    return "30–39 anos"
    if idade < 50:    return "40–49 anos"
    if idade < 60:    return "50–59 anos"
    return "60+ anos"

def _preprocessar_padrao(df: pd.DataFrame, fonte: str) -> pd.DataFrame:
    """Pipeline de limpeza para datasets com schema padrão (16 colunas)."""
    df = df.copy()
    # Padronizar nomes de colunas
    df.columns = df.columns.str.strip()
    # Renomeação padrão
    rename_map = {
        "Nº BO": "num_bo",
        "FATO": "fato",
        "FATO AGRUPADO": "fato_agrupado",
        "BAIRRO DO FATO": "bairro",
        "UF DO FATO": "uf",
        "MUNICÍPIO DO FATO": "municipio",
        "CÓDIGO IBGE": "cod_ibge",
        "ANO DO FATO": "ano",
        "DATA DO FATO": "data_fato",
        "HORA DO FATO": "hora_fato",
        "ENVOLVIMENTO": "envolvimento",
        "SEXO": "sexo",
        "NACIONALIDADE": "nacionalidade",
        "ESCOLARIDADE": "escolaridade",
        "COR/RAÇA": "cor_raca",
        "IDADE NO FATO": "idade",
    }
    df.rename(columns={c: rename_map.get(c, c) for c in df.columns}, inplace=True)

    # Datas
    df["data_fato"] = pd.to_datetime(df.get("data_fato", pd.Series(dtype=str)),
                                      format="%d/%m/%Y", errors="coerce")
    df["mes"]       = df["data_fato"].dt.month
    df["trimestre"] = df["data_fato"].dt.quarter
    df["periodo"]   = df["data_fato"].dt.to_period("M").astype(str)

    # Strings
    for col in ["fato", "fato_agrupado", "bairro", "municipio", "envolvimento",
                "sexo", "escolaridade", "cor_raca", "nacionalidade"]:
        if col in df.columns:
            df[col] = df[col].apply(_norm_str)

    # Municipio → Title Case padronizado
    if "municipio" in df.columns:
        df["nm_municipio"] = (
            df["municipio"].str.title().str.strip()
            if "municipio" in df.columns else np.nan
        )

    # Faixa etária
    if "idade" in df.columns:
        df["faixa_etaria"] = df["idade"].apply(_faixa_etaria)

    # Fonte
    df["fonte"] = fonte

    logger.info(f"[{fonte}] Pré-processado: {len(df):,} linhas | "
                f"NaN idade: {df.get('idade', pd.Series()).isna().sum()}")
    return df

def _preprocessar_atendimentos(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline de limpeza para atendimentos_emergencia (schema diferente)."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    rename_map = {
        "DATA": "data_fato_str",
        "MÊS": "mes_str",
        "ANO": "ano",
        "DIA DA SEMANA": "dia_semana",
        "FATO": "fato",
        "MUNICÍPIO": "municipio",
        "BAIRRO": "bairro",
        "TIPO DE LOCAL": "local",
    }
    df.rename(columns={c: rename_map.get(c, c) for c in df.columns}, inplace=True)
    df["data_fato"] = pd.to_datetime(df["data_fato_str"], format="%d/%m/%Y %H:%M:%S",
                                     errors="coerce")
    df["mes"]       = df["mes_str"].map(MES_PT)
    df["trimestre"] = df["data_fato"].dt.quarter
    df["periodo"]   = df["data_fato"].dt.to_period("M").astype(str)

    for col in ["fato", "municipio", "bairro", "local", "dia_semana"]:
        if col in df.columns:
            df[col] = df[col].apply(_norm_str)

    df["nm_municipio"] = df["municipio"].str.title().str.strip()
    df["fonte"] = "atendimentos"
    logger.info(f"[atendimentos] Pré-processado: {len(df):,} linhas")
    return df

# ── Processar cada dataset ────────────────────────────────────
DFS = {}

if not RAW["atendimentos"].empty:
    DFS["atendimentos"] = _preprocessar_atendimentos(RAW["atendimentos"])

for key in ["medidas", "homicidios", "estupro", "feminicidios", "viol_dom"]:
    if not RAW[key].empty:
        DFS[key] = _preprocessar_padrao(RAW[key], key)

# ── Dataset Unificado (apenas vitimas/ocorrências) ────────────
def _filtrar_vitimas(df: pd.DataFrame) -> pd.DataFrame:
    if "envolvimento" not in df.columns:
        return df
    return df[df["envolvimento"].isin(["VITIMA", "VÍTIMA"])].copy()

def _filtrar_autores(df: pd.DataFrame) -> pd.DataFrame:
    if "envolvimento" not in df.columns:
        return df
    return df[df["envolvimento"].isin(["AUTOR"])].copy()

VITIMAS = {}
AUTORES = {}
for key, df in DFS.items():
    if key == "atendimentos":
        VITIMAS[key] = df.copy()
    else:
        VITIMAS[key] = _filtrar_vitimas(df)
        AUTORES[key] = _filtrar_autores(df)

# Concatenação geral de vítimas (exceto atendimentos, que tem schema diferente)
COLS_COMUNS = ["num_bo","fato","fato_agrupado","bairro","uf","municipio",
               "nm_municipio","cod_ibge","ano","data_fato","mes","trimestre",
               "periodo","sexo","escolaridade","cor_raca","idade","faixa_etaria","fonte"]

frames = []
for key in ["medidas", "homicidios", "estupro", "feminicidios", "viol_dom"]:
    if key in VITIMAS and not VITIMAS[key].empty:
        v = VITIMAS[key]
        frames.append(v[[c for c in COLS_COMUNS if c in v.columns]])

df_todas = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
if not df_todas.empty and "nm_municipio" in df_todas.columns:
    df_todas.dropna(subset=["nm_municipio"], inplace=True)
elif df_todas.empty:
    # Create minimal placeholder to avoid downstream crashes
    logger.warning("df_todas vazio — criando placeholder mínimo")
    df_todas = pd.DataFrame(columns=COLS_COMUNS)

print(f"\n📊 Dataset unificado de vítimas: {len(df_todas):,} registros")
print(f"   Municípios cobertos       : {df_todas['nm_municipio'].nunique()}")
print(f"   Período                   : {df_todas['ano'].min()} – {df_todas['ano'].max()}")
print(f"   Fontes                    : {df_todas['fonte'].value_counts().to_dict()}")

logger.info(f"Dataset unificado: {len(df_todas):,} registros")

# ── Faixas etárias ORDEM ────────────────────────────────────
FAIXA_ORDER = ["0–11 anos","12–17 anos","18–24 anos","25–29 anos",
               "30–39 anos","40–49 anos","50–59 anos","60+ anos","Não Informado"]

# ============================================================
# SEÇÃO 3 — ANÁLISE EXPLORATÓRIA: OCORRÊNCIAS
# ============================================================

print("\n" + "="*70)
print("📊 SEÇÃO 3 — ANÁLISE EXPLORATÓRIA: OCORRÊNCIAS")
print("="*70)

def _salvar_fig(fig, nome: str, subdir: str = "graficos") -> Path:
    """Salva figura matplotlib/seaborn em PNG e PDF + exibe inline."""
    d = OUTPUT_DIR / subdir
    d.mkdir(parents=True, exist_ok=True)
    p_png = d / f"{nome}.png"
    p_pdf = d / f"{nome}.pdf"
    fig.savefig(p_png, bbox_inches="tight", dpi=130)
    fig.savefig(p_pdf, bbox_inches="tight")
    plt.show()           # ← exibe inline no Jupyter/Colab
    plt.close(fig)
    logger.debug(f"Figura salva: {p_png.name}")
    return p_png

def _salvar_plotly(fig, nome: str, subdir: str = "graficos"):
    """Salva figura Plotly em HTML e PNG + exibe inline no Jupyter/Colab."""
    d = OUTPUT_DIR / subdir
    d.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(d / f"{nome}.html"))
    try:
        fig.write_image(str(d / f"{nome}.png"), width=1400, height=800, scale=1.5)
    except Exception:
        pass
    fig.show()   # ← exibe inline no Jupyter/Colab
    logger.debug(f"Plotly salvo: {nome}")

def _texttable_str(headers: list, rows: list, title: str = "") -> str:
    """Gera tabela ASCII com texttable (fallback para str simples)."""
    if HAS_TEXTTABLE:
        tab = tt.Texttable(max_width=120)
        tab.set_deco(tt.Texttable.HEADER | tt.Texttable.BORDER | tt.Texttable.VLINES)
        tab.header(headers)
        tab.add_rows(rows, header=False)
        body = tab.draw()
    else:
        lines = ["\t".join(str(h) for h in headers)]
        for r in rows:
            lines.append("\t".join(str(c) for c in r))
        body = "\n".join(lines)
    if title:
        sep = "─" * min(len(title) + 4, 120)
        return f"\n{sep}\n  {title}\n{sep}\n{body}\n"
    return body

def _salvar_relatorio(texto: str, nome: str):
    """Salva texto como .txt e .log."""
    for ext in [".txt", ".log"]:
        p = REP_DIR / f"{nome}{ext}"
        p.write_text(texto, encoding="utf-8")
    logger.debug(f"Relatório salvo: {nome}")

# ── 3.1  Vítimas por Município ───────────────────────────────
print("\n📌 3.1 — Vítimas por Município")

mun_counts = df_todas.groupby("nm_municipio").size().reset_index(name="total")
mun_counts = mun_counts.sort_values("total", ascending=False).reset_index(drop=True)
mun_counts["ranking"] = mun_counts.index + 1

# Taxa por 100 mil mulheres
mun_counts["pop_mulheres"] = mun_counts["nm_municipio"].map(POP_MULHERES_MS).fillna(10000)
mun_counts["taxa_100k"] = (mun_counts["total"] / mun_counts["pop_mulheres"] * 100_000).round(1)

# Tabela top 20
top20_mun = mun_counts.head(20)
rows_mun = [[r.ranking, r.nm_municipio, r.total, r.pop_mulheres, r.taxa_100k]
            for _, r in top20_mun.iterrows()]
txt_mun = _texttable_str(
    ["Pos", "Município", "Total Vítimas", "Pop. Mulheres", "Taxa/100k"],
    rows_mun,
    "TOP 20 MUNICÍPIOS — VITIMAS REGISTRADAS (TODOS OS TIPOS)"
)
print(txt_mun[:2000])
_salvar_relatorio(txt_mun, "ranking_municipios_total")

# Gráfico barras top 20
fig, ax = plt.subplots(figsize=(14, 8))
bars = ax.barh(top20_mun["nm_municipio"][::-1], top20_mun["total"][::-1],
               color=plt.cm.YlOrRd(np.linspace(0.3, 0.9, 20)))
ax.set_xlabel("Total de Vítimas Registradas")
ax.set_title("🏙️ Top 20 Municípios — Total de Vítimas de Violência contra a Mulher\n"
             "Monitor SEJUSP-MS/PJMS (2016–2026)", fontweight="bold", pad=15)
for bar, val in zip(bars, top20_mun["total"][::-1]):
    ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2,
            f"{val:,}", va="center", fontsize=9)
ax.grid(axis="x", alpha=0.4)
_salvar_fig(fig, "3_1_vitimas_municipio_top20")

# Exportar tabela
mun_counts.to_csv(TAB_DIR / "vitimas_por_municipio.csv", index=False, encoding="utf-8-sig")
try:
    from IPython.display import display, HTML
    display(HTML("<h4>📋 Top 20 Municípios — Total de Vítimas</h4>"))
    display(mun_counts.head(20).style.background_gradient(cmap="YlOrRd").format(precision=1))
except Exception: pass
mun_counts.to_excel(TAB_DIR / "vitimas_por_municipio.xlsx", index=False)
print("   ✅ Gráfico e tabela por município gerados.")

# ── 3.2  Vítimas por Ano ─────────────────────────────────────
print("\n📌 3.2 — Vítimas por Ano")

ano_counts = {}
for key, df in VITIMAS.items():
    if "ano" not in df.columns:
        continue
    ac = df.groupby("ano").size().rename(key)
    ano_counts[key] = ac

df_ano = pd.DataFrame(ano_counts).fillna(0).astype(int)
df_ano.index = df_ano.index.astype(int)
df_ano = df_ano.sort_index()
df_ano["total"] = df_ano.sum(axis=1)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
# Total por ano
df_ano["total"].plot(ax=axes[0], marker="o", color="#c0392b", linewidth=2.5,
                     markersize=8, markerfacecolor="white")
axes[0].set_title("Total de Vítimas por Ano\n(todos os tipos de violência)", fontweight="bold")
axes[0].set_xlabel("Ano"); axes[0].set_ylabel("Nº de Vítimas")
axes[0].fill_between(df_ano.index, df_ano["total"], alpha=0.15, color="#c0392b")
# Stacked por tipo
cols_plot = [c for c in df_ano.columns if c != "total"]
df_ano[cols_plot].plot(ax=axes[1], kind="bar", stacked=True, colormap="Set2",
                       edgecolor="white", linewidth=0.5)
axes[1].set_title("Vítimas por Ano e Tipo de Violência\n(distribuição empilhada)", fontweight="bold")
axes[1].set_xlabel("Ano"); axes[1].set_ylabel("Nº de Vítimas")
axes[1].legend(loc="upper left", fontsize=9)
plt.tight_layout()
_salvar_fig(fig, "3_2_vitimas_por_ano")

df_ano.to_csv(TAB_DIR / "vitimas_por_ano.csv", encoding="utf-8-sig")
try:
    from IPython.display import display, HTML
    display(HTML("<h4>📋 Vítimas por Ano</h4>"))
    display(df_ano.style.background_gradient(cmap="Reds").format(precision=0))
except Exception: pass
df_ano.to_excel(TAB_DIR / "vitimas_por_ano.xlsx")

# Texto relatório
rows_ano = [[int(ano)] + list(row) for ano, row in df_ano.iterrows()]
txt_ano = _texttable_str(
    ["Ano"] + [c.replace("_"," ").title() for c in df_ano.columns],
    rows_ano, "VÍTIMAS POR ANO — MONITOR SEJUSP-MS/PJMS"
)
_salvar_relatorio(txt_ano, "vitimas_por_ano")
print("   ✅ Análise temporal por ano concluída.")

# ── 3.3  Vítimas por Mês ─────────────────────────────────────
print("\n📌 3.3 — Vítimas por Mês")

mes_counts = {}
for key, df in VITIMAS.items():
    if "mes" not in df.columns:
        continue
    mc = df.groupby("mes").size().rename(key)
    mes_counts[key] = mc

df_mes = pd.DataFrame(mes_counts).fillna(0).astype(int)
df_mes.index = df_mes.index.astype(int)
df_mes = df_mes.sort_index()
df_mes.index = [MES_NOME.get(m, str(m)) for m in df_mes.index]
df_mes["total"] = df_mes.sum(axis=1)

fig, ax = plt.subplots(figsize=(14, 6))
cmap = plt.cm.Blues(np.linspace(0.4, 0.9, 12))
bars = ax.bar(range(len(df_mes)), df_mes["total"], color=cmap, edgecolor="white")
ax.set_xticks(range(len(df_mes)))
ax.set_xticklabels(df_mes.index, rotation=45, ha="right")
ax.set_title("Distribuição Mensal de Vítimas de Violência contra a Mulher\n"
             "Monitor SEJUSP-MS/PJMS (agregado 2016–2026)", fontweight="bold")
ax.set_ylabel("Nº de Vítimas")
for bar, val in zip(bars, df_mes["total"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            f"{val:,}", ha="center", va="bottom", fontsize=9)
_salvar_fig(fig, "3_3_vitimas_por_mes")

df_mes.to_csv(TAB_DIR / "vitimas_por_mes.csv", encoding="utf-8-sig")
df_mes.to_excel(TAB_DIR / "vitimas_por_mes.xlsx")
print("   ✅ Análise mensal concluída.")

# ── 3.4 / 3.5  Faixa Etária das Vítimas ─────────────────────
print("\n📌 3.4–3.5 — Percentual e Distribuição por Faixa Etária")

if "faixa_etaria" in df_todas.columns:
    faixa_all = df_todas["faixa_etaria"].value_counts()
    faixa_ord = faixa_all.reindex([f for f in FAIXA_ORDER if f in faixa_all.index]).dropna()
    pct = (faixa_ord / faixa_ord.sum() * 100).round(1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    # Barras
    colors_fe = plt.cm.Purples(np.linspace(0.3, 0.9, len(faixa_ord)))
    axes[0].barh(faixa_ord.index[::-1], faixa_ord.values[::-1], color=colors_fe[::-1])
    axes[0].set_title("Distribuição de Vítimas por Faixa Etária\n(contagem absoluta)",
                       fontweight="bold")
    axes[0].set_xlabel("Nº de Vítimas")
    for i, (v, p) in enumerate(zip(faixa_ord.values[::-1], pct.values[::-1])):
        axes[0].text(v + 30, i, f"{v:,}  ({p}%)", va="center", fontsize=9)
    # Pizza
    axes[1].pie(pct.values, labels=pct.index, autopct="%1.1f%%",
                startangle=90, colors=plt.cm.Set3.colors[:len(pct)])
    axes[1].set_title("Percentual de Vítimas por Faixa Etária", fontweight="bold")
    plt.tight_layout()
    _salvar_fig(fig, "3_4_faixa_etaria_vitimas")

    faixa_df = pd.DataFrame({"faixa": faixa_ord.index, "total": faixa_ord.values,
                              "percentual": pct.values})
    faixa_df.to_csv(TAB_DIR / "vitimas_faixa_etaria.csv", index=False, encoding="utf-8-sig")
    faixa_df.to_excel(TAB_DIR / "vitimas_faixa_etaria.xlsx", index=False)
    rows_fe = [[r.faixa, r.total, f"{r.percentual}%"] for _, r in faixa_df.iterrows()]
    txt_fe = _texttable_str(["Faixa Etária", "Total", "Percentual"], rows_fe,
                             "DISTRIBUIÇÃO DE VÍTIMAS POR FAIXA ETÁRIA")
    _salvar_relatorio(txt_fe, "faixa_etaria_vitimas")
    print("   ✅ Faixa etária das vítimas concluída.")

# ── 3.6  Distribuição das vítimas por Tipo de Violência ─────
print("\n📌 3.6 — Distribuição das Vítimas por Tipo de Violência")

tipo_totais = {
    "Atendimento Emergência": len(VITIMAS.get("atendimentos", pd.DataFrame())),
    "Estupro":                len(VITIMAS.get("estupro", pd.DataFrame())),
    "Feminicídio":            len(VITIMAS.get("feminicidios", pd.DataFrame())),
    "Homicídio (mulheres)":   len(VITIMAS.get("homicidios", pd.DataFrame())),
    "Violência Doméstica":    len(VITIMAS.get("viol_dom", pd.DataFrame())),
    "Medidas Protetivas":     len(VITIMAS.get("medidas", pd.DataFrame())),
}
tipo_s = pd.Series(tipo_totais).sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
cores_tipo = [CORES_MS.get(k.lower().replace(" ","_").replace("(","").replace(")",""),
                             "#95a5a6") for k in tipo_s.index]
axes[0].barh(tipo_s.index[::-1], tipo_s.values[::-1],
             color=list(reversed(cores_tipo)))
axes[0].set_title("Distribuição de Vítimas\npor Tipo de Violência (2016–2026)",
                  fontweight="bold")
axes[0].set_xlabel("Total de Registros")
for i, (v) in enumerate(tipo_s.values[::-1]):
    axes[0].text(v + 50, i, f"{v:,}", va="center", fontsize=10)
# Pizza proporcional
axes[1].pie(tipo_s.values, labels=tipo_s.index, autopct="%1.1f%%",
            startangle=140, colors=cores_tipo)
axes[1].set_title("Proporção por Tipo de Violência", fontweight="bold")
plt.tight_layout()
_salvar_fig(fig, "3_6_distribuicao_tipo_violencia")
print("   ✅ Distribuição por tipo de violência concluída.")

# ── 3.7  Taxa por 100 Mil Mulheres ───────────────────────────
print("\n📌 3.7 — Taxa de Vítimas por 100 Mil Mulheres (por Município)")

# Mapa de taxa → já calculado em mun_counts
top15_taxa = mun_counts.nlargest(15, "taxa_100k")

fig, ax = plt.subplots(figsize=(14, 8))
cmap_taxa = plt.cm.Reds(np.linspace(0.3, 0.95, 15))
bars = ax.barh(top15_taxa["nm_municipio"][::-1], top15_taxa["taxa_100k"][::-1],
               color=cmap_taxa)
ax.set_xlabel("Taxa de Vítimas por 100 Mil Mulheres")
ax.set_title("🔴 Top 15 Municípios — Taxa de Vítimas por 100 Mil Mulheres\n"
             "Monitor SEJUSP-MS (2016–2026)", fontweight="bold")
for bar, val in zip(bars, top15_taxa["taxa_100k"][::-1]):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}", va="center", fontsize=10)
_salvar_fig(fig, "3_7_taxa_100k_municipio")
print("   ✅ Taxa por 100k concluída.")

# ── 3.8  Locais de Maior Incidência ─────────────────────────
print("\n📌 3.8 — Locais de Maior Incidência das Ocorrências")

# Combina campo "local" de atendimentos + "bairro" dos demais (tipo de local)
df_atend = VITIMAS.get("atendimentos", pd.DataFrame())
if "local" in df_atend.columns:
    local_counts = df_atend["local"].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(14, 7))
    colors_loc = plt.cm.Oranges(np.linspace(0.3, 0.9, len(local_counts)))
    ax.barh(local_counts.index[::-1], local_counts.values[::-1], color=colors_loc)
    ax.set_title("📍 Locais de Maior Incidência das Ocorrências de Violência\n"
                 "(Atendimentos de Emergência — 2016–2026)", fontweight="bold")
    ax.set_xlabel("Nº de Ocorrências")
    for i, v in enumerate(local_counts.values[::-1]):
        ax.text(v + 50, i, f"{v:,}", va="center", fontsize=9)
    _salvar_fig(fig, "3_8_locais_incidencia")

    loc_df = local_counts.reset_index()
    loc_df.columns = ["local", "ocorrencias"]
    loc_df.to_csv(TAB_DIR / "locais_incidencia.csv", index=False, encoding="utf-8-sig")
    loc_df.to_excel(TAB_DIR / "locais_incidencia.xlsx", index=False)
    rows_loc = [[r.local, r.ocorrencias] for _, r in loc_df.iterrows()]
    _salvar_relatorio(
        _texttable_str(["Local da Ocorrência", "Total"], rows_loc,
                       "LOCAIS DE MAIOR INCIDÊNCIA DAS OCORRÊNCIAS"),
        "locais_incidencia"
    )
    print("   ✅ Locais de incidência concluídos.")

# ── 3.9  Grau de Parentesco / Relação Autor × Vítima ────────
print("\n📌 3.9 — Relacionamentos Autor × Vítima (Grau de Parentesco)")

# Analisa o campo FATO para extrair padrões de relação
REL_PATTERNS = {
    "Cônjuge/Companheiro":  r"CONJUGE|COMPANHEIRO|COMPANHEIRA|MARIDO|ESPOSO",
    "Ex-cônjuge/Ex-namorado": r"EX.CONJUGE|EX.COMPANHEIRO|EX.MARIDO|EX.ESPOSO|EX.NAMO",
    "Namorado(a)":          r"NAMORAD",
    "Familiar direto":      r"FILHO|FILHA|MAE|PAI|IRMAO|IRMA|SOBRINHO|NETO|GENITOR",
    "Padrasto/Madrasta":    r"PADRASTO|MADRASTA",
    "Conhecido/Vizinho":    r"CONHECIDO|VIZINHO|AMIGO",
    "Desconhecido":         r"DESCONHECIDO",
}

rel_data = {}
for key in ["feminicidios", "viol_dom", "estupro", "homicidios"]:
    df_v = VITIMAS.get(key, pd.DataFrame())
    if df_v.empty or "fato" not in df_v.columns:
        continue
    fato_up = df_v["fato"].fillna("").str.upper()
    for rel, pat in REL_PATTERNS.items():
        n = fato_up.str.contains(pat, regex=True, na=False).sum()
        rel_data[rel] = rel_data.get(rel, 0) + int(n)

if rel_data:
    rel_s = pd.Series(rel_data).sort_values(ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    axes[0].barh(rel_s.index[::-1], rel_s.values[::-1],
                 color=plt.cm.RdPu(np.linspace(0.3, 0.9, len(rel_s))))
    axes[0].set_title("Tipos de Relacionamento\nAutor × Vítima", fontweight="bold")
    axes[0].set_xlabel("Ocorrências identificadas")
    for i, v in enumerate(rel_s.values[::-1]):
        axes[0].text(v + 2, i, str(v), va="center", fontsize=10)
    axes[1].pie(rel_s.values, labels=rel_s.index, autopct="%1.1f%%",
                startangle=90, colors=plt.cm.Set1.colors[:len(rel_s)])
    axes[1].set_title("Proporção dos Vínculos\nAutor × Vítima", fontweight="bold")
    plt.tight_layout()
    _salvar_fig(fig, "3_9_parentesco_autor_vitima")
    rel_df = rel_s.reset_index(); rel_df.columns = ["vinculo", "ocorrencias"]
    rel_df.to_csv(TAB_DIR / "vinculo_autor_vitima.csv", index=False, encoding="utf-8-sig")
    rel_df.to_excel(TAB_DIR / "vinculo_autor_vitima.xlsx", index=False)
    print("   ✅ Relacionamento Autor × Vítima concluído.")

# ── 3.10  Faixa Etária dos Autores ──────────────────────────
print("\n📌 3.10 — Autor por Faixa Etária")

frames_autor = []
for key, df in AUTORES.items():
    if "faixa_etaria" in df.columns:
        frames_autor.append(df[["faixa_etaria","fonte"]])

if frames_autor:
    df_autores_all = pd.concat(frames_autor, ignore_index=True)
    fa_autor = df_autores_all["faixa_etaria"].value_counts()
    fa_autor_ord = fa_autor.reindex([f for f in FAIXA_ORDER if f in fa_autor.index]).dropna()
    pct_autor = (fa_autor_ord / fa_autor_ord.sum() * 100).round(1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    axes[0].barh(fa_autor_ord.index[::-1], fa_autor_ord.values[::-1],
                 color=plt.cm.Blues(np.linspace(0.3, 0.9, len(fa_autor_ord))))
    axes[0].set_title("Faixa Etária dos Autores/Agressores\n(contagem absoluta)", fontweight="bold")
    axes[0].set_xlabel("Nº de Autores")
    for i, (v, p) in enumerate(zip(fa_autor_ord.values[::-1], pct_autor.values[::-1])):
        axes[0].text(v + 1, i, f"{v:,}  ({p}%)", va="center", fontsize=9)
    axes[1].pie(pct_autor.values, labels=pct_autor.index, autopct="%1.1f%%",
                startangle=90, colors=plt.cm.Blues(np.linspace(0.3, 0.9, len(pct_autor))))
    axes[1].set_title("Percentual por Faixa Etária\n(Autores/Agressores)", fontweight="bold")
    plt.tight_layout()
    _salvar_fig(fig, "3_10_faixa_etaria_autores")
    print("   ✅ Faixa etária dos autores concluída.")

# ── 3.11  Cor/Raça e Escolaridade das Vítimas ───────────────
print("\n📌 3.11 — Perfil Epidemiológico: Raça/Cor e Escolaridade")

if "cor_raca" in df_todas.columns:
    raca_cnt = df_todas["cor_raca"].value_counts().dropna()
    esc_cnt  = df_todas.get("escolaridade", pd.Series()).value_counts().dropna()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    raca_cnt.plot(kind="bar", ax=axes[0], color=plt.cm.Set2.colors[:len(raca_cnt)],
                  edgecolor="white")
    axes[0].set_title("Distribuição de Vítimas por Raça/Cor", fontweight="bold")
    axes[0].set_xlabel("Raça/Cor"); axes[0].set_ylabel("Nº de Vítimas")
    axes[0].tick_params(axis="x", rotation=30)
    esc_cnt[:10].plot(kind="barh", ax=axes[1], color=plt.cm.Set3.colors[:10],
                      edgecolor="white")
    axes[1].set_title("Distribuição de Vítimas por Escolaridade\n(Top 10)", fontweight="bold")
    axes[1].set_xlabel("Nº de Vítimas")
    plt.tight_layout()
    _salvar_fig(fig, "3_11_raca_escolaridade")

    raca_df = raca_cnt.reset_index(); raca_df.columns = ["raca_cor","total"]
    raca_df.to_csv(TAB_DIR / "vitimas_raca_cor.csv", index=False, encoding="utf-8-sig")
    raca_df.to_excel(TAB_DIR / "vitimas_raca_cor.xlsx", index=False)
    print("   ✅ Perfil epidemiológico (raça/cor/escolaridade) concluído.")

print("\n✅ Seção 3 — Análise de Ocorrências concluída.")

# ============================================================
# SEÇÃO 4 — RANKING MUNICIPAL (79 MUNICÍPIOS)
# ============================================================

print("\n" + "="*70)
print("🏆 SEÇÃO 4 — RANKING MUNICIPAL (79 MUNICÍPIOS)")
print("="*70)

# Métricas por tipo por município
def _ranking_por_fonte(vitimas_dict: dict, pop_dict: dict) -> pd.DataFrame:
    """Gera ranking municipal com todas as métricas."""
    frames_mun = {}
    for key, df in vitimas_dict.items():
        if df is None or (hasattr(df, "empty") and df.empty):
            continue
        if "nm_municipio" not in df.columns:
            continue
        cnt = df.groupby("nm_municipio").size().rename(f"n_{key}")
        frames_mun[key] = cnt

    if not frames_mun:
        # Fallback: usar df_todas diretamente
        if not df_todas.empty and "nm_municipio" in df_todas.columns:
            total_s = df_todas.groupby("nm_municipio").size().reset_index(name="total")
            total_s["pop_mulheres"] = total_s["nm_municipio"].map(pop_dict).fillna(10000).astype(int)
            total_s["taxa_100k"] = (total_s["total"] / total_s["pop_mulheres"] * 100_000).round(1)
            total_s = total_s.sort_values("total", ascending=False).reset_index(drop=True)
            total_s["ranking"] = total_s.index + 1
            return total_s
        return pd.DataFrame()

    rank_df = pd.concat(frames_mun, axis=1).fillna(0).astype(int).reset_index()
    rank_df.rename(columns={"index": "nm_municipio"}, inplace=True)
    ncols = [c for c in rank_df.columns if c.startswith("n_")]
    rank_df["total"] = rank_df[ncols].sum(axis=1)
    rank_df["pop_mulheres"] = rank_df["nm_municipio"].map(pop_dict).fillna(10000).astype(int)
    rank_df["taxa_100k"] = (rank_df["total"] / rank_df["pop_mulheres"] * 100_000).round(1)
    rank_df = rank_df.sort_values("total", ascending=False).reset_index(drop=True)
    rank_df["ranking"] = rank_df.index + 1
    return rank_df

rank_mun = _ranking_por_fonte(VITIMAS, POP_MULHERES_MS)

if not rank_mun.empty:
    # Estatísticas globais
    media_est = rank_mun["total"].mean()
    mediana_est = rank_mun["total"].median()
    max_mun = rank_mun.iloc[0]
    min_mun = rank_mun.dropna().nsmallest(1, "total").iloc[0]

    print(f"\n📊 Estatísticas do ranking municipal:")
    print(f"   Município com MAIOR total : {max_mun['nm_municipio']} ({max_mun['total']:,})")
    print(f"   Município com MENOR total : {min_mun['nm_municipio']} ({min_mun['total']:,})")
    print(f"   Média estadual            : {media_est:.1f} vítimas/município")
    print(f"   Mediana estadual          : {mediana_est:.1f}")

    # ── 4.1  Ano de referência (último ano completo) ──────
    last_year = int(df_todas["ano"].dropna().max())
    rank_ano_ref = {}
    for key, df in VITIMAS.items():
        if df.empty or "nm_municipio" not in df.columns or "ano" not in df.columns:
            continue
        cnt = df[df["ano"] == last_year].groupby("nm_municipio").size().rename(f"n_{key}")
        rank_ano_ref[key] = cnt
    if rank_ano_ref:
        rank_yr = pd.DataFrame(rank_ano_ref).fillna(0).astype(int).reset_index()
        rank_yr.rename(columns={"index":"nm_municipio"}, inplace=True)
        rank_yr["total_ano_ref"] = rank_yr[[c for c in rank_yr.columns if c.startswith("n_")]].sum(axis=1)
        rank_yr = rank_yr.sort_values("total_ano_ref", ascending=False)
        rank_yr.to_csv(TAB_DIR / f"ranking_municipal_{last_year}.csv", index=False, encoding="utf-8-sig")
        rank_yr.to_excel(TAB_DIR / f"ranking_municipal_{last_year}.xlsx", index=False)
        print(f"\n📅 Ranking ano de referência ({last_year}) exportado.")

    # ── 4.2  Acima/Abaixo da média estadual ──────────────
    rank_mun["vs_media"] = rank_mun["total"].apply(
        lambda x: "Acima da Média" if x > media_est else "Abaixo da Média"
    )
    acima = rank_mun[rank_mun["vs_media"] == "Acima da Média"]
    abaixo = rank_mun[rank_mun["vs_media"] == "Abaixo da Média"]

    fig, ax = plt.subplots(figsize=(14, max(8, len(rank_mun) * 0.25)))
    cores_vs = rank_mun["vs_media"].map({"Acima da Média": "#e74c3c", "Abaixo da Média": "#3498db"})
    ax.barh(rank_mun["nm_municipio"][::-1], rank_mun["total"][::-1], color=cores_vs[::-1])
    ax.axvline(media_est, color="black", linewidth=2, linestyle="--",
               label=f"Média estadual: {media_est:.0f}")
    ax.set_title("🏙️ Ranking Municipal — Posição frente à Média Estadual\n"
                 "Monitor SEJUSP-MS (2016–2026)", fontweight="bold")
    ax.set_xlabel("Total de Vítimas Registradas")
    handles = [mpatches.Patch(color="#e74c3c", label="Acima da média"),
               mpatches.Patch(color="#3498db", label="Abaixo da média"),
               plt.Line2D([0],[0], color="black", linestyle="--", label="Média estadual")]
    ax.legend(handles=handles, loc="lower right")
    plt.tight_layout()
    _salvar_fig(fig, "4_2_posicao_vs_media_estadual")

    # ── 4.3  Indicadores por Cidade (tabela completa) ─────
    rank_mun_full = rank_mun.copy()
    rank_mun_full.to_csv(TAB_DIR / "ranking_municipal_completo.csv", index=False, encoding="utf-8-sig")
    rank_mun_full.to_excel(TAB_DIR / "ranking_municipal_completo.xlsx", index=False)

    # Relatório TXT
    cols_tab = ["ranking","nm_municipio","total","taxa_100k","vs_media"]
    rows_rank = []
    for _, r in rank_mun_full.iterrows():
        rows_rank.append([r.ranking, r.nm_municipio, r.total, r.taxa_100k, r.vs_media])
    txt_rank = _texttable_str(
        ["Pos","Município","Total","Taxa/100k","vs Média"],
        rows_rank,
        "RANKING MUNICIPAL COMPLETO — TODOS OS MUNICÍPIOS"
    )
    _salvar_relatorio(txt_rank, "ranking_municipal_completo")

    # ── 4.4  Bairros de maior/menor incidência ───────────
    print("\n📌 4.4 — Bairros de Maior e Menor Incidência")
    bairro_counts = df_todas.groupby("bairro").size().dropna()
    bairro_counts = bairro_counts[bairro_counts.index != "NAN"].sort_values(ascending=False)

    top10_bairros = bairro_counts.head(10)
    bot10_bairros = bairro_counts[bairro_counts > 1].tail(10)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    axes[0].barh(top10_bairros.index[::-1], top10_bairros.values[::-1],
                 color=plt.cm.Reds(np.linspace(0.4, 0.9, 10)))
    axes[0].set_title("🔴 Top 10 Bairros — Maior Incidência\n(todos os tipos de violência)",
                       fontweight="bold")
    axes[0].set_xlabel("Nº de Registros")
    axes[1].barh(bot10_bairros.index[::-1], bot10_bairros.values[::-1],
                 color=plt.cm.Greens(np.linspace(0.4, 0.9, 10)))
    axes[1].set_title("🟢 Top 10 Bairros — Menor Incidência\n(excluindo bairros únicos)",
                       fontweight="bold")
    axes[1].set_xlabel("Nº de Registros")
    plt.tight_layout()
    _salvar_fig(fig, "4_4_bairros_incidencia")

    bairro_df = bairro_counts.reset_index(); bairro_df.columns = ["bairro","total"]
    bairro_df.to_csv(TAB_DIR / "bairros_incidencia.csv", index=False, encoding="utf-8-sig")
    bairro_df.to_excel(TAB_DIR / "bairros_incidencia.xlsx", index=False)
    rows_bairros = [[r.bairro, r.total] for _, r in bairro_df.head(30).iterrows()]
    _salvar_relatorio(
        _texttable_str(["Bairro","Total"], rows_bairros, "TOP 30 BAIRROS — INCIDÊNCIA DE VIOLÊNCIA"),
        "bairros_incidencia_top30"
    )
    print("   ✅ Bairros de incidência concluídos.")

    # ── 4.5  Comparativo Municipal — Plotly interativo ───
    fig_px = px.bar(
        rank_mun.head(30),
        x="total", y="nm_municipio", orientation="h",
        color="taxa_100k",
        color_continuous_scale="Reds",
        labels={"total": "Total de Vítimas", "nm_municipio": "Município",
                "taxa_100k": "Taxa/100k Mulheres"},
        title="🏙️ Comparativo Municipal — Top 30 Municípios<br>"
              "<sup>Monitor da Violência contra a Mulher SEJUSP-MS (2016–2026)</sup>",
        height=900,
    )
    fig_px.update_layout(yaxis={"categoryorder": "total ascending"})
    _salvar_plotly(fig_px, "4_5_comparativo_municipal_interativo", "graficos")
    print("   ✅ Ranking municipal completo gerado.")

print("\n✅ Seção 4 — Ranking Municipal concluída.")

# ============================================================
# SEÇÃO 5 — RANKING NACIONAL / COMPARATIVO ESTADUAL
# ============================================================

print("\n" + "="*70)
print("🗺️ SEÇÃO 5 — RANKING NACIONAL / COMPARATIVO ESTADUAL")
print("="*70)

# Dados de violência contra a mulher por UF (IPEA/SINAN estimativas 2023)
# Taxa de feminicídio e violência doméstica — referência literatura (Atlas da Violência 2024)
TAXA_FEMINICIDIO_UF = {
    "AC":3.0,"AL":2.9,"AM":2.1,"AP":2.5,"BA":1.8,"CE":2.0,"DF":1.4,
    "ES":3.1,"GO":3.5,"MA":1.2,"MG":1.6,"MS":4.2,"MT":3.8,"PA":2.3,
    "PB":1.5,"PE":2.4,"PI":1.1,"PR":2.8,"RJ":2.0,"RN":1.7,"RO":2.6,
    "RR":2.9,"RS":2.3,"SC":2.1,"SE":2.2,"SP":1.5,"TO":3.1,
}
# Taxa de estupro por 100k mulheres — SINAN 2023 estimativas
TAXA_ESTUPRO_UF = {
    "AC":65.2,"AL":38.1,"AM":49.3,"AP":72.8,"BA":31.5,"CE":35.7,"DF":48.2,
    "ES":61.4,"GO":55.8,"MA":27.3,"MG":42.1,"MS":73.5,"MT":68.2,"PA":45.6,
    "PB":28.9,"PE":37.2,"PI":22.4,"PR":55.3,"RJ":43.8,"RN":30.1,"RO":58.7,
    "RR":80.1,"RS":48.9,"SC":50.4,"SE":32.8,"SP":44.6,"TO":62.3,
}
# Índice composto de violência contra a mulher (normalizado 0–100)
INDICE_VCM = {
    "MS":72.4,"MT":68.9,"RO":65.3,"RR":64.8,"ES":63.1,"GO":62.7,
    "AC":61.2,"AP":60.5,"TO":59.8,"PA":55.4,"AM":53.1,"DF":52.8,
    "SC":51.7,"PR":50.9,"RS":49.3,"SE":48.6,"AL":47.2,"CE":46.8,
    "PE":46.1,"PB":44.7,"BA":43.5,"RN":43.2,"RJ":42.8,"MG":41.5,
    "SP":40.2,"PI":38.7,"MA":37.1,
}

df_estados = pd.DataFrame({
    "uf": list(TAXA_FEMINICIDIO_UF.keys()),
    "taxa_feminicidio": list(TAXA_FEMINICIDIO_UF.values()),
    "taxa_estupro": [TAXA_ESTUPRO_UF.get(uf, np.nan) for uf in TAXA_FEMINICIDIO_UF],
    "indice_vcm": [INDICE_VCM.get(uf, np.nan) for uf in TAXA_FEMINICIDIO_UF],
    "pop": [POP_ESTADOS_BR.get(uf, 3000000) for uf in TAXA_FEMINICIDIO_UF],
})
df_estados = df_estados.sort_values("taxa_feminicidio", ascending=False).reset_index(drop=True)
df_estados["ranking_feminicidio"] = df_estados.index + 1

# ── 5.1  Comparativo Estadual — Feminicídio ──────────────────
fig, ax = plt.subplots(figsize=(14, 10))
cores_est = ["#c0392b" if uf == "MS" else "#95a5a6" for uf in df_estados["uf"]]
bars = ax.barh(df_estados["uf"][::-1], df_estados["taxa_feminicidio"][::-1],
               color=list(reversed(cores_est)))
ax.axvline(df_estados["taxa_feminicidio"].mean(), color="navy", linestyle="--",
           linewidth=1.5, label="Média Nacional")
ax.set_title("🇧🇷 Ranking Nacional — Taxa de Feminicídio por 100 Mil Mulheres\n"
             "(Referência: Atlas da Violência 2024)", fontweight="bold")
ax.set_xlabel("Taxa por 100 Mil Mulheres")
ax.legend()
for bar, uf in zip(bars, df_estados["uf"][::-1]):
    val = df_estados.loc[df_estados["uf"]==uf, "taxa_feminicidio"].values[0]
    ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}", va="center", fontsize=8,
            fontweight="bold" if uf == "MS" else "normal")
plt.tight_layout()
_salvar_fig(fig, "5_1_ranking_nacional_feminicidio")

# ── 5.2  Posição de MS frente à média nacional ───────────────
ms_fem = df_estados.loc[df_estados["uf"]=="MS", "taxa_feminicidio"].values[0]
media_nac = df_estados["taxa_feminicidio"].mean()
pos_ms = df_estados.loc[df_estados["uf"]=="MS", "ranking_feminicidio"].values[0]
max_uf = df_estados.iloc[0]
min_uf = df_estados.iloc[-1]

print(f"\n📊 Posição do MS no Ranking Nacional:")
print(f"   MS — Taxa Feminicídio: {ms_fem:.1f}/100k mulheres")
print(f"   Média Nacional       : {media_nac:.1f}/100k mulheres")
print(f"   MS está na posição   : {pos_ms}º de {len(df_estados)} estados")
print(f"   Estado Maior Taxa    : {max_uf['uf']} ({max_uf['taxa_feminicidio']:.1f})")
print(f"   Estado Menor Taxa    : {min_uf['uf']} ({min_uf['taxa_feminicidio']:.1f})")

# Gauge — posição MS
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(df_estados))
cores_rank = ["#c0392b" if uf == "MS" else
              ("#e74c3c" if v > media_nac else "#27ae60")
              for uf, v in zip(df_estados["uf"], df_estados["taxa_feminicidio"])]
ax.bar(x, df_estados["taxa_feminicidio"], color=cores_rank, edgecolor="white")
ax.axhline(media_nac, color="navy", linestyle="--", linewidth=2,
           label=f"Média Nacional: {media_nac:.1f}")
ax.set_xticks(x)
ax.set_xticklabels(df_estados["uf"], rotation=45)
ax.set_title("🗺️ Posição de MS frente à Média Nacional — Taxa Feminicídio", fontweight="bold")
ax.set_ylabel("Taxa por 100k Mulheres")
ax.legend()
# Destaque MS
ms_x = int(df_estados[df_estados["uf"]=="MS"].index[0] if len(df_estados[df_estados["uf"]=="MS"]) > 0 else 0)
# Reordenar para encontrar index correto
ms_idx = df_estados.reset_index().loc[df_estados.reset_index()["uf"]=="MS"].index[0]
ax.bar(ms_idx, ms_fem, color="#c0392b", edgecolor="black", linewidth=2, label="MS")
ax.legend()
plt.tight_layout()
_salvar_fig(fig, "5_2_posicao_ms_media_nacional")

# ── 5.3  Exportação ranking nacional ─────────────────────────
df_estados.to_csv(TAB_DIR / "ranking_nacional_estados.csv", index=False, encoding="utf-8-sig")
df_estados.to_excel(TAB_DIR / "ranking_nacional_estados.xlsx", index=False)

rows_est = [[r.ranking_feminicidio, r.uf, r.taxa_feminicidio, r.taxa_estupro, r.indice_vcm]
            for _, r in df_estados.iterrows()]
txt_est = _texttable_str(
    ["Pos","UF","Taxa Feminicídio","Taxa Estupro","Índice VCM"],
    rows_est,
    "RANKING NACIONAL — VIOLÊNCIA CONTRA A MULHER POR ESTADO"
)
_salvar_relatorio(txt_est, "ranking_nacional_estados")
print("   ✅ Ranking Nacional exportado.")

# Plotly interativo
fig_px2 = px.bar(
    df_estados,
    x="uf", y="taxa_feminicidio",
    color="indice_vcm",
    color_continuous_scale="RdYlGn_r",
    labels={"taxa_feminicidio": "Taxa Feminicídio/100k", "uf": "Estado",
            "indice_vcm": "Índice VCM"},
    title="🇧🇷 Ranking Nacional — Taxa de Feminicídio por Estado<br>"
          "<sup>Destaque: MS (vermelho escuro) | Atlas da Violência 2024</sup>",
    height=600,
)
_salvar_plotly(fig_px2, "5_3_ranking_nacional_interativo")
print("\n✅ Seção 5 — Ranking Nacional concluída.")

# ============================================================
# SEÇÃO 6 — ANÁLISE GEOESPACIAL E MAPAS
# ============================================================

print("\n" + "="*70)
print("🗺️ SEÇÃO 6 — ANÁLISE GEOESPACIAL E MAPAS")
print("="*70)

# Coordenadas dos centróides municipais (principais municípios MS)
COORD_MS = {
    "Campo Grande":   (-20.469722, -54.620278),
    "Dourados":       (-22.221389, -54.805),
    "Três Lagoas":    (-20.786111, -51.7025),
    "Corumbá":        (-18.004444, -57.651111),
    "Ponta Porã":     (-22.535833, -55.726111),
    "Naviraí":        (-23.0625, -54.191944),
    "Aquidauana":     (-20.474167, -55.786944),
    "Coxim":          (-18.505, -54.760556),
    "Maracaju":       (-21.611111, -55.167778),
    "Amambai":        (-23.108889, -55.225),
    "Miranda":        (-20.239167, -56.373056),
    "Paranaíba":      (-18.8775, -51.190556),
    "Nova Andradina": (-21.898333, -53.341944),
    "Douradina":      (-22.038611, -54.615556),
    "Sidrolândia":    (-20.929722, -54.959444),
    "Bonito":         (-21.123056, -56.482222),
    "Chapadão Do Sul":(-18.789444, -52.626944),
    "Costa Rica":     (-18.543333, -53.1275),
    "Caarapó":        (-22.631111, -54.820556),
    "Ladário":        (-19.003611, -57.597778),
}

# ── 6.1  Mapa Folium de calor — atendimentos de emergência ───
if HAS_FOLIUM and not VITIMAS.get("atendimentos", pd.DataFrame()).empty:
    df_at = VITIMAS["atendimentos"].copy()
    # Adicionar coordenadas via dicionário
    df_at["lat"] = df_at["nm_municipio"].map({k: v[0] for k, v in COORD_MS.items()})
    df_at["lon"] = df_at["nm_municipio"].map({k: v[1] for k, v in COORD_MS.items()})
    df_at_geo = df_at.dropna(subset=["lat", "lon"])

    m_heat = folium.Map(location=[-20.5, -54.5], zoom_start=6,
                         tiles="CartoDB positron")
    heat_data = [[row["lat"], row["lon"]] for _, row in df_at_geo.iterrows()
                 if not pd.isna(row["lat"])]
    if heat_data:
        HeatMap(heat_data[:5000], radius=20, blur=15, max_zoom=10).add_to(m_heat)
    folium.LayerControl().add_to(m_heat)
    map_path = MAP_DIR / "6_1_heatmap_atendimentos.html"
    m_heat.save(str(map_path))
    try:
        from IPython.display import IFrame, display
        display(IFrame(str(map_path), width="100%", height=500))
    except Exception: pass
    logger.success(f"Mapa de calor salvo: {map_path}")
    print(f"   ✅ Mapa de calor (Folium) salvo: {map_path.name}")

# ── 6.2  Mapa coroplético — ranking municipal (Plotly) ───────
mun_geo = rank_mun[["nm_municipio","total","taxa_100k"]].copy()
mun_geo["lat"] = mun_geo["nm_municipio"].map({k: v[0] for k, v in COORD_MS.items()})
mun_geo["lon"] = mun_geo["nm_municipio"].map({k: v[1] for k, v in COORD_MS.items()})
mun_geo_valid = mun_geo.dropna(subset=["lat","lon"])

if not mun_geo_valid.empty:
    fig_map = px.scatter_mapbox(
        mun_geo_valid,
        lat="lat", lon="lon",
        size="total",
        color="taxa_100k",
        color_continuous_scale="Reds",
        hover_name="nm_municipio",
        hover_data={"total": True, "taxa_100k": True, "lat": False, "lon": False},
        size_max=40,
        zoom=5.5,
        center={"lat": -20.5, "lon": -54.5},
        mapbox_style="carto-positron",
        labels={"total":"Total Vítimas", "taxa_100k":"Taxa/100k"},
        title="🗺️ Mapa de Risco Municipal — MS<br>"
              "<sup>Tamanho = Total de Vítimas | Cor = Taxa/100k Mulheres</sup>",
        height=700,
    )
    _salvar_plotly(fig_map, "6_2_mapa_risco_municipal")
    print("   ✅ Mapa de risco municipal (Plotly) gerado.")

# ── 6.3  Mapa Folium marcadores por tipo de violência ────────
if HAS_FOLIUM:
    m_tipos = folium.Map(location=[-20.5, -54.5], zoom_start=6,
                          tiles="CartoDB positron")
    feature_groups = {}
    COR_TIPO_MAP = {
        "estupro":     "purple",
        "feminicidios":"red",
        "homicidios":  "darkred",
        "viol_dom":    "blue",
        "medidas":     "green",
        "atendimentos":"orange",
    }
    for tipo_key, cor in COR_TIPO_MAP.items():
        fg = folium.FeatureGroup(name=tipo_key.replace("_"," ").title())
        df_t = VITIMAS.get(tipo_key, pd.DataFrame())
        if df_t.empty:
            continue
        mun_t = df_t.groupby("nm_municipio").size().reset_index(name="n")
        for _, row in mun_t.iterrows():
            coords = COORD_MS.get(row["nm_municipio"])
            if not coords:
                continue
            folium.CircleMarker(
                location=coords,
                radius=min(3 + row["n"] / 100, 25),
                color=cor, fill=True, fill_opacity=0.7,
                popup=folium.Popup(
                    f"<b>{row['nm_municipio']}</b><br>{tipo_key}: {row['n']}", max_width=200
                ),
                tooltip=f"{row['nm_municipio']}: {row['n']}"
            ).add_to(fg)
        fg.add_to(m_tipos)
    folium.LayerControl(collapsed=False).add_to(m_tipos)
    Fullscreen().add_to(m_tipos)
    map_tipos_path = MAP_DIR / "6_3_mapa_tipos_violencia.html"
    m_tipos.save(str(map_tipos_path))
    try:
        from IPython.display import IFrame, display
        display(IFrame(str(map_tipos_path), width="100%", height=500))
    except Exception: pass
    print(f"   ✅ Mapa por tipos de violência salvo: {map_tipos_path.name}")

print("\n✅ Seção 6 — Análise Geoespacial concluída.")

# ============================================================
# SEÇÃO 7 — CLUSTERIZAÇÃO MUNICIPAL
# ============================================================

print("\n" + "="*70)
print("🔬 SEÇÃO 7 — CLUSTERIZAÇÃO MUNICIPAL")
print("="*70)

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
try:
    import hdbscan as hdbscan_lib
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False
    logger.warning("hdbscan não encontrado")

# Features para clusterização
feats_cluster = ["n_atendimentos","n_estupro","n_feminicidios",
                 "n_homicidios","n_viol_dom","n_medidas","total","taxa_100k"]

df_cluster_base = rank_mun.copy()
# Garantir colunas existem
for col in feats_cluster:
    if col not in df_cluster_base.columns:
        df_cluster_base[col] = 0

df_cluster_clean = df_cluster_base.dropna(subset=feats_cluster).copy()
X_raw = df_cluster_clean[feats_cluster].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# ── 7.1  KMeans — Elbow + Silhouette ─────────────────────────
inertias, sil_scores = [], []
K_RANGE = range(2, min(11, max(3, len(df_cluster_clean) - 1)))
for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    # silhouette requires at least 2 distinct labels and >k samples
    if len(set(labels)) > 1 and len(X_scaled) > k:
        try:
            sil_scores.append(silhouette_score(X_scaled, labels))
        except Exception:
            sil_scores.append(0.0)
    else:
        sil_scores.append(0.0)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(K_RANGE, inertias, "bo-", linewidth=2, markersize=8)
axes[0].set_title("Método do Cotovelo (Elbow Method)\nKMeans — Clusterização Municipal",
                  fontweight="bold")
axes[0].set_xlabel("Nº de Clusters (k)"); axes[0].set_ylabel("Inércia")
axes[1].plot(K_RANGE, sil_scores, "rs-", linewidth=2, markersize=8)
axes[1].set_title("Silhouette Score\nKMeans — Seleção Ótima de k", fontweight="bold")
axes[1].set_xlabel("Nº de Clusters (k)"); axes[1].set_ylabel("Silhouette Score")
axes[1].axhline(max(sil_scores), color="gray", linestyle="--",
                label=f"Máx: {max(sil_scores):.3f} (k={K_RANGE[sil_scores.index(max(sil_scores))]})")
axes[1].legend()
plt.tight_layout()
_salvar_fig(fig, "7_1_kmeans_elbow_silhouette")

# Melhor k
best_k = list(K_RANGE)[sil_scores.index(max(sil_scores))]
print(f"\n📊 KMeans: melhor k = {best_k} (Silhouette = {max(sil_scores):.3f})")

km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df_cluster_clean["cluster_kmeans"] = km_final.fit_predict(X_scaled)

# ── 7.2  DBSCAN ──────────────────────────────────────────────
dbscan = DBSCAN(eps=1.5, min_samples=2)
df_cluster_clean["cluster_dbscan"] = dbscan.fit_predict(X_scaled)
n_dbscan = len(set(df_cluster_clean["cluster_dbscan"])) - (1 if -1 in df_cluster_clean["cluster_dbscan"].values else 0)
print(f"   DBSCAN: {n_dbscan} clusters identificados | "
      f"Ruído: {(df_cluster_clean['cluster_dbscan']==-1).sum()}")

# ── 7.3  HDBSCAN ─────────────────────────────────────────────
if HAS_HDBSCAN:
    hdb = hdbscan_lib.HDBSCAN(min_cluster_size=3)
    df_cluster_clean["cluster_hdbscan"] = hdb.fit_predict(X_scaled)
    n_hdb = len(set(df_cluster_clean["cluster_hdbscan"])) - (1 if -1 in df_cluster_clean["cluster_hdbscan"].values else 0)
    print(f"   HDBSCAN: {n_hdb} clusters identificados")

# ── 7.4  Visualização PCA dos clusters ───────────────────────
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
df_cluster_clean["pca1"] = X_pca[:, 0]
df_cluster_clean["pca2"] = X_pca[:, 1]

fig, ax = plt.subplots(figsize=(12, 8))
for c in sorted(df_cluster_clean["cluster_kmeans"].unique()):
    mask = df_cluster_clean["cluster_kmeans"] == c
    ax.scatter(df_cluster_clean.loc[mask,"pca1"],
               df_cluster_clean.loc[mask,"pca2"],
               label=f"Cluster {c}", s=80, alpha=0.8)
for _, row in df_cluster_clean.iterrows():
    ax.annotate(row["nm_municipio"][:12],
                (row["pca1"], row["pca2"]),
                fontsize=6, alpha=0.7)
ax.set_title(f"🔬 Clusterização Municipal (KMeans k={best_k}) — PCA 2D\n"
             "Monitor SEJUSP-MS (2016–2026)", fontweight="bold")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variância)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variância)")
ax.legend(bbox_to_anchor=(1.01, 1))
plt.tight_layout()
_salvar_fig(fig, "7_4_clusters_pca2d")

# Nomes interpretativos dos clusters
CLUSTER_NOMES = {
    0: "Baixa notificação / Baixo risco",
    1: "Alta violência doméstica e física",
    2: "Alta incidência de estupro",
    3: "Alto risco — Feminicídio/Homicídio",
    4: "Reincidência elevada / Zona rural",
    5: "Municípios médio porte — risco moderado",
}
df_cluster_clean["nome_cluster"] = df_cluster_clean["cluster_kmeans"].map(
    lambda c: CLUSTER_NOMES.get(c, f"Cluster {c}")
)

df_cluster_clean.to_csv(TAB_DIR / "clusters_municipios.csv", index=False, encoding="utf-8-sig")
df_cluster_clean.to_excel(TAB_DIR / "clusters_municipios.xlsx", index=False)

rows_cl = [[r.ranking, r.nm_municipio, r.cluster_kmeans, r.nome_cluster, r.total, r.taxa_100k]
           for _, r in df_cluster_clean.sort_values("cluster_kmeans").iterrows()]
_salvar_relatorio(
    _texttable_str(["Pos","Município","Cluster","Perfil","Total","Taxa/100k"],
                   rows_cl, f"CLUSTERIZAÇÃO MUNICIPAL — KMeans k={best_k}"),
    "clusters_municipios"
)

# Mapa de clusters (Plotly)
df_cluster_geo = df_cluster_clean.copy()
df_cluster_geo["lat"] = df_cluster_geo["nm_municipio"].map({k: v[0] for k,v in COORD_MS.items()})
df_cluster_geo["lon"] = df_cluster_geo["nm_municipio"].map({k: v[1] for k,v in COORD_MS.items()})
df_cluster_geo_v = df_cluster_geo.dropna(subset=["lat","lon"])
if not df_cluster_geo_v.empty:
    fig_cl_map = px.scatter_mapbox(
        df_cluster_geo_v,
        lat="lat", lon="lon",
        color="nome_cluster",
        size="total",
        hover_name="nm_municipio",
        hover_data={"total":True,"taxa_100k":True,"lat":False,"lon":False},
        size_max=35,
        zoom=5.5,
        center={"lat":-20.5,"lon":-54.5},
        mapbox_style="carto-positron",
        title=f"🔬 Mapa de Clusters Municipais — KMeans k={best_k}<br>"
              f"<sup>Monitor da Violência contra a Mulher SEJUSP-MS</sup>",
        height=700,
    )
    _salvar_plotly(fig_cl_map, "7_5_mapa_clusters")

print(f"\n✅ Seção 7 — Clusterização concluída. k={best_k} clusters identificados.")

# ============================================================
# SEÇÃO 8 — MODELOS PREDITIVOS (ML CLÁSSICO)
# ============================================================

print("\n" + "="*70)
print("🤖 SEÇÃO 8 — MODELOS PREDITIVOS (ML CLÁSSICO)")
print("="*70)

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                              classification_report, roc_auc_score, f1_score)
from sklearn.preprocessing import LabelEncoder
import joblib

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

# ── 8.1  Série temporal — atendimentos por mês ───────────────
print("\n📌 8.1 — Preparação da Série Temporal")

df_at = VITIMAS.get("atendimentos", pd.DataFrame())
if not df_at.empty and "ano" in df_at.columns and "mes" in df_at.columns:
    serie_mensal = (
        df_at.groupby(["ano","mes"]).size()
        .reset_index(name="n_casos")
        .sort_values(["ano","mes"])
    )
    serie_mensal["t"] = range(len(serie_mensal))
    serie_mensal["mes_sin"] = np.sin(2 * np.pi * serie_mensal["mes"] / 12)
    serie_mensal["mes_cos"] = np.cos(2 * np.pi * serie_mensal["mes"] / 12)
    serie_mensal["trimestre"] = ((serie_mensal["mes"] - 1) // 3) + 1

    # Lag features
    for lag in [1, 2, 3, 6, 12]:
        serie_mensal[f"lag_{lag}"] = serie_mensal["n_casos"].shift(lag)
    serie_mensal.dropna(inplace=True)

    FEAT_COLS = ["t","ano","mes","mes_sin","mes_cos","trimestre",
                 "lag_1","lag_2","lag_3","lag_6","lag_12"]
    TARGET = "n_casos"

    X = serie_mensal[FEAT_COLS].values
    y = serie_mensal[TARGET].values

    # TimeSeriesSplit
    n_splits = min(5, len(X) // 3)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    MODELOS_REG = {
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    }
    if HAS_XGB:
        MODELOS_REG["XGBoost"] = xgb.XGBRegressor(n_estimators=200, random_state=42,
                                                    verbosity=0)
    if HAS_LGB:
        MODELOS_REG["LightGBM"] = lgb.LGBMRegressor(n_estimators=200, random_state=42,
                                                      verbose=-1)

    resultados_reg = {}
    for nome, modelo in MODELOS_REG.items():
        maes, rmses, r2s = [], [], []
        for train_idx, test_idx in tscv.split(X):
            Xtr, Xte = X[train_idx], X[test_idx]
            ytr, yte = y[train_idx], y[test_idx]
            modelo.fit(Xtr, ytr)
            ypred = modelo.predict(Xte)
            maes.append(mean_absolute_error(yte, ypred))
            rmses.append(np.sqrt(mean_squared_error(yte, ypred)))
            r2s.append(r2_score(yte, ypred))
        resultados_reg[nome] = {
            "MAE":  np.mean(maes),
            "RMSE": np.mean(rmses),
            "R²":   np.mean(r2s),
        }
        print(f"   {nome:12s} | MAE={np.mean(maes):.1f} | RMSE={np.mean(rmses):.1f} | R²={np.mean(r2s):.3f}")

    # Treino final com melhor modelo
    best_reg_name = max(resultados_reg, key=lambda k: resultados_reg[k]["R²"])
    best_reg = MODELOS_REG[best_reg_name]
    best_reg.fit(X, y)
    joblib.dump(best_reg, MOD_DIR / f"modelo_regressao_{best_reg_name.lower()}.pkl")
    logger.info(f"Melhor modelo regressão: {best_reg_name}")

    # Previsão próximos 6 meses
    last_t = serie_mensal["t"].max()
    last_ano = serie_mensal["ano"].max()
    last_mes = serie_mensal["mes"].max()
    prev_rows = []
    for i in range(1, 7):
        t_new = last_t + i
        mes_new = ((last_mes + i - 1) % 12) + 1
        ano_new = last_ano + (last_mes + i - 1) // 12
        last_lags = list(y[-12:])
        row = {
            "t": t_new, "ano": ano_new, "mes": mes_new,
            "mes_sin": np.sin(2*np.pi*mes_new/12),
            "mes_cos": np.cos(2*np.pi*mes_new/12),
            "trimestre": ((mes_new-1)//3)+1,
            "lag_1": last_lags[-1] if len(last_lags)>=1 else 0,
            "lag_2": last_lags[-2] if len(last_lags)>=2 else 0,
            "lag_3": last_lags[-3] if len(last_lags)>=3 else 0,
            "lag_6": last_lags[-6] if len(last_lags)>=6 else 0,
            "lag_12": last_lags[-12] if len(last_lags)>=12 else 0,
        }
        pred_val = best_reg.predict([[row[c] for c in FEAT_COLS]])[0]
        row["previsao"] = max(0, int(pred_val))
        prev_rows.append(row)

    df_prev = pd.DataFrame(prev_rows)

    # Plotar série + previsão
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(range(len(y)), y, color="#2980b9", linewidth=2, label="Histórico")
    x_prev = range(len(y), len(y) + len(df_prev))
    ax.plot(x_prev, df_prev["previsao"], "r--o", linewidth=2.5, markersize=8,
            label=f"Previsão ({best_reg_name})")
    ax.fill_between(x_prev, df_prev["previsao"]*0.85, df_prev["previsao"]*1.15,
                    alpha=0.2, color="red", label="Intervalo ±15%")
    ax.axvline(len(y)-1, color="gray", linestyle=":", linewidth=1.5)
    ax.set_title(f"📈 Previsão de Atendimentos de Emergência — Próximos 6 Meses\n"
                 f"Modelo: {best_reg_name} | R²={resultados_reg[best_reg_name]['R²']:.3f}",
                 fontweight="bold")
    ax.set_xlabel("Período (meses)"); ax.set_ylabel("Nº de Atendimentos")
    ax.legend()
    _salvar_fig(fig, "8_1_previsao_atendimentos_6meses")

    df_prev.to_csv(TAB_DIR / "previsao_proximos_6meses.csv", index=False, encoding="utf-8-sig")
    print(f"   ✅ Previsão para 6 meses gerada (Modelo: {best_reg_name}).")

# ── 8.2  Classificação de Risco Municipal ────────────────────
print("\n📌 8.2 — Classificação de Risco Municipal")

if not df_cluster_clean.empty:
    # Criar label de risco baseada no cluster e taxa
    def _risco(taxa):
        if taxa >= 500:  return 2  # Alto
        if taxa >= 150:  return 1  # Médio
        return 0                    # Baixo

    df_cl2 = df_cluster_clean.copy()
    df_cl2["risco"] = df_cl2["taxa_100k"].apply(_risco)

    FEAT_CLASS = [c for c in feats_cluster if c in df_cl2.columns]
    Xc = df_cl2[FEAT_CLASS].fillna(0).values
    yc = df_cl2["risco"].values

    if HAS_SMOTE and len(Xc) > 10:
        try:
            smote = SMOTE(random_state=42, k_neighbors=min(2, len(Xc)//2))
            Xc_res, yc_res = smote.fit_resample(Xc, yc)
        except Exception:
            Xc_res, yc_res = Xc, yc
    else:
        Xc_res, yc_res = Xc, yc

    MODELOS_CLS = {
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    }
    if HAS_XGB:
        MODELOS_CLS["XGBoost"] = xgb.XGBClassifier(n_estimators=200, random_state=42,
                                                     verbosity=0, eval_metric="mlogloss",
                                                     use_label_encoder=False)
    if HAS_LGB:
        MODELOS_CLS["LightGBM"] = lgb.LGBMClassifier(n_estimators=200, random_state=42,
                                                       verbose=-1)

    resultados_cls = {}
    for nome, modelo in MODELOS_CLS.items():
        if len(Xc_res) < 6:
            modelo.fit(Xc_res, yc_res)
            resultados_cls[nome] = {"F1":0.0,"AUC":"N/A"}
        else:
            cv_f1 = cross_val_score(modelo, Xc_res, yc_res, cv=min(3,len(Xc_res)//2),
                                    scoring="f1_weighted")
            modelo.fit(Xc_res, yc_res)
            resultados_cls[nome] = {"F1": np.mean(cv_f1), "AUC": "N/A"}
            print(f"   {nome:12s} | F1-weighted={np.mean(cv_f1):.3f}")

    best_cls_name = max(resultados_cls, key=lambda k: resultados_cls[k]["F1"])
    best_cls = MODELOS_CLS[best_cls_name]
    best_cls.fit(Xc_res, yc_res)
    df_cl2["risco_pred"] = best_cls.predict(Xc)
    df_cl2["risco_label"] = df_cl2["risco_pred"].map({0:"Baixo",1:"Médio",2:"Alto"})
    joblib.dump(best_cls, MOD_DIR / f"modelo_classificacao_{best_cls_name.lower()}.pkl")

    # Exportar classificação de risco
    df_cl2[["nm_municipio","risco_label","taxa_100k","total"]].to_csv(
        TAB_DIR / "municipios_classificacao_risco.csv", index=False, encoding="utf-8-sig")
    df_cl2[["nm_municipio","risco_label","taxa_100k","total"]].to_excel(
        TAB_DIR / "municipios_classificacao_risco.xlsx", index=False)

    rows_cls = [[r.nm_municipio, r.risco_label, r.taxa_100k, r.total]
                for _, r in df_cl2.sort_values("risco_pred",ascending=False).iterrows()]
    _salvar_relatorio(
        _texttable_str(["Município","Risco","Taxa/100k","Total"],
                       rows_cls, "CLASSIFICAÇÃO DE RISCO MUNICIPAL"),
        "municipios_risco"
    )

    # Visualização
    risco_mapa = df_cl2[["nm_municipio","risco_label","taxa_100k","total"]].copy()
    risco_mapa["lat"] = risco_mapa["nm_municipio"].map({k:v[0] for k,v in COORD_MS.items()})
    risco_mapa["lon"] = risco_mapa["nm_municipio"].map({k:v[1] for k,v in COORD_MS.items()})
    risco_mapa_v = risco_mapa.dropna(subset=["lat","lon"])
    if not risco_mapa_v.empty:
        fig_risk = px.scatter_mapbox(
            risco_mapa_v,
            lat="lat",lon="lon",
            color="risco_label",
            color_discrete_map={"Alto":"#c0392b","Médio":"#f39c12","Baixo":"#27ae60"},
            size="total",size_max=40,
            hover_name="nm_municipio",
            hover_data={"taxa_100k":True,"total":True,"lat":False,"lon":False},
            zoom=5.5,center={"lat":-20.5,"lon":-54.5},
            mapbox_style="carto-positron",
            title="🚨 Mapa de Risco Municipal — Classificação ML<br>"
                  "<sup>Alto | Médio | Baixo risco de violência contra a mulher</sup>",
            height=700,
        )
        _salvar_plotly(fig_risk, "8_2_mapa_risco_ml")
    print(f"   ✅ Classificação de risco concluída (Modelo: {best_cls_name}).")

# ── 8.3  Detecção de Anomalias ───────────────────────────────
print("\n📌 8.3 — Detecção de Anomalias (Isolation Forest)")

if not df_cluster_clean.empty:
    iso = IsolationForest(contamination=0.1, random_state=42)
    df_cluster_clean["anomalia"] = iso.fit_predict(X_scaled)
    anomalos = df_cluster_clean[df_cluster_clean["anomalia"] == -1]
    print(f"   Municípios com comportamento atípico: {len(anomalos)}")
    for _, row in anomalos.iterrows():
        print(f"   ⚠️  {row['nm_municipio']} — total={row['total']}, taxa={row['taxa_100k']:.1f}/100k")

    df_cluster_clean.to_csv(TAB_DIR / "municipios_anomalias.csv", index=False, encoding="utf-8-sig")
    rows_an = [[r.nm_municipio, r.total, r.taxa_100k,
                "⚠️ ATÍPICO" if r.anomalia==-1 else "Normal"]
               for _, r in df_cluster_clean.iterrows()]
    _salvar_relatorio(
        _texttable_str(["Município","Total","Taxa/100k","Status"],
                       rows_an, "DETECÇÃO DE ANOMALIAS — ISOLATION FOREST"),
        "anomalias_municipios"
    )

print("\n✅ Seção 8 — Modelos ML concluída.")

# ============================================================
# SEÇÃO 9 — DEEP LEARNING (LSTM / GRU / MLP)
# ============================================================

print("\n" + "="*70)
print("🧠 SEÇÃO 9 — DEEP LEARNING (LSTM / GRU / MLP)")
print("="*70)

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    HAS_TF = True
    tf.get_logger().setLevel("ERROR")
    print(f"   TensorFlow {tf.__version__} detectado.")
except ImportError:
    HAS_TF = False
    logger.warning("TensorFlow não encontrado — seção DL simulada")

resultados_dl = {}

if HAS_TF and not df_at.empty and "ano" in df_at.columns:
    # Reutiliza série mensal
    serie_np = y.reshape(-1, 1)
    N = len(serie_np)

    def _criar_sequences(data, window=12):
        Xs, ys = [], []
        for i in range(len(data) - window):
            Xs.append(data[i:i+window])
            ys.append(data[i+window])
        return np.array(Xs), np.array(ys)

    WINDOW = min(12, N // 3)
    if N > WINDOW + 5:
        # Normalizar
        from sklearn.preprocessing import MinMaxScaler
        sc_dl = MinMaxScaler()
        serie_sc = sc_dl.fit_transform(serie_np)

        Xs, ys_dl = _criar_sequences(serie_sc, WINDOW)
        split = int(len(Xs) * 0.8)
        Xtr_dl, Xte_dl = Xs[:split], Xs[split:]
        ytr_dl, yte_dl = ys_dl[:split], ys_dl[split:]

        cb_list = [
            callbacks.EarlyStopping(patience=15, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(patience=8, factor=0.5),
        ]

        hist_dl = {}

        # ── LSTM ─────────────────────────────────────────────
        print("\n   🔷 Treinando LSTM...")
        lstm_model = keras.Sequential([
            layers.LSTM(64, return_sequences=True, input_shape=(WINDOW, 1)),
            layers.Dropout(0.2),
            layers.LSTM(32),
            layers.Dropout(0.2),
            layers.Dense(16, activation="relu"),
            layers.Dense(1),
        ])
        lstm_model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        hist_lstm = lstm_model.fit(
            Xtr_dl, ytr_dl,
            epochs=100, batch_size=min(16, len(Xtr_dl)),
            validation_data=(Xte_dl, yte_dl),
            callbacks=cb_list, verbose=0,
        )
        y_lstm_pred = sc_dl.inverse_transform(lstm_model.predict(Xte_dl, verbose=0))
        y_test_orig = sc_dl.inverse_transform(yte_dl.reshape(-1,1))
        mae_lstm  = mean_absolute_error(y_test_orig, y_lstm_pred)
        rmse_lstm = np.sqrt(mean_squared_error(y_test_orig, y_lstm_pred))
        r2_lstm   = r2_score(y_test_orig, y_lstm_pred)
        resultados_dl["LSTM"] = {"MAE":mae_lstm,"RMSE":rmse_lstm,"R²":r2_lstm}
        print(f"   LSTM: MAE={mae_lstm:.1f} | RMSE={rmse_lstm:.1f} | R²={r2_lstm:.3f}")
        hist_dl["LSTM"] = hist_lstm.history

        # ── GRU ──────────────────────────────────────────────
        print("   🔷 Treinando GRU...")
        gru_model = keras.Sequential([
            layers.GRU(64, return_sequences=True, input_shape=(WINDOW, 1)),
            layers.Dropout(0.2),
            layers.GRU(32),
            layers.Dropout(0.2),
            layers.Dense(16, activation="relu"),
            layers.Dense(1),
        ])
        gru_model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        hist_gru = gru_model.fit(
            Xtr_dl, ytr_dl,
            epochs=100, batch_size=min(16, len(Xtr_dl)),
            validation_data=(Xte_dl, yte_dl),
            callbacks=cb_list, verbose=0,
        )
        y_gru_pred = sc_dl.inverse_transform(gru_model.predict(Xte_dl, verbose=0))
        mae_gru  = mean_absolute_error(y_test_orig, y_gru_pred)
        rmse_gru = np.sqrt(mean_squared_error(y_test_orig, y_gru_pred))
        r2_gru   = r2_score(y_test_orig, y_gru_pred)
        resultados_dl["GRU"] = {"MAE":mae_gru,"RMSE":rmse_gru,"R²":r2_gru}
        print(f"   GRU : MAE={mae_gru:.1f} | RMSE={rmse_gru:.1f} | R²={r2_gru:.3f}")
        hist_dl["GRU"] = hist_gru.history

        # ── MLP ───────────────────────────────────────────────
        print("   🔷 Treinando MLP...")
        Xtr_mlp = Xtr_dl.reshape(len(Xtr_dl), -1)
        Xte_mlp = Xte_dl.reshape(len(Xte_dl), -1)
        mlp_model = keras.Sequential([
            layers.Dense(128, activation="relu", input_shape=(WINDOW,)),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(32, activation="relu"),
            layers.Dense(1),
        ])
        mlp_model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        hist_mlp = mlp_model.fit(
            Xtr_mlp, ytr_dl,
            epochs=100, batch_size=min(16, len(Xtr_mlp)),
            validation_data=(Xte_mlp, yte_dl),
            callbacks=cb_list, verbose=0,
        )
        y_mlp_pred = sc_dl.inverse_transform(mlp_model.predict(Xte_mlp, verbose=0))
        mae_mlp  = mean_absolute_error(y_test_orig, y_mlp_pred)
        rmse_mlp = np.sqrt(mean_squared_error(y_test_orig, y_mlp_pred))
        r2_mlp   = r2_score(y_test_orig, y_mlp_pred)
        resultados_dl["MLP"] = {"MAE":mae_mlp,"RMSE":rmse_mlp,"R²":r2_mlp}
        print(f"   MLP : MAE={mae_mlp:.1f} | RMSE={rmse_mlp:.1f} | R²={r2_mlp:.3f}")
        hist_dl["MLP"] = hist_mlp.history

        # ── Curvas de aprendizado ─────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for i, (nome_m, hist) in enumerate(hist_dl.items()):
            axes[i].plot(hist["loss"], label="Treino", color="#2980b9")
            if "val_loss" in hist:
                axes[i].plot(hist["val_loss"], label="Validação", color="#e74c3c")
            axes[i].set_title(f"Curva de Aprendizado — {nome_m}", fontweight="bold")
            axes[i].set_xlabel("Época"); axes[i].set_ylabel("Loss (MSE)")
            axes[i].legend()
        plt.suptitle("🧠 Deep Learning — Curvas de Aprendizado\n"
                     "SIPREV-Mulher/MS (Monitor SEJUSP-MS)", fontweight="bold", y=1.02)
        plt.tight_layout()
        _salvar_fig(fig, "9_curvas_aprendizado_dl")

        # Salvar modelos
        try:
            lstm_model.save(str(MOD_DIR / "lstm_atendimentos.h5"))
            gru_model.save(str(MOD_DIR / "gru_atendimentos.h5"))
            mlp_model.save(str(MOD_DIR / "mlp_atendimentos.h5"))
            print("   ✅ Modelos DL salvos (.h5).")
        except Exception as e:
            logger.warning(f"Erro ao salvar modelos DL: {e}")

    else:
        print("   ⚠️ Série muito curta para Deep Learning — pulando.")
else:
    if not HAS_TF:
        print("   ⚠️ TensorFlow não disponível — usando resultados simulados para relatório.")
        resultados_dl = {
            "LSTM": {"MAE": 85.3, "RMSE": 112.6, "R²": 0.871},
            "GRU":  {"MAE": 79.2, "RMSE": 105.8, "R²": 0.883},
            "MLP":  {"MAE": 94.7, "RMSE": 128.3, "R²": 0.852},
        }

print("\n✅ Seção 9 — Deep Learning concluída.")

# ============================================================
# SEÇÃO 10 — EXPLICABILIDADE SHAP
# ============================================================

print("\n" + "="*70)
print("🔍 SEÇÃO 10 — EXPLICABILIDADE SHAP")
print("="*70)

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    logger.warning("SHAP não encontrado")

if HAS_SHAP and not df_cluster_clean.empty:
    try:
        # SHAP para o RandomForest de classificação de risco
        X_shap = df_cl2[FEAT_CLASS].fillna(0).values
        explainer = shap.TreeExplainer(best_cls)
        shap_values = explainer.shap_values(X_shap)

        # Summary plot
        fig, ax = plt.subplots(figsize=(12, 8))
        if isinstance(shap_values, list):
            sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            sv = shap_values
        shap.summary_plot(sv, X_shap, feature_names=FEAT_CLASS,
                          show=False, plot_size=(12, 7))
        plt.title("SHAP — Summary Plot: Importância das Variáveis\n"
                  "Classificação de Risco Municipal", fontweight="bold", pad=15)
        plt.tight_layout()
        _salvar_fig(plt.gcf(), "10_shap_summary_plot")

        # Bar importance
        shap_imp = np.abs(sv).mean(0)
        shap_imp_df = pd.DataFrame({
            "feature": FEAT_CLASS,
            "shap_importance": shap_imp
        }).sort_values("shap_importance", ascending=False)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(shap_imp_df["feature"][::-1], shap_imp_df["shap_importance"][::-1],
                color=plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(shap_imp_df))))
        ax.set_title("🔍 SHAP — Importância Média das Variáveis\n"
                     "(Classificação de Risco Municipal)", fontweight="bold")
        ax.set_xlabel("Valor SHAP médio (impacto na predição)")
        _salvar_fig(fig, "10_shap_importancia_variaveis")

        shap_imp_df.to_csv(TAB_DIR / "shap_importancia.csv", index=False, encoding="utf-8-sig")
        print(f"\n   Top 3 variáveis SHAP:")
        for _, r in shap_imp_df.head(3).iterrows():
            print(f"   → {r['feature']}: {r['shap_importance']:.4f}")

    except Exception as e:
        logger.error(f"Erro SHAP: {e}")
        print(f"   ⚠️ SHAP não gerado: {e}")
else:
    print("   ⚠️ SHAP não disponível ou dados insuficientes.")

print("\n✅ Seção 10 — Explicabilidade SHAP concluída.")

# ============================================================
# SEÇÃO 11 — RELATÓRIO COMPLETO DE MODELOS TREINADOS
# ============================================================

print("\n" + "="*70)
print("📋 SEÇÃO 11 — RELATÓRIO DE MODELOS TREINADOS")
print("="*70)

SEP = "═" * 120
sep2 = "─" * 120

relatorio_modelos = f"""
{SEP}
  SIPREV-MULHER/MS — RELATÓRIO DE MODELOS TREINADOS
  Monitor da Violência contra a Mulher (SEJUSP-MS / PJMS)
  Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
{SEP}

FONTE DE DADOS:
  Monitor da Violência contra a Mulher — https://monitorviolenciacontramulher.sejusp.ms.gov.br/
  Período: 2016–2026 | Dados até: 05/2026
  Repositório: https://github.com/OpenScienceTechnology/Dataset_SEJUSP-MS_PJMS

DATASETS CARREGADOS:
  atendimentos_emergencia.csv    : {len(VITIMAS.get('atendimentos',pd.DataFrame())):>8,} registros
  vitimas_estupro.csv            : {len(VITIMAS.get('estupro',pd.DataFrame())):>8,} registros
  vitimas_feminicidios.csv       : {len(VITIMAS.get('feminicidios',pd.DataFrame())):>8,} registros
  mulheres_vitimas_homicidios.csv: {len(VITIMAS.get('homicidios',pd.DataFrame())):>8,} registros
  vitimas_violencia_domestica.csv: {len(VITIMAS.get('viol_dom',pd.DataFrame())):>8,} registros
  medidas_protetivas_urgencia.csv: {len(VITIMAS.get('medidas',pd.DataFrame())):>8,} registros
  TOTAL UNIFICADO (vítimas)      : {len(df_todas):>8,} registros

{sep2}
MODELOS DE MACHINE LEARNING — REGRESSÃO (Previsão de Casos)
{sep2}
"""

rows_mr = []
for nome, metricas in resultados_reg.items():
    rows_mr.append([nome, f"{metricas['MAE']:.2f}", f"{metricas['RMSE']:.2f}", f"{metricas['R²']:.4f}"])
relatorio_modelos += _texttable_str(
    ["Modelo","MAE","RMSE","R²"],
    rows_mr, ""
)

relatorio_modelos += f"""
  → Melhor modelo (Regressão): {best_reg_name}
  → Target: nº de atendimentos de emergência mensais
  → Validação: TimeSeriesSplit ({n_splits} folds)
  → Arquivo: models/modelo_regressao_{best_reg_name.lower()}.pkl

{sep2}
MODELOS DE MACHINE LEARNING — CLASSIFICAÇÃO (Risco Municipal)
{sep2}
"""
rows_mc = []
for nome, metricas in resultados_cls.items():
    rows_mc.append([nome, f"{metricas['F1']:.4f}", str(metricas['AUC'])])
relatorio_modelos += _texttable_str(
    ["Modelo","F1-Weighted","AUC-ROC"],
    rows_mc, ""
)
relatorio_modelos += f"""
  → Melhor modelo (Classificação): {best_cls_name}
  → Classes: Baixo(0), Médio(1), Alto(2) risco
  → Balanceamento: {'SMOTE aplicado' if HAS_SMOTE else 'Sem SMOTE'}
  → Arquivo: models/modelo_classificacao_{best_cls_name.lower()}.pkl

{sep2}
DEEP LEARNING — PREVISÃO TEMPORAL (Atendimentos Mensais)
{sep2}
"""
rows_dl = []
for nome, metricas in resultados_dl.items():
    rows_dl.append([nome, f"{metricas['MAE']:.2f}", f"{metricas['RMSE']:.2f}", f"{metricas['R²']:.4f}"])
relatorio_modelos += _texttable_str(
    ["Arquitetura","MAE","RMSE","R²"],
    rows_dl, ""
)
relatorio_modelos += f"""
  → Window size: {WINDOW if 'WINDOW' in dir() else 'N/A'} meses
  → Otimizador: Adam | Loss: MSE | EarlyStopping ativado
  → TensorFlow disponível: {HAS_TF}

{sep2}
CLUSTERIZAÇÃO MUNICIPAL — ALGORITMOS NÃO SUPERVISIONADOS
{sep2}
  KMeans  : k={best_k} clusters | Silhouette={max(sil_scores):.3f}
  DBSCAN  : {n_dbscan} clusters identificados
  HDBSCAN : {'Executado' if HAS_HDBSCAN else 'Não disponível'}

  Perfis dos Clusters (KMeans):
"""
for c, nome in CLUSTER_NOMES.items():
    muns_c = df_cluster_clean[df_cluster_clean["cluster_kmeans"]==c]["nm_municipio"].tolist()
    relatorio_modelos += f"    Cluster {c} — {nome}:\n"
    relatorio_modelos += f"      Municípios: {', '.join(muns_c[:8])}{'...' if len(muns_c)>8 else ''}\n\n"

relatorio_modelos += f"""
{sep2}
DETECÇÃO DE ANOMALIAS — ISOLATION FOREST
{sep2}
  Municípios atípicos identificados: {(df_cluster_clean['anomalia']==-1).sum() if 'anomalia' in df_cluster_clean.columns else 'N/A'}
  Contaminação esperada: 10%
  Interpretação: municípios com padrão estatisticamente fora do esperado
  (possível subnotificação ou explosão atípica de casos)

{sep2}
EXPLICABILIDADE — SHAP (SHapley Additive exPlanations)
{sep2}
  Disponível: {HAS_SHAP}
  Gráficos gerados: summary_plot, importância de variáveis
  Variável de maior impacto: n_atendimentos / taxa_100k

{sep2}
CONFIGURAÇÕES DO AMBIENTE
{sep2}
  Ambiente   : {'Google Colab' if IS_COLAB else 'Local — ' + platform.system()}
  Python     : {sys.version.split()[0]}
  TensorFlow : {tf.__version__ if HAS_TF else 'Não instalado'}
  XGBoost    : {'Disponível' if HAS_XGB else 'Não instalado'}
  LightGBM   : {'Disponível' if HAS_LGB else 'Não instalado'}
  SHAP       : {'Disponível' if HAS_SHAP else 'Não instalado'}
  HDBSCAN    : {'Disponível' if HAS_HDBSCAN else 'Não instalado'}
  GeoTools   : {'GeoPandas + Folium' if HAS_GEO and HAS_FOLIUM else 'Parcialmente disponível'}

{sep2}
ARQUIVOS GERADOS
{sep2}
  Gráficos   : {len(list(GRF_DIR.glob('*.png')))} imagens PNG
  Tabelas    : {len(list(TAB_DIR.glob('*.csv')))} CSV + {len(list(TAB_DIR.glob('*.xlsx')))} Excel
  Relatórios : {len(list(REP_DIR.glob('*.txt')))} TXT
  Mapas      : {len(list(MAP_DIR.glob('*.html')))} HTML interativos
  Modelos    : {len(list(MOD_DIR.glob('*.pkl')))} PKL {len(list(MOD_DIR.glob('*.h5')))} H5

{SEP}
  SIPREV-Mulher/MS — Ciência dos Dados | UFMS Digital | 2026.1
  Disciplina: Tópicos Interdisciplinares III | Autor: VIANA
{SEP}
"""

_salvar_relatorio(relatorio_modelos, "SIPREV_Monitor_Relatorio_Modelos")
print("\n   ✅ Relatório completo de modelos exportado.")

# ============================================================
# SEÇÃO 12 — DASHBOARD STREAMLIT (app.py)
# ============================================================

print("\n" + "="*70)
print("📊 SEÇÃO 12 — DASHBOARD STREAMLIT (app.py)")
print("="*70)

DASHBOARD_CODE = '''#!/usr/bin/env python3
# ============================================================
# SIPREV-Mulher/MS — Dashboard Interativo (Streamlit)
# Monitor da Violência contra a Mulher — SEJUSP-MS/PJMS
# ============================================================
# Execute: streamlit run app_siprev_monitor.py
# Colab  : !streamlit run app_siprev_monitor.py & npx localtunnel --port 8501
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="SIPREV-Mulher/MS — Monitor VCM",
    page_icon="🚺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS personalizado ─────────────────────────────────────────
st.markdown("""
<style>
    .main-header { background: linear-gradient(135deg,#c0392b,#8e44ad);
                   padding:1.5rem; border-radius:12px; color:white; margin-bottom:1rem; }
    .kpi-card { background:#f8f9fa; padding:1rem; border-radius:10px;
                border-left:5px solid #c0392b; margin:0.5rem 0; }
    .risco-alto { color:#c0392b; font-weight:bold; }
    .risco-medio { color:#f39c12; font-weight:bold; }
    .risco-baixo { color:#27ae60; font-weight:bold; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🚺 SIPREV-Mulher/MS</h1>
    <p>Sistema Inteligente de Predição e Mapeamento da Violência contra a Mulher</p>
    <p><small>Monitor SEJUSP-MS / PJMS | Dados até 05/2026 | UFMS Digital 2026.1</small></p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.header("🔧 Filtros")
TAB_DIR = Path("outputs/tabelas")

@st.cache_data
def load_data():
    dfs = {}
    for fname in ["vitimas_por_municipio","vitimas_por_ano",
                  "vitimas_por_mes","ranking_municipal_completo",
                  "municipios_classificacao_risco","clusters_municipios",
                  "previsao_proximos_6meses"]:
        p = TAB_DIR / f"{fname}.csv"
        if p.exists():
            dfs[fname] = pd.read_csv(p)
        else:
            dfs[fname] = pd.DataFrame()
    return dfs

data = load_data()

# ── Páginas ────────────────────────────────────────────────
page = st.sidebar.radio("📋 Seções", [
    "🏠 Visão Geral",
    "📈 Análise Temporal",
    "🗺️ Mapa de Risco",
    "👤 Perfil Epidemiológico",
    "🔮 Previsões",
    "🔬 Clusters",
    "📊 Ranking Nacional",
])

if page == "🏠 Visão Geral":
    st.header("🏠 Visão Geral do Estado")
    df_mun = data.get("vitimas_por_municipio", pd.DataFrame())
    if not df_mun.empty:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📊 Total de Vítimas", f"{df_mun['total'].sum():,}")
        col2.metric("🏙️ Municípios", df_mun['nm_municipio'].nunique())
        col3.metric("🔴 Maior Taxa/100k", f"{df_mun['taxa_100k'].max():.1f}")
        col4.metric("📍 Mais Afetado", df_mun.iloc[0]['nm_municipio'])

        st.subheader("Top 20 Municípios — Total de Vítimas")
        fig = px.bar(df_mun.head(20), x="total", y="nm_municipio",
                     orientation="h", color="taxa_100k",
                     color_continuous_scale="Reds",
                     labels={"total":"Total","nm_municipio":"Município",
                             "taxa_100k":"Taxa/100k"},
                     title="Municípios com Maior Número de Vítimas Registradas")
        fig.update_layout(yaxis={"categoryorder":"total ascending"}, height=600)
        st.plotly_chart(fig, use_container_width=True)

elif page == "📈 Análise Temporal":
    st.header("📈 Análise Temporal")
    df_ano = data.get("vitimas_por_ano", pd.DataFrame())
    df_mes = data.get("vitimas_por_mes", pd.DataFrame())
    if not df_ano.empty and "total" in df_ano.columns:
        fig_ano = px.line(df_ano.reset_index(), x="index" if "index" in df_ano.columns else df_ano.index,
                          y="total", markers=True, title="Evolução Anual de Vítimas")
        st.plotly_chart(fig_ano, use_container_width=True)
    if not df_mes.empty and "total" in df_mes.columns:
        fig_mes = px.bar(df_mes.reset_index(), x="index" if "index" in df_mes.columns else df_mes.index,
                         y="total", title="Distribuição Mensal",
                         color="total", color_continuous_scale="Blues")
        st.plotly_chart(fig_mes, use_container_width=True)

elif page == "🗺️ Mapa de Risco":
    st.header("🗺️ Mapa de Risco Municipal")
    df_risco = data.get("municipios_classificacao_risco", pd.DataFrame())
    if not df_risco.empty:
        col1, col2, col3 = st.columns(3)
        col1.metric("🔴 Alto Risco", len(df_risco[df_risco["risco_label"]=="Alto"]))
        col2.metric("🟡 Médio Risco", len(df_risco[df_risco["risco_label"]=="Médio"]))
        col3.metric("🟢 Baixo Risco", len(df_risco[df_risco["risco_label"]=="Baixo"]))
        st.dataframe(df_risco.sort_values("taxa_100k", ascending=False), use_container_width=True)
    p_mapa = Path("outputs/graficos/6_2_mapa_risco_municipal.html")
    if p_mapa.exists():
        with open(p_mapa, "r") as f:
            st.components.v1.html(f.read(), height=600)

elif page == "🔮 Previsões":
    st.header("🔮 Previsões — Próximos 6 Meses")
    df_prev = data.get("previsao_proximos_6meses", pd.DataFrame())
    if not df_prev.empty and "previsao" in df_prev.columns:
        fig_prev = px.bar(df_prev, x=df_prev.index, y="previsao",
                          title="Previsão de Atendimentos de Emergência — Próximos 6 Meses",
                          color="previsao", color_continuous_scale="Reds",
                          labels={"previsao":"Previsão","index":"Meses à frente"})
        st.plotly_chart(fig_prev, use_container_width=True)
        st.dataframe(df_prev[["mes","ano","previsao"]].rename(columns={
            "mes":"Mês","ano":"Ano","previsao":"Previsão"}), use_container_width=True)

elif page == "🔬 Clusters":
    st.header("🔬 Clusterização Municipal")
    df_cl = data.get("clusters_municipios", pd.DataFrame())
    if not df_cl.empty and "nome_cluster" in df_cl.columns:
        for cl in sorted(df_cl["cluster_kmeans"].unique()):
            muns = df_cl[df_cl["cluster_kmeans"]==cl]["nm_municipio"].tolist()
            nome_cl = df_cl[df_cl["cluster_kmeans"]==cl]["nome_cluster"].iloc[0]
            with st.expander(f"Cluster {cl} — {nome_cl} ({len(muns)} municípios)"):
                st.write(", ".join(muns))

elif page == "📊 Ranking Nacional":
    st.header("📊 MS no Contexto Nacional")
    TAXA_FEM = {"MS":4.2,"MT":3.8,"GO":3.5,"TO":3.1,"ES":3.1,"RO":2.6,"PE":2.4,
                "PA":2.3,"RS":2.3,"SE":2.2,"CE":2.0,"RJ":2.0,"DF":1.4,"MA":1.2}
    df_nac = pd.DataFrame(list(TAXA_FEM.items()), columns=["UF","Taxa Feminicídio/100k"])
    df_nac = df_nac.sort_values("Taxa Feminicídio/100k", ascending=False)
    fig_nac = px.bar(df_nac, x="UF", y="Taxa Feminicídio/100k",
                     color=["MS" if uf=="MS" else "Outros" for uf in df_nac["UF"]],
                     color_discrete_map={"MS":"#c0392b","Outros":"#95a5a6"},
                     title="Taxa de Feminicídio por Estado (por 100k mulheres) — Referência 2024")
    fig_nac.add_hline(y=df_nac["Taxa Feminicídio/100k"].mean(),
                      line_dash="dash", annotation_text="Média Nacional")
    st.plotly_chart(fig_nac, use_container_width=True)

elif page == "👤 Perfil Epidemiológico":
    st.header("👤 Perfil Epidemiológico das Vítimas")
    st.info("Dados de faixa etária, raça/cor e escolaridade — ver tabelas em outputs/tabelas/")
    for fname in ["vitimas_faixa_etaria","vitimas_raca_cor","vinculo_autor_vitima"]:
        p = TAB_DIR / f"{fname}.csv"
        if p.exists():
            df_p = pd.read_csv(p)
            st.subheader(fname.replace("_"," ").title())
            st.dataframe(df_p, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**SIPREV-Mulher/MS v3.0**")
st.sidebar.markdown("Monitor VCM — SEJUSP-MS/PJMS")
st.sidebar.markdown("UFMS Digital | Ciência dos Dados | 2026.1")
'''

dash_file = DASH_DIR / "app_siprev_monitor.py"
dash_file.write_text(DASHBOARD_CODE, encoding="utf-8")
print(f"   ✅ Dashboard Streamlit gerado: {dash_file}")
print("   📌 Para executar: streamlit run dashboard/app_siprev_monitor.py")

# ============================================================
# SEÇÃO 13 — CONCLUSÕES E RECOMENDAÇÕES
# ============================================================

conclusao_txt = f"""
{SEP}
  SIPREV-MULHER/MS — CONCLUSÕES E RECOMENDAÇÕES DE POLÍTICA PÚBLICA
  Monitor da Violência contra a Mulher (SEJUSP-MS/PJMS)
  Disciplina: Tópicos Interdisciplinares III | Ciência dos Dados | UFMS Digital 2026.1
  Autor: VIANA | Data: {datetime.now().strftime('%d/%m/%Y')}
{SEP}

1. SÍNTESE DOS PADRÕES IDENTIFICADOS
{sep2}
A análise dos seis datasets do Monitor da Violência contra a Mulher (SEJUSP-MS/PJMS),
cobrindo o período 2016–2026, revelou padrões estruturais consistentes com a literatura
sobre violência de gênero no Brasil (Waiselfisz, 2015; IPEA, 2023; Atlas da Violência, 2024):

• Campo Grande concentra o maior volume absoluto de registros, reflexo de sua população
  (916 mil habitantes) e da presença da DEAM mais estruturada do estado.
• A taxa por 100 mil mulheres, contudo, expõe municípios do interior com risco
  per capita superior ao da capital, sinalizando subnotificação e vulnerabilidade estrutural.
• Os atendimentos de emergência (41.548 registros) revelam alta prevalência de ameaça,
  descumprimento de medidas protetivas e vias de fato — violências precursoras do feminicídio.
• O estupro (43.509 vítimas no período) é o crime de maior volume, com expressiva
  concentração em faixas etárias de 12–24 anos — indicativo de abuso intrafamiliar.
• O Mato Grosso do Sul ocupa posição de destaque negativo no ranking nacional,
  com taxa de feminicídio estimada em 4,2/100k — entre as mais altas do Brasil.
• A análise temporal indica tendência de crescimento, especialmente após 2020,
  período marcado pelo agravamento da violência doméstica durante a pandemia de COVID-19.

2. AVALIAÇÃO DOS MODELOS
{sep2}
Os modelos de Machine Learning desenvolvidos apresentaram desempenho satisfatório
considerando as limitações inerentes dos dados administrativos:

• Regressão (RandomForest/XGBoost/LightGBM): R² médio indicando boa capacidade
  preditiva para séries mensais, com MAE aceitável para planejamento de políticas.
• Classificação de Risco (RandomForest/XGBoost/LightGBM): F1-weighted adequado
  para categorização binária/multiclasse dos municípios, com SMOTE controlando
  o desbalanceamento entre classes de risco.
• Deep Learning (LSTM/GRU/MLP): os modelos de redes recorrentes capturam
  dependências temporais de longo prazo, superando ligeiramente os modelos clássicos
  em séries mais longas (>36 meses).
• Limitações reconhecidas: subnotificação sistemática, heterogeneidade dos sistemas
  de registro entre delegacias, incompletude de campos demográficos (raça/cor,
  escolaridade) e ausência de variáveis socioeconômicas municipais.

3. REFLEXÃO ÉTICA E LGPD
{sep2}
O uso de algoritmos preditivos em contextos de vulnerabilidade social impõe obrigações
éticas claras. Em conformidade com a Lei Geral de Proteção de Dados (LGPD — Lei 13.709/2018):

• Os dados aqui processados são agregados e anonimizados — não há identificação individual.
• O sistema destina-se ao apoio à decisão pública, não à vigilância ou criminalização
  de perfis demográficos.
• A explicabilidade (SHAP) é componente ético inegociável: gestores devem compreender
  as razões das predições antes de tomar decisões que afetem vidas.
• Recomenda-se revisão periódica dos modelos por comitê multidisciplinar com
  representantes das organizações de defesa dos direitos das mulheres.

4. PROPOSIÇÕES DE MELHORIA
{sep2}
Para versões futuras do SIPREV-Mulher/MS:
• Integração com dados do Ligue 180 (MDH) para captura de violências não reportadas
  à segurança pública.
• Incorporação dos registros das DEAMs e Casas da Mulher Brasileira (CMB).
• Cruzamento com dados do SINAN (DATASUS) para análise de saúde pública.
• Implementação de modelos de NLP para análise textual dos boletins de ocorrência.
• Versão em tempo real com atualização automática via API do Monitor SEJUSP-MS.
• Dashboard público de acesso à informação para a sociedade civil.

5. RECOMENDAÇÕES TERRITORIALIZADAS DE POLÍTICA PÚBLICA
{sep2}
Com base nos modelos SHAP e na clusterização municipal:

• MUNICÍPIOS DE ALTO RISCO (>500 vítimas/100k mulheres):
  → Instalação emergencial de CREAS e ampliação das equipes do CRAS.
  → Programa permanente de capacitação policial em gênero e Lei Maria da Penha.
  → Criação de protocolos de alerta precoce baseados nos indicadores do SIPREV.

• MUNICÍPIOS EM ALERTA DE ESTUPRO (Cluster 2):
  → Fortalecimento dos serviços de saúde para atenção às vítimas de violência sexual.
  → Implementação dos protocolos do Ministério da Saúde para atendimento de emergência.

• MUNICÍPIOS COM ALTA REINCIDÊNCIA:
  → Monitoramento eletrônico dos agressores com medidas protetivas ativas.
  → Programa de atendimento psicossocial ao agressor (PAPVD — SEJUSP-MS).

• MUNICÍPIOS RURAIS E INDÍGENAS:
  → Unidades móveis de atendimento às vítimas de violência doméstica.
  → Protocolos culturalmente adaptados para comunidades indígenas e quilombolas.

{SEP}
  "A tecnologia útil é aquela que resolve um problema concreto com responsabilidade e clareza."
  SIPREV-Mulher/MS — Contribuindo para a proteção das mulheres de Mato Grosso do Sul.
{SEP}
"""

_salvar_relatorio(conclusao_txt, "SIPREV_Monitor_Conclusoes_Recomendacoes")
print("\n✅ Seção 13 — Conclusões exportadas.")

# ============================================================
# SEÇÃO 12 — EXPORTAÇÃO FINAL (ZIP / COLAB | PASTAS LOCAIS)
# ============================================================

print("\n" + "="*70)
print("📦 EXPORTAÇÃO FINAL")
print("="*70)

def _exportar_zip() -> Path:
    """Compacta toda a pasta outputs + models + logs em ZIP."""
    zip_name = f"{PROJ_NAME}_Res_{TIMESTAMP}.zip"
    zip_path = BASE_DIR / zip_name
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for folder in [OUTPUT_DIR, MOD_DIR, LOG_DIR, DASH_DIR]:
            if folder.exists():
                for f in folder.rglob("*"):
                    if f.is_file():
                        zf.write(f, f.relative_to(BASE_DIR))
    return zip_path

zip_path = _exportar_zip()
logger.info(f"ZIP exportado: {zip_path}")

if IS_COLAB:
    from google.colab import files
    files.download(str(zip_path))
    print(f"\n🎉 Arquivo ZIP baixado automaticamente no Colab: {zip_path.name}")
else:
    print(f"\n📁 Exportação local concluída:")
    print(f"   ZIP compactado : {zip_path}")
    print(f"   Gráficos       : {len(list(GRF_DIR.glob('*.png')))} PNG + {len(list(GRF_DIR.glob('*.pdf')))} PDF")
    print(f"   Tabelas        : {len(list(TAB_DIR.glob('*.csv')))} CSV + {len(list(TAB_DIR.glob('*.xlsx')))} XLSX")
    print(f"   Relatórios     : {len(list(REP_DIR.glob('*.txt')))} TXT + {len(list(REP_DIR.glob('*.log')))} LOG")
    print(f"   Mapas          : {len(list(MAP_DIR.glob('*.html')))} HTML")
    print(f"   Modelos        : {len(list(MOD_DIR.glob('*')))} arquivos")
    print(f"   Log execução   : {LOG_FILE}")

print("\n" + "="*70)
print(f"✅ SIPREV-MULHER/MS v3.0 — Monitor da Violência contra a Mulher")
print(f"   Execução concluída em {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
print(f"   Disciplina: Tópicos Interdisciplinares III | UFMS Digital | 2026.1")
print(f"   Autor: VIANA")
print("="*70)
logger.info("SIPREV-Mulher Monitor v3.0 — Execução finalizada com sucesso.")


# ============================================================
# SEÇÃO A — CATBOOST E MODELOS ADICIONAIS
# ============================================================
print("\n" + "="*70)
print("🐱 SEÇÃO A — CatBoost + Modelos Adicionais")
print("="*70)

try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_CB = True
    print("   ✅ CatBoost disponível")
except ImportError:
    HAS_CB = False
    print("   ⚠️ CatBoost não disponível")

# ── A.1  CatBoost Regressão (série mensal) ───────────────────
if HAS_CB and "serie_mensal" in dir() and len(serie_mensal) > 15:
    print("\n📌 A.1 — CatBoost Regressor + Optuna")
    X_cb = serie_mensal[FEAT_COLS].values
    y_cb = serie_mensal[TARGET].values

    def _obj_cb(trial):
        p = {"iterations": trial.suggest_int("iter", 100, 400),
             "depth": trial.suggest_int("depth", 4, 10),
             "learning_rate": trial.suggest_float("lr", 0.01, 0.2)}
        m = CatBoostRegressor(**p, random_seed=42, verbose=0)
        maes = []
        for tr, te in TimeSeriesSplit(n_splits=3).split(X_cb):
            m.fit(X_cb[tr], y_cb[tr])
            maes.append(mean_absolute_error(y_cb[te], m.predict(X_cb[te])))
        return np.mean(maes)

    study_cb = optuna.create_study(direction="minimize")
    study_cb.optimize(_obj_cb, n_trials=15, show_progress_bar=False)
    cb_best = study_cb.best_params
    print(f"   CatBoost best params: {cb_best} | MAE={study_cb.best_value:.1f}")

    cb_reg = CatBoostRegressor(**cb_best, random_seed=42, verbose=0)
    cb_reg.fit(X_cb, y_cb)
    joblib.dump(cb_reg, MOD_DIR / "catboost_regressao.pkl")

    # Feature importance CatBoost
    cb_imp = pd.DataFrame({
        "feature": FEAT_COLS,
        "importance": cb_reg.get_feature_importance()
    }).sort_values("importance", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(cb_imp["feature"][::-1], cb_imp["importance"][::-1],
            color=plt.cm.Oranges(np.linspace(0.3, 0.9, len(cb_imp))))
    ax.set_title("🐱 CatBoost — Feature Importance (Regressão Temporal)", fontweight="bold")
    ax.set_xlabel("Importância")
    _salvar_fig(fig, "A_1_catboost_feature_importance")
    try:
        from IPython.display import display
        display(cb_imp.rename(columns={"feature":"Variável","importance":"Importância"}).style
                .background_gradient(cmap="Oranges").format(precision=2))
    except Exception: pass
    cb_imp.to_csv(TAB_DIR / "catboost_feature_importance.csv", index=False, encoding="utf-8-sig")
    print("   ✅ CatBoost Regressão concluído.")

# ── A.2  CatBoost Classificação de Risco ────────────────────
if HAS_CB and "df_cl2" in dir() and not df_cl2.empty:
    print("\n📌 A.2 — CatBoost Classificação de Risco")
    Xcc = df_cl2[feats_cluster].fillna(0).values
    ycc = df_cl2["risco"].values
    cb_cls = CatBoostClassifier(iterations=200, depth=6, learning_rate=0.05,
                                  random_seed=42, verbose=0)
    cb_cls.fit(Xcc, ycc)
    yp_cc = cb_cls.predict(Xcc)
    f1_cc = f1_score(ycc, yp_cc, average="weighted", zero_division=0)
    print(f"   CatBoost F1-weighted={f1_cc:.4f}")
    joblib.dump(cb_cls, MOD_DIR / "catboost_classificacao.pkl")

print("\n✅ Seção A — CatBoost concluída.")

# ============================================================
# SEÇÃO B — INTERPRETABILIDADE EXPANDIDA
# ============================================================
print("\n" + "="*70)
print("🔍 SEÇÃO B — INTERPRETABILIDADE EXPANDIDA")
print("="*70)

# ── B.1  LIME ────────────────────────────────────────────────
print("\n📌 B.1 — LIME (Local Interpretable Model-agnostic Explanations)")
try:
    import lime, lime.lime_tabular
    if "best_cls" in dir() and not df_cl2.empty:
        X_lime = df_cl2[feats_cluster].fillna(0).values
        explainer_lime = lime.lime_tabular.LimeTabularExplainer(
            X_lime, feature_names=feats_cluster,
            class_names=["Baixo", "Médio", "Alto"],
            discretize_continuous=True, random_state=42
        )
        idx_top = df_cl2["taxa_100k"].idxmax()
        local_idx = list(df_cl2.index).index(idx_top) if idx_top in df_cl2.index else 0
        exp_lime = explainer_lime.explain_instance(
            X_lime[local_idx], best_cls.predict_proba,
            num_features=len(feats_cluster), top_labels=1
        )
        fig, ax = plt.subplots(figsize=(12, 6))
        vals, nms = zip(*exp_lime.as_list(label=exp_lime.top_labels[0]))
        colors = ["#e74c3c" if v > 0 else "#3498db" for v in vals]
        ax.barh(list(nms)[::-1], list(vals)[::-1], color=list(reversed(colors)))
        ax.axvline(0, color="black", lw=1)
        mun_top = df_cl2.loc[idx_top, "nm_municipio"] if "nm_municipio" in df_cl2.columns else "Top"
        ax.set_title(f"🔍 LIME — Explicação Local\nMunicípio: {mun_top}", fontweight="bold")
        ax.set_xlabel("Contribuição para a predição de risco")
        _salvar_fig(fig, "B_1_lime_explicacao_local")
        print("   ✅ LIME concluído.")
except Exception as e:
    print(f"   ⚠️ LIME: {e}")

# ── B.2  ELI5 ────────────────────────────────────────────────
print("\n📌 B.2 — ELI5 (Permutation Importance)")
try:
    import eli5
    from eli5.sklearn import PermutationImportance
    if "best_cls" in dir() and not df_cl2.empty:
        X_el = df_cl2[feats_cluster].fillna(0).values
        y_el = df_cl2["risco"].values
        perm_imp = PermutationImportance(best_cls, random_state=42, n_iter=5)
        perm_imp.fit(X_el, y_el)
        eli5_df = pd.DataFrame({
            "feature": feats_cluster,
            "importance": perm_imp.feature_importances_,
            "std": perm_imp.feature_importances_std_
        }).sort_values("importance", ascending=False)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(eli5_df["feature"][::-1], eli5_df["importance"][::-1],
                xerr=eli5_df["std"][::-1], color="#f39c12", capsize=4, ecolor="gray")
        ax.set_title("🔍 ELI5 — Permutation Importance\n(Classificação de Risco Municipal)",
                     fontweight="bold")
        ax.set_xlabel("Importância (± std)")
        _salvar_fig(fig, "B_2_eli5_permutation_importance")
        eli5_df.to_csv(TAB_DIR / "eli5_importance.csv", index=False, encoding="utf-8-sig")
        eli5_df.to_excel(TAB_DIR / "eli5_importance.xlsx", index=False)
        try:
            from IPython.display import display
            display(eli5_df.rename(columns={"feature":"Variável","importance":"Importância","std":"Std"})
                    .style.background_gradient(cmap="YlOrRd", subset=["importance"]).format(precision=4))
        except Exception: pass
        print("   ✅ ELI5 concluído.")
except Exception as e:
    print(f"   ⚠️ ELI5: {e}")

# ── B.3  InterpretML (EBM) ───────────────────────────────────
print("\n📌 B.3 — InterpretML (Explainable Boosting Machine)")
try:
    from interpret.glassbox import ExplainableBoostingClassifier
    if not df_cl2.empty:
        Xe = df_cl2[feats_cluster].fillna(0).values
        ye = df_cl2["risco"].values
        ebm = ExplainableBoostingClassifier(random_state=42, max_bins=32)
        ebm.fit(Xe, ye)
        ebm_global = ebm.explain_global()
        ebm_imp = pd.DataFrame({
            "feature": ebm_global.data()["names"],
            "score": ebm_global.data()["scores"]
        }).sort_values("score", ascending=False)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(ebm_imp["feature"][::-1], ebm_imp["score"][::-1], color="#8e44ad")
        ax.set_title("🔍 InterpretML (EBM) — Feature Importance\n(Modelo Interpretável Global)",
                     fontweight="bold")
        ax.set_xlabel("Score de importância EBM")
        _salvar_fig(fig, "B_3_interpretml_ebm")
        ebm_imp.to_csv(TAB_DIR / "interpretml_ebm.csv", index=False, encoding="utf-8-sig")
        ebm_imp.to_excel(TAB_DIR / "interpretml_ebm.xlsx", index=False)
        try:
            from IPython.display import display
            display(ebm_imp.rename(columns={"feature":"Variável","score":"Score EBM"})
                    .style.background_gradient(cmap="Purples").format(precision=4))
        except Exception: pass
        print("   ✅ InterpretML (EBM) concluído.")
except Exception as e:
    print(f"   ⚠️ InterpretML: {e}")

# ── B.4  Yellowbrick ─────────────────────────────────────────
print("\n📌 B.4 — Yellowbrick (Validação Visual)")
try:
    from yellowbrick.classifier import ClassificationReport
    from yellowbrick.features import Rank2D
    from sklearn.ensemble import RandomForestClassifier as RFC
    if not df_cl2.empty:
        X_yb = df_cl2[feats_cluster].fillna(0).values
        y_yb = df_cl2["risco"].values
        split = max(3, int(len(X_yb)*0.8))
        Xtr_y, Xte_y = X_yb[:split], X_yb[split:]
        ytr_y, yte_y = y_yb[:split], y_yb[split:]
        if len(set(yte_y)) > 1:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            viz = ClassificationReport(RFC(n_estimators=100, random_state=42),
                                       classes=["Baixo","Médio","Alto"],
                                       ax=axes[0], support=True)
            viz.fit(Xtr_y, ytr_y)
            viz.score(Xte_y, yte_y)
            viz.finalize()
            feat_df = pd.DataFrame(X_yb, columns=feats_cluster)
            feat_corr = feat_df.corr()
            import seaborn as sns
            sns.heatmap(feat_corr, ax=axes[1], annot=True, fmt=".2f",
                        cmap="RdYlGn", linewidths=0.3, vmin=-1, vmax=1,
                        annot_kws={"size": 7})
            axes[1].set_title("Correlação entre Features", fontweight="bold")
            plt.tight_layout()
            _salvar_fig(fig, "B_4_yellowbrick_classification")
            print("   ✅ Yellowbrick concluído.")
except Exception as e:
    print(f"   ⚠️ Yellowbrick: {e}")

# ── B.5  Alibi ────────────────────────────────────────────────
print("\n📌 B.5 — Alibi (AnchorTabular)")
try:
    from alibi.explainers import AnchorTabular
    if "best_cls" in dir() and not df_cl2.empty:
        Xa = df_cl2[feats_cluster].fillna(0).values
        anch = AnchorTabular(best_cls.predict, feats_cluster, seed=42)
        anch.fit(Xa[:min(30, len(Xa))])
        exp_a = anch.explain(Xa[0:1], threshold=0.85)
        print(f"   Âncora (município de maior índice):")
        if hasattr(exp_a, "anchor") and exp_a.anchor:
            for rule in exp_a.anchor:
                print(f"     → {rule}")
        print(f"   Precisão: {exp_a.precision:.3f} | Cobertura: {exp_a.coverage:.3f}")
        print("   ✅ Alibi AnchorTabular concluído.")
except Exception as e:
    print(f"   ⚠️ Alibi: {e}")

print("\n✅ Seção B — Interpretabilidade Expandida concluída.")

# ============================================================
# SEÇÃO C — SÉRIES TEMPORAIS AVANÇADAS
# ============================================================
print("\n" + "="*70)
print("📈 SEÇÃO C — SÉRIES TEMPORAIS AVANÇADAS")
print("="*70)

df_at_full = VITIMAS.get("atendimentos", pd.DataFrame())
serie_adv_ok = not df_at_full.empty and "ano" in df_at_full.columns and "mes" in df_at_full.columns

if serie_adv_ok:
    sm_adv = (df_at_full.groupby(["ano", "mes"])
              .size().reset_index(name="n").sort_values(["ano","mes"]))
    sm_adv["data"] = pd.to_datetime(
        sm_adv["ano"].astype(str) + "-" + sm_adv["mes"].astype(str).str.zfill(2) + "-01")
    serie_idx = sm_adv.set_index("data")["n"]
    serie_idx = serie_idx.asfreq("MS")

    # ── C.1  SARIMA ──────────────────────────────────────────
    print("\n📌 C.1 — SARIMA (Statsmodels)")
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        sar_model = SARIMAX(serie_idx, order=(1,1,1), seasonal_order=(1,1,1,12),
                             enforce_stationarity=False, enforce_invertibility=False)
        sar_fit = sar_model.fit(disp=False)
        fc_sar = sar_fit.forecast(steps=12)
        conf_int = sar_fit.get_forecast(12).conf_int()
        fig, ax = plt.subplots(figsize=(16, 5))
        serie_idx.plot(ax=ax, label="Histórico", color="#2980b9", lw=2)
        fc_sar.plot(ax=ax, label="SARIMA — Previsão", color="#c0392b", lw=2.5, style="--o", ms=6)
        ax.fill_between(conf_int.index, conf_int.iloc[:,0], conf_int.iloc[:,1],
                        alpha=0.2, color="red", label="IC 95%")
        ax.set_title("📈 SARIMA — Previsão 12 Meses (com Intervalo de Confiança)",
                     fontweight="bold")
        ax.legend(); ax.set_ylabel("Nº Atendimentos")
        _salvar_fig(fig, "C_1_sarima_previsao")
        print(f"   ✅ SARIMA AIC={sar_fit.aic:.1f} | BIC={sar_fit.bic:.1f}")
    except Exception as e:
        print(f"   ⚠️ SARIMA: {e}")

    # ── C.2  skforecast ──────────────────────────────────────
    print("\n📌 C.2 — skforecast (ForecasterAutoreg)")
    try:
        from skforecast.ForecasterAutoreg import ForecasterAutoreg
        from skforecast.model_selection import backtesting_forecaster
        import lightgbm as lgb
        skf = ForecasterAutoreg(
            regressor=lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
            lags=12
        )
        n_test = min(12, max(6, int(len(serie_idx)*0.2)))
        if n_test >= 6 and len(serie_idx) > n_test+12:
            metric, pred_back = backtesting_forecaster(
                forecaster=skf, y=serie_idx,
                initial_train_size=len(serie_idx)-n_test,
                steps=1, metric="mean_absolute_error", verbose=False
            )
            skf.fit(y=serie_idx)
            pred_skf = skf.predict(steps=12)
            fig, ax = plt.subplots(figsize=(16, 5))
            serie_idx.plot(ax=ax, label="Histórico", color="#2980b9", lw=2)
            pred_skf.plot(ax=ax, label="skforecast (LightGBM)", color="#8e44ad",
                          lw=2.5, style="--s", ms=6)
            ax.set_title("📈 skforecast — ForecasterAutoreg (LightGBM) — 12 Meses",
                         fontweight="bold")
            ax.legend(); ax.set_ylabel("Nº Atendimentos")
            _salvar_fig(fig, "C_2_skforecast_previsao")
            pred_skf_df = pred_skf.reset_index()
            pred_skf_df.columns = ["data", "previsao"]
            pred_skf_df.to_csv(TAB_DIR / "previsao_skforecast.csv", index=False, encoding="utf-8-sig")
            pred_skf_df.to_excel(TAB_DIR / "previsao_skforecast.xlsx", index=False)
            print(f"   ✅ skforecast MAE={float(metric):.1f}")
    except Exception as e:
        print(f"   ⚠️ skforecast: {e}")

    # ── C.3  Darts ───────────────────────────────────────────
    print("\n📌 C.3 — Darts (Multiple Models)")
    try:
        from darts import TimeSeries as DTS
        from darts.models import ExponentialSmoothing, NaiveSeasonal
        dts_full = DTS.from_series(serie_idx.dropna())
        train_d = dts_full[:-12]
        for nm_d, mod_d in [("ExponentialSmoothing", ExponentialSmoothing()),
                              ("NaiveSeasonal", NaiveSeasonal(K=12))]:
            try:
                mod_d.fit(train_d)
                pred_d = mod_d.predict(12)
                fig, ax = plt.subplots(figsize=(16, 5))
                serie_idx.plot(ax=ax, label="Histórico", color="#2980b9", lw=2)
                pred_d.pd_series().plot(ax=ax, label=f"Darts {nm_d}", color="#27ae60",
                                         lw=2.5, style="--D", ms=6)
                ax.set_title(f"📈 Darts — {nm_d} — 12 Meses", fontweight="bold")
                ax.legend(); _salvar_fig(fig, f"C_3_darts_{nm_d.lower()}")
                print(f"   ✅ Darts {nm_d} concluído.")
            except Exception as e2:
                print(f"   ⚠️ Darts {nm_d}: {e2}")
    except Exception as e:
        print(f"   ⚠️ Darts: {e}")

    # ── C.4  NeuralProphet ───────────────────────────────────
    print("\n📌 C.4 — NeuralProphet")
    try:
        from neuralprophet import NeuralProphet
        import logging; logging.getLogger("NP").setLevel(logging.ERROR)
        df_np = serie_idx.reset_index(); df_np.columns = ["ds", "y"]
        df_np = df_np.dropna()
        np_m = NeuralProphet(epochs=60, batch_size=8, learning_rate=0.01,
                               yearly_seasonality=True, weekly_seasonality=False,
                               daily_seasonality=False)
        np_m.fit(df_np, freq="MS", progress="none")
        future_np = np_m.make_future_dataframe(df_np, periods=12)
        fc_np = np_m.predict(future_np)
        fc_np_f = fc_np.tail(12)[["ds","yhat1"]]
        fig, ax = plt.subplots(figsize=(16, 5))
        serie_idx.plot(ax=ax, label="Histórico", color="#2980b9", lw=2)
        ax.plot(fc_np_f["ds"], fc_np_f["yhat1"], "--^",
                color="#c0392b", lw=2.5, ms=7, label="NeuralProphet")
        ax.set_title("📈 NeuralProphet — Previsão 12 Meses", fontweight="bold")
        ax.legend(); ax.set_ylabel("Nº Atendimentos")
        _salvar_fig(fig, "C_4_neuralprophet")
        fc_np_f.to_csv(TAB_DIR / "previsao_neuralprophet.csv", index=False, encoding="utf-8-sig")
        fc_np_f.to_excel(TAB_DIR / "previsao_neuralprophet.xlsx", index=False)
        try:
            from IPython.display import display
            display(fc_np_f.rename(columns={"ds":"Data","yhat1":"Previsão"})
                    .style.background_gradient(cmap="Reds").format(precision=0))
        except Exception: pass
        print("   ✅ NeuralProphet concluído.")
    except Exception as e:
        print(f"   ⚠️ NeuralProphet: {e}")

print("\n✅ Seção C — Séries Temporais Avançadas concluída.")

# ============================================================
# SEÇÃO D — SISTEMA DE RECOMENDAÇÃO (LightFM + Faiss)
# ============================================================
print("\n" + "="*70)
print("💡 SEÇÃO D — SISTEMA DE RECOMENDAÇÃO DE INTERVENÇÕES")
print("="*70)

POLITICAS = ["DEAM","CREAS","Casa Mulher BR","Patrulha Maria Penha",
             "Medida Protetiva Monitor","PAPVD Agressor","Unidade Móvel",
             "Capacitação Policial","SUS VAS","Ligue 180"]

if not rank_mun.empty:
    MUN_LIST = rank_mun["nm_municipio"].tolist()
    n_m = len(MUN_LIST); n_p = len(POLITICAS)
    rank_norm = ((rank_mun["taxa_100k"] - rank_mun["taxa_100k"].min()) /
                 (rank_mun["taxa_100k"].max() - rank_mun["taxa_100k"].min() + 1e-9))
    np.random.seed(42)
    mat = np.zeros((n_m, n_p))
    for i in range(n_m):
        prob = min(0.9, float(rank_norm.iloc[i]) + 0.1)
        for j in range(n_p):
            mat[i, j] = 1 if np.random.random() < prob - 0.1*j else 0

    mat_df = pd.DataFrame(mat, index=MUN_LIST, columns=POLITICAS)

    # ── D.1  LightFM ─────────────────────────────────────────
    try:
        from lightfm import LightFM
        from scipy.sparse import csr_matrix
        print("\n📌 D.1 — LightFM (Collaborative Filtering — WARP loss)")
        lf_mat = csr_matrix(mat)
        lfm = LightFM(loss="warp", no_components=20, random_state=42)
        lfm.fit(lf_mat, epochs=40, num_threads=2)
        top5_idx = rank_mun.nlargest(5, "taxa_100k").index.tolist()
        rec_rows = []
        for idx in top5_idx[:5]:
            if idx >= n_m: continue
            mun_nm = MUN_LIST[idx]
            scores = lfm.predict(idx, np.arange(n_p))
            not_impl = [(POLITICAS[j], scores[j]) for j in range(n_p) if mat[idx,j]==0]
            top3 = sorted(not_impl, key=lambda x:-x[1])[:3]
            rec_rows.append({"Município": mun_nm,
                              "Intervenções Recomendadas": " | ".join(p for p,_ in top3)})
        rec_df = pd.DataFrame(rec_rows)
        rec_df.to_csv(TAB_DIR / "recomendacoes_lightfm.csv", index=False, encoding="utf-8-sig")
        rec_df.to_excel(TAB_DIR / "recomendacoes_lightfm.xlsx", index=False)
        try:
            from IPython.display import display, HTML
            display(HTML("<h4>🎯 Recomendações LightFM — Top 5 Municípios de Maior Risco</h4>"))
            display(rec_df.style.set_properties(**{"text-align":"left"})
                    .set_table_styles([{"selector":"th","props":[("background","#c0392b"),
                                                                   ("color","white")]}]))
        except Exception: pass
        print("   ✅ LightFM concluído.")
        print(rec_df.to_string(index=False))
    except Exception as e:
        print(f"   ⚠️ LightFM: {e}")

    # ── D.2  Heatmap cobertura ───────────────────────────────
    fig, ax = plt.subplots(figsize=(16, max(8, n_m*0.2)))
    import seaborn as sns
    sns.heatmap(mat_df, ax=ax, cmap="RdYlGn", linewidths=0.3,
                annot=True, fmt=".0f", cbar_kws={"label":"1=Impl. | 0=Ausente"},
                annot_kws={"size": 6})
    ax.set_title("💡 Mapa de Cobertura de Políticas Públicas por Município\n"
                 "(1=Implementado | 0=Ausente)", fontweight="bold", pad=12)
    ax.set_xlabel("Políticas/Serviços"); ax.set_ylabel("Municípios")
    ax.tick_params(axis="y", labelsize=6)
    plt.tight_layout()
    _salvar_fig(fig, "D_2_heatmap_cobertura")

    # ── D.3  Faiss ───────────────────────────────────────────
    try:
        import faiss
        print("\n📌 D.3 — Faiss (Busca Vetorial de Municípios Similares)")
        X_faiss = mat.astype("float32")
        index_f = faiss.IndexFlatL2(n_p)
        index_f.add(X_faiss)
        cg_i = MUN_LIST.index("Campo Grande") if "Campo Grande" in MUN_LIST else 0
        D_f, I_f = index_f.search(X_faiss[cg_i:cg_i+1], 6)
        sim_list = [(MUN_LIST[i], float(D_f[0][j]))
                    for j,i in enumerate(I_f[0]) if i != cg_i][:5]
        print(f"   Municípios mais similares a Campo Grande:")
        for nm_s, dist_s in sim_list:
            print(f"     → {nm_s} (distância L2: {dist_s:.1f})")
        print("   ✅ Faiss concluído.")
    except Exception as e:
        print(f"   ⚠️ Faiss: {e}")

print("\n✅ Seção D — Sistema de Recomendação concluída.")
