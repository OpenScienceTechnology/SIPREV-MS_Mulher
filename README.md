# 🚺 SIPREV-Mulher/MS — Monitor da Violência contra a Mulher

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-purple)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-TensorFlow-red)
![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-ff4b4b)
![License](https://img.shields.io/badge/Licença-Acadêmica-lightgrey)

## 📌 Descrição do Projeto

O **SIPREV-Mulher/MS — Sistema Inteligente de Predição e Mapeamento da Violência contra a Mulher em Mato Grosso do Sul** é um notebook analítico desenvolvido em Python para realizar uma análise integrada, exploratória, estatística, geoespacial e preditiva sobre dados públicos relacionados à violência contra a mulher no Estado de Mato Grosso do Sul.

O projeto utiliza dados do **Monitor da Violência contra a Mulher**, mantido pela **SEJUSP-MS** e pelo **PJMS**, com foco em eventos como atendimentos de emergência, estupro, feminicídio, homicídios de mulheres, violência doméstica e medidas protetivas de urgência.

🔗 Fonte oficial dos dados:  
https://monitorviolenciacontramulher.sejusp.ms.gov.br/

---

## 🎯 Objetivo Geral

Desenvolver uma solução de análise de dados capaz de:

- 📊 Explorar padrões temporais, territoriais e sociodemográficos da violência contra a mulher;
- 🗺️ Mapear municípios com maior concentração absoluta e relativa de registros;
- 🔬 Aplicar técnicas de clusterização para agrupamento de municípios por perfil de risco;
- 🤖 Construir modelos preditivos com Machine Learning e Deep Learning;
- 🔍 Utilizar métodos de explicabilidade para interpretar variáveis relevantes;
- 📈 Gerar relatórios, gráficos, mapas, tabelas e dashboards interativos;
- 💡 Apoiar recomendações de política pública e prevenção da violência de gênero.

---

## 🏛️ Contexto Institucional

| Campo | Informação |
|---|---|
| 📚 Disciplina | Tópicos Interdisciplinares III |
| 🎓 Curso | Ciência dos Dados |
| 🏫 Instituição | UFMS Digital |
| 📅 Semestre | 2026.1 |
| 👤 Autor | VIANA |
| 🌐 Fonte | Monitor da Violência contra a Mulher — SEJUSP-MS/PJMS |
| 📍 Recorte territorial | Mato Grosso do Sul |
| 📆 Período analisado | 2016–2026, com dados parciais até maio de 2026 |
| 🧪 Tipo de projeto | Análise de dados, predição, visualização e apoio à decisão |

---

## 📂 Datasets Utilizados

O notebook trabalha com **seis conjuntos de dados principais**, extraídos do Monitor da Violência contra a Mulher.

| Dataset | Descrição | Período |
|---|---|---|
| `atendimentos_emergencia.csv` | Registros de atendimentos de emergência relacionados à violência contra a mulher | 2016–2026 |
| `vitimas_estupro.csv` | Registros de vítimas de estupro | 2016–2026 |
| `vitimas_feminicidios.csv` | Registros de vítimas de feminicídio | 2016–2026 |
| `mulheres_vitimas_homicidios.csv` | Registros de mulheres vítimas de homicídio | 2016–2026 |
| `vitimas_violencia_domestica.csv` | Registros de vítimas de violência doméstica | 2026 |
| `medidas_protetivas_urgencia.csv` | Registros de medidas protetivas de urgência | 2026 |

---

## 🧱 Estrutura Geral do Notebook

O notebook está organizado em seções sequenciais, formando um pipeline completo de Ciência de Dados.

| Seção | Nome | Finalidade |
|---|---|---|
| ⚙️ Seção 0 | Configuração do Ambiente | Instalação de bibliotecas, criação de diretórios e preparação do ambiente |
| 📥 Seção 1 | Carregamento e Ingestão dos Dados | Leitura dos arquivos CSV locais ou via URLs do GitHub |
| 🔧 Seção 2 | Limpeza e Pré-processamento | Padronização de colunas, datas, municípios, faixas etárias e variáveis categóricas |
| 📊 Seção 3 | Análise Exploratória | Geração de gráficos, tabelas e estatísticas descritivas |
| 🏆 Seção 4 | Ranking Municipal | Cálculo de totais, taxas por 100 mil mulheres e classificação dos municípios |
| 🗺️ Seção 5 | Ranking Nacional / Comparativo Estadual | Comparação de Mato Grosso do Sul com outras unidades federativas |
| 🧭 Seção 6 | Análise Geoespacial | Mapas interativos, mapas de calor e visualização territorial |
| 🔬 Seção 7 | Clusterização Municipal | Agrupamento de municípios por perfis de risco |
| 🤖 Seção 8 | Machine Learning Clássico | Modelos de regressão, classificação e detecção de anomalias |
| 🧠 Seção 9 | Deep Learning | Modelos LSTM, GRU, CNN-LSTM e MLP para séries temporais |
| 🔍 Seção 10 | Explicabilidade SHAP | Interpretação dos modelos preditivos |
| 📋 Seção 11 | Relatório de Modelos | Síntese textual dos modelos treinados e métricas obtidas |
| 📊 Seção 12 | Dashboard Streamlit | Geração automática de aplicação interativa |
| 📝 Seção 13 | Conclusões e Recomendações | Síntese analítica e propostas de política pública |
| 📦 Exportação Final | ZIP de Resultados | Compactação dos gráficos, tabelas, relatórios, mapas e modelos |
| 🐱 Seção A | CatBoost + Optuna | Modelos adicionais com otimização de hiperparâmetros |
| 🔎 Seção B | Interpretabilidade Expandida | LIME, ELI5, InterpretML, Yellowbrick e Alibi |
| 📈 Seção C | Séries Temporais Avançadas | SARIMA, skforecast, Darts e NeuralProphet |
| 💡 Seção D | Sistema de Recomendação | Recomendação de intervenções com LightFM e Faiss |

---

## 🛠️ Tecnologias e Bibliotecas Utilizadas

O notebook utiliza bibliotecas robustas para análise de dados, aprendizado de máquina, aprendizado profundo, visualização, mapas, explicabilidade e dashboards.

### 📊 Análise e Manipulação de Dados

- `pandas`
- `numpy`
- `polars`
- `duckdb`
- `openpyxl`
- `texttable`

### 📈 Visualização de Dados

- `matplotlib`
- `seaborn`
- `plotly`
- `missingno`
- `kaleido`

### 🗺️ Geoprocessamento e Mapas

- `folium`
- `geopandas`
- `shapely`
- `contextily`
- `plotly.express`

### 🤖 Machine Learning

- `scikit-learn`
- `xgboost`
- `lightgbm`
- `catboost`
- `imbalanced-learn`
- `hdbscan`
- `joblib`

### 🧠 Deep Learning

- `tensorflow`
- `keras`

### 🔍 Explicabilidade e Interpretabilidade

- `shap`
- `lime`
- `eli5`
- `interpret`
- `yellowbrick`
- `alibi`

### 📈 Séries Temporais

- `statsmodels`
- `skforecast`
- `darts`
- `neuralprophet`

### 💡 Sistemas de Recomendação

- `lightfm`
- `faiss-cpu`

### 🌐 Dashboard e Aplicação Web

- `streamlit`
- `plotly`

### 🧾 Logs e Utilidades

- `loguru`
- `tqdm`
- `requests`
- `unidecode`
- `rapidfuzz`

---

## ⚙️ Instalação do Ambiente

### 1. Clone o repositório

```bash
git clone https://github.com/OpenScienceTechnology/Dataset_SEJUSP-MS_PJMS.git
cd Dataset_SEJUSP-MS_PJMS
2. Crie um ambiente virtual
python -m venv .venv
3. Ative o ambiente virtual
Windows — PowerShell
.venv\Scripts\Activate.ps1
Linux/macOS
source .venv/bin/activate
4. Instale as dependências principais
pip install --upgrade pip
pip install pandas numpy matplotlib seaborn plotly scikit-learn openpyxl jupyter notebook
5. Instale as dependências avançadas
pip install polars duckdb folium geopandas shapely contextily missingno texttable loguru unidecode rapidfuzz tqdm requests kaleido
pip install xgboost lightgbm catboost imbalanced-learn hdbscan shap lime eli5 interpret yellowbrick alibi
pip install tensorflow streamlit statsmodels skforecast darts neuralprophet lightfm faiss-cpu optuna mlflow joblib

⚠️ Algumas bibliotecas avançadas podem exigir dependências adicionais do sistema operacional. Caso ocorra erro de instalação, recomenda-se instalar gradualmente os pacotes e verificar a compatibilidade com a versão do Python utilizada.

▶️ Como Executar o Notebook
Opção 1 — Jupyter Notebook local
jupyter notebook SIPREV_Mulher_MS_Monitor_v4.ipynb
Opção 2 — Jupyter Lab
jupyter lab
Opção 3 — Google Colab
Acesse https://colab.research.google.com/
Faça upload do arquivo SIPREV_Mulher_MS_Monitor_v4.ipynb
Execute as células sequencialmente
Aguarde a instalação das bibliotecas e o carregamento dos dados
Ao final, o notebook gera arquivos em outputs/, models/, logs/ e dashboard/
📁 Estrutura Esperada do Projeto
SIPREV-Mulher-MS/
│
├── SIPREV_Mulher_MS_Monitor_v4.ipynb
├── README.md
│
├── data/
│   └── raw/
│       ├── atendimentos_emergencia.csv
│       ├── medidas_protetivas_urgencia.csv
│       ├── mulheres_vitimas_homicidios.csv
│       ├── vitimas_estupro.csv
│       ├── vitimas_feminicidios.csv
│       └── vitimas_violencia_domestica.csv
│
├── outputs/
│   ├── graficos/
│   ├── tabelas/
│   ├── mapas/
│   └── relatorios/
│
├── models/
│   └── modelos_treinados.pkl
│
├── logs/
│   └── execucao.log
│
└── dashboard/
    └── app_siprev_monitor.py
📥 Carregamento dos Dados

O notebook possui duas estratégias de carregamento:

Leitura local, caso os arquivos CSV estejam disponíveis no diretório do projeto;
Leitura remota, utilizando URLs dos arquivos hospedados no GitHub.

Os arquivos esperados são:

atendimentos_emergencia.csv
medidas_protetivas_urgencia.csv
mulheres_vitimas_homicidios.csv
vitimas_estupro.csv
vitimas_feminicidios.csv
vitimas_violencia_domestica.csv
🔧 Pré-processamento dos Dados

Durante a etapa de limpeza, o notebook executa:

Padronização dos nomes das colunas;
Normalização de textos com letras maiúsculas e remoção de acentos;
Conversão de datas;
Extração de ano, mês e dia;
Padronização de municípios;
Criação de variáveis auxiliares;
Tratamento de valores ausentes;
Criação de faixas etárias;
Integração dos seis datasets em uma base unificada;
Geração de variáveis para análise municipal e temporal.
📊 Análises Exploratórias Realizadas

O notebook gera análises sobre:

📅 Evolução anual dos registros;
📆 Distribuição mensal;
🏙️ Distribuição por município;
👩 Perfil das vítimas;
🎨 Cor/raça;
🎓 Escolaridade;
🔢 Faixa etária;
🧾 Tipo de ocorrência;
⚖️ Comparação entre crimes;
📍 Concentração espacial;
🚨 Indicadores de risco municipal.

As visualizações são exibidas diretamente no notebook e também salvas em arquivos.

🏆 Ranking Municipal

O ranking municipal calcula indicadores como:

Total de registros por município;
Número de atendimentos de emergência;
Número de vítimas de estupro;
Número de feminicídios;
Número de homicídios de mulheres;
Número de vítimas de violência doméstica;
Número de medidas protetivas;
População feminina estimada;
Taxa por 100 mil mulheres;
Classificação geral de risco.

Essa etapa permite identificar municípios com maior volume absoluto e municípios com maior risco proporcional.

🗺️ Análise Geoespacial

A seção geoespacial utiliza coordenadas municipais para produzir:

Mapas interativos;
Heatmaps;
Marcadores por município;
Visualizações territoriais dos registros;
Análises de concentração espacial.

As bibliotecas principais utilizadas são folium, geopandas, shapely e plotly.

🔬 Clusterização Municipal

A etapa de clusterização agrupa municípios segundo padrões semelhantes de violência registrada.

Modelos utilizados:

KMeans
DBSCAN
HDBSCAN
Clusterização hierárquica
PCA para redução dimensional
Dendrograma

As variáveis consideradas incluem:

n_atendimentos
n_estupro
n_feminicidios
n_homicidios
n_viol_dom
n_medidas
total
taxa_100k

Essa análise permite identificar grupos de municípios com perfis semelhantes de vulnerabilidade.

🤖 Modelos de Machine Learning

O notebook implementa modelos clássicos para previsão, classificação e identificação de padrões.

Modelos de Regressão
Random Forest Regressor
Extra Trees Regressor
Gradient Boosting
Ridge Regression
ElasticNet
XGBoost
LightGBM
CatBoost
Modelos de Classificação
Random Forest Classifier
Extra Trees Classifier
Gradient Boosting Classifier
AdaBoost
XGBoost
LightGBM
CatBoost
Técnicas de validação
Validação cruzada;
TimeSeriesSplit;
MAE;
RMSE;
R²;
F1-score;
ROC-AUC;
Classification Report.
🧠 Deep Learning

A seção de Deep Learning aplica redes neurais para modelagem de séries temporais e previsão de casos.

Arquiteturas utilizadas:

LSTM;
GRU;
CNN-LSTM;
MLP.

Recursos empregados:

Normalização com MinMaxScaler;
Criação de janelas temporais;
Separação treino/teste;
Early Stopping;
Curvas de aprendizado;
Comparação entre valores reais e previstos.
📈 Séries Temporais Avançadas

O notebook também inclui modelos avançados de previsão temporal:

SARIMA;
skforecast com LightGBM;
Darts;
NeuralProphet.

Esses modelos são aplicados para analisar a evolução mensal dos atendimentos e projetar tendências futuras.

🔍 Explicabilidade dos Modelos

Para tornar os modelos mais interpretáveis, o projeto utiliza técnicas de explicabilidade.

Métodos utilizados:

SHAP;
LIME;
ELI5;
InterpretML;
Yellowbrick;
Alibi AnchorTabular.

Essas ferramentas ajudam a responder perguntas como:

Quais variáveis mais influenciam a classificação de risco?
Quais fatores explicam o aumento da taxa municipal?
Como cada variável contribui para determinada previsão?
Quais municípios apresentam padrões atípicos?
💡 Sistema de Recomendação de Intervenções

O notebook inclui uma seção experimental para recomendação de políticas públicas e intervenções.

Tecnologias utilizadas:

LightFM
Faiss
Matriz município × política pública
Busca vetorial
Recomendação de ações prioritárias

Exemplos de políticas consideradas:

DEAM;
CREAS;
Casa da Mulher Brasileira;
Patrulha Maria da Penha;
Monitoramento de medidas protetivas;
Programa de atendimento ao agressor;
Unidade móvel;
Capacitação policial;
Serviços de saúde;
Ligue 180.
📊 Dashboard Streamlit

O notebook gera automaticamente um arquivo de dashboard:

dashboard/app_siprev_monitor.py

Para executar:

streamlit run dashboard/app_siprev_monitor.py

O dashboard permite visualizar:

Indicadores gerais;
Rankings municipais;
Distribuição temporal;
Gráficos interativos;
Mapas;
Indicadores de risco;
Comparações entre categorias.
📦 Arquivos Gerados

Ao final da execução, o notebook exporta automaticamente os resultados em diferentes formatos.

outputs/
├── graficos/
│   ├── *.png
│   ├── *.pdf
│   └── *.html
│
├── tabelas/
│   ├── *.csv
│   └── *.xlsx
│
├── mapas/
│   └── *.html
│
└── relatorios/
    ├── *.txt
    └── *.log

Também são gerados:

models/
└── modelos treinados e serializados

logs/
└── registros de execução

dashboard/
└── app_siprev_monitor.py

Além disso, o notebook compacta os resultados em um arquivo .zip.

🧾 Principais Produtos Analíticos

O projeto entrega:

📊 Gráficos estatísticos;
🗺️ Mapas interativos;
🏆 Rankings municipais;
📈 Modelos preditivos;
🧠 Redes neurais;
🔍 Explicações com SHAP e LIME;
📋 Relatórios técnicos;
📦 Arquivo ZIP final;
🌐 Dashboard Streamlit;
💡 Recomendações de intervenção pública.
🧠 Possíveis Aplicações

Este projeto pode apoiar:

Planejamento de políticas públicas;
Diagnóstico territorial da violência contra a mulher;
Priorização de municípios vulneráveis;
Apoio à gestão de segurança pública;
Apoio à rede de proteção social;
Estudos acadêmicos em Ciência dos Dados;
Projetos de extensão universitária;
Sistemas de monitoramento e alerta preventivo;
Pesquisas sobre violência de gênero;
Estudos sobre desigualdade territorial e vulnerabilidade social.
⚖️ Aspectos Éticos e LGPD

Embora os dados utilizados sejam públicos, este projeto trata de tema sensível e deve observar princípios éticos e legais.

Recomenda-se:

Não expor dados pessoais identificáveis;
Utilizar apenas bases públicas e autorizadas;
Evitar interpretações discriminatórias;
Não culpabilizar vítimas;
Interpretar resultados com responsabilidade estatística;
Considerar subnotificação e desigualdade de acesso aos serviços;
Utilizar os modelos como apoio à decisão, não como decisão automática;
Respeitar a Lei Geral de Proteção de Dados Pessoais — LGPD, Lei nº 13.709/2018.
⚠️ Limitações do Projeto

O projeto apresenta limitações importantes:

Os dados podem conter subnotificação;
Alguns municípios podem ter registros incompletos;
As bases de 2026 são parciais;
A qualidade das previsões depende da qualidade dos dados;
Modelos preditivos não substituem análise humana especializada;
Indicadores de risco não devem ser usados isoladamente para decisões públicas;
Algumas bibliotecas avançadas podem não estar disponíveis em todos os ambientes;
Comparações nacionais podem depender de estimativas ou bases complementares.
🚀 Melhorias Futuras

Sugestões para evolução do projeto:

Integrar dados do SINAN/DATASUS;
Integrar dados do IBGE;
Integrar dados socioeconômicos municipais;
Incluir dados de escolaridade, renda, raça/cor e faixa etária de forma mais aprofundada;
Criar API para consulta dos indicadores;
Implantar dashboard em nuvem;
Automatizar atualização dos dados;
Criar modelo de alerta preventivo;
Aplicar análise espacial com shapefiles oficiais dos municípios;
Incorporar indicadores de rede de atendimento;
Adicionar testes automatizados;
Criar documentação técnica da arquitetura do sistema.
🧪 Exemplo de Execução
# Abrir o notebook
jupyter notebook SIPREV_Mulher_MS_Monitor_v4.ipynb

# Executar todas as células em sequência
# Aguardar geração de gráficos, tabelas, mapas, modelos e relatórios

Para executar o dashboard:

streamlit run dashboard/app_siprev_monitor.py
📚 Fontes de Dados
Monitor da Violência contra a Mulher — SEJUSP-MS/PJMS
https://monitorviolenciacontramulher.sejusp.ms.gov.br/
Secretaria de Estado de Justiça e Segurança Pública de Mato Grosso do Sul — SEJUSP-MS
Poder Judiciário de Mato Grosso do Sul — PJMS
Repositório de dados utilizado no notebook:
https://github.com/OpenScienceTechnology/Dataset_SEJUSP-MS_PJMS
📖 Referências Recomendadas

BRASIL. Lei nº 11.340, de 7 de agosto de 2006. Cria mecanismos para coibir a violência doméstica e familiar contra a mulher. Diário Oficial da União, Brasília, DF, 2006.

BRASIL. Lei nº 13.104, de 9 de março de 2015. Altera o Código Penal para prever o feminicídio como circunstância qualificadora do crime de homicídio. Diário Oficial da União, Brasília, DF, 2015.

BRASIL. Lei nº 13.709, de 14 de agosto de 2018. Lei Geral de Proteção de Dados Pessoais — LGPD. Diário Oficial da União, Brasília, DF, 2018.

IPEA. Atlas da Violência. Instituto de Pesquisa Econômica Aplicada. Brasília: IPEA, 2024.

SEJUSP-MS. Monitor da Violência contra a Mulher. Secretaria de Estado de Justiça e Segurança Pública de Mato Grosso do Sul. Disponível em: https://monitorviolenciacontramulher.sejusp.ms.gov.br/

PJMS. Poder Judiciário de Mato Grosso do Sul. Dados e informações institucionais sobre medidas protetivas e violência contra a mulher.

WAISELFISZ, Julio Jacobo. Mapa da Violência 2015: homicídio de mulheres no Brasil. Brasília: FLACSO Brasil, 2015.

👨‍💻 Autor

VIANA
Curso: Ciência dos Dados
Disciplina: Tópicos Interdisciplinares III
Projeto: SIPREV-Mulher/MS — Sistema Inteligente de Predição e Mapeamento da Violência contra a Mulher em Mato Grosso do Sul

📌 Status do Projeto

🚧 Projeto acadêmico em desenvolvimento, com aplicação de técnicas de Ciência dos Dados, Machine Learning, Deep Learning, visualização interativa e apoio à formulação de políticas públicas.

🏷️ Palavras-chave

violência contra a mulher · feminicídio · Mato Grosso do Sul · SEJUSP-MS · PJMS · ciência dos dados · machine learning · deep learning · geoprocessamento · dashboard · políticas públicas

📄 Licença

Este projeto possui finalidade acadêmica, científica e educacional.
Os dados pertencem às respectivas fontes públicas oficiais.
O uso, reprodução e adaptação devem respeitar a legislação vigente, a LGPD e a correta citação das fontes.
