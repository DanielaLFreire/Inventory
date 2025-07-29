# utils/arquivo_handler.py
"""
Utilitário para carregar e processar diferentes tipos de arquivos
Suporta CSV (com diferentes separadores), Excel e outros formatos
"""

import pandas as pd
import streamlit as st
import io
import zipfile
import tempfile
import os
from pathlib import Path


def detectar_separador_csv(conteudo_texto):
    """
    Detecta automaticamente o separador de um arquivo CSV

    Args:
        conteudo_texto: Conteúdo do arquivo como string

    Returns:
        str: Separador detectado
    """
    separadores = [';', ',', '\t', '|']
    contadores = {}

    # Analisar as primeiras 5 linhas
    linhas = conteudo_texto.split('\n')[:5]
    texto_amostra = '\n'.join(linhas)

    for sep in separadores:
        contadores[sep] = texto_amostra.count(sep)

    # Retornar o separador mais comum
    separador_detectado = max(contadores, key=contadores.get)

    # Se nenhum separador foi encontrado em quantidade significativa, usar ';'
    if contadores[separador_detectado] == 0:
        return ';'

    return separador_detectado


def detectar_encoding_arquivo(arquivo_bytes):
    """
    Detecta encoding do arquivo

    Args:
        arquivo_bytes: Bytes do arquivo

    Returns:
        str: Encoding detectado
    """
    import chardet

    try:
        # Analisar uma amostra dos bytes
        amostra = arquivo_bytes[:10000]  # Primeiros 10KB
        resultado = chardet.detect(amostra)
        encoding_detectado = resultado['encoding']

        # Encodings comuns para fallback
        encodings_fallback = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

        if encoding_detectado and encoding_detectado.lower() not in ['ascii']:
            return encoding_detectado
        else:
            return 'utf-8'  # Default seguro

    except Exception:
        return 'utf-8'


def carregar_csv(arquivo, encoding=None, separador=None):
    """
    Carrega arquivo CSV com detecção automática de separador e encoding

    Args:
        arquivo: Arquivo uploaded pelo Streamlit
        encoding: Encoding específico (opcional)
        separador: Separador específico (opcional)

    Returns:
        DataFrame ou None se erro
    """
    try:
        # Ler bytes do arquivo
        arquivo_bytes = arquivo.read()
        arquivo.seek(0)  # Reset para reutilizar

        # Detectar encoding se não especificado
        if encoding is None:
            encoding = detectar_encoding_arquivo(arquivo_bytes)

        # Converter bytes para string
        try:
            conteudo_str = arquivo_bytes.decode(encoding)
        except UnicodeDecodeError:
            # Fallback para latin-1 que aceita qualquer byte
            conteudo_str = arquivo_bytes.decode('latin-1')
            encoding = 'latin-1'

        # Detectar separador se não especificado
        if separador is None:
            separador = detectar_separador_csv(conteudo_str)

        # Criar StringIO para pandas
        buffer_str = io.StringIO(conteudo_str)

        # Tentar carregar com pandas
        df = pd.read_csv(
            buffer_str,
            sep=separador,
            encoding=None,  # Já decodificamos
            low_memory=False,
            on_bad_lines='skip'  # Pular linhas problemáticas
        )

        st.success(f"✅ CSV carregado: {len(df)} linhas, separador '{separador}', encoding '{encoding}'")

        return df

    except Exception as e:
        st.error(f"❌ Erro ao carregar CSV: {e}")

        # Tentar fallbacks
        try:
            st.info("🔄 Tentando métodos alternativos...")

            # Fallback 1: Separadores comuns
            for sep in [';', ',', '\t']:
                try:
                    arquivo.seek(0)
                    df = pd.read_csv(arquivo, sep=sep, low_memory=False, on_bad_lines='skip')
                    if len(df.columns) > 1:  # Se conseguiu separar em colunas
                        st.success(f"✅ CSV carregado com separador '{sep}'")
                        return df
                except:
                    continue

            # Fallback 2: Sem separador específico (deixar pandas detectar)
            arquivo.seek(0)
            df = pd.read_csv(arquivo, low_memory=False, on_bad_lines='skip')
            st.success(f"✅ CSV carregado com detecção automática")
            return df

        except Exception as fallback_error:
            st.error(f"❌ Todos os métodos falharam: {fallback_error}")
            return None


def carregar_excel(arquivo):
    """
    Carrega arquivo Excel (.xlsx, .xls, .xlsb)

    Args:
        arquivo: Arquivo uploaded pelo Streamlit

    Returns:
        DataFrame ou None se erro
    """
    try:
        # Detectar tipo de arquivo Excel
        nome_arquivo = arquivo.name.lower()

        if nome_arquivo.endswith('.xlsb'):
            engine = 'pyxlsb'
        elif nome_arquivo.endswith('.xls'):
            engine = 'xlrd'
        else:
            engine = 'openpyxl'

        # Tentar carregar
        try:
            df = pd.read_excel(arquivo, engine=engine)
        except Exception as e:
            # Fallback para outros engines
            st.warning(f"⚠️ Engine {engine} falhou, tentando alternativas...")

            engines_fallback = ['openpyxl', 'xlrd', 'pyxlsb']
            for alt_engine in engines_fallback:
                if alt_engine != engine:
                    try:
                        arquivo.seek(0)
                        df = pd.read_excel(arquivo, engine=alt_engine)
                        engine = alt_engine
                        break
                    except:
                        continue
            else:
                raise e

        st.success(f"✅ Excel carregado: {len(df)} linhas, engine '{engine}'")
        return df

    except Exception as e:
        st.error(f"❌ Erro ao carregar Excel: {e}")
        return None


def carregar_shapefile(arquivo_zip):
    """
    Carrega shapefile de um arquivo ZIP

    Args:
        arquivo_zip: Arquivo ZIP contendo shapefile

    Returns:
        GeoDataFrame ou None se erro
    """
    try:
        import geopandas as gpd

        # Criar diretório temporário
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extrair ZIP
            with zipfile.ZipFile(io.BytesIO(arquivo_zip.read())) as zip_ref:
                zip_ref.extractall(temp_dir)

            # Procurar arquivo .shp
            shp_files = list(Path(temp_dir).glob("*.shp"))

            if not shp_files:
                st.error("❌ Arquivo .shp não encontrado no ZIP")
                return None

            # Carregar shapefile
            gdf = gpd.read_file(shp_files[0])

            st.success(f"✅ Shapefile carregado: {len(gdf)} polígonos")
            return gdf

    except ImportError:
        st.error("❌ GeoPandas não instalado. Instale com: pip install geopandas")
        return None
    except Exception as e:
        st.error(f"❌ Erro ao carregar shapefile: {e}")
        return None


def carregar_arquivo(arquivo):
    """
    Função principal para carregar qualquer tipo de arquivo suportado

    Args:
        arquivo: Arquivo uploaded pelo Streamlit

    Returns:
        DataFrame ou GeoDataFrame
    """
    if arquivo is None:
        return None

    try:
        # Obter extensão do arquivo
        nome_arquivo = arquivo.name.lower()
        extensao = Path(nome_arquivo).suffix.lower()

        # Log do arquivo sendo processado
        st.info(f"📁 Processando: {arquivo.name} ({arquivo.size:,} bytes)")

        # Roteamento por tipo de arquivo
        if extensao == '.csv':
            return carregar_csv(arquivo)

        elif extensao in ['.xlsx', '.xls', '.xlsb']:
            return carregar_excel(arquivo)

        elif extensao == '.zip':
            # Verificar se é shapefile
            try:
                with zipfile.ZipFile(io.BytesIO(arquivo.read())) as zip_ref:
                    arquivos_no_zip = zip_ref.namelist()
                    arquivo.seek(0)  # Reset

                    # Se tem .shp, é shapefile
                    if any(f.lower().endswith('.shp') for f in arquivos_no_zip):
                        return carregar_shapefile(arquivo)
                    else:
                        st.error("❌ ZIP não contém shapefile (.shp)")
                        return None
            except:
                st.error("❌ Arquivo ZIP inválido")
                return None

        elif extensao == '.shp':
            st.error(
                "❌ Para shapefiles, faça upload do arquivo ZIP contendo todos os componentes (.shp, .shx, .dbf, etc.)")
            return None

        else:
            st.error(f"❌ Extensão não suportada: {extensao}")
            st.info("📋 Formatos suportados: CSV, Excel (.xlsx, .xls, .xlsb), Shapefile (.zip)")
            return None

    except Exception as e:
        st.error(f"❌ Erro geral ao carregar arquivo: {e}")
        with st.expander("🔍 Detalhes do erro"):
            st.code(str(e))
        return None


def validar_estrutura_arquivo(df, colunas_obrigatorias, nome_tipo="arquivo"):
    """
    Valida se o DataFrame possui as colunas obrigatórias

    Args:
        df: DataFrame a ser validado
        colunas_obrigatorias: Lista de colunas obrigatórias
        nome_tipo: Nome do tipo de arquivo para mensagens

    Returns:
        dict: Resultado da validação
    """
    if df is None:
        return {
            'valido': False,
            'erros': [f"{nome_tipo} não foi carregado"],
            'alertas': []
        }

    resultado = {
        'valido': True,
        'erros': [],
        'alertas': []
    }

    # Verificar se DataFrame não está vazio
    if len(df) == 0:
        resultado['erros'].append(f"{nome_tipo} está vazio")
        resultado['valido'] = False
        return resultado

    # Verificar colunas obrigatórias
    colunas_faltantes = []
    for coluna in colunas_obrigatorias:
        if coluna not in df.columns:
            colunas_faltantes.append(coluna)

    if colunas_faltantes:
        resultado['erros'].append(f"Colunas obrigatórias faltantes: {colunas_faltantes}")
        resultado['valido'] = False

    # Verificar se há dados válidos nas colunas obrigatórias
    for coluna in colunas_obrigatorias:
        if coluna in df.columns:
            valores_nulos = df[coluna].isna().sum()
            total_linhas = len(df)

            if valores_nulos == total_linhas:
                resultado['erros'].append(f"Coluna '{coluna}' está completamente vazia")
                resultado['valido'] = False
            elif valores_nulos > total_linhas * 0.5:
                resultado['alertas'].append(
                    f"Coluna '{coluna}' tem muitos valores nulos ({valores_nulos}/{total_linhas})")

    return resultado


def exportar_dataframe(df, formato='csv', nome_base="dados_exportados"):
    """
    Exporta DataFrame em diferentes formatos

    Args:
        df: DataFrame a ser exportado
        formato: Formato de exportação ('csv', 'excel', 'json')
        nome_base: Nome base do arquivo

    Returns:
        tuple: (dados_bytes, nome_arquivo, mime_type)
    """
    try:
        if formato.lower() == 'csv':
            # CSV com separador ponto e vírgula (padrão brasileiro)
            dados = df.to_csv(index=False, sep=';', encoding='utf-8')
            nome_arquivo = f"{nome_base}.csv"
            mime_type = "text/csv"

        elif formato.lower() == 'excel':
            # Excel
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Dados')
            dados = buffer.getvalue()
            nome_arquivo = f"{nome_base}.xlsx"
            mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

        elif formato.lower() == 'json':
            # JSON
            dados = df.to_json(orient='records', indent=2, force_ascii=False)
            nome_arquivo = f"{nome_base}.json"
            mime_type = "application/json"

        else:
            raise ValueError(f"Formato não suportado: {formato}")

        return dados, nome_arquivo, mime_type

    except Exception as e:
        st.error(f"❌ Erro ao exportar dados: {e}")
        return None, None, None


def criar_template_csv(colunas, nome_arquivo="template"):
    """
    Cria template CSV com colunas especificadas

    Args:
        colunas: Lista de nomes das colunas
        nome_arquivo: Nome do arquivo template

    Returns:
        str: Conteúdo CSV do template
    """
    # Criar DataFrame vazio com as colunas
    df_template = pd.DataFrame(columns=colunas)

    # Adicionar uma linha de exemplo (valores fictícios)
    if colunas:
        exemplo = {}
        for coluna in colunas:
            if 'D_cm' in coluna or 'dap' in coluna.lower():
                exemplo[coluna] = 15.5
            elif 'H_m' in coluna or 'altura' in coluna.lower():
                exemplo[coluna] = 18.2
            elif 'talhao' in coluna.lower():
                exemplo[coluna] = 1
            elif 'parcela' in coluna.lower():
                exemplo[coluna] = 1
            elif 'cod' in coluna.lower():
                exemplo[coluna] = 'D'
            elif 'idade' in coluna.lower():
                exemplo[coluna] = 5
            elif 'arv' in coluna.lower():
                exemplo[coluna] = 1
            else:
                exemplo[coluna] = 'exemplo'

        df_exemplo = pd.DataFrame([exemplo])
        df_template = pd.concat([df_template, df_exemplo], ignore_index=True)

    return df_template.to_csv(index=False, sep=';')


def verificar_qualidade_dados(df, nome_tipo="dados"):
    """
    Verifica qualidade geral dos dados

    Args:
        df: DataFrame a ser analisado
        nome_tipo: Nome do tipo de dados

    Returns:
        dict: Relatório de qualidade
    """
    if df is None or len(df) == 0:
        return {
            'qualidade_geral': 'Ruim',
            'problemas': ['DataFrame vazio ou nulo'],
            'sugestoes': ['Verificar arquivo de entrada']
        }

    relatorio = {
        'total_linhas': len(df),
        'total_colunas': len(df.columns),
        'problemas': [],
        'sugestoes': [],
        'detalhes_colunas': {}
    }

    # Analisar cada coluna
    for coluna in df.columns:
        detalhes = {
            'tipo': str(df[coluna].dtype),
            'valores_nulos': df[coluna].isna().sum(),
            'valores_unicos': df[coluna].nunique(),
            'percentual_nulos': (df[coluna].isna().sum() / len(df)) * 100
        }

        # Identificar problemas
        if detalhes['percentual_nulos'] > 50:
            relatorio['problemas'].append(f"Coluna '{coluna}' tem >50% valores nulos")

        if detalhes['valores_unicos'] == 1:
            relatorio['problemas'].append(f"Coluna '{coluna}' tem valor constante")

        relatorio['detalhes_colunas'][coluna] = detalhes

    # Classificar qualidade geral
    num_problemas = len(relatorio['problemas'])
    if num_problemas == 0:
        relatorio['qualidade_geral'] = 'Excelente'
    elif num_problemas <= 2:
        relatorio['qualidade_geral'] = 'Boa'
    elif num_problemas <= 5:
        relatorio['qualidade_geral'] = 'Regular'
    else:
        relatorio['qualidade_geral'] = 'Ruim'

    # Gerar sugestões
    if relatorio['problemas']:
        relatorio['sugestoes'].append("Revisar colunas com muitos valores nulos")
        relatorio['sugestoes'].append("Verificar consistência dos dados de entrada")

    return relatorio


# Funções auxiliares para tipos específicos de arquivo

def normalizar_nomes_colunas(df):
    """
    Normaliza nomes das colunas removendo espaços e caracteres especiais

    Args:
        df: DataFrame

    Returns:
        DataFrame com colunas normalizadas
    """
    df_normalizado = df.copy()

    # Dicionário de mapeamento comum
    mapeamento_comum = {
        'diametro': 'D_cm',
        'dap': 'D_cm',
        'altura': 'H_m',
        'height': 'H_m',
        'talhão': 'talhao',
        'stand': 'talhao',
        'plot': 'parcela',
        'codigo': 'cod',
        'code': 'cod',
        'idade': 'idade_anos',
        'age': 'idade_anos',
        'arvore': 'arv',
        'tree': 'arv'
    }

    # Normalizar nomes
    nomes_novos = {}
    for coluna in df.columns:
        nome_limpo = str(coluna).strip().lower()
        nome_limpo = nome_limpo.replace(' ', '_').replace('-', '_')

        # Verificar mapeamento comum
        for padrao, nome_padrao in mapeamento_comum.items():
            if padrao in nome_limpo:
                nomes_novos[coluna] = nome_padrao
                break
        else:
            nomes_novos[coluna] = nome_limpo

    df_normalizado = df_normalizado.rename(columns=nomes_novos)

    return df_normalizado