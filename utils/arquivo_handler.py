# utils/arquivo_handler.py
'''
Funções para carregamento e processamento de arquivos
'''

import pandas as pd
import streamlit as st
from config.config import SEPARADORES_CSV, ENGINES_EXCEL


def carregar_arquivo(arquivo):
    '''
    Carrega arquivo CSV ou Excel com máxima compatibilidade

    Args:
        arquivo: Arquivo uploaded via Streamlit

    Returns:
        DataFrame ou None se erro
    '''
    try:
        if arquivo.name.endswith('.csv'):
            return _carregar_csv(arquivo)
        elif arquivo.name.endswith(('.xlsx', '.xls', '.xlsb')):
            return _carregar_excel(arquivo)
        else:
            st.error("❌ Formato não suportado. Use .csv, .xlsx, .xls ou .xlsb")
            return None
    except Exception as e:
        st.error(f"❌ Erro inesperado: {e}")
        return None


def _carregar_csv(arquivo):
    '''Carrega arquivo CSV testando diferentes separadores'''
    # Tentar diferentes separadores
    for sep in SEPARADORES_CSV:
        try:
            df = pd.read_csv(arquivo, sep=sep)
            if len(df.columns) > 1:  # Se tem múltiplas colunas, provavelmente acertou
                return df
        except:
            continue

    # Fallback final
    try:
        df = pd.read_csv(arquivo)
        return df
    except Exception as e:
        st.error(f"❌ Erro ao ler CSV: {e}")
        return None


def _carregar_excel(arquivo):
    '''Carrega arquivo Excel testando diferentes engines'''
    # Verificar engines disponíveis
    engines_disponiveis = []

    for engine in ENGINES_EXCEL:
        try:
            __import__(engine)
            engines_disponiveis.append(engine)
        except ImportError:
            continue

    # Tentar cada engine disponível
    if engines_disponiveis:
        for engine in engines_disponiveis:
            try:
                # Verificar compatibilidade engine/extensão
                if arquivo.name.endswith('.xlsx') and engine == 'openpyxl':
                    return pd.read_excel(arquivo, engine=engine)
                elif arquivo.name.endswith('.xls') and engine == 'xlrd':
                    return pd.read_excel(arquivo, engine=engine)
                elif arquivo.name.endswith('.xlsb') and engine == 'pyxlsb':
                    return pd.read_excel(arquivo, engine=engine)
                else:
                    # Tentar qualquer engine com qualquer arquivo
                    return pd.read_excel(arquivo, engine=engine)
            except Exception:
                continue

    # Tentativa final: pandas padrão
    try:
        return pd.read_excel(arquivo)
    except Exception:
        _mostrar_erro_excel(engines_disponiveis)
        return None


def _mostrar_erro_excel(engines_disponiveis):
    '''Mostra mensagens de erro específicas para Excel'''
    st.error("❌ Não foi possível ler o arquivo Excel")
    st.error("🔧 **Soluções rápidas:**")

    if not engines_disponiveis:
        st.error("• Nenhuma engine Excel encontrada")
        st.code("pip install openpyxl xlrd")
    else:
        st.error(f"• Engines disponíveis: {', '.join(engines_disponiveis)}")
        st.error("• Arquivo pode estar corrompido ou em formato não suportado")

    st.error("• **Alternativa**: Converta para CSV no Excel:")
    st.error("  Arquivo → Salvar Como → CSV UTF-8")


def processar_shapefile(arquivo_shp):
    '''
    Processa shapefile para extrair áreas dos talhões

    Args:
        arquivo_shp: Arquivo shapefile (.shp ou .zip)

    Returns:
        DataFrame com talhao e area_ha ou None se erro
    '''
    try:
        import geopandas as gpd
        import zipfile
        import tempfile
        import os

        if arquivo_shp.name.endswith('.zip'):
            # Processar ZIP
            with tempfile.TemporaryDirectory() as temp_dir:
                with zipfile.ZipFile(arquivo_shp, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)

                # Procurar arquivo .shp
                shp_files = [f for f in os.listdir(temp_dir) if f.endswith('.shp')]
                if shp_files:
                    shp_path = os.path.join(temp_dir, shp_files[0])
                    gdf = gpd.read_file(shp_path)
                else:
                    raise Exception("Arquivo .shp não encontrado no ZIP")
        else:
            # Arquivo .shp direto
            gdf = gpd.read_file(arquivo_shp)

        return _extrair_areas_shapefile(gdf)

    except ImportError:
        st.error("❌ GeoPandas não está instalado")
        st.error("🔧 Execute: pip install geopandas")
        return None
    except Exception as e:
        st.error(f"❌ Erro ao processar shapefile: {e}")
        st.info("💡 Verifique se o arquivo contém colunas 'talhao' e 'area_ha'")
        return None


def _extrair_areas_shapefile(gdf):
    '''Extrai áreas dos talhões do GeoDataFrame'''
    from config.config import NOMES_ALTERNATIVOS

    # Procurar coluna de talhão
    col_talhao = None
    for col in gdf.columns:
        if col.lower() in NOMES_ALTERNATIVOS['talhao']:
            col_talhao = col
            break

    if col_talhao is None:
        raise Exception("Coluna de talhão não encontrada")

    # Procurar coluna de área
    col_area = None
    for col in gdf.columns:
        if col.lower() in NOMES_ALTERNATIVOS['area']:
            col_area = col
            break

    if col_area is None:
        # Calcular área da geometria
        gdf['area_ha'] = gdf.geometry.area / 10000  # Converter m² para ha
        col_area = 'area_ha'

    # Agrupar por talhão e somar áreas
    areas_df = gdf.groupby(col_talhao)[col_area].sum().reset_index()
    areas_df.columns = ['talhao', 'area_ha']
    areas_df['talhao'] = areas_df['talhao'].astype(int)

    return areas_df


def processar_coordenadas(arquivo_coord, raio_parcela):
    '''
    Processa coordenadas para calcular áreas dos talhões

    Args:
        arquivo_coord: Arquivo com coordenadas
        raio_parcela: Raio da parcela em metros

    Returns:
        DataFrame com talhao e area_ha ou None se erro
    '''
    try:
        from config.config import NOMES_ALTERNATIVOS

        # Carregar arquivo
        df_coord = carregar_arquivo(arquivo_coord)
        if df_coord is None:
            return None

        # Verificar colunas de coordenadas
        colunas_coord = None
        for coord_set in NOMES_ALTERNATIVOS['coordenadas']:
            if all(col in df_coord.columns for col in coord_set):
                colunas_coord = coord_set
                break

        if not colunas_coord:
            st.error("❌ Coordenadas: colunas X,Y ou lon,lat não encontradas")
            return None

        # Verificar coluna de talhão
        col_talhao = None
        for col in df_coord.columns:
            if col.lower() in NOMES_ALTERNATIVOS['talhao']:
                col_talhao = col
                break

        if col_talhao is None:
            st.error("❌ Coordenadas: coluna 'talhao' não encontrada")
            return None

        # Calcular área
        area_parcela_ha = 3.14159 * (raio_parcela ** 2) / 10000

        # Contar parcelas por talhão
        parcelas_por_talhao = df_coord.groupby(col_talhao).size().reset_index()
        parcelas_por_talhao.columns = ['talhao', 'num_parcelas']
        parcelas_por_talhao['area_ha'] = parcelas_por_talhao['num_parcelas'] * area_parcela_ha

        areas_df = parcelas_por_talhao[['talhao', 'area_ha']].copy()
        areas_df['talhao'] = areas_df['talhao'].astype(int)

        return areas_df

    except Exception as e:
        st.error(f"❌ Erro ao processar coordenadas: {e}")
        return None