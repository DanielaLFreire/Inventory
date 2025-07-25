# utils/arquivo_handler.py
'''
Funções para carregamento e processamento de arquivos
'''

import pandas as pd
import streamlit as st
from config.config import SEPARADORES_CSV, ENGINES_EXCEL
import re
import numpy as np


def converter_dms_para_decimal(coord_dms):
    """
    Converte coordenadas DMS para decimal com tratamento robusto

    Args:
        coord_dms: String no formato "22°57'23.38\"S"

    Returns:
        float: Coordenada em formato decimal
    """
    try:
        if pd.isna(coord_dms) or coord_dms == '':
            return None

        coord_str = str(coord_dms).strip()

        # Padrão regex mais flexível
        pattern = r"(\d+)°(\d+)'([\d.]+)\"([NSEOOW])"
        match = re.match(pattern, coord_str)

        if not match:
            # Tentar padrão alternativo sem símbolos
            pattern2 = r"(\d+)\s*(\d+)\s*([\d.]+)\s*([NSEOOW])"
            match = re.match(pattern2, coord_str)

        if not match:
            # Se ainda não funcionar, tentar apenas números
            try:
                return float(coord_str)
            except:
                return None

        graus = float(match.group(1))
        minutos = float(match.group(2))
        segundos = float(match.group(3))
        direcao = match.group(4).upper()

        # Converter para decimal
        decimal = graus + (minutos / 60) + (segundos / 3600)

        # Aplicar sinal baseado na direção
        if direcao in ['S', 'O', 'W']:  # Sul ou Oeste = negativo
            decimal = -decimal

        return decimal

    except Exception as e:
        st.warning(f"⚠️ Erro ao converter coordenada '{coord_dms}': {e}")
        return None


def converter_dms_para_decimal(coord_dms):
    """
    Converte coordenadas DMS (Graus, Minutos, Segundos) para formato decimal

    Args:
        coord_dms: String no formato "22°57'23.38\"S" ou "49°14'4.46\"O"

    Returns:
        float: Coordenada em formato decimal
    """
    try:
        # Padrão regex para capturar graus, minutos, segundos e direção
        pattern = r"(\d+)°(\d+)'([\d.]+)\"([NSEO])"
        match = re.match(pattern, coord_dms.strip())

        if not match:
            return None

        graus = float(match.group(1))
        minutos = float(match.group(2))
        segundos = float(match.group(3))
        direcao = match.group(4)

        # Converter para decimal
        decimal = graus + (minutos / 60) + (segundos / 3600)

        # Aplicar sinal baseado na direção
        if direcao in ['S', 'O', 'W']:  # Sul ou Oeste = negativo
            decimal = -decimal

        return decimal

    except Exception as e:
        print(f"Erro ao converter {coord_dms}: {e}")
        return None


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
    '''Carrega arquivo CSV testando diferentes separadores - VERSÃO MELHORADA'''

    # Lista de separadores em ordem de prioridade
    separadores = [',', ';', '\t', '|']

    for sep in separadores:
        try:
            # Resetar ponteiro do arquivo
            arquivo.seek(0)

            # Tentar ler com encoding UTF-8
            df = pd.read_csv(arquivo, sep=sep, encoding='utf-8')

            # Verificar se o DataFrame é válido
            if len(df.columns) > 1 and len(df) > 0:
                st.success(f"✅ CSV carregado com separador '{sep}' e encoding UTF-8")
                return df

        except UnicodeDecodeError:
            # Tentar com encoding latin-1
            try:
                arquivo.seek(0)
                df = pd.read_csv(arquivo, sep=sep, encoding='latin-1')
                if len(df.columns) > 1 and len(df) > 0:
                    st.success(f"✅ CSV carregado com separador '{sep}' e encoding latin-1")
                    return df
            except:
                continue
        except Exception as e:
            continue

    # Fallback com pandas padrão
    try:
        arquivo.seek(0)
        df = pd.read_csv(arquivo)
        if len(df.columns) > 0 and len(df) > 0:
            st.warning("⚠️ CSV carregado com configuração padrão")
            return df
    except Exception as e:
        st.error(f"❌ Erro final ao ler CSV: {e}")
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


def processar_coordenadas_dms(df_coord):
    """
    Processa arquivo de coordenadas com formato DMS

    Args:
        df_coord: DataFrame com coordenadas originais

    Returns:
        DataFrame com coordenadas convertidas para decimal
    """
    df_processado = df_coord.copy()

    # Detectar se as coordenadas estão em formato DMS
    primeira_lat = str(df_coord['latitude'].iloc[0])
    primeira_lon = str(df_coord['longitude'].iloc[0])

    if '°' in primeira_lat and '°' in primeira_lon:
        print("🔄 Coordenadas em formato DMS detectadas. Convertendo para decimal...")

        # Converter latitude
        df_processado['latitude_decimal'] = df_coord['latitude'].apply(converter_dms_para_decimal)

        # Converter longitude
        df_processado['longitude_decimal'] = df_coord['longitude'].apply(converter_dms_para_decimal)

        # Substituir colunas originais
        df_processado['latitude'] = df_processado['latitude_decimal']
        df_processado['longitude'] = df_processado['longitude_decimal']

        # Remover colunas temporárias
        df_processado = df_processado.drop(['latitude_decimal', 'longitude_decimal'], axis=1)

        print("✅ Conversão DMS → Decimal concluída!")

        # Mostrar exemplos
        print(f"Exemplo: {primeira_lat} → {df_processado['latitude'].iloc[0]:.6f}")
        print(f"Exemplo: {primeira_lon} → {df_processado['longitude'].iloc[0]:.6f}")

    else:
        print("ℹ️ Coordenadas já estão em formato decimal")

    return df_processado


def processar_coordenadas(arquivo_coord, raio_parcela):
    '''
    Processa coordenadas para calcular áreas dos talhões - VERSÃO CORRIGIDA
    '''
    try:
        # Carregar arquivo com função melhorada
        df_coord = carregar_arquivo(arquivo_coord)
        if df_coord is None:
            st.error("❌ Não foi possível carregar o arquivo de coordenadas")
            return None

        st.info(f"📊 Arquivo carregado: {len(df_coord)} registros, {len(df_coord.columns)} colunas")

        # Mostrar preview dos dados
        with st.expander("👀 Preview dos dados carregados"):
            st.dataframe(df_coord.head())
            st.write("Colunas:", list(df_coord.columns))

        # Verificar se as coordenadas estão em formato DMS
        primeira_lat = str(df_coord['latitude'].iloc[0])
        if '°' in primeira_lat:
            st.info("🔄 Convertendo coordenadas DMS para decimal...")

            # Converter coordenadas
            df_coord['lat_decimal'] = df_coord['latitude'].apply(converter_dms_para_decimal)
            df_coord['lon_decimal'] = df_coord['longitude'].apply(converter_dms_para_decimal)

            # Verificar conversões válidas
            validas_lat = df_coord['lat_decimal'].notna().sum()
            validas_lon = df_coord['lon_decimal'].notna().sum()

            st.success(
                f"✅ Conversão concluída: {validas_lat}/{len(df_coord)} latitudes, {validas_lon}/{len(df_coord)} longitudes")

            if validas_lat == 0 or validas_lon == 0:
                st.error("❌ Nenhuma coordenada foi convertida com sucesso")
                return None
        else:
            # Coordenadas já em decimal
            df_coord['lat_decimal'] = pd.to_numeric(df_coord['latitude'], errors='coerce')
            df_coord['lon_decimal'] = pd.to_numeric(df_coord['longitude'], errors='coerce')

        # Limpar nomes dos talhões para números
        df_coord['talhao_num'] = df_coord['talhao'].astype(str).str.extract(r'(\d+)').astype(int)

        # Calcular área da parcela
        area_parcela_ha = np.pi * (raio_parcela ** 2) / 10000

        # Contar parcelas por talhão
        parcelas_por_talhao = df_coord.groupby('talhao_num').size().reset_index()
        parcelas_por_talhao.columns = ['talhao', 'num_parcelas']
        parcelas_por_talhao['area_ha'] = parcelas_por_talhao['num_parcelas'] * area_parcela_ha

        st.success(f"✅ Áreas calculadas: {len(parcelas_por_talhao)} talhões")
        st.info(f"📐 Área por parcela: {area_parcela_ha:.4f} ha (raio {raio_parcela}m)")

        # Mostrar resultado
        with st.expander("📊 Resultado do cálculo de áreas"):
            st.dataframe(parcelas_por_talhao)
            st.metric("Área Total", f"{parcelas_por_talhao['area_ha'].sum():.3f} ha")

        return parcelas_por_talhao[['talhao', 'area_ha']]

    except Exception as e:
        st.error(f"❌ Erro ao processar coordenadas: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None