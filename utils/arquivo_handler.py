# utils/arquivo_handler.py - VERSÃO CORRIGIDA
"""
Utilitário para carregar e processar diferentes tipos de arquivos
CORREÇÃO: Tratamento adequado de DataFrames vs arquivos
"""

import pandas as pd
import streamlit as st
import io
import zipfile
import tempfile
import os
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')



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
    try:
        import chardet
        # Analisar uma amostra dos bytes
        amostra = arquivo_bytes[:10000]  # Primeiros 10KB
        resultado = chardet.detect(amostra)
        encoding_detectado = resultado['encoding']

        # Encodings comuns para fallback
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

    Returns:
        DataFrame ou None se erro
    """
    try:
        # Verificar se é um objeto de arquivo válido
        if not hasattr(arquivo, 'read'):
            st.error("❌ Objeto inválido para leitura de arquivo")
            return None

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

        nome_arquivo = getattr(arquivo, 'name', 'arquivo_csv')
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
        # Verificar se é um objeto de arquivo válido
        if not hasattr(arquivo, 'read'):
            st.error("❌ Objeto inválido para leitura de arquivo Excel")
            return None

        # Detectar tipo de arquivo Excel
        nome_arquivo = getattr(arquivo, 'name', 'arquivo.xlsx').lower()

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

        # Verificar se é um objeto de arquivo válido
        if not hasattr(arquivo_zip, 'read'):
            st.error("❌ Objeto inválido para leitura de shapefile")
            return None

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
    VERSÃO CORRIGIDA: Trata adequadamente objetos DataFrame vs arquivos

    Args:
        arquivo: Arquivo uploaded pelo Streamlit OU DataFrame

    Returns:
        DataFrame ou GeoDataFrame
    """
    # CORREÇÃO PRINCIPAL: Verificar se já é um DataFrame
    if isinstance(arquivo, pd.DataFrame):
        st.info("📊 DataFrame já carregado - retornando diretamente")
        return arquivo

    if arquivo is None:
        return None

    # Verificar se é um objeto de arquivo válido
    if not hasattr(arquivo, 'read') or not hasattr(arquivo, 'name'):
        st.error("❌ Objeto inválido - esperado arquivo do Streamlit ou DataFrame")
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
    VERSÃO CORRIGIDA: Melhores verificações de validade

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

    if not isinstance(df, pd.DataFrame):
        return {
            'valido': False,
            'erros': [f"{nome_tipo} não é um DataFrame válido"],
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


def criar_arquivo_exemplo_para_debug(tipo_arquivo="inventario"):
    """
    Cria arquivo de exemplo para debug quando há problemas de carregamento

    Args:
        tipo_arquivo: Tipo do arquivo a criar exemplo

    Returns:
        DataFrame: Exemplo de dados
    """
    if tipo_arquivo == "inventario":
        dados_exemplo = {
            'talhao': [1, 1, 1, 2, 2, 2],
            'parcela': [1, 2, 3, 1, 2, 3],
            'D_cm': [15.2, 12.8, 18.5, 14.1, 16.9, 13.7],
            'H_m': [18.5, 16.2, 22.1, 17.8, 19.4, 16.5],
            'cod': ['D', 'D', 'D', 'D', 'D', 'D'],
            'idade_anos': [5, 5, 5, 5, 5, 5]
        }
    elif tipo_arquivo == "cubagem":
        dados_exemplo = {
            'arv': [1, 1, 1, 2, 2, 2],
            'talhao': [1, 1, 1, 1, 1, 1],
            'd_cm': [12.5, 10.2, 8.1, 14.8, 11.9, 9.5],
            'h_m': [2.5, 5.0, 7.5, 2.5, 5.0, 7.5],
            'D_cm': [15.2, 15.2, 15.2, 16.8, 16.8, 16.8],
            'H_m': [18.5, 18.5, 18.5, 19.2, 19.2, 19.2]
        }
    else:
        dados_exemplo = {'exemplo': [1, 2, 3]}

    return pd.DataFrame(dados_exemplo)


def diagnosticar_problema_arquivo(arquivo, erro_original):
    """
    Diagnostica problemas no carregamento de arquivos

    Args:
        arquivo: Objeto que causou erro
        erro_original: Erro original capturado

    Returns:
        str: Diagnóstico do problema
    """
    diagnostico = ["🔍 **Diagnóstico do Problema:**"]

    # Verificar tipo do objeto
    tipo_objeto = type(arquivo).__name__
    diagnostico.append(f"- Tipo do objeto: {tipo_objeto}")

    # Verificar se é DataFrame
    if isinstance(arquivo, pd.DataFrame):
        diagnostico.append("- ✅ É um DataFrame válido")
        diagnostico.append(f"- Shape: {arquivo.shape}")
        diagnostico.append(f"- Colunas: {list(arquivo.columns)}")
        diagnostico.append("- **Solução:** DataFrame não precisa ser 'carregado', use diretamente")

    # Verificar se tem atributos de arquivo
    elif hasattr(arquivo, 'read'):
        diagnostico.append("- ✅ Tem método 'read'")
        if hasattr(arquivo, 'name'):
            diagnostico.append(f"- ✅ Nome: {arquivo.name}")
        else:
            diagnostico.append("- ❌ Sem atributo 'name'")

        if hasattr(arquivo, 'size'):
            diagnostico.append(f"- ✅ Tamanho: {arquivo.size} bytes")
        else:
            diagnostico.append("- ❌ Sem atributo 'size'")

    else:
        diagnostico.append("- ❌ Não é um arquivo válido do Streamlit")
        diagnostico.append("- **Possíveis causas:**")
        diagnostico.append("  - Objeto já processado anteriormente")
        diagnostico.append("  - Problema no upload do Streamlit")
        diagnostico.append("  - Tentativa de reprocessar DataFrame")

    # Adicionar erro original
    diagnostico.append(f"\n**Erro Original:** {erro_original}")

    return "\n".join(diagnostico)


# Função auxiliar para sidebar e outras interfaces
def carregar_arquivo_seguro(arquivo, nome_tipo="arquivo"):
    """
    Versão segura da função carregar_arquivo com melhor tratamento de erros

    Args:
        arquivo: Arquivo ou DataFrame
        nome_tipo: Nome para mensagens de erro

    Returns:
        DataFrame ou None
    """
    try:
        # Se já é DataFrame, retornar diretamente
        if isinstance(arquivo, pd.DataFrame):
            return arquivo

        # Se é None, retornar None
        if arquivo is None:
            return None

        # Tentar carregar normalmente
        resultado = carregar_arquivo(arquivo)
        return resultado

    except Exception as e:
        # Diagnóstico detalhado em caso de erro
        st.error(f"❌ Erro ao carregar {nome_tipo}")

        with st.expander("🔍 Diagnóstico Detalhado"):
            diagnostico = diagnosticar_problema_arquivo(arquivo, str(e))
            st.markdown(diagnostico)

            # Oferecer exemplo para teste
            if st.button(f"📋 Criar Exemplo de {nome_tipo.title()}", key=f"exemplo_{nome_tipo}"):
                exemplo = criar_arquivo_exemplo_para_debug(nome_tipo)
                st.success(f"✅ Exemplo de {nome_tipo} criado!")
                st.dataframe(exemplo)
                return exemplo

        return None


def exportar_dataframe(df, formato='csv', nome_base="dados_exportados"):
    """
    Exporta DataFrame em diferentes formatos
    VERSÃO CORRIGIDA: Melhor tratamento de tipos

    Args:
        df: DataFrame a ser exportado
        formato: Formato de exportação ('csv', 'excel', 'json')
        nome_base: Nome base do arquivo

    Returns:
        tuple: (dados_bytes, nome_arquivo, mime_type)
    """
    try:
        if not isinstance(df, pd.DataFrame):
            st.error("❌ Objeto não é um DataFrame válido para exportação")
            return None, None, None

        if len(df) == 0:
            st.warning("⚠️ DataFrame vazio - nada para exportar")
            return None, None, None

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


# utils/arquivo_handler.py - FUNÇÃO CRIAR_TEMPLATE_CSV
"""
Função para criar templates CSV de exemplo para o sistema GreenVista
Gera arquivos de exemplo com estrutura correta para upload
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
import streamlit as st
from datetime import datetime


def criar_template_csv(tipo_template: str, n_registros: int = 20) -> Optional[str]:
    """
    Cria template CSV para diferentes tipos de dados do sistema

    Args:
        tipo_template: Tipo do template ('inventario', 'cubagem', 'lidar_metricas', 'coordenadas')
        n_registros: Número de registros de exemplo para gerar

    Returns:
        str: CSV formatado como string ou None se erro
    """
    try:
        if tipo_template == 'inventario':
            return _criar_template_inventario(n_registros)
        elif tipo_template == 'cubagem':
            return _criar_template_cubagem(n_registros)
        elif tipo_template == 'lidar_metricas':
            return _criar_template_lidar_metricas(n_registros)
        elif tipo_template == 'coordenadas':
            return _criar_template_coordenadas(n_registros)
        elif tipo_template == 'shapefile_areas':
            return _criar_template_areas_talhoes(n_registros)
        else:
            st.error(f"❌ Tipo de template não reconhecido: {tipo_template}")
            return None

    except Exception as e:
        st.error(f"❌ Erro ao criar template {tipo_template}: {e}")
        return None


def _criar_template_inventario(n_registros: int) -> str:
    """Cria template para dados de inventário florestal"""

    # Simular dados realísticos de inventário
    np.random.seed(42)  # Para reprodutibilidade

    dados = []

    for i in range(n_registros):
        # Distribuir entre talhões
        talhao = np.random.choice([1, 2, 3, 4, 5], p=[0.3, 0.25, 0.2, 0.15, 0.1])

        # Parcelas dentro do talhão
        parcela = (i % 10) + 1

        # Simular DAP com base na idade do talhão
        idade_base = {1: 7, 2: 6, 3: 5, 4: 4, 5: 3}
        idade = idade_base[talhao] + np.random.normal(0, 0.5)

        # DAP baseado na idade (modelo simplificado)
        dap_medio = 2.5 * idade + np.random.normal(0, 2)
        dap = max(4.0, dap_medio)  # Mínimo 4cm

        # Altura baseada no DAP (relação hipsométrica simulada)
        altura = 1.3 + (25 * (1 - np.exp(-0.05 * dap))) + np.random.normal(0, 1.5)
        altura = max(2.0, altura)  # Mínimo 2m

        # Código da árvore
        cod = f"T{talhao}P{parcela:02d}A{(i % 20) + 1:03d}"

        dados.append({
            'talhao': int(talhao),
            'parcela': int(parcela),
            'cod': cod,
            'D_cm': round(dap, 1),
            'H_m': round(altura, 1),
            'idade_anos': round(idade, 1)
        })

    # Criar DataFrame
    df = pd.DataFrame(dados)

    # Adicionar cabeçalho explicativo como comentário
    comentario = f"""# TEMPLATE - DADOS DE INVENTÁRIO FLORESTAL
# Arquivo gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# 
# COLUNAS OBRIGATÓRIAS:
# - talhao: Número do talhão (inteiro)
# - parcela: Número da parcela dentro do talhão (inteiro) 
# - D_cm: Diâmetro à altura do peito em centímetros (decimal)
# - H_m: Altura total em metros (decimal)
#
# COLUNAS OPCIONAIS:
# - cod: Código identificador da árvore (texto)
# - idade_anos: Idade do povoamento em anos (decimal)
# - x, y: Coordenadas UTM (se disponíveis)
#
# IMPORTANTE:
# - Use ponto (.) como separador decimal
# - Use ponto e vírgula (;) como separador de colunas
# - Não deixe células vazias nas colunas obrigatórias
# - DAP mínimo recomendado: 4.0 cm
# - Altura mínima recomendada: 1.3 m
#
"""

    # Converter para CSV
    csv_content = df.to_csv(index=False, sep=';', decimal=',')

    return comentario + csv_content


def _criar_template_cubagem(n_registros: int) -> str:
    """Cria template para dados de cubagem rigorosa"""

    np.random.seed(42)

    dados = []

    # Simular 5 árvores com diferentes números de seções
    n_arvores = min(5, max(1, n_registros // 10))

    for arv_id in range(1, n_arvores + 1):
        # Talhão aleatório
        talhao = np.random.choice([1, 2, 3])

        # DAP e altura da árvore
        dap = np.random.uniform(15, 35)  # DAP entre 15-35 cm
        altura_total = 15 + (dap - 15) * 0.8 + np.random.normal(0, 2)
        altura_total = max(10, altura_total)

        # Número de seções (baseado na altura)
        n_secoes = max(5, int(altura_total / 2))  # Aproximadamente a cada 2m

        for secao in range(n_secoes):
            # Altura da seção
            h_secao = (secao + 1) * (altura_total / n_secoes)

            # Diâmetro da seção (afilamento)
            fator_afilamento = 1 - (h_secao / altura_total) * 0.7  # Redução até 70%
            d_secao = dap * fator_afilamento + np.random.normal(0, 0.5)
            d_secao = max(1.0, d_secao)  # Mínimo 1cm

            dados.append({
                'arv': int(arv_id),
                'talhao': int(talhao),
                'secao': int(secao + 1),
                'd_cm': round(d_secao, 1),
                'h_m': round(h_secao, 2),
                'D_cm': round(dap, 1),
                'H_m': round(altura_total, 1)
            })

    # Completar com mais registros se necessário
    while len(dados) < n_registros:
        ultimo_registro = dados[-1].copy()
        ultimo_registro['secao'] += 1
        ultimo_registro['h_m'] = round(ultimo_registro['h_m'] + 2.0, 2)
        ultimo_registro['d_cm'] = max(1.0, ultimo_registro['d_cm'] - 1.0)
        dados.append(ultimo_registro)

    # Limitar ao número solicitado
    dados = dados[:n_registros]

    df = pd.DataFrame(dados)

    comentario = f"""# TEMPLATE - DADOS DE CUBAGEM RIGOROSA
# Arquivo gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
#
# COLUNAS OBRIGATÓRIAS:
# - arv: Número identificador da árvore (inteiro)
# - talhao: Número do talhão (inteiro)
# - d_cm: Diâmetro da seção em centímetros (decimal)
# - h_m: Altura da seção em metros (decimal)
# - D_cm: DAP da árvore em centímetros (decimal)
# - H_m: Altura total da árvore em metros (decimal)
#
# COLUNAS OPCIONAIS:
# - secao: Número da seção na árvore (inteiro)
# - cod_arv: Código da árvore (texto)
#
# OBSERVAÇÕES:
# - Cada linha representa uma seção de medição na árvore
# - D_cm e H_m devem ser iguais para todas as seções da mesma árvore
# - d_cm e h_m variam para cada seção
# - Recomenda-se medições a cada 1-2 metros de altura
# - d_cm deve ser <= D_cm (diâmetro da seção <= DAP)
#
"""

    csv_content = df.to_csv(index=False, sep=';', decimal=',')
    return comentario + csv_content


def _criar_template_lidar_metricas(n_registros: int) -> str:
    """Cria template para métricas LiDAR pré-processadas"""

    np.random.seed(42)

    dados = []

    for i in range(n_registros):
        talhao = (i // 10) + 1
        parcela = (i % 10) + 1

        # Simular métricas LiDAR realísticas
        altura_media = np.random.uniform(18, 28)
        altura_maxima = altura_media + np.random.uniform(5, 12)
        altura_minima = max(1.3, altura_media - np.random.uniform(8, 15))

        desvio_altura = np.random.uniform(2, 6)
        cobertura = np.random.uniform(75, 95)  # % de cobertura
        densidade = np.random.uniform(5, 25)  # pontos/m²

        # Percentis de altura
        altura_p95 = altura_maxima - np.random.uniform(1, 3)
        altura_p75 = altura_media + np.random.uniform(2, 5)
        altura_p50 = altura_media + np.random.uniform(-2, 2)
        altura_p25 = altura_media - np.random.uniform(3, 6)

        # Métricas derivadas
        complexidade = np.random.uniform(0.2, 0.8)
        rugosidade = np.random.uniform(1, 4)
        shannon_height = np.random.uniform(1.5, 2.5)

        dados.append({
            'talhao': int(talhao),
            'parcela': int(parcela),
            'altura_media': round(altura_media, 2),
            'altura_maxima': round(altura_maxima, 2),
            'altura_minima': round(altura_minima, 2),
            'desvio_altura': round(desvio_altura, 2),
            'altura_p95': round(altura_p95, 2),
            'altura_p75': round(altura_p75, 2),
            'altura_p50': round(altura_p50, 2),
            'altura_p25': round(altura_p25, 2),
            'cobertura': round(cobertura, 1),
            'densidade': round(densidade, 2),
            'complexidade': round(complexidade, 3),
            'rugosidade': round(rugosidade, 2),
            'shannon_height': round(shannon_height, 3),
            'n_pontos': int(np.random.uniform(50, 500))
        })

    df = pd.DataFrame(dados)

    comentario = f"""# TEMPLATE - MÉTRICAS LIDAR PRÉ-PROCESSADAS
# Arquivo gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
#
# COLUNAS OBRIGATÓRIAS:
# - talhao: Número do talhão (inteiro)
# - parcela: Número da parcela (inteiro)
#
# MÉTRICAS PRINCIPAIS (pelo menos uma obrigatória):
# - altura_media: Altura média LiDAR em metros
# - altura_maxima: Altura máxima LiDAR em metros  
# - altura_minima: Altura mínima LiDAR em metros
# - desvio_altura: Desvio padrão das alturas em metros
#
# PERCENTIS DE ALTURA (opcionais):
# - altura_p95, altura_p75, altura_p50, altura_p25: Percentis em metros
#
# MÉTRICAS ESTRUTURAIS (opcionais):
# - cobertura: Percentual de cobertura do dossel (0-100)
# - densidade: Densidade de pontos por m² 
# - complexidade: Índice de complexidade estrutural (0-1)
# - rugosidade: Rugosidade da superfície em metros
# - shannon_height: Diversidade de Shannon para alturas
# - n_pontos: Número total de pontos LiDAR na parcela
#
# NOMES ALTERNATIVOS ACEITOS:
# - zmean, zmax, zmin (para alturas média, máxima, mínima)
# - zsd (para desvio padrão)
# - cover, coverage (para cobertura)
# - point_density, dens (para densidade)
#
"""

    csv_content = df.to_csv(index=False, sep=';', decimal=',')
    return comentario + csv_content


def _criar_template_coordenadas(n_registros: int) -> str:
    """Cria template para coordenadas das parcelas"""

    np.random.seed(42)

    # Simular coordenadas UTM (zona 23S - região sudeste Brasil)
    x_base = 200000  # Coordenada X base
    y_base = 7500000  # Coordenada Y base

    dados = []

    for i in range(n_registros):
        talhao = (i // 10) + 1
        parcela = (i % 10) + 1

        # Distribuir parcelas em grid aproximado
        offset_x = (talhao - 1) * 1000 + (parcela - 1) * 100
        offset_y = (i % 5) * 100

        x = x_base + offset_x + np.random.uniform(-50, 50)
        y = y_base + offset_y + np.random.uniform(-50, 50)

        dados.append({
            'talhao': int(talhao),
            'parcela': int(parcela),
            'x': round(x, 2),
            'y': round(y, 2),
            'zona_utm': '23S',
            'datum': 'SIRGAS2000'
        })

    df = pd.DataFrame(dados)

    comentario = f"""# TEMPLATE - COORDENADAS DAS PARCELAS
# Arquivo gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
#
# COLUNAS OBRIGATÓRIAS:
# - talhao: Número do talhão (inteiro)
# - parcela: Número da parcela (inteiro)  
# - x: Coordenada X em metros (decimal)
# - y: Coordenada Y em metros (decimal)
#
# COLUNAS RECOMENDADAS:
# - zona_utm: Zona UTM (ex: 23S, 24K)
# - datum: Sistema de referência (ex: SIRGAS2000, WGS84)
# - precisao: Precisão da medição em metros
#
# OBSERVAÇÕES:
# - Use coordenadas no sistema UTM
# - Coordenadas devem estar em metros
# - Uma linha para cada parcela do inventário
# - Coordenadas representam o centro da parcela
# - Certifique-se da zona UTM correta para sua região
#
# SISTEMAS MAIS USADOS NO BRASIL:
# - SIRGAS2000 / UTM zone 23S (EPSG:31983) - SP, RJ, ES
# - SIRGAS2000 / UTM zone 24S (EPSG:31984) - MG, BA parte
# - SIRGAS2000 / UTM zone 22S (EPSG:31982) - MS, MT, GO
#
"""

    csv_content = df.to_csv(index=False, sep=';', decimal=',')
    return comentario + csv_content


def _criar_template_areas_talhoes(n_registros: int) -> str:
    """Cria template para áreas dos talhões (alternativa ao shapefile)"""

    np.random.seed(42)

    dados = []

    # Simular diferentes talhões com áreas variadas
    for talhao in range(1, min(n_registros + 1, 11)):
        # Área em hectares (variação realística)
        area_ha = np.random.uniform(5, 50)  # Entre 5 e 50 hectares
        area_m2 = area_ha * 10000  # Converter para m²

        # Informações adicionais
        perimetro = 4 * np.sqrt(area_m2)  # Aproximação para perímetro

        dados.append({
            'talhao': int(talhao),
            'area_ha': round(area_ha, 2),
            'area_m2': round(area_m2, 0),
            'perimetro_m': round(perimetro, 0),
            'forma': np.random.choice(['Regular', 'Irregular']),
            'observacoes': f'Talhão {talhao} - Eucalipto'
        })

    df = pd.DataFrame(dados)

    comentario = f"""# TEMPLATE - ÁREAS DOS TALHÕES
# Arquivo gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
#
# ALTERNATIVA AO SHAPEFILE:
# Use este template quando não tiver shapefile das áreas
# dos talhões, mas souber as áreas de cada um.
#
# COLUNAS OBRIGATÓRIAS:
# - talhao: Número do talhão (inteiro)
# - area_ha: Área em hectares (decimal) OU
# - area_m2: Área em metros quadrados (decimal)
#
# COLUNAS OPCIONAIS:
# - perimetro_m: Perímetro em metros
# - forma: Tipo de forma (Regular/Irregular)
# - observacoes: Observações gerais
# - especie: Espécie plantada
# - idade_plantio: Data do plantio
#
# OBSERVAÇÕES:
# - Use APENAS quando não tiver shapefile
# - Se tiver ambos (area_ha e area_m2), será usado area_ha
# - Área será aplicada para cálculo de volume/ha
# - Mais preciso usar shapefile quando disponível
#
"""

    csv_content = df.to_csv(index=False, sep=';', decimal=',')
    return comentario + csv_content


def obter_informacoes_template(tipo_template: str) -> Dict:
    """
    Retorna informações sobre um tipo de template

    Args:
        tipo_template: Tipo do template

    Returns:
        dict: Informações do template
    """

    templates_info = {
        'inventario': {
            'nome': 'Dados de Inventário Florestal',
            'descricao': 'Template para dados de parcelas do inventário com DAP e altura',
            'colunas_obrigatorias': ['talhao', 'parcela', 'D_cm', 'H_m'],
            'colunas_opcionais': ['cod', 'idade_anos', 'x', 'y'],
            'tamanho_recomendado': 50,
            'formato': 'CSV com separador ;'
        },

        'cubagem': {
            'nome': 'Dados de Cubagem Rigorosa',
            'descricao': 'Template para medições detalhadas de seções das árvores',
            'colunas_obrigatorias': ['arv', 'talhao', 'd_cm', 'h_m', 'D_cm', 'H_m'],
            'colunas_opcionais': ['secao', 'cod_arv'],
            'tamanho_recomendado': 100,
            'formato': 'CSV com separador ;'
        },

        'lidar_metricas': {
            'nome': 'Métricas LiDAR Pré-processadas',
            'descricao': 'Template para métricas já extraídas de dados LiDAR',
            'colunas_obrigatorias': ['talhao', 'parcela'],
            'colunas_opcionais': ['altura_media', 'altura_maxima', 'desvio_altura', 'cobertura', 'densidade'],
            'tamanho_recomendado': 30,
            'formato': 'CSV com separador ;'
        },

        'coordenadas': {
            'nome': 'Coordenadas das Parcelas',
            'descricao': 'Template para coordenadas UTM das parcelas do inventário',
            'colunas_obrigatorias': ['talhao', 'parcela', 'x', 'y'],
            'colunas_opcionais': ['zona_utm', 'datum', 'precisao'],
            'tamanho_recomendado': 30,
            'formato': 'CSV com separador ;'
        },

        'shapefile_areas': {
            'nome': 'Áreas dos Talhões',
            'descricao': 'Template alternativo ao shapefile com áreas dos talhões',
            'colunas_obrigatorias': ['talhao', 'area_ha'],
            'colunas_opcionais': ['area_m2', 'perimetro_m', 'forma', 'observacoes'],
            'tamanho_recomendado': 10,
            'formato': 'CSV com separador ;'
        }
    }

    return templates_info.get(tipo_template, {})


def listar_templates_disponiveis() -> List[str]:
    """
    Lista todos os templates disponíveis

    Returns:
        list: Lista com nomes dos templates
    """
    return ['inventario', 'cubagem', 'lidar_metricas', 'coordenadas', 'shapefile_areas']


def gerar_template_personalizado(colunas: Dict[str, str], n_registros: int = 20) -> str:
    """
    Gera template personalizado baseado em especificação de colunas

    Args:
        colunas: Dicionário {nome_coluna: tipo_dados}
        n_registros: Número de registros

    Returns:
        str: CSV formatado
    """
    try:
        np.random.seed(42)
        dados = []

        for i in range(n_registros):
            registro = {}

            for nome_col, tipo_dados in colunas.items():
                if tipo_dados == 'int':
                    registro[nome_col] = np.random.randint(1, 100)
                elif tipo_dados == 'float':
                    registro[nome_col] = round(np.random.uniform(1, 100), 2)
                elif tipo_dados == 'str':
                    registro[nome_col] = f"Item_{i + 1:03d}"
                else:
                    registro[nome_col] = "N/A"

            dados.append(registro)

        df = pd.DataFrame(dados)

        comentario = f"""# TEMPLATE PERSONALIZADO
# Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Colunas: {', '.join(colunas.keys())}
#
"""

        csv_content = df.to_csv(index=False, sep=';', decimal=',')
        return comentario + csv_content

    except Exception as e:
        st.error(f"❌ Erro ao gerar template personalizado: {e}")
        return None


def criar_interface_templates_streamlit():
    """
    Cria interface Streamlit para download de templates
    Para usar em páginas do sistema
    """
    st.subheader("📥 Download de Templates")

    # Informações sobre templates
    st.info("""
    **📋 Templates Disponíveis:**
    Baixe arquivos de exemplo com a estrutura correta para upload no sistema.
    """)

    # Seleção do tipo de template
    col1, col2 = st.columns([2, 1])

    with col1:
        templates_opcoes = {
            'inventario': '📋 Inventário Florestal',
            'cubagem': '📏 Cubagem Rigorosa',
            'lidar_metricas': '🛩️ Métricas LiDAR',
            'coordenadas': '📍 Coordenadas das Parcelas',
            'shapefile_areas': '📐 Áreas dos Talhões'
        }

        template_selecionado = st.selectbox(
            "Tipo de Template",
            options=list(templates_opcoes.keys()),
            format_func=lambda x: templates_opcoes[x],
            key="template_selector"
        )

    with col2:
        n_registros = st.number_input(
            "Nº de Registros",
            min_value=5,
            max_value=100,
            value=20,
            step=5,
            key="n_registros_template"
        )

    # Mostrar informações do template selecionado
    info_template = obter_informacoes_template(template_selecionado)

    if info_template:
        with st.expander(f"ℹ️ Sobre: {info_template['nome']}", expanded=True):
            st.markdown(f"**Descrição:** {info_template['descricao']}")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**📋 Colunas Obrigatórias:**")
                for col in info_template['colunas_obrigatorias']:
                    st.markdown(f"- `{col}`")

            with col2:
                st.markdown("**📝 Colunas Opcionais:**")
                for col in info_template['colunas_opcionais']:
                    st.markdown(f"- `{col}`")

            st.markdown(f"**📄 Formato:** {info_template['formato']}")

    # Botões de ação
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("👀 Preview Template", key="preview_template"):
            try:
                template_content = criar_template_csv(template_selecionado, min(n_registros, 10))
                if template_content:
                    st.code(template_content[:1000] + "..." if len(template_content) > 1000 else template_content)
            except Exception as e:
                st.error(f"❌ Erro no preview: {e}")

    with col2:
        # Botão de download
        try:
            template_content = criar_template_csv(template_selecionado, n_registros)
            if template_content:
                nome_arquivo = f"template_{template_selecionado}_{n_registros}reg.csv"

                st.download_button(
                    label="📥 Download Template",
                    data=template_content,
                    file_name=nome_arquivo,
                    mime="text/csv",
                    key="download_template",
                    help=f"Baixar template de {info_template.get('nome', template_selecionado)}"
                )
        except Exception as e:
            st.error(f"❌ Erro ao gerar template: {e}")

    with col3:
        if st.button("📖 Ver Instruções", key="instrucoes_template"):
            mostrar_instrucoes_template(template_selecionado)


def mostrar_instrucoes_template(tipo_template: str):
    """Mostra instruções específicas para cada tipo de template"""

    instrucoes = {
        'inventario': """
        ### 📋 Instruções - Template de Inventário

        **1. Preenchimento:**
        - Uma linha para cada árvore medida
        - `talhao` e `parcela`: Números inteiros identificando a localização
        - `D_cm`: Diâmetro à altura do peito (1,30m) em centímetros
        - `H_m`: Altura total da árvore em metros

        **2. Valores Recomendados:**
        - DAP mínimo: 4,0 cm (incluir apenas árvores mensuráveis)
        - Altura mínima: 1,3 m (altura do peito)
        - Use ponto como separador decimal: `15.2` (não `15,2`)

        **3. Colunas Opcionais:**
        - `cod`: Código único da árvore (ex: T1P01A001)
        - `idade_anos`: Idade do povoamento
        - `x`, `y`: Coordenadas UTM se disponíveis

        **4. Dicas:**
        - Mantenha consistência nos códigos dos talhões/parcelas
        - Verifique outliers antes do upload
        - Use o mesmo número de parcelas por talhão quando possível
        """,

        'cubagem': """
        ### 📏 Instruções - Template de Cubagem

        **1. Estrutura dos Dados:**
        - Cada linha = uma seção medida em uma árvore
        - Várias linhas por árvore (uma para cada seção)
        - `arv`: Número único da árvore cubada
        - `talhao`: Talhão onde a árvore está localizada

        **2. Medições das Seções:**
        - `d_cm`: Diâmetro na altura da seção
        - `h_m`: Altura da seção (acumulada desde a base)
        - `D_cm`: DAP da árvore (mesmo valor para todas as seções)
        - `H_m`: Altura total da árvore (mesmo valor para todas as seções)

        **3. Validações Automáticas:**
        - `d_cm` deve ser ≤ `D_cm` (diâmetro da seção ≤ DAP)
        - `h_m` deve ser crescente para a mesma árvore
        - `H_m` deve ser igual à maior `h_m` da árvore

        **4. Recomendações:**
        - Meça seções a cada 1-2 metros
        - Include medição na base (h=0) e no topo
        - Mínimo 5 seções por árvore para bom ajuste
        """,

        'lidar_metricas': """
        ### 🛩️ Instruções - Template Métricas LiDAR

        **1. Origem dos Dados:**
        - Use este template se já processou dados LiDAR em outro software
        - Métricas por parcela (não por árvore individual)
        - Uma linha para cada parcela com dados LiDAR

        **2. Métricas Principais:**
        - `altura_media`: Altura média dos pontos acima de 1,3m
        - `altura_maxima`: Altura do ponto mais alto
        - `desvio_altura`: Desvio padrão das alturas
        - `cobertura`: % de área coberta por pontos de vegetação

        **3. Nomes Alternativos Aceitos:**
        - `zmean` = `altura_media`
        - `zmax` = `altura_maxima`
        - `zsd` = `desvio_altura`
        - `cover` = `cobertura`

        **4. Integração:**
        - Parcelas devem coincidir com o inventário (`talhao`, `parcela`)
        - Sistema fará integração automática
        - Métricas faltantes não impedem o processamento
        """,

        'coordenadas': """
        ### 📍 Instruções - Template de Coordenadas

        **1. Sistema de Coordenadas:**
        - Use coordenadas UTM em metros
        - Especifique a zona UTM correta
        - Prefira SIRGAS2000 para dados do Brasil

        **2. Posicionamento:**
        - `x`, `y`: Centro da parcela (não canto)
        - Uma linha para cada parcela do inventário
        - Coordenadas devem corresponder exatamente às parcelas

        **3. Zonas UTM Brasileiras Comuns:**
        - 22S: MT, MS, GO (oeste)
        - 23S: SP, RJ, ES, MG (sudeste)
        - 24S: BA, SE (nordeste)

        **4. Precisão:**
        - GPS: ±3-5 metros (adequado)
        - GPS RTK: ±0,1 metro (ideal)
        - Evite coordenadas geográficas (lat/lon)
        """,

        'shapefile_areas': """
        ### 📐 Instruções - Template Áreas dos Talhões

        **1. Quando Usar:**
        - Alternativa quando não tem shapefile
        - Você conhece a área exata de cada talhão
        - Para cálculos de volume por hectare

        **2. Unidades:**
        - `area_ha`: Área em hectares (recomendado)
        - `area_m2`: Área em metros quadrados
        - Se fornecer ambos, será usado `area_ha`

        **3. Fonte dos Dados:**
        - Levantamento topográfico
        - GPS de perímetro
        - Mapas oficiais
        - Sistemas CAD/GIS

        **4. Importante:**
        - Um registro para cada talhão
        - Áreas devem ser realísticas (0,1 - 1000 ha)
        - Sistema preferirá shapefile se disponível
        """
    }

    instrucao = instrucoes.get(tipo_template, "Instruções não disponíveis para este template.")
    st.markdown(instrucao)


def validar_template_carregado(df: pd.DataFrame, tipo_template: str) -> Dict:
    """
    Valida se arquivo carregado corresponde ao template esperado

    Args:
        df: DataFrame carregado
        tipo_template: Tipo esperado do template

    Returns:
        dict: Resultado da validação
    """
    info_template = obter_informacoes_template(tipo_template)

    if not info_template:
        return {'valido': False, 'erros': ['Tipo de template não reconhecido']}

    erros = []
    alertas = []

    # Verificar colunas obrigatórias
    colunas_faltantes = []
    for col in info_template['colunas_obrigatorias']:
        if col not in df.columns:
            colunas_faltantes.append(col)

    if colunas_faltantes:
        erros.append(f"Colunas obrigatórias faltantes: {colunas_faltantes}")

    # Verificar se há dados
    if len(df) == 0:
        erros.append("Arquivo não contém dados")

    # Validações específicas por tipo
    if tipo_template == 'inventario':
        if 'D_cm' in df.columns:
            daps_invalidos = df['D_cm'].le(0).sum()
            if daps_invalidos > 0:
                alertas.append(f"{daps_invalidos} registros com DAP ≤ 0")

        if 'H_m' in df.columns:
            alturas_invalidas = df['H_m'].le(1.3).sum()
            if alturas_invalidas > 0:
                alertas.append(f"{alturas_invalidas} registros com altura ≤ 1.3m")

    elif tipo_template == 'cubagem':
        if 'd_cm' in df.columns and 'D_cm' in df.columns:
            inconsistencias = (df['d_cm'] > df['D_cm']).sum()
            if inconsistencias > 0:
                erros.append(f"{inconsistencias} registros com diâmetro seção > DAP")

    elif tipo_template == 'coordenadas':
        if 'x' in df.columns and 'y' in df.columns:
            coords_invalidas = (df['x'].le(0) | df['y'].le(0)).sum()
            if coords_invalidas > 0:
                alertas.append(f"{coords_invalidas} coordenadas suspeitas (≤ 0)")

    return {
        'valido': len(erros) == 0,
        'erros': erros,
        'alertas': alertas,
        'info': info_template
    }


def criar_template_baseado_em_dados(df_exemplo: pd.DataFrame, tipo_sugerido: str = None) -> str:
    """
    Cria template baseado em dados existentes (engenharia reversa)

    Args:
        df_exemplo: DataFrame de exemplo
        tipo_sugerido: Tipo sugerido de template

    Returns:
        str: Template CSV gerado
    """
    try:
        # Analisar estrutura dos dados
        colunas_numericas = df_exemplo.select_dtypes(include=[np.number]).columns.tolist()
        colunas_texto = df_exemplo.select_dtypes(include=['object']).columns.tolist()

        # Criar amostra representativa (5-10 registros)
        if len(df_exemplo) > 10:
            df_amostra = df_exemplo.sample(n=10, random_state=42)
        else:
            df_amostra = df_exemplo.copy()

        # Limpar dados sensíveis (opcional)
        for col in colunas_texto:
            if 'cod' in col.lower() or 'id' in col.lower():
                df_amostra[col] = df_amostra[col].apply(lambda x: f"EXEMPLO_{hash(str(x)) % 1000:03d}")

        # Adicionar cabeçalho explicativo
        comentario = f"""# TEMPLATE BASEADO EM DADOS EXISTENTES
# Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Baseado em arquivo com {len(df_exemplo)} registros
# Tipo sugerido: {tipo_sugerido or 'Não especificado'}
#
# ESTRUTURA DETECTADA:
# - Colunas numéricas: {', '.join(colunas_numericas)}
# - Colunas de texto: {', '.join(colunas_texto)}
#
# IMPORTANTE:
# - Este é um exemplo baseado em seus dados
# - Substitua os valores pelos dados reais
# - Mantenha a estrutura das colunas
# - Use separador ; (ponto e vírgula)
#
"""

        csv_content = df_amostra.to_csv(index=False, sep=';', decimal=',')
        return comentario + csv_content

    except Exception as e:
        st.error(f"❌ Erro ao criar template baseado em dados: {e}")
        return None


# Função auxiliar para integração fácil em outras páginas
def mostrar_secao_templates_rapida():
    """
    Seção compacta de templates para usar em outras páginas
    """
    with st.expander("📥 Download de Templates", expanded=False):
        st.info("Baixe arquivos de exemplo com a estrutura correta:")

        col1, col2 = st.columns(2)

        templates_principais = {
            'inventario': '📋 Inventário',
            'cubagem': '📏 Cubagem',
            'lidar_metricas': '🛩️ LiDAR',
            'coordenadas': '📍 Coordenadas'
        }

        for i, (template_id, nome) in enumerate(templates_principais.items()):
            col = col1 if i % 2 == 0 else col2

            with col:
                try:
                    template_content = criar_template_csv(template_id, 10)
                    if template_content:
                        st.download_button(
                            label=nome,
                            data=template_content,
                            file_name=f"template_{template_id}.csv",
                            mime="text/csv",
                            key=f"download_template_{template_id}_rapido",
                            use_container_width=True
                        )
                except Exception:
                    st.error(f"Erro em {nome}")


def verificar_qualidade_dados(df: pd.DataFrame, tipo_dados: str = 'inventario') -> Dict:
    """
    Verifica a qualidade dos dados carregados e gera relatório completo

    Args:
        df: DataFrame com os dados a serem verificados
        tipo_dados: Tipo dos dados ('inventario', 'cubagem', 'lidar_metricas', 'coordenadas')

    Returns:
        dict: Relatório completo de qualidade dos dados
    """
    try:
        if df is None or df.empty:
            return {
                'qualidade_geral': 'CRÍTICA',
                'pontuacao': 0,
                'erros_criticos': ['DataFrame vazio ou nulo'],
                'alertas': [],
                'sugestoes': ['Verifique se o arquivo foi carregado corretamente'],
                'estatisticas': {},
                'detalhes': {}
            }

        # Inicializar relatório
        relatorio = {
            'qualidade_geral': 'BOA',
            'pontuacao': 100,
            'erros_criticos': [],
            'alertas': [],
            'sugestoes': [],
            'estatisticas': {},
            'detalhes': {},
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # Análises gerais
        relatorio = _analisar_estrutura_geral(df, relatorio)
        relatorio = _analisar_valores_faltantes(df, relatorio)
        relatorio = _analisar_duplicatas(df, relatorio)
        relatorio = _analisar_tipos_dados(df, relatorio)

        # Análises específicas por tipo
        if tipo_dados == 'inventario':
            relatorio = _analisar_qualidade_inventario(df, relatorio)
        elif tipo_dados == 'cubagem':
            relatorio = _analisar_qualidade_cubagem(df, relatorio)
        elif tipo_dados == 'lidar_metricas':
            relatorio = _analisar_qualidade_lidar(df, relatorio)
        elif tipo_dados == 'coordenadas':
            relatorio = _analisar_qualidade_coordenadas(df, relatorio)

        # Análise de outliers gerais
        relatorio = _analisar_outliers_gerais(df, relatorio)

        # Calcular qualidade geral final
        relatorio = _calcular_qualidade_final(relatorio)

        return relatorio

    except Exception as e:
        return {
            'qualidade_geral': 'ERRO',
            'pontuacao': 0,
            'erros_criticos': [f'Erro na análise de qualidade: {str(e)}'],
            'alertas': [],
            'sugestoes': ['Verifique o formato e estrutura do arquivo'],
            'estatisticas': {},
            'detalhes': {}
        }


def _analisar_estrutura_geral(df: pd.DataFrame, relatorio: Dict) -> Dict:
    """Analisa estrutura geral do DataFrame"""

    # Estatísticas básicas
    relatorio['estatisticas']['total_registros'] = len(df)
    relatorio['estatisticas']['total_colunas'] = len(df.columns)
    relatorio['estatisticas']['memoria_mb'] = round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)

    # Verificar tamanho mínimo
    if len(df) < 5:
        relatorio['erros_criticos'].append(f'Poucos registros: {len(df)} (mínimo recomendado: 5)')
        relatorio['pontuacao'] -= 20
    elif len(df) < 10:
        relatorio['alertas'].append(f'Poucos registros para análise robusta: {len(df)}')
        relatorio['pontuacao'] -= 5

    # Verificar colunas duplicadas
    colunas_duplicadas = df.columns[df.columns.duplicated()].tolist()
    if colunas_duplicadas:
        relatorio['erros_criticos'].append(f'Colunas duplicadas: {colunas_duplicadas}')
        relatorio['pontuacao'] -= 15

    # Verificar nomes de colunas
    colunas_problema = []
    for col in df.columns:
        if col.strip() != col:  # Espaços no início/fim
            colunas_problema.append(f'{col} (espaços)')
        elif any(char in col for char in ['/', '\\', '?', '*', ':', '|', '<', '>']):
            colunas_problema.append(f'{col} (caracteres especiais)')

    if colunas_problema:
        relatorio['alertas'].append(f'Nomes de colunas com problemas: {colunas_problema[:3]}')
        relatorio['sugestoes'].append('Padronize nomes de colunas (sem espaços, caracteres especiais)')
        relatorio['pontuacao'] -= 3

    return relatorio


def _analisar_valores_faltantes(df: pd.DataFrame, relatorio: Dict) -> Dict:
    """Analisa valores faltantes (NaN, vazios)"""

    valores_faltantes = df.isnull().sum()
    percentual_faltantes = (valores_faltantes / len(df)) * 100

    # Estatísticas de valores faltantes
    relatorio['detalhes']['valores_faltantes'] = {
        'por_coluna': valores_faltantes.to_dict(),
        'percentual_por_coluna': percentual_faltantes.round(2).to_dict(),
        'total_celulas_faltantes': valores_faltantes.sum(),
        'percentual_total': round((valores_faltantes.sum() / (len(df) * len(df.columns))) * 100, 2)
    }

    # Alertas por percentual de valores faltantes
    colunas_criticas = percentual_faltantes[percentual_faltantes > 50].index.tolist()
    colunas_problema = percentual_faltantes[(percentual_faltantes > 20) & (percentual_faltantes <= 50)].index.tolist()
    colunas_atencao = percentual_faltantes[(percentual_faltantes > 5) & (percentual_faltantes <= 20)].index.tolist()

    if colunas_criticas:
        relatorio['erros_criticos'].append(f'Colunas com >50% dados faltantes: {colunas_criticas}')
        relatorio['pontuacao'] -= 25

    if colunas_problema:
        relatorio['alertas'].append(f'Colunas com 20-50% dados faltantes: {colunas_problema}')
        relatorio['pontuacao'] -= 10

    if colunas_atencao:
        relatorio['alertas'].append(f'Colunas com 5-20% dados faltantes: {colunas_atencao}')
        relatorio['pontuacao'] -= 5

    # Linhas completamente vazias
    linhas_vazias = df.isnull().all(axis=1).sum()
    if linhas_vazias > 0:
        relatorio['alertas'].append(f'Linhas completamente vazias: {linhas_vazias}')
        relatorio['sugestoes'].append('Remova linhas completamente vazias')
        relatorio['pontuacao'] -= 5

    return relatorio


def _analisar_duplicatas(df: pd.DataFrame, relatorio: Dict) -> Dict:
    """Analisa registros duplicados"""

    # Duplicatas completas
    duplicatas_completas = df.duplicated().sum()
    if duplicatas_completas > 0:
        relatorio['alertas'].append(f'Registros completamente duplicados: {duplicatas_completas}')
        relatorio['sugestoes'].append('Remova registros duplicados')
        relatorio['pontuacao'] -= 8

    # Duplicatas em colunas-chave (se existirem)
    colunas_chave_possiveis = ['talhao', 'parcela', 'cod', 'arv', 'id']
    colunas_chave_existentes = [col for col in colunas_chave_possiveis if col in df.columns]

    if len(colunas_chave_existentes) >= 2:
        duplicatas_chave = df.duplicated(subset=colunas_chave_existentes).sum()
        if duplicatas_chave > 0:
            relatorio['alertas'].append(f'Duplicatas em chaves {colunas_chave_existentes}: {duplicatas_chave}')
            relatorio['pontuacao'] -= 10

    relatorio['detalhes']['duplicatas'] = {
        'completas': duplicatas_completas,
        'chaves_verificadas': colunas_chave_existentes
    }

    return relatorio


def _analisar_tipos_dados(df: pd.DataFrame, relatorio: Dict) -> Dict:
    """Analisa tipos de dados e conversões necessárias"""

    tipos_por_coluna = df.dtypes.to_dict()
    relatorio['detalhes']['tipos_dados'] = {str(k): str(v) for k, v in tipos_por_coluna.items()}

    # Verificar colunas que deveriam ser numéricas mas estão como texto
    colunas_numericas_esperadas = ['D_cm', 'H_m', 'd_cm', 'h_m', 'x', 'y', 'area_ha', 'idade_anos',
                                   'altura_media', 'altura_maxima', 'densidade', 'cobertura']

    problemas_tipo = []
    for col in colunas_numericas_esperadas:
        if col in df.columns:
            if df[col].dtype == 'object':  # String
                # Tentar converter para verificar se é numérico
                try:
                    pd.to_numeric(df[col].dropna().head(10), errors='raise')
                except:
                    problemas_tipo.append(f'{col} (deveria ser numérico)')

    if problemas_tipo:
        relatorio['alertas'].append(f'Colunas com tipo incorreto: {problemas_tipo}')
        relatorio['sugestoes'].append('Verifique formatação numérica (use ponto como decimal)')
        relatorio['pontuacao'] -= 8

    # Verificar valores não-numéricos em colunas numéricas esperadas
    for col in df.select_dtypes(include=[np.number]).columns:
        valores_infinitos = np.isinf(df[col]).sum()
        if valores_infinitos > 0:
            relatorio['alertas'].append(f'Valores infinitos em {col}: {valores_infinitos}')
            relatorio['pontuacao'] -= 5

    return relatorio


def _analisar_qualidade_inventario(df: pd.DataFrame, relatorio: Dict) -> Dict:
    """Análise específica para dados de inventário"""

    # Verificar colunas obrigatórias
    colunas_obrigatorias = ['talhao', 'parcela', 'D_cm', 'H_m']
    colunas_faltantes = [col for col in colunas_obrigatorias if col not in df.columns]

    if colunas_faltantes:
        relatorio['erros_criticos'].append(f'Colunas obrigatórias faltantes: {colunas_faltantes}')
        relatorio['pontuacao'] -= 30
        return relatorio

    # Análises específicas do inventário
    if 'D_cm' in df.columns:
        dap_stats = _analisar_coluna_numerica(df['D_cm'], 'DAP (D_cm)', min_esperado=4.0, max_esperado=100.0)
        relatorio['detalhes']['DAP'] = dap_stats

        if dap_stats['valores_negativos'] > 0:
            relatorio['erros_criticos'].append(f'DAP com valores negativos: {dap_stats["valores_negativos"]}')
            relatorio['pontuacao'] -= 20

        if dap_stats['valores_zero'] > 0:
            relatorio['alertas'].append(f'DAP com valores zero: {dap_stats["valores_zero"]}')
            relatorio['pontuacao'] -= 10

        if dap_stats['outliers_baixos'] > 0:
            relatorio['alertas'].append(f'DAP muito baixos (<4cm): {dap_stats["outliers_baixos"]}')
            relatorio['pontuacao'] -= 5

        if dap_stats['outliers_altos'] > 0:
            relatorio['alertas'].append(f'DAP muito altos (>100cm): {dap_stats["outliers_altos"]}')
            relatorio['pontuacao'] -= 5

    if 'H_m' in df.columns:
        altura_stats = _analisar_coluna_numerica(df['H_m'], 'Altura (H_m)', min_esperado=1.3, max_esperado=60.0)
        relatorio['detalhes']['Altura'] = altura_stats

        if altura_stats['valores_negativos'] > 0:
            relatorio['erros_criticos'].append(f'Altura com valores negativos: {altura_stats["valores_negativos"]}')
            relatorio['pontuacao'] -= 20

        if altura_stats['outliers_baixos'] > 0:
            relatorio['alertas'].append(f'Alturas muito baixas (<1.3m): {altura_stats["outliers_baixos"]}')
            relatorio['pontuacao'] -= 5

        if altura_stats['outliers_altos'] > 0:
            relatorio['alertas'].append(f'Alturas muito altas (>60m): {altura_stats["outliers_altos"]}')
            relatorio['pontuacao'] -= 5

    # Verificar relação DAP x Altura
    if 'D_cm' in df.columns and 'H_m' in df.columns:
        relacao_stats = _analisar_relacao_dap_altura(df)
        relatorio['detalhes']['relacao_DAP_Altura'] = relacao_stats

        if relacao_stats['correlacao'] < 0.3:
            relatorio['alertas'].append(f'Correlação baixa DAP-Altura: {relacao_stats["correlacao"]:.3f}')
            relatorio['pontuacao'] -= 5

    # Análise de distribuição por talhão
    if 'talhao' in df.columns:
        talhao_stats = _analisar_distribuicao_talhoes(df)
        relatorio['detalhes']['distribuicao_talhoes'] = talhao_stats

        if talhao_stats['talhoes_poucos_dados'] > 0:
            relatorio['alertas'].append(
                f'Talhões com poucos dados (<5 registros): {talhao_stats["talhoes_poucos_dados"]}')
            relatorio['pontuacao'] -= 3

    return relatorio


def _analisar_qualidade_cubagem(df: pd.DataFrame, relatorio: Dict) -> Dict:
    """Análise específica para dados de cubagem"""

    # Verificar colunas obrigatórias
    colunas_obrigatorias = ['arv', 'talhao', 'd_cm', 'h_m', 'D_cm', 'H_m']
    colunas_faltantes = [col for col in colunas_obrigatorias if col not in df.columns]

    if colunas_faltantes:
        relatorio['erros_criticos'].append(f'Colunas obrigatórias faltantes: {colunas_faltantes}')
        relatorio['pontuacao'] -= 30
        return relatorio

    # Verificar consistência entre d_cm e D_cm
    if 'd_cm' in df.columns and 'D_cm' in df.columns:
        inconsistencias = (df['d_cm'] > df['D_cm']).sum()
        if inconsistencias > 0:
            relatorio['erros_criticos'].append(f'Diâmetro seção > DAP em {inconsistencias} registros')
            relatorio['pontuacao'] -= 25

    # Verificar consistência entre h_m e H_m
    if 'h_m' in df.columns and 'H_m' in df.columns:
        inconsistencias_h = (df['h_m'] > df['H_m']).sum()
        if inconsistencias_h > 0:
            relatorio['erros_criticos'].append(f'Altura seção > Altura total em {inconsistencias_h} registros')
            relatorio['pontuacao'] -= 25

    # Análise por árvore
    if 'arv' in df.columns:
        arvore_stats = _analisar_distribuicao_arvores_cubagem(df)
        relatorio['detalhes']['distribuicao_arvores'] = arvore_stats

        if arvore_stats['arvores_poucas_secoes'] > 0:
            relatorio['alertas'].append(f'Árvores com poucas seções (<3): {arvore_stats["arvores_poucas_secoes"]}')
            relatorio['pontuacao'] -= 5

    return relatorio


def _analisar_qualidade_lidar(df: pd.DataFrame, relatorio: Dict) -> Dict:
    """Análise específica para métricas LiDAR"""

    # Verificar colunas básicas
    colunas_basicas = ['talhao', 'parcela']
    colunas_faltantes = [col for col in colunas_basicas if col not in df.columns]

    if colunas_faltantes:
        relatorio['erros_criticos'].append(f'Colunas básicas faltantes: {colunas_faltantes}')
        relatorio['pontuacao'] -= 30

    # Verificar se há pelo menos uma métrica LiDAR
    metricas_lidar = ['altura_media', 'altura_maxima', 'zmean', 'zmax', 'desvio_altura', 'zsd']
    metricas_encontradas = [col for col in metricas_lidar if col in df.columns]

    if not metricas_encontradas:
        relatorio['erros_criticos'].append('Nenhuma métrica LiDAR reconhecida encontrada')
        relatorio['pontuacao'] -= 40
    else:
        relatorio['detalhes']['metricas_lidar_encontradas'] = metricas_encontradas

    # Analisar métricas específicas
    if 'altura_media' in df.columns or 'zmean' in df.columns:
        col_altura = 'altura_media' if 'altura_media' in df.columns else 'zmean'
        altura_stats = _analisar_coluna_numerica(df[col_altura], 'Altura LiDAR', min_esperado=2.0, max_esperado=80.0)
        relatorio['detalhes']['altura_lidar'] = altura_stats

        if altura_stats['outliers_baixos'] > 0:
            relatorio['alertas'].append(f'Alturas LiDAR muito baixas: {altura_stats["outliers_baixos"]}')

    return relatorio


def _analisar_qualidade_coordenadas(df: pd.DataFrame, relatorio: Dict) -> Dict:
    """Análise específica para coordenadas"""

    # Verificar colunas obrigatórias
    colunas_obrigatorias = ['talhao', 'parcela', 'x', 'y']
    colunas_faltantes = [col for col in colunas_obrigatorias if col not in df.columns]

    if colunas_faltantes:
        relatorio['erros_criticos'].append(f'Colunas obrigatórias faltantes: {colunas_faltantes}')
        relatorio['pontuacao'] -= 30
        return relatorio

    # Verificar coordenadas válidas
    if 'x' in df.columns and 'y' in df.columns:
        coords_zero = ((df['x'] == 0) | (df['y'] == 0)).sum()
        coords_negativas = ((df['x'] < 0) | (df['y'] < 0)).sum()

        if coords_zero > 0:
            relatorio['alertas'].append(f'Coordenadas zero: {coords_zero}')
            relatorio['pontuacao'] -= 10

        if coords_negativas > 0:
            relatorio['alertas'].append(f'Coordenadas negativas: {coords_negativas}')
            relatorio['pontuacao'] -= 5

        # Verificar se coordenadas estão em faixa realística (UTM Brasil)
        x_fora_faixa = ((df['x'] < 100000) | (df['x'] > 1000000)).sum()
        y_fora_faixa = ((df['y'] < 7000000) | (df['y'] > 10000000)).sum()

        if x_fora_faixa > 0 or y_fora_faixa > 0:
            relatorio['alertas'].append('Coordenadas fora da faixa UTM brasileira')
            relatorio['pontuacao'] -= 5

    return relatorio


def _analisar_outliers_gerais(df: pd.DataFrame, relatorio: Dict) -> Dict:
    """Análise geral de outliers usando IQR"""

    outliers_por_coluna = {}

    for col in df.select_dtypes(include=[np.number]).columns:
        if col in ['talhao', 'parcela', 'arv']:  # Pular colunas ID
            continue

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        if IQR > 0:  # Evitar divisão por zero
            limite_inferior = Q1 - 1.5 * IQR
            limite_superior = Q3 + 1.5 * IQR

            outliers_baixos = (df[col] < limite_inferior).sum()
            outliers_altos = (df[col] > limite_superior).sum()
            total_outliers = outliers_baixos + outliers_altos

            if total_outliers > 0:
                percentual = (total_outliers / len(df)) * 100
                outliers_por_coluna[col] = {
                    'total': total_outliers,
                    'percentual': round(percentual, 2),
                    'baixos': outliers_baixos,
                    'altos': outliers_altos
                }

                if percentual > 10:  # Mais de 10% outliers
                    relatorio['alertas'].append(f'Muitos outliers em {col}: {percentual:.1f}%')
                    relatorio['pontuacao'] -= 3

    relatorio['detalhes']['outliers'] = outliers_por_coluna

    return relatorio


def _analisar_coluna_numerica(serie: pd.Series, nome_coluna: str, min_esperado: float = None,
                              max_esperado: float = None) -> Dict:
    """Análise detalhada de uma coluna numérica"""

    stats = {
        'nome': nome_coluna,
        'total_valores': len(serie),
        'valores_validos': serie.notna().sum(),
        'valores_faltantes': serie.isna().sum(),
        'valores_zero': (serie == 0).sum(),
        'valores_negativos': (serie < 0).sum(),
        'media': round(serie.mean(), 3) if serie.notna().any() else None,
        'mediana': round(serie.median(), 3) if serie.notna().any() else None,
        'desvio_padrao': round(serie.std(), 3) if serie.notna().any() else None,
        'minimo': round(serie.min(), 3) if serie.notna().any() else None,
        'maximo': round(serie.max(), 3) if serie.notna().any() else None
    }

    if min_esperado is not None:
        stats['outliers_baixos'] = (serie < min_esperado).sum()
    else:
        stats['outliers_baixos'] = 0

    if max_esperado is not None:
        stats['outliers_altos'] = (serie > max_esperado).sum()
    else:
        stats['outliers_altos'] = 0

    return stats


def _analisar_relacao_dap_altura(df: pd.DataFrame) -> Dict:
    """Analisa relação entre DAP e altura"""

    dados_validos = df[['D_cm', 'H_m']].dropna()

    if len(dados_validos) < 3:
        return {'correlacao': None, 'observacoes': 'Poucos dados válidos'}

    correlacao = dados_validos['D_cm'].corr(dados_validos['H_m'])

    return {
        'correlacao': round(correlacao, 3) if not np.isnan(correlacao) else None,
        'n_pares_validos': len(dados_validos),
        'dap_medio': round(dados_validos['D_cm'].mean(), 2),
        'altura_media': round(dados_validos['H_m'].mean(), 2)
    }


def _analisar_distribuicao_talhoes(df: pd.DataFrame) -> Dict:
    """Analisa distribuição de dados por talhão"""

    dist_talhao = df['talhao'].value_counts().sort_index()

    return {
        'total_talhoes': len(dist_talhao),
        'registros_por_talhao': dist_talhao.to_dict(),
        'media_registros_por_talhao': round(dist_talhao.mean(), 1),
        'talhoes_poucos_dados': (dist_talhao < 5).sum(),
        'talhao_mais_dados': dist_talhao.idxmax(),
        'max_registros': dist_talhao.max()
    }


def _analisar_distribuicao_arvores_cubagem(df: pd.DataFrame) -> Dict:
    """Analisa distribuição de seções por árvore na cubagem"""

    secoes_por_arvore = df.groupby(['talhao', 'arv']).size()

    return {
        'total_arvores': len(secoes_por_arvore),
        'media_secoes_por_arvore': round(secoes_por_arvore.mean(), 1),
        'min_secoes': secoes_por_arvore.min(),
        'max_secoes': secoes_por_arvore.max(),
        'arvores_poucas_secoes': (secoes_por_arvore < 3).sum()
    }


def _calcular_qualidade_final(relatorio: Dict) -> Dict:
    """Calcula qualidade final baseada na pontuação"""

    pontuacao = max(0, relatorio['pontuacao'])  # Não deixar negativo

    if len(relatorio['erros_criticos']) > 0:
        relatorio['qualidade_geral'] = 'CRÍTICA'
    elif pontuacao >= 90:
        relatorio['qualidade_geral'] = 'EXCELENTE'
    elif pontuacao >= 80:
        relatorio['qualidade_geral'] = 'BOA'
    elif pontuacao >= 70:
        relatorio['qualidade_geral'] = 'REGULAR'
    elif pontuacao >= 60:
        relatorio['qualidade_geral'] = 'RUIM'
    else:
        relatorio['qualidade_geral'] = 'CRÍTICA'

    relatorio['pontuacao'] = pontuacao

    return relatorio


def gerar_relatorio_qualidade_formatado(relatorio: Dict) -> str:
    """
    Gera relatório de qualidade formatado em texto

    Args:
        relatorio: Dicionário com resultado da análise

    Returns:
        str: Relatório formatado
    """

    texto = f"""
# RELATÓRIO DE QUALIDADE DOS DADOS
**Gerado em:** {relatorio.get('timestamp', 'N/A')}

## RESUMO GERAL
- **Qualidade:** {relatorio['qualidade_geral']} ({relatorio['pontuacao']}/100 pontos)
- **Total de Registros:** {relatorio['estatisticas'].get('total_registros', 'N/A')}
- **Total de Colunas:** {relatorio['estatisticas'].get('total_colunas', 'N/A')}

## PROBLEMAS IDENTIFICADOS

### 🔴 Erros Críticos ({len(relatorio['erros_criticos'])})
"""

    for erro in relatorio['erros_criticos']:
        texto += f"- {erro}\n"

    texto += f"""
### ⚠️ Alertas ({len(relatorio['alertas'])})
"""

    for alerta in relatorio['alertas']:
        texto += f"- {alerta}\n"

    texto += f"""
### 💡 Sugestões de Melhoria ({len(relatorio['sugestoes'])})
"""

    for sugestao in relatorio['sugestoes']:
        texto += f"- {sugestao}\n"

    # Adicionar detalhes se disponíveis
    if 'valores_faltantes' in relatorio['detalhes']:
        vf = relatorio['detalhes']['valores_faltantes']
        texto += f"""
## VALORES FALTANTES
- **Total de células faltantes:** {vf['total_celulas_faltantes']} ({vf['percentual_total']}%)
- **Colunas mais afetadas:**
"""

        # Mostrar top 5 colunas com mais valores faltantes
        percentuais = vf['percentual_por_coluna']
        top_faltantes = sorted(percentuais.items(), key=lambda x: x[1], reverse=True)[:5]

        for col, pct in top_faltantes:
            if pct > 0:
                texto += f"  - {col}: {pct}%\n"

    # Adicionar informações sobre outliers
    if 'outliers' in relatorio['detalhes']:
        outliers = relatorio['detalhes']['outliers']
        if outliers:
            texto += f"""
## OUTLIERS DETECTADOS
"""
            for col, info in outliers.items():
                texto += f"- **{col}:** {info['total']} outliers ({info['percentual']}%)\n"

    # Adicionar estatísticas específicas se disponíveis
    if 'DAP' in relatorio['detalhes']:
        dap_info = relatorio['detalhes']['DAP']
        texto += f"""
## ANÁLISE DO DAP
- **Média:** {dap_info.get('media', 'N/A')} cm
- **Faixa:** {dap_info.get('minimo', 'N/A')} - {dap_info.get('maximo', 'N/A')} cm
- **Valores questionáveis:** {dap_info.get('outliers_baixos', 0) + dap_info.get('outliers_altos', 0)}
"""

    if 'Altura' in relatorio['detalhes']:
        altura_info = relatorio['detalhes']['Altura']
        texto += f"""
## ANÁLISE DA ALTURA
- **Média:** {altura_info.get('media', 'N/A')} m
- **Faixa:** {altura_info.get('minimo', 'N/A')} - {altura_info.get('maximo', 'N/A')} m
- **Valores questionáveis:** {altura_info.get('outliers_baixos', 0) + altura_info.get('outliers_altos', 0)}
"""

    if 'relacao_DAP_Altura' in relatorio['detalhes']:
        relacao = relatorio['detalhes']['relacao_DAP_Altura']
        correlacao = relacao.get('correlacao')
        if correlacao is not None:
            texto += f"""
## RELAÇÃO DAP x ALTURA
- **Correlação:** {correlacao} ({"Boa" if correlacao > 0.7 else "Regular" if correlacao > 0.5 else "Fraca"})
- **Pares válidos:** {relacao.get('n_pares_validos', 'N/A')}
"""

    texto += f"""
---
*Relatório gerado automaticamente pelo Sistema GreenVista*
*Para melhorar a qualidade, corrija os problemas identificados e recarregue os dados*
"""

    return texto


def mostrar_relatorio_qualidade_streamlit(relatorio: Dict):
    """
    Mostra relatório de qualidade na interface Streamlit

    Args:
        relatorio: Dicionário com resultado da análise
    """

    # Determinar cor baseada na qualidade
    cores_qualidade = {
        'EXCELENTE': 'success',
        'BOA': 'success',
        'REGULAR': 'warning',
        'RUIM': 'warning',
        'CRÍTICA': 'error',
        'ERRO': 'error'
    }

    cor = cores_qualidade.get(relatorio['qualidade_geral'], 'info')

    # Cabeçalho com qualidade geral
    if cor == 'success':
        st.success(f"🎉 **Qualidade dos Dados: {relatorio['qualidade_geral']}** ({relatorio['pontuacao']}/100 pontos)")
    elif cor == 'warning':
        st.warning(f"⚠️ **Qualidade dos Dados: {relatorio['qualidade_geral']}** ({relatorio['pontuacao']}/100 pontos)")
    else:
        st.error(f"❌ **Qualidade dos Dados: {relatorio['qualidade_geral']}** ({relatorio['pontuacao']}/100 pontos)")

    # Estatísticas gerais
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📊 Registros", relatorio['estatisticas'].get('total_registros', 0))

    with col2:
        st.metric("📋 Colunas", relatorio['estatisticas'].get('total_colunas', 0))

    with col3:
        st.metric("🔴 Erros Críticos", len(relatorio['erros_criticos']))

    with col4:
        st.metric("⚠️ Alertas", len(relatorio['alertas']))

    # Problemas identificados
    if relatorio['erros_criticos']:
        st.subheader("🔴 Erros Críticos")
        for erro in relatorio['erros_criticos']:
            st.error(f"• {erro}")

    if relatorio['alertas']:
        st.subheader("⚠️ Alertas")
        for alerta in relatorio['alertas']:
            st.warning(f"• {alerta}")

    if relatorio['sugestoes']:
        st.subheader("💡 Sugestões de Melhoria")
        for sugestao in relatorio['sugestoes']:
            st.info(f"• {sugestao}")

    # Detalhes em expanders
    if 'valores_faltantes' in relatorio['detalhes']:
        with st.expander("📊 Análise de Valores Faltantes"):
            vf = relatorio['detalhes']['valores_faltantes']

            st.metric("Total de células faltantes",
                      f"{vf['total_celulas_faltantes']} ({vf['percentual_total']}%)")

            # Gráfico de valores faltantes por coluna
            percentuais_df = pd.DataFrame(list(vf['percentual_por_coluna'].items()),
                                          columns=['Coluna', 'Percentual'])
            percentuais_df = percentuais_df[percentuais_df['Percentual'] > 0].sort_values('Percentual', ascending=False)

            if not percentuais_df.empty:
                st.bar_chart(percentuais_df.set_index('Coluna')['Percentual'])

    if 'outliers' in relatorio['detalhes'] and relatorio['detalhes']['outliers']:
        with st.expander("📈 Análise de Outliers"):
            outliers = relatorio['detalhes']['outliers']

            outliers_df = pd.DataFrame([
                {'Coluna': col, 'Total Outliers': info['total'],
                 'Percentual': info['percentual'], 'Baixos': info['baixos'], 'Altos': info['altos']}
                for col, info in outliers.items()
            ])

            st.dataframe(outliers_df, use_container_width=True, hide_index=True)

    # Análises específicas
    analises_especificas = ['DAP', 'Altura', 'relacao_DAP_Altura', 'distribuicao_talhoes']

    for analise in analises_especificas:
        if analise in relatorio['detalhes']:
            with st.expander(f"🔍 Análise: {analise.replace('_', ' ').title()}"):
                dados = relatorio['detalhes'][analise]

                if isinstance(dados, dict):
                    for chave, valor in dados.items():
                        if valor is not None:
                            st.write(f"**{chave.replace('_', ' ').title()}:** {valor}")

    # Botão para download do relatório
    st.subheader("📥 Download do Relatório")

    relatorio_texto = gerar_relatorio_qualidade_formatado(relatorio)

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="📄 Baixar Relatório (TXT)",
            data=relatorio_texto,
            file_name=f"relatorio_qualidade_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            key="download_relatorio_qualidade"
        )

    with col2:
        # Converter relatório para JSON para download
        import json
        relatorio_json = json.dumps(relatorio, indent=2, ensure_ascii=False, default=str)

        st.download_button(
            label="📊 Baixar Dados (JSON)",
            data=relatorio_json,
            file_name=f"dados_qualidade_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            key="download_dados_qualidade"
        )


def verificar_qualidade_rapida(df: pd.DataFrame) -> Dict:
    """
    Versão rápida da verificação de qualidade para uso em interfaces
    Retorna apenas informações essenciais

    Args:
        df: DataFrame a ser analisado

    Returns:
        dict: Resumo rápido da qualidade
    """

    if df is None or df.empty:
        return {
            'qualidade': 'CRÍTICA',
            'pontuacao': 0,
            'problemas_principais': ['DataFrame vazio'],
            'pode_prosseguir': False
        }

    problemas = []
    pontuacao = 100

    # Verificações básicas
    if len(df) < 5:
        problemas.append(f'Poucos registros: {len(df)}')
        pontuacao -= 30

    # Valores faltantes críticos
    valores_faltantes_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    if valores_faltantes_pct > 30:
        problemas.append(f'Muitos valores faltantes: {valores_faltantes_pct:.1f}%')
        pontuacao -= 25

    # Duplicatas
    duplicatas = df.duplicated().sum()
    if duplicatas > len(df) * 0.1:  # Mais de 10% duplicatas
        problemas.append(f'Muitas duplicatas: {duplicatas}')
        pontuacao -= 15

    # Determinar qualidade
    if pontuacao >= 80:
        qualidade = 'BOA'
    elif pontuacao >= 60:
        qualidade = 'REGULAR'
    else:
        qualidade = 'CRÍTICA'

    pode_prosseguir = len(problemas) == 0 or qualidade != 'CRÍTICA'

    return {
        'qualidade': qualidade,
        'pontuacao': max(0, pontuacao),
        'problemas_principais': problemas,
        'pode_prosseguir': pode_prosseguir,
        'total_registros': len(df),
        'total_colunas': len(df.columns),
        'valores_faltantes_pct': round(valores_faltantes_pct, 1)
    }


def sugerir_limpeza_automatica(relatorio: Dict) -> List[str]:
    """
    Sugere ações de limpeza automática baseadas no relatório de qualidade

    Args:
        relatorio: Relatório de qualidade completo

    Returns:
        list: Lista de sugestões de limpeza
    """

    sugestoes = []

    # Sugestões baseadas nos problemas identificados
    for erro in relatorio['erros_criticos']:
        if 'duplicatas' in erro.lower():
            sugestoes.append("🔄 Remover registros duplicados")
        elif 'valores negativos' in erro.lower():
            sugestoes.append("🔢 Corrigir valores negativos nas colunas numéricas")
        elif 'inconsistências' in erro.lower():
            sugestoes.append("✅ Validar consistência entre colunas relacionadas")

    for alerta in relatorio['alertas']:
        if 'outliers' in alerta.lower():
            sugestoes.append("📊 Revisar e filtrar outliers extremos")
        elif 'valores faltantes' in alerta.lower():
            sugestoes.append("📝 Tratar valores faltantes (remover ou imputar)")
        elif 'tipo incorreto' in alerta.lower():
            sugestoes.append("🔤 Corrigir tipos de dados das colunas")

    # Sugestões específicas baseadas em detalhes
    if 'valores_faltantes' in relatorio['detalhes']:
        vf = relatorio['detalhes']['valores_faltantes']
        if vf['percentual_total'] > 5:
            sugestoes.append("🧹 Limpeza geral de valores faltantes recomendada")

    # Remover duplicatas
    sugestoes = list(set(sugestoes))  # Remove duplicatas

    if not sugestoes:
        sugestoes.append("✅ Dados em boa qualidade - nenhuma limpeza crítica necessária")

    return sugestoes