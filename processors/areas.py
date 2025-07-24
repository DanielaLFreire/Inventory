# processors/areas.py
'''
Processamento de áreas dos talhões usando diferentes métodos
'''

import pandas as pd
import numpy as np
import streamlit as st


def processar_areas_por_metodo(metodo, **kwargs):
    '''
    Processa áreas dos talhões baseado no método selecionado

    Args:
        metodo: Método de cálculo ('manual', 'shapefile', 'coordenadas', 'simulacao')
        **kwargs: Argumentos específicos para cada método

    Returns:
        DataFrame com talhao e area_ha ou None se erro
    '''
    if metodo == 'manual':
        return processar_areas_manuais(kwargs.get('areas_dict'), kwargs.get('talhoes'))

    elif metodo == 'shapefile':
        return processar_shapefile_areas(kwargs.get('arquivo_shp'))

    elif metodo == 'coordenadas':
        return processar_coordenadas_areas(
            kwargs.get('arquivo_coord'),
            kwargs.get('raio_parcela')
        )

    elif metodo == 'simulacao':
        return processar_areas_simuladas(
            kwargs.get('df_inventario'),
            kwargs.get('config')
        )

    else:
        st.error(f"Método '{metodo}' não reconhecido")
        return None


def processar_areas_manuais(areas_dict, talhoes_lista):
    '''
    Processa áreas informadas manualmente

    Args:
        areas_dict: Dicionário {talhao: area_ha}
        talhoes_lista: Lista de talhões disponíveis

    Returns:
        DataFrame com áreas manuais
    '''
    try:
        if not areas_dict:
            st.error("❌ Nenhuma área manual foi informada")
            return None

        # Criar DataFrame a partir do dicionário
        df_areas = pd.DataFrame([
            {'talhao': int(talhao), 'area_ha': float(area)}
            for talhao, area in areas_dict.items()
        ])

        # Verificar se todos os talhões têm área
        talhoes_sem_area = set(talhoes_lista) - set(df_areas['talhao'])
        if talhoes_sem_area:
            st.warning(f"⚠️ Talhões sem área informada: {list(talhoes_sem_area)}")

            # Adicionar área padrão para talhões faltantes
            for talhao in talhoes_sem_area:
                df_areas = pd.concat([
                    df_areas,
                    pd.DataFrame({'talhao': [int(talhao)], 'area_ha': [25.0]})
                ], ignore_index=True)

        st.success(f"✅ Áreas manuais configuradas para {len(df_areas)} talhões")

        return df_areas

    except Exception as e:
        st.error(f"❌ Erro ao processar áreas manuais: {e}")
        return None


def processar_shapefile_areas(arquivo_shp):
    '''
    Processa shapefile para extrair áreas dos talhões

    Args:
        arquivo_shp: Arquivo shapefile carregado

    Returns:
        DataFrame com talhao e area_ha ou None se erro
    '''
    try:
        import geopandas as gpd
        import zipfile
        import tempfile
        import os

        if arquivo_shp.name.endswith('.zip'):
            # Processar arquivo ZIP com shapefile
            with tempfile.TemporaryDirectory() as temp_dir:
                with zipfile.ZipFile(arquivo_shp, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)

                # Procurar arquivo .shp
                shp_files = [f for f in os.listdir(temp_dir) if f.endswith('.shp')]
                if not shp_files:
                    raise Exception("Arquivo .shp não encontrado no ZIP")

                shp_path = os.path.join(temp_dir, shp_files[0])
                gdf = gpd.read_file(shp_path)
        else:
            # Arquivo .shp direto
            gdf = gpd.read_file(arquivo_shp)

        # Extrair áreas do GeoDataFrame
        df_areas = extrair_areas_geodataframe(gdf)

        if df_areas is not None:
            st.success(f"✅ Áreas extraídas do shapefile: {len(df_areas)} talhões")

        return df_areas

    except ImportError:
        st.error("❌ GeoPandas não está instalado")
        st.code("pip install geopandas")
        return None

    except Exception as e:
        st.error(f"❌ Erro ao processar shapefile: {e}")
        st.info("💡 Verifique se o arquivo contém colunas 'talhao' e 'area_ha'")
        return None


def extrair_areas_geodataframe(gdf):
    '''
    Extrai áreas dos talhões do GeoDataFrame

    Args:
        gdf: GeoDataFrame do shapefile

    Returns:
        DataFrame com talhao e area_ha
    '''
    # Procurar coluna de talhão (variações possíveis)
    colunas_talhao_possiveis = ['talhao', 'talhão', 'talh', 'plot', 'stand', 'id', 'ID']
    col_talhao = None

    for col in gdf.columns:
        if col.lower() in [c.lower() for c in colunas_talhao_possiveis]:
            col_talhao = col
            break

    if col_talhao is None:
        raise Exception("Coluna de talhão não encontrada. Procuradas: " + str(colunas_talhao_possiveis))

    # Procurar coluna de área
    colunas_area_possiveis = ['area_ha', 'area', 'hectares', 'ha', 'area_m2', 'superficie']
    col_area = None

    for col in gdf.columns:
        if col.lower() in [c.lower() for c in colunas_area_possiveis]:
            col_area = col
            break

    if col_area is None:
        # Calcular área da geometria se não existe coluna específica
        try:
            # Verificar se geometria está em coordenadas geográficas
            if gdf.crs and gdf.crs.is_geographic:
                # Converter para projeção adequada (UTM ou similar)
                gdf_proj = gdf.to_crs('EPSG:3857')  # Web Mercator
                gdf['area_calculada'] = gdf_proj.geometry.area / 10000  # m² para ha
            else:
                gdf['area_calculada'] = gdf.geometry.area / 10000  # Assumir que já está em metros

            col_area = 'area_calculada'
            st.info("ℹ️ Área calculada automaticamente da geometria")

        except Exception as e:
            raise Exception(f"Não foi possível calcular área da geometria: {e}")

    # Agrupar por talhão e somar áreas (caso existam múltiplos polígonos por talhão)
    try:
        df_areas = gdf.groupby(col_talhao)[col_area].sum().reset_index()
        df_areas.columns = ['talhao', 'area_ha']

        # Converter talhão para inteiro
        df_areas['talhao'] = pd.to_numeric(df_areas['talhao'], errors='coerce').astype('Int64')

        # Remover linhas com talhão inválido
        df_areas = df_areas.dropna(subset=['talhao'])

        # Converter área para float e garantir valores positivos
        df_areas['area_ha'] = pd.to_numeric(df_areas['area_ha'], errors='coerce')
        df_areas = df_areas[df_areas['area_ha'] > 0]

        if len(df_areas) == 0:
            raise Exception("Nenhuma área válida encontrada no shapefile")

        return df_areas

    except Exception as e:
        raise Exception(f"Erro ao processar dados do shapefile: {e}")


def processar_coordenadas_areas(arquivo_coord, raio_parcela):
    '''
    Processa coordenadas para calcular áreas dos talhões

    Args:
        arquivo_coord: Arquivo com coordenadas das parcelas
        raio_parcela: Raio da parcela em metros

    Returns:
        DataFrame com talhao e area_ha
    '''
    try:
        from utils.arquivo_handler import carregar_arquivo

        # Carregar arquivo de coordenadas
        df_coord = carregar_arquivo(arquivo_coord)
        if df_coord is None:
            return None

        # Verificar colunas de coordenadas
        colunas_coord = None
        coord_possiveis = [['x', 'y'], ['X', 'Y'], ['lon', 'lat'], ['longitude', 'latitude']]

        for coord_set in coord_possiveis:
            if all(col in df_coord.columns for col in coord_set):
                colunas_coord = coord_set
                break

        if not colunas_coord:
            st.error("❌ Colunas de coordenadas não encontradas. Esperadas: X,Y ou lon,lat")
            return None

        # Verificar coluna de talhão
        colunas_talhao_possiveis = ['talhao', 'talhão', 'talh', 'plot', 'stand']
        col_talhao = None

        for col in df_coord.columns:
            if col.lower() in [c.lower() for c in colunas_talhao_possiveis]:
                col_talhao = col
                break

        if col_talhao is None:
            st.error("❌ Coluna de talhão não encontrada nas coordenadas")
            return None

        # Calcular área circular da parcela
        area_parcela_ha = np.pi * (raio_parcela ** 2) / 10000  # Converter m² para ha

        # Contar parcelas por talhão
        parcelas_por_talhao = df_coord.groupby(col_talhao).size().reset_index()
        parcelas_por_talhao.columns = ['talhao', 'num_parcelas']

        # Calcular área total do talhão
        parcelas_por_talhao['area_ha'] = parcelas_por_talhao['num_parcelas'] * area_parcela_ha

        # Preparar DataFrame final
        df_areas = parcelas_por_talhao[['talhao', 'area_ha']].copy()
        df_areas['talhao'] = df_areas['talhao'].astype(int)

        st.success(f"✅ Áreas calculadas das coordenadas: {len(df_areas)} talhões")
        st.info(f"📐 Área por parcela: {area_parcela_ha:.4f} ha (raio {raio_parcela}m)")

        return df_areas

    except Exception as e:
        st.error(f"❌ Erro ao processar coordenadas: {e}")
        return None


def processar_areas_simuladas(df_inventario, config):
    '''
    Simula áreas realísticas para os talhões

    Args:
        df_inventario: DataFrame do inventário
        config: Configurações do sistema

    Returns:
        DataFrame com áreas simuladas
    '''
    try:
        talhoes_unicos = sorted(df_inventario['talhao'].unique())
        area_parcela_m2 = config.get('area_parcela', 400)

        # Configurações de simulação
        fator_min = config.get('simulacao_fator_min', 2.5)
        fator_max = config.get('simulacao_fator_max', 4.0)
        variacao = config.get('simulacao_variacao', 0.2)

        # Calcular áreas baseadas no número de parcelas
        areas_calculadas = []

        # Definir seed para reprodutibilidade
        np.random.seed(42)

        for talhao in talhoes_unicos:
            # Contar parcelas únicas no talhão
            parcelas_talhao = df_inventario[df_inventario['talhao'] == talhao]['parcela'].nunique()

            # Método 1: Baseado no número de parcelas (mais realístico)
            if parcelas_talhao > 0:
                # Assumir que cada parcela representa uma amostra de fator_min a fator_max hectares
                fator_expansao = np.random.uniform(fator_min, fator_max)
                area_estimada = parcelas_talhao * fator_expansao
            else:
                area_estimada = 25.0  # Área padrão

            # Adicionar variação realística
            multiplicador_variacao = np.random.uniform(1 - variacao, 1 + variacao)
            area_final = area_estimada * multiplicador_variacao

            # Arredondar para valores realísticos
            area_final = round(area_final, 1)

            # Garantir mínimo e máximo realísticos
            area_final = max(5.0, min(area_final, 200.0))

            areas_calculadas.append({
                'talhao': talhao,
                'area_ha': area_final,
                'parcelas': parcelas_talhao,
                'fator_expansao': fator_expansao
            })

        df_areas = pd.DataFrame(areas_calculadas)[['talhao', 'area_ha']]

        st.success(f"✅ Áreas simuladas para {len(df_areas)} talhões")
        st.info("🎲 Simulação baseada no número de parcelas por talhão")

        # Mostrar resumo da simulação
        with st.expander("📊 Resumo da Simulação de Áreas"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Área Total", f"{df_areas['area_ha'].sum():.1f} ha")
            with col2:
                st.metric("Área Média", f"{df_areas['area_ha'].mean():.1f} ha")
            with col3:
                st.metric("Variação", f"{df_areas['area_ha'].min():.1f} - {df_areas['area_ha'].max():.1f} ha")

            # Mostrar detalhes por talhão
            df_detalhes = pd.DataFrame(areas_calculadas)
            df_detalhes_formatado = df_detalhes.round(2)
            st.dataframe(df_detalhes_formatado, hide_index=True)

        return df_areas

    except Exception as e:
        st.error(f"❌ Erro ao simular áreas: {e}")
        return None


def validar_areas_processadas(df_areas, df_inventario):
    '''
    Valida as áreas processadas

    Args:
        df_areas: DataFrame com áreas dos talhões
        df_inventario: DataFrame do inventário

    Returns:
        dict: Resultado da validação
    '''
    validacao = {
        'valido': True,
        'alertas': [],
        'erros': []
    }

    try:
        # Verificar se DataFrame não está vazio
        if df_areas is None or len(df_areas) == 0:
            validacao['erros'].append("Nenhuma área foi processada")
            validacao['valido'] = False
            return validacao

        # Verificar colunas obrigatórias
        if not all(col in df_areas.columns for col in ['talhao', 'area_ha']):
            validacao['erros'].append("Colunas obrigatórias faltantes: talhao, area_ha")
            validacao['valido'] = False
            return validacao

        # Verificar valores válidos
        areas_invalidas = df_areas[df_areas['area_ha'] <= 0]
        if len(areas_invalidas) > 0:
            validacao['erros'].append(f"Áreas inválidas (≤0) encontradas: {len(areas_invalidas)} talhões")
            validacao['valido'] = False

        # Verificar valores extremos
        area_max = df_areas['area_ha'].max()
        area_min = df_areas['area_ha'].min()

        if area_max > 500:
            validacao['alertas'].append(f"Área muito grande detectada: {area_max:.1f} ha")

        if area_min < 1:
            validacao['alertas'].append(f"Área muito pequena detectada: {area_min:.1f} ha")

        # Verificar se todos os talhões do inventário têm área
        talhoes_inventario = set(df_inventario['talhao'].unique())
        talhoes_areas = set(df_areas['talhao'].unique())

        talhoes_sem_area = talhoes_inventario - talhoes_areas
        if talhoes_sem_area:
            validacao['alertas'].append(f"Talhões sem área definida: {list(talhoes_sem_area)}")

        talhoes_sem_inventario = talhoes_areas - talhoes_inventario
        if talhoes_sem_inventario:
            validacao['alertas'].append(f"Áreas sem dados de inventário: {list(talhoes_sem_inventario)}")

        # Verificar distribuição das áreas
        cv_areas = (df_areas['area_ha'].std() / df_areas['area_ha'].mean()) * 100
        if cv_areas > 100:
            validacao['alertas'].append(f"Alta variabilidade nas áreas: CV = {cv_areas:.1f}%")

        # Verificar duplicatas
        duplicatas = df_areas[df_areas.duplicated(subset=['talhao'])]
        if len(duplicatas) > 0:
            validacao['erros'].append(f"Talhões duplicados encontrados: {duplicatas['talhao'].tolist()}")
            validacao['valido'] = False

        # Verificar valores nulos
        nulos_talhao = df_areas['talhao'].isna().sum()
        nulos_area = df_areas['area_ha'].isna().sum()

        if nulos_talhao > 0:
            validacao['erros'].append(f"Valores nulos em 'talhao': {nulos_talhao}")
            validacao['valido'] = False

        if nulos_area > 0:
            validacao['erros'].append(f"Valores nulos em 'area_ha': {nulos_area}")
            validacao['valido'] = False

    except Exception as e:
        validacao['erros'].append(f"Erro na validação: {e}")
        validacao['valido'] = False

    return validacao


def gerar_resumo_areas(df_areas, metodo_utilizado):
    '''
    Gera resumo das áreas processadas

    Args:
        df_areas: DataFrame com áreas dos talhões
        metodo_utilizado: Método usado para calcular as áreas

    Returns:
        dict: Resumo das áreas
    '''
    resumo = {
        'metodo': metodo_utilizado,
        'total_talhoes': len(df_areas),
        'area_total_ha': df_areas['area_ha'].sum(),
        'area_media_ha': df_areas['area_ha'].mean(),
        'area_min_ha': df_areas['area_ha'].min(),
        'area_max_ha': df_areas['area_ha'].max(),
        'cv_areas': (df_areas['area_ha'].std() / df_areas['area_ha'].mean()) * 100,
        'mediana_ha': df_areas['area_ha'].median()
    }

    # Classificação dos talhões por tamanho
    q33 = df_areas['area_ha'].quantile(0.33)
    q67 = df_areas['area_ha'].quantile(0.67)

    resumo['pequenos'] = (df_areas['area_ha'] < q33).sum()
    resumo['medios'] = ((df_areas['area_ha'] >= q33) & (df_areas['area_ha'] < q67)).sum()
    resumo['grandes'] = (df_areas['area_ha'] >= q67).sum()
    resumo['q33'] = q33
    resumo['q67'] = q67

    # Estatísticas adicionais
    resumo['desvio_padrao'] = df_areas['area_ha'].std()
    resumo['amplitude'] = resumo['area_max_ha'] - resumo['area_min_ha']

    return resumo


def exportar_areas_para_csv(df_areas, metodo):
    '''
    Prepara DataFrame de áreas para exportação

    Args:
        df_areas: DataFrame com áreas
        metodo: Método utilizado

    Returns:
        str: CSV formatado para download
    '''
    df_export = df_areas.copy()

    # Adicionar informações adicionais
    df_export['metodo_calculo'] = metodo
    df_export['data_processamento'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

    # Adicionar estatísticas por talhão
    df_export['percentil_area'] = df_export['area_ha'].rank(pct=True) * 100
    df_export['classificacao_tamanho'] = pd.cut(
        df_export['area_ha'],
        bins=3,
        labels=['Pequeno', 'Médio', 'Grande']
    )

    # Reordenar colunas
    colunas_ordenadas = [
        'talhao', 'area_ha', 'percentil_area', 'classificacao_tamanho',
        'metodo_calculo', 'data_processamento'
    ]
    df_export = df_export[colunas_ordenadas]

    # Renomear colunas para português
    df_export = df_export.rename(columns={
        'talhao': 'Talhão',
        'area_ha': 'Área (ha)',
        'percentil_area': 'Percentil Área (%)',
        'classificacao_tamanho': 'Classificação Tamanho',
        'metodo_calculo': 'Método de Cálculo',
        'data_processamento': 'Data Processamento'
    })

    return df_export.to_csv(index=False)


def criar_interface_areas_configuracao():
    '''
    Cria interface para configuração de áreas (para usar em ui/configuracoes.py)

    Returns:
        dict: Configurações de área selecionadas
    '''
    st.subheader("📏 Configurações de Área")

    col1, col2, col3 = st.columns(3)

    with col1:
        metodo_area = st.selectbox(
            "🗺️ Método para Área dos Talhões",
            [
                "Simular automaticamente",
                "Valores informados manualmente",
                "Upload shapefile",
                "Coordenadas das parcelas"
            ],
            help="Como calcular as áreas dos talhões"
        )

    with col2:
        area_parcela = st.number_input(
            "📐 Área da Parcela (m²)",
            min_value=100,
            max_value=2000,
            value=400,
            step=100,
            help="Área padrão: 400m² (20x20m)"
        )

    with col3:
        if metodo_area == "Coordenadas das parcelas":
            raio_parcela = st.number_input(
                "📐 Raio da Parcela (m)",
                min_value=5.0,
                max_value=30.0,
                value=11.28,
                step=0.1,
                help="Raio para calcular área circular"
            )
            area_calculada = np.pi * (raio_parcela ** 2)
            st.write(f"**Área**: {area_calculada:.0f} m²")
        else:
            raio_parcela = 11.28  # Valor padrão

    # Configurações específicas por método
    config_especifica = {}

    if metodo_area == "Valores informados manualmente":
        st.write("**📝 Informe as áreas por talhão:**")
        st.info("💡 As áreas serão coletadas na interface principal")
        config_especifica['tipo'] = 'manual'

    elif metodo_area == "Upload shapefile":
        st.success("📁 Shapefile será processado automaticamente")
        config_especifica['tipo'] = 'shapefile'

    elif metodo_area == "Coordenadas das parcelas":
        st.success("📍 Coordenadas serão processadas automaticamente")
        config_especifica['tipo'] = 'coordenadas'
        config_especifica['raio_parcela'] = raio_parcela

    else:
        st.info("🎲 Áreas serão simuladas baseadas no número de parcelas")
        config_especifica['tipo'] = 'simulacao'

    return {
        'metodo_area': metodo_area,
        'area_parcela': area_parcela,
        'raio_parcela': raio_parcela,
        'config_especifica': config_especifica
    }


def mostrar_estatisticas_areas(df_areas):
    '''
    Mostra estatísticas das áreas processadas

    Args:
        df_areas: DataFrame com áreas dos talhões
    '''
    st.subheader("📊 Estatísticas das Áreas")

    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total de Talhões", len(df_areas))

    with col2:
        st.metric("Área Total", f"{df_areas['area_ha'].sum():.1f} ha")

    with col3:
        st.metric("Área Média", f"{df_areas['area_ha'].mean():.1f} ha")

    with col4:
        cv = (df_areas['area_ha'].std() / df_areas['area_ha'].mean()) * 100
        st.metric("CV", f"{cv:.1f}%")

    # Distribuição
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Distribuição por Tamanho:**")
        q33 = df_areas['area_ha'].quantile(0.33)
        q67 = df_areas['area_ha'].quantile(0.67)

        pequenos = (df_areas['area_ha'] < q33).sum()
        medios = ((df_areas['area_ha'] >= q33) & (df_areas['area_ha'] < q67)).sum()
        grandes = (df_areas['area_ha'] >= q67).sum()

        st.write(f"- Pequenos (< {q33:.1f} ha): {pequenos} talhões")
        st.write(f"- Médios ({q33:.1f} - {q67:.1f} ha): {medios} talhões")
        st.write(f"- Grandes (≥ {q67:.1f} ha): {grandes} talhões")

    with col2:
        st.write("**Estatísticas Descritivas:**")
        st.write(f"- Mínimo: {df_areas['area_ha'].min():.1f} ha")
        st.write(f"- Máximo: {df_areas['area_ha'].max():.1f} ha")
        st.write(f"- Mediana: {df_areas['area_ha'].median():.1f} ha")
        st.write(f"- Desvio padrão: {df_areas['area_ha'].std():.1f} ha")


def gerar_relatorio_areas(df_areas, metodo, validacao=None):
    '''
    Gera relatório das áreas processadas

    Args:
        df_areas: DataFrame com áreas
        metodo: Método utilizado
        validacao: Resultado da validação (opcional)

    Returns:
        str: Relatório em markdown
    '''
    resumo = gerar_resumo_areas(df_areas, metodo)

    relatorio = f'''
                # RELATÓRIO DE ÁREAS DOS TALHÕES
                
                ## Informações Gerais
                - **Data/Hora**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
                - **Método utilizado**: {metodo}
                - **Total de talhões**: {resumo['total_talhoes']}
                
                ## Estatísticas das Áreas
                - **Área total**: {resumo['area_total_ha']:.1f} ha
                - **Área média**: {resumo['area_media_ha']:.1f} ha
                - **Área mínima**: {resumo['area_min_ha']:.1f} ha
                - **Área máxima**: {resumo['area_max_ha']:.1f} ha
                - **Mediana**: {resumo['mediana_ha']:.1f} ha
                - **Desvio padrão**: {resumo['desvio_padrao']:.1f} ha
                - **Coeficiente de variação**: {resumo['cv_areas']:.1f}%
                
                ## Distribuição por Tamanho
                - **Talhões pequenos** (< {resumo['q33']:.1f} ha): {resumo['pequenos']} talhões
                - **Talhões médios** ({resumo['q33']:.1f} - {resumo['q67']:.1f} ha): {resumo['medios']} talhões  
                - **Talhões grandes** (≥ {resumo['q67']:.1f} ha): {resumo['grandes']} talhões
                
                ## Método Específico
                '''

    if metodo == "Simular automaticamente":
        relatorio += '''
                ### Simulação Automática
                - Baseada no número de parcelas por talhão
                - Fator de expansão: 2,5 a 4,0 ha por parcela
                - Variação aleatória aplicada: ±20%
                - Seed fixo para reprodutibilidade
                '''
    elif metodo == "Upload shapefile":
        relatorio += '''
                    ### Shapefile
                    - Áreas extraídas da geometria dos polígonos
                    - Agrupamento automático por talhão
                    - Conversão para hectares quando necessário
                    '''
    elif metodo == "Coordenadas das parcelas":
        relatorio += '''
                    ### Coordenadas das Parcelas
                    - Cálculo baseado em parcelas circulares
                    - Contagem de parcelas por talhão
                    - Área expandida conforme raio definido
                    '''
    elif metodo == "Valores informados manualmente":
        relatorio += '''
                    ### Entrada Manual
                    - Áreas informadas diretamente pelo usuário
                    - Valores validados antes do processamento
                    '''

    # Adicionar validação se disponível
    if validacao:
        relatorio += f'''
                        ## Validação
                        - **Status**: {'✅ Válido' if validacao['valido'] else '❌ Inválido'}
                        '''
        if validacao['alertas']:
            relatorio += "### Alertas:\n"
            for alerta in validacao['alertas']:
                relatorio += f"- ⚠️ {alerta}\n"

        if validacao['erros']:
            relatorio += "### Erros:\n"
            for erro in validacao['erros']:
                relatorio += f"- ❌ {erro}\n"

    relatorio += '''
                ---
                *Relatório gerado pelo Sistema Modular de Inventário Florestal*
                '''

    return relatorio

    if len(df_areas) == 0:
        raise Exception("Nenhuma área válida encontrada no shapefile")

    return df_areas




def processar_coordenadas_areas(arquivo_coord, raio_parcela):
    '''
    Processa coordenadas para calcular áreas dos talhões

    Args:
        arquivo_coord: Arquivo com coordenadas das parcelas
        raio_parcela: Raio da parcela em metros

    Returns:'
        DataFrame com talhao e area_ha
    '''
    try:
        from utils.arquivo_handler import carregar_arquivo

        # Carregar arquivo de coordenadas
        df_coord = carregar_arquivo(arquivo_coord)
        if df_coord is None:
            return None

        # Verificar colunas de coordenadas
        colunas_coord = None
        coord_possiveis = [['x', 'y'], ['X', 'Y'], ['lon', 'lat'], ['longitude', 'latitude']]

        for coord_set in coord_possiveis:
            if all(col in df_coord.columns for col in coord_set):
                colunas_coord = coord_set
                break

        if not colunas_coord:
            st.error("❌ Colunas de coordenadas não encontradas. Esperadas: X,Y ou lon,lat")
            return None

        # Verificar coluna de talhão
        colunas_talhao_possiveis = ['talhao', 'talhão', 'talh', 'plot', 'stand']
        col_talhao = None

        for col in df_coord.columns:
            if col.lower() in [c.lower() for c in colunas_talhao_possiveis]:
                col_talhao = col
                break

        if col_talhao is None:
            st.error("❌ Coluna de talhão não encontrada nas coordenadas")
            return None

        # Calcular área circular da parcela
        area_parcela_ha = np.pi * (raio_parcela ** 2) / 10000  # Converter m² para ha

        # Contar parcelas por talhão
        parcelas_por_talhao = df_coord.groupby(col_talhao).size().reset_index()
        parcelas_por_talhao.columns = ['talhao', 'num_parcelas']

        # Calcular área total do talhão
        parcelas_por_talhao['area_ha'] = parcelas_por_talhao['num_parcelas'] * area_parcela_ha

        # Preparar DataFrame final
        df_areas = parcelas_por_talhao[['talhao', 'area_ha']].copy()
        df_areas['talhao'] = df_areas['talhao'].astype(int)

        st.success(f"✅ Áreas calculadas das coordenadas: {len(df_areas)} talhões")
        st.info(f"📐 Área por parcela: {area_parcela_ha:.4f} ha (raio {raio_parcela}m)")

        return df_areas

    except Exception as e:
        st.error(f"❌ Erro ao processar coordenadas: {e}")
        return None


def processar_areas_simuladas(df_inventario, config):
    '''
    Simula áreas realísticas para os talhões

    Args:
        df_inventario: DataFrame do inventário
        config: Configurações do sistema

    Returns:
        DataFrame com áreas simuladas
    '''
    try:
        talhoes_unicos = sorted(df_inventario['talhao'].unique())
        area_parcela_m2 = config.get('area_parcela', 400)

        # Calcular áreas baseadas no número de parcelas
        areas_calculadas = []

        for talhao in talhoes_unicos:
            # Contar parcelas únicas no talhão
            parcelas_talhao = df_inventario[df_inventario['talhao'] == talhao]['parcela'].nunique()

            # Método 1: Baseado no número de parcelas (mais realístico)
            if parcelas_talhao > 0:
                # Assumir que cada parcela representa uma amostra de 2-5 hectares
                fator_expansao = np.random.uniform(2.0, 5.0)
                area_estimada = parcelas_talhao * fator_expansao
            else:
                area_estimada = 25.0  # Área padrão

            # Adicionar variação realística (±20%)
            variacao = np.random.uniform(0.8, 1.2)
            area_final = area_estimada * variacao

            # Arredondar para valores realísticos
            area_final = round(area_final, 1)

            # Garantir mínimo e máximo realísticos
            area_final = max(5.0, min(area_final, 200.0))

            areas_calculadas.append({
                'talhao': talhao,
                'area_ha': area_final,
                'parcelas': parcelas_talhao
            })

        df_areas = pd.DataFrame(areas_calculadas)[['talhao', 'area_ha']]

        st.success(f"✅ Áreas simuladas para {len(df_areas)} talhões")
        st.info("🎲 Simulação baseada no número de parcelas por talhão")

        # Mostrar resumo da simulação
        with st.expander("📊 Resumo da Simulação de Áreas"):
            st.write(f"**Área total**: {df_areas['area_ha'].sum():.1f} ha")
            st.write(f"**Área média por talhão**: {df_areas['area_ha'].mean():.1f} ha")
            st.write(f"**Variação**: {df_areas['area_ha'].min():.1f} - {df_areas['area_ha'].max():.1f} ha")

        return df_areas

    except Exception as e:
        st.error(f"❌ Erro ao simular áreas: {e}")
        return None


def validar_areas_processadas(df_areas, df_inventario):
    '''
    Valida as áreas processadas

    Args:
        df_areas: DataFrame com áreas dos talhões
        df_inventario: DataFrame do inventário

    Returns:
        dict: Resultado da validação
    '''
    validacao = {
        'valido': True,
        'alertas': [],
        'erros': []
    }

    try:
        # Verificar se DataFrame não está vazio
        if df_areas is None or len(df_areas) == 0:
            validacao['erros'].append("Nenhuma área foi processada")
            validacao['valido'] = False
            return validacao

        # Verificar colunas obrigatórias
        if not all(col in df_areas.columns for col in ['talhao', 'area_ha']):
            validacao['erros'].append("Colunas obrigatórias faltantes: talhao, area_ha")
            validacao['valido'] = False
            return validacao

        # Verificar valores válidos
        areas_invalidas = df_areas[df_areas['area_ha'] <= 0]
        if len(areas_invalidas) > 0:
            validacao['erros'].append(f"Áreas inválidas (≤0) encontradas: {len(areas_invalidas)} talhões")
            validacao['valido'] = False

        # Verificar valores extremos
        area_max = df_areas['area_ha'].max()
        area_min = df_areas['area_ha'].min()

        if area_max > 500:
            validacao['alertas'].append(f"Área muito grande detectada: {area_max:.1f} ha")

        if area_min < 1:
            validacao['alertas'].append(f"Área muito pequena detectada: {area_min:.1f} ha")

        # Verificar se todos os talhões do inventário têm área
        talhoes_inventario = set(df_inventario['talhao'].unique())
        talhoes_areas = set(df_areas['talhao'].unique())

        talhoes_sem_area = talhoes_inventario - talhoes_areas
        if talhoes_sem_area:
            validacao['alertas'].append(f"Talhões sem área definida: {list(talhoes_sem_area)}")

        talhoes_sem_inventario = talhoes_areas - talhoes_inventario
        if talhoes_sem_inventario:
            validacao['alertas'].append(f"Áreas sem dados de inventário: {list(talhoes_sem_inventario)}")

        # Verificar distribuição das áreas
        cv_areas = (df_areas['area_ha'].std() / df_areas['area_ha'].mean()) * 100
        if cv_areas > 100:
            validacao['alertas'].append(f"Alta variabilidade nas áreas: CV = {cv_areas:.1f}%")

    except Exception as e:
        validacao['erros'].append(f"Erro na validação: {e}")
        validacao['valido'] = False

    return validacao


def gerar_resumo_areas(df_areas, metodo_utilizado):
    '''
    Gera resumo das áreas processadas

    Args:
        df_areas: DataFrame com áreas dos talhões
        metodo_utilizado: Método usado para calcular as áreas

    Returns:
        dict: Resumo das áreas
    '''
    resumo = {
        'metodo': metodo_utilizado,
        'total_talhoes': len(df_areas),
        'area_total_ha': df_areas['area_ha'].sum(),
        'area_media_ha': df_areas['area_ha'].mean(),
        'area_min_ha': df_areas['area_ha'].min(),
        'area_max_ha': df_areas['area_ha'].max(),
        'cv_areas': (df_areas['area_ha'].std() / df_areas['area_ha'].mean()) * 100
    }

    # Classificação dos talhões por tamanho
    q33 = df_areas['area_ha'].quantile(0.33)
    q67 = df_areas['area_ha'].quantile(0.67)

    resumo['pequenos'] = (df_areas['area_ha'] < q33).sum()
    resumo['medios'] = ((df_areas['area_ha'] >= q33) & (df_areas['area_ha'] < q67)).sum()
    resumo['grandes'] = (df_areas['area_ha'] >= q67).sum()
    resumo['q33'] = q33
    resumo['q67'] = q67

    return resumo


def exportar_areas_para_csv(df_areas, metodo):
    '''
    Prepara DataFrame de áreas para exportação

    Args:
        df_areas: DataFrame com áreas
        metodo: Método utilizado

    Returns:
        str: CSV formatado para download
    '''
    df_export = df_areas.copy()

    # Adicionar informações adicionais
    df_export['metodo_calculo'] = metodo
    df_export['data_processamento'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

    # Reordenar colunas
    df_export = df_export[['talhao', 'area_ha', 'metodo_calculo', 'data_processamento']]

    # Renomear colunas para português
    df_export = df_export.rename(columns={
        'talhao': 'Talhão',
        'area_ha': 'Área (ha)',
        'metodo_calculo': 'Método de Cálculo',
        'data_processamento': 'Data Processamento'
    })

    return df_export.to_csv(index=False)


def criar_interface_areas_configuracao():
    '''
    Cria interface para configuração de áreas (para usar em ui/configuracoes.py)

    Returns:
        dict: Configurações de área selecionadas
    '''
    st.subheader("📏 Configurações de Área")

    col1, col2, col3 = st.columns(3)

    with col1:
        metodo_area = st.selectbox(
            "🗺️ Método para Área dos Talhões",
            [
                "Simular automaticamente",
                "Valores informados manualmente",
                "Upload shapefile",
                "Coordenadas das parcelas"
            ],
            help="Como calcular as áreas dos talhões"
        )

    with col2:
        area_parcela = st.number_input(
            "📐 Área da Parcela (m²)",
            min_value=100,
            max_value=2000,
            value=400,
            step=100,
            help="Área padrão: 400m² (20x20m)"
        )

    with col3:
        if metodo_area == "Coordenadas das parcelas":
            raio_parcela = st.number_input(
                "📐 Raio da Parcela (m)",
                min_value=5.0,
                max_value=30.0,
                value=11.28,
                step=0.1,
                help="Raio para calcular área circular"
            )
            area_calculada = np.pi * (raio_parcela ** 2)
            st.write(f"**Área**: {area_calculada:.0f} m²")
        else:
            raio_parcela = 11.28  # Valor padrão

    # Configurações específicas por método
    config_especifica = {}

    if metodo_area == "Valores informados manualmente":
        st.write("**📝 Informe as áreas por talhão:**")
        st.info("💡 As áreas serão coletadas na interface principal")
        config_especifica['tipo'] = 'manual'

    elif metodo_area == "Upload shapefile":
        st.success("📁 Shapefile será processado automaticamente")
        config_especifica['tipo'] = 'shapefile'

    elif metodo_area == "Coordenadas das parcelas":
        st.success("📍 Coordenadas serão processadas automaticamente")
        config_especifica['tipo'] = 'coordenadas'
        config_especifica['raio_parcela'] = raio_parcela

    else:
        st.info("🎲 Áreas serão simuladas baseadas no número de parcelas")
        config_especifica['tipo'] = 'simulacao'

    return {
        'metodo_area': metodo_area,
        'area_parcela': area_parcela,
        'raio_parcela': raio_parcela,
        'config_especifica': config_especifica
    }