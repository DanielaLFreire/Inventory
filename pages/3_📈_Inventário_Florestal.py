# pages/3_📈_Inventário_Florestal.py - VERSÃO MELHORADA COM FORMATAÇÃO BRASILEIRA
"""
Etapa 3: Inventário Florestal
Processamento completo e relatórios finais com métricas detalhadas e formatação brasileira
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import traceback

# Importar funções de formatação brasileira
from utils.formatacao import (
    formatar_brasileiro,
    formatar_dataframe_brasileiro,
    formatar_numero_inteligente
)

st.set_page_config(
    page_title="Inventário Florestal",
    page_icon="📈",
    layout="wide"
)


def verificar_prerequisitos():
    """Verifica se as etapas anteriores foram concluídas"""
    problemas = []

    if not hasattr(st.session_state, 'dados_inventario') or st.session_state.dados_inventario is None:
        problemas.append("Dados de inventário não disponíveis")

    if not st.session_state.get('resultados_hipsometricos'):
        problemas.append("Etapa 1 (Hipsométricos) não concluída")

    if not st.session_state.get('resultados_volumetricos'):
        problemas.append("Etapa 2 (Volumétricos) não concluída")

    if problemas:
        st.error("❌ Pré-requisitos não atendidos:")
        for problema in problemas:
            st.error(f"• {problema}")

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🏠 Página Principal", key="btn_principal_req"):
                st.switch_page("Principal.py")
        with col2:
            if st.button("🌳 Hipsométricos", key="btn_hip_req"):
                st.switch_page("pages/1_🌳_Modelos_Hipsométricos.py")
        with col3:
            if st.button("📊 Volumétricos", key="btn_vol_req"):
                st.switch_page("pages/2_📊_Modelos_Volumétricos.py")

        return False

    return True


def mostrar_status_etapas():
    """Mostra status das etapas anteriores"""
    st.subheader("✅ Status das Etapas Anteriores")

    col1, col2 = st.columns(2)

    with col1:
        melhor_hip = st.session_state.resultados_hipsometricos.get('melhor_modelo', 'N/A')
        r2_hip = st.session_state.resultados_hipsometricos.get('resultados', {}).get(melhor_hip, {}).get('r2g', 0)
        st.success(f"🌳 **Etapa 1 Concluída** - {melhor_hip} (R² = {r2_hip:.3f})")

    with col2:
        melhor_vol = st.session_state.resultados_volumetricos.get('melhor_modelo', 'N/A')
        r2_vol = st.session_state.resultados_volumetricos.get('resultados', {}).get(melhor_vol, {}).get('r2', 0)
        st.success(f"📊 **Etapa 2 Concluída** - {melhor_vol} (R² = {r2_vol:.3f})")


def configurar_areas_talhoes():
    """Configura áreas dos talhões com interface melhorada"""
    st.header("📏 Configuração de Áreas dos Talhões")

    df_inventario = st.session_state.dados_inventario
    talhoes_disponiveis = sorted(df_inventario['talhao'].unique())

    # Método de cálculo das áreas
    metodo_area = st.selectbox(
        "🗺️ Método para Cálculo das Áreas",
        ["Área fixa para todos", "Valores específicos por talhão", "Simulação baseada em parcelas"],
        key="selectbox_metodo_area"
    )

    config_areas = {'metodo': metodo_area}

    if metodo_area == "Valores específicos por talhão":
        st.write("**📝 Informe as áreas por talhão (hectares):**")

        areas_manuais = {}
        n_colunas = min(4, len(talhoes_disponiveis))
        colunas = st.columns(n_colunas)

        for i, talhao in enumerate(talhoes_disponiveis):
            col_idx = i % n_colunas
            with colunas[col_idx]:
                areas_manuais[talhao] = st.number_input(
                    f"Talhão {talhao}",
                    min_value=0.1,
                    max_value=1000.0,
                    value=25.0,
                    step=0.1,
                    key=f"area_talhao_{talhao}"
                )

        config_areas['areas_manuais'] = areas_manuais

        if areas_manuais:
            area_total = sum(areas_manuais.values())
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Área Total", f"{formatar_brasileiro(area_total, 1)} ha")
            with col2:
                st.metric("Área Média", f"{formatar_brasileiro(np.mean(list(areas_manuais.values())), 1)} ha")
            with col3:
                st.metric("Talhões", len(areas_manuais))

    elif metodo_area == "Simulação baseada em parcelas":
        st.info("🎲 **Simulação Inteligente de Áreas**")

        col1, col2 = st.columns(2)
        with col1:
            fator_expansao = st.slider(
                "Fator de expansão (ha por parcela)",
                min_value=1.0,
                max_value=10.0,
                value=3.0,
                step=0.5,
                help="Cada parcela representa quantos hectares"
            )

        with col2:
            variacao_percentual = st.slider(
                "Variação aleatória (%)",
                min_value=0,
                max_value=50,
                value=20,
                step=5,
                help="Variação para simular heterogeneidade"
            )

        config_areas['fator_expansao'] = fator_expansao
        config_areas['variacao'] = variacao_percentual / 100

        # Preview da simulação
        np.random.seed(42)
        areas_simuladas = {}
        for talhao in talhoes_disponiveis:
            parcelas_talhao = df_inventario[df_inventario['talhao'] == talhao]['parcela'].nunique()
            area_base = parcelas_talhao * fator_expansao
            variacao = np.random.uniform(1 - config_areas['variacao'], 1 + config_areas['variacao'])
            areas_simuladas[talhao] = area_base * variacao

        area_total_sim = sum(areas_simuladas.values())
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Área Total (Preview)", f"{formatar_brasileiro(area_total_sim, 1)} ha")
        with col2:
            st.metric("Área Média", f"{formatar_brasileiro(np.mean(list(areas_simuladas.values())), 1)} ha")
        with col3:
            st.metric("Talhões", len(areas_simuladas))

        config_areas['areas_simuladas'] = areas_simuladas

    else:
        # Área fixa para todos
        area_fixa = st.number_input(
            "Área por talhão (hectares)",
            min_value=0.1,
            max_value=1000.0,
            value=25.0,
            step=0.1,
            key="area_fixa_todos"
        )

        config_areas['area_fixa'] = area_fixa
        config_areas['talhoes'] = talhoes_disponiveis

        area_total = area_fixa * len(talhoes_disponiveis)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Área Total", f"{formatar_brasileiro(area_total, 1)} ha")
        with col2:
            st.metric("Área por Talhão", f"{formatar_brasileiro(area_fixa, 1)} ha")
        with col3:
            st.metric("Total de Talhões", len(talhoes_disponiveis))

    return config_areas


def configurar_parametros_avancados():
    """Configura parâmetros avançados do inventário"""
    with st.expander("⚙️ Parâmetros Avançados"):
        col1, col2, col3 = st.columns(3)

        with col1:
            area_parcela = st.number_input(
                "📐 Área da Parcela (m²)",
                min_value=100,
                max_value=2000,
                value=400,
                step=50,
                help="Área padrão: 400m² (20x20m)"
            )

            idade_padrao = st.number_input(
                "📅 Idade Padrão (anos)",
                min_value=1.0,
                max_value=50.0,
                value=5.0,
                step=0.5,
                help="Idade usada quando não informada"
            )

        with col2:
            densidade_plantio = st.number_input(
                "🌱 Densidade de Plantio (árv/ha)",
                min_value=500,
                max_value=5000,
                value=1667,
                step=50,
                help="Densidade inicial de plantio (3x2m = 1667 árv/ha)"
            )

            sobrevivencia = st.slider(
                "🌲 Taxa de Sobrevivência (%)",
                min_value=50,
                max_value=100,
                value=85,
                step=5,
                help="Percentual de árvores que sobreviveram"
            )

        with col3:
            fator_forma = st.number_input(
                "📊 Fator de Forma",
                min_value=0.3,
                max_value=0.8,
                value=0.5,
                step=0.05,
                help="Fator de forma médio (0.5 = típico para eucalipto)"
            )

            densidade_madeira = st.number_input(
                "🌱 Densidade da Madeira (kg/m³)",
                min_value=300,
                max_value=800,
                value=500,
                step=25,
                help="Densidade básica da madeira"
            )

        return {
            'area_parcela': area_parcela,
            'idade_padrao': idade_padrao,
            'densidade_plantio': densidade_plantio,
            'sobrevivencia': sobrevivencia / 100,
            'fator_forma': fator_forma,
            'densidade_madeira': densidade_madeira
        }


def criar_df_areas(config_areas):
    """Cria DataFrame de áreas baseado na configuração"""
    if config_areas['metodo'] == "Valores específicos por talhão":
        areas_dict = config_areas.get('areas_manuais', {})
        df_areas = pd.DataFrame([
            {'talhao': int(talhao), 'area_ha': float(area)}
            for talhao, area in areas_dict.items()
        ])
    elif config_areas['metodo'] == "Simulação baseada em parcelas":
        areas_dict = config_areas.get('areas_simuladas', {})
        df_areas = pd.DataFrame([
            {'talhao': int(talhao), 'area_ha': float(area)}
            for talhao, area in areas_dict.items()
        ])
    else:
        # Área fixa
        area_fixa = config_areas['area_fixa']
        talhoes = config_areas['talhoes']
        df_areas = pd.DataFrame([
            {'talhao': talhao, 'area_ha': area_fixa}
            for talhao in talhoes
        ])

    return df_areas


def estimar_alturas_inventario(df, melhor_modelo):
    """Estima alturas usando o melhor modelo hipsométrico"""
    df = df.copy()

    def estimar_altura_arvore(row):
        try:
            if pd.notna(row.get('H_m')) and row.get('H_m', 0) > 1.3:
                return row['H_m']

            D = row['D_cm']

            if melhor_modelo == "Curtis":
                return np.exp(3.2 - 8.5 / max(D, 1))
            elif melhor_modelo == "Campos":
                h_dom = 25.0
                return np.exp(2.8 - 7.2 / max(D, 1) + 0.7 * np.log(h_dom))
            elif melhor_modelo == "Henri":
                return 1.3 + 8.5 * np.log(max(D, 1))
            elif melhor_modelo == "Prodan":
                prod = 0.8 * D + 0.002 * (D ** 2)
                return max((D ** 2) / max(prod, 0.1) + 1.3, 1.5)
            elif melhor_modelo == "Chapman":
                return 30 * (1 - np.exp(-0.08 * D)) ** 1.2
            elif melhor_modelo == "Weibull":
                return 32 * (1 - np.exp(-0.05 * (D ** 1.1)))
            elif melhor_modelo == "Mononuclear":
                return 28 * (1 - 0.9 * np.exp(-0.12 * D))
            else:
                return 1.3 + 0.8 * D
        except:
            return 1.3 + 0.8 * row['D_cm']

    df['H_est'] = df.apply(estimar_altura_arvore, axis=1)
    df['H_est'] = df['H_est'].clip(lower=1.5)

    return df


def estimar_volumes_inventario(df, melhor_modelo):
    """Estima volumes usando o melhor modelo volumétrico"""
    df = df.copy()

    def estimar_volume_arvore(row):
        try:
            D = row['D_cm']
            H = row['H_est']

            if D <= 0 or H <= 1.3:
                return 0.001

            if melhor_modelo == 'Schumacher':
                return np.exp(-9.5 + 1.8 * np.log(D) + 1.1 * np.log(H))
            elif melhor_modelo == 'G1':
                return np.exp(-8.8 + 2.2 * np.log(D) - 1.2 / max(D, 1))
            elif melhor_modelo == 'G2':
                D2 = D ** 2
                return max(-0.05 + 0.0008 * D2 + 0.000045 * D2 * H + 0.008 * H, 0.001)
            elif melhor_modelo == 'G3':
                D2H = (D ** 2) * H
                return np.exp(-10.2 + 0.95 * np.log(max(D2H, 1)))
            else:
                return 0.0008 * (D ** 2) * H
        except:
            return 0.0008 * (row['D_cm'] ** 2) * row['H_est']

    df['V_est'] = df.apply(estimar_volume_arvore, axis=1)
    df['V_est'] = df['V_est'].clip(lower=0.001)

    return df


def calcular_metricas_adicionais(df, parametros):
    """Calcula métricas florestais adicionais"""
    df = df.copy()

    # Área basal individual (m²)
    df['G_ind'] = np.pi * (df['D_cm'] / 200) ** 2  # /200 para converter cm para m e dividir por 2 para raio

    # Biomassa estimada (usando fator de forma e densidade)
    df['biomassa_kg'] = df['V_est'] * parametros['fator_forma'] * parametros['densidade_madeira']

    # Volume comercial (assumindo 85% do volume total)
    df['V_comercial'] = df['V_est'] * 0.85

    # Classe diamétrica
    df['classe_dap'] = pd.cut(df['D_cm'],
                              bins=[0, 5, 10, 15, 20, 25, 30, 999],
                              labels=['<5cm', '5-10cm', '10-15cm', '15-20cm', '20-25cm', '25-30cm', '>30cm'])

    return df


def calcular_resumo_por_parcela(df, parametros):
    """Calcula resumo detalhado por parcela"""
    area_parcela_m2 = parametros['area_parcela']

    resumo = df.groupby(['talhao', 'parcela']).agg({
        'area_ha': 'first',
        'D_cm': ['mean', 'std', 'min', 'max'],
        'H_est': ['mean', 'std', 'min', 'max'],
        'V_est': 'sum',
        'V_comercial': 'sum',
        'G_ind': 'sum',
        'biomassa_kg': 'sum',
        'cod': 'count'
    }).reset_index()

    # Achatar colunas multi-nível
    resumo.columns = [
        'talhao', 'parcela', 'area_ha',
        'dap_medio', 'dap_desvio', 'dap_min', 'dap_max',
        'altura_media', 'altura_desvio', 'altura_min', 'altura_max',
        'volume_parcela', 'volume_comercial_parcela', 'area_basal_parcela', 'biomassa_parcela',
        'n_arvores'
    ]

    # Calcular métricas por hectare
    resumo['vol_ha'] = resumo['volume_parcela'] * (10000 / area_parcela_m2)
    resumo['vol_comercial_ha'] = resumo['volume_comercial_parcela'] * (10000 / area_parcela_m2)
    resumo['area_basal_ha'] = resumo['area_basal_parcela'] * (10000 / area_parcela_m2)
    resumo['biomassa_ha'] = resumo['biomassa_parcela'] * (10000 / area_parcela_m2)
    resumo['densidade_ha'] = resumo['n_arvores'] * (10000 / area_parcela_m2)

    # Calcular idade (se disponível)
    if 'idade_anos' in df.columns:
        idade_por_parcela = df.groupby(['talhao', 'parcela'])['idade_anos'].mean()
        resumo = resumo.merge(idade_por_parcela.reset_index(), on=['talhao', 'parcela'], how='left')
        resumo['idade_anos'] = resumo['idade_anos'].fillna(parametros['idade_padrao'])
    else:
        resumo['idade_anos'] = parametros['idade_padrao']

    # Calcular IMA e outras métricas temporais
    resumo['ima_vol'] = resumo['vol_ha'] / resumo['idade_anos']
    resumo['ima_area_basal'] = resumo['area_basal_ha'] / resumo['idade_anos']
    resumo['ima_biomassa'] = resumo['biomassa_ha'] / resumo['idade_anos']

    # Índices de sítio e qualidade
    resumo['indice_sitio'] = resumo['altura_media'] / resumo['idade_anos']  # Simplificado
    resumo['mortalidade_estimada'] = (1 - resumo['densidade_ha'] / parametros['densidade_plantio']) * 100

    # Classificação de produtividade
    q75_vol = resumo['vol_ha'].quantile(0.75)
    q25_vol = resumo['vol_ha'].quantile(0.25)

    def classificar_produtividade(vol):
        if vol >= q75_vol:
            return "Alta"
        elif vol >= q25_vol:
            return "Média"
        else:
            return "Baixa"

    resumo['classe_produtividade'] = resumo['vol_ha'].apply(classificar_produtividade)

    return resumo


def calcular_resumo_por_talhao(resumo_parcelas):
    """Calcula resumo detalhado por talhão"""

    # Verificar quais colunas existem para evitar erros
    colunas_disponiveis = resumo_parcelas.columns.tolist()

    # Configurar agregações baseado nas colunas disponíveis
    agg_dict = {
        'area_ha': 'first',
        'vol_ha': ['mean', 'std', 'min', 'max'],
        'dap_medio': 'mean',
        'altura_media': 'mean',
        'idade_anos': 'mean',
        'n_arvores': 'mean'  # Usar n_arvores em vez de cod
    }

    # Adicionar colunas opcionais se existirem
    colunas_opcionais = [
        'vol_comercial_ha', 'area_basal_ha', 'biomassa_ha', 'densidade_ha',
        'ima_vol', 'ima_area_basal', 'ima_biomassa', 'indice_sitio', 'mortalidade_estimada'
    ]

    for col in colunas_opcionais:
        if col in colunas_disponiveis:
            agg_dict[col] = 'mean'

    # Contar parcelas por talhão separadamente
    n_parcelas = resumo_parcelas.groupby('talhao').size().reset_index(name='n_parcelas')

    resumo_talhao = resumo_parcelas.groupby('talhao').agg(agg_dict).round(2)

    # Achatar colunas multi-nível dinamicamente
    new_columns = []
    for col in resumo_talhao.columns:
        if isinstance(col, tuple):
            if col[1] == 'first' or col[1] == 'mean':
                new_columns.append(col[0])
            else:
                new_columns.append(f"{col[0]}_{col[1]}")
        else:
            new_columns.append(col)

    resumo_talhao.columns = new_columns
    resumo_talhao = resumo_talhao.reset_index()

    # Merge com contagem de parcelas
    resumo_talhao = resumo_talhao.merge(n_parcelas, on='talhao', how='left')

    # Calcular estoques totais por talhão (apenas se colunas existirem)
    if 'vol_ha' in resumo_talhao.columns:
        resumo_talhao['estoque_total_m3'] = resumo_talhao['area_ha'] * resumo_talhao['vol_ha']

    if 'vol_comercial_ha' in resumo_talhao.columns:
        resumo_talhao['estoque_comercial_m3'] = resumo_talhao['area_ha'] * resumo_talhao['vol_comercial_ha']

    if 'biomassa_ha' in resumo_talhao.columns:
        resumo_talhao['biomassa_total_ton'] = resumo_talhao['area_ha'] * resumo_talhao['biomassa_ha'] / 1000

    if 'area_basal_ha' in resumo_talhao.columns:
        resumo_talhao['area_basal_total_m2'] = resumo_talhao['area_ha'] * resumo_talhao['area_basal_ha']

    # Calcular CV de produtividade (apenas se colunas existirem)
    if 'vol_ha_std' in resumo_talhao.columns and 'vol_ha' in resumo_talhao.columns:
        resumo_talhao['cv_volume'] = (resumo_talhao['vol_ha_std'] / resumo_talhao['vol_ha']) * 100

    # Classificação dos talhões
    def classificar_talhao(row):
        vol_ha = row.get('vol_ha', 0)
        ima_vol = row.get('ima_vol', 0)

        if vol_ha >= 150 and ima_vol >= 25:
            return "Excelente"
        elif vol_ha >= 120 and ima_vol >= 20:
            return "Muito Bom"
        elif vol_ha >= 100 and ima_vol >= 15:
            return "Bom"
        elif vol_ha >= 80 and ima_vol >= 12:
            return "Regular"
        else:
            return "Baixo"

    resumo_talhao['classificacao_geral'] = resumo_talhao.apply(classificar_talhao, axis=1)

    return resumo_talhao


def calcular_estatisticas_gerais(resumo_parcelas, resumo_talhoes):
    """Calcula estatísticas gerais detalhadas do inventário"""
    stats = {
        'total_parcelas': len(resumo_parcelas),
        'total_talhoes': resumo_parcelas['talhao'].nunique(),
        'area_total_ha': resumo_talhoes['area_ha'].sum(),

        # Métricas de volume
        'vol_medio_ha': resumo_parcelas['vol_ha'].mean(),
        'vol_min_ha': resumo_parcelas['vol_ha'].min(),
        'vol_max_ha': resumo_parcelas['vol_ha'].max(),
        'cv_volume': (resumo_parcelas['vol_ha'].std() / resumo_parcelas['vol_ha'].mean()) * 100,
        'estoque_total_m3': resumo_talhoes['estoque_total_m3'].sum(),

        # Métricas volumétricas comerciais
        'vol_comercial_medio_ha': resumo_parcelas['vol_comercial_ha'].mean(),
        'estoque_comercial_total_m3': resumo_talhoes['estoque_comercial_m3'].sum(),

        # Métricas dendrométricas
        'dap_medio': resumo_parcelas['dap_medio'].mean(),
        'dap_min': resumo_parcelas['dap_min'].min(),
        'dap_max': resumo_parcelas['dap_max'].max(),
        'altura_media': resumo_parcelas['altura_media'].mean(),
        'altura_min': resumo_parcelas['altura_min'].min(),
        'altura_max': resumo_parcelas['altura_max'].max(),

        # Métricas de crescimento
        'idade_media': resumo_parcelas['idade_anos'].mean(),
        'ima_vol_medio': resumo_parcelas['ima_vol'].mean(),
        'ima_area_basal_medio': resumo_parcelas['ima_area_basal'].mean(),
        'ima_biomassa_medio': resumo_parcelas['ima_biomassa'].mean(),

        # Métricas de densidade e estrutura
        'densidade_media_ha': resumo_parcelas['densidade_ha'].mean(),
        'area_basal_media_ha': resumo_parcelas['area_basal_ha'].mean(),
        'mortalidade_media': resumo_parcelas['mortalidade_estimada'].mean(),

        # Métricas ambientais
        'biomassa_total_ton': resumo_talhoes['biomassa_total_ton'].sum(),
        'carbono_estimado_ton': resumo_talhoes['biomassa_total_ton'].sum() * 0.47,  # 47% da biomassa é carbono

        # Índices de qualidade
        'indice_sitio_medio': resumo_parcelas['indice_sitio'].mean(),
        'arvores_por_parcela': resumo_parcelas['n_arvores'].mean()
    }

    # Classificação de produtividade
    q25 = resumo_parcelas['vol_ha'].quantile(0.25)
    q75 = resumo_parcelas['vol_ha'].quantile(0.75)

    stats['classe_alta'] = (resumo_parcelas['vol_ha'] >= q75).sum()
    stats['classe_media'] = ((resumo_parcelas['vol_ha'] >= q25) & (resumo_parcelas['vol_ha'] < q75)).sum()
    stats['classe_baixa'] = (resumo_parcelas['vol_ha'] < q25).sum()
    stats['q25_volume'] = q25
    stats['q75_volume'] = q75

    # Classificação de IMA
    ima_excelente = (resumo_parcelas['ima_vol'] >= 25).sum()
    ima_bom = ((resumo_parcelas['ima_vol'] >= 15) & (resumo_parcelas['ima_vol'] < 25)).sum()
    ima_regular = (resumo_parcelas['ima_vol'] < 15).sum()

    stats['ima_excelente'] = ima_excelente
    stats['ima_bom'] = ima_bom
    stats['ima_regular'] = ima_regular

    # Potencial de colheita (assumindo ciclo de 7 anos)
    anos_restantes = max(0, 7 - stats['idade_media'])
    volume_final_estimado = stats['vol_medio_ha'] + (stats['ima_vol_medio'] * anos_restantes)
    stats['volume_final_estimado_ha'] = volume_final_estimado
    stats['potencial_colheita_m3'] = stats['area_total_ha'] * volume_final_estimado

    return stats


def executar_inventario_completo(config_areas, parametros):
    """Executa o inventário completo"""
    st.header("🚀 Executando Inventário Completo")

    # Barra de progresso
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.text("Processando áreas dos talhões...")
        progress_bar.progress(0.1)

        # Criar DataFrame de áreas
        df_areas = criar_df_areas(config_areas)
        #st.success(f"✅ Áreas processadas: {len(df_areas)} talhões")

        status_text.text("Preparando dados do inventário...")
        progress_bar.progress(0.2)

        # Obter modelos selecionados
        melhor_hip = st.session_state.resultados_hipsometricos['melhor_modelo']
        melhor_vol = st.session_state.resultados_volumetricos['melhor_modelo']

        # Filtrar dados do inventário
        df_inventario = st.session_state.dados_inventario.copy()
        df_filtrado = df_inventario[
            (df_inventario['D_cm'].notna()) &
            (df_inventario['D_cm'] > 0) &
            (df_inventario['D_cm'] >= 4.0)
            ]

        # Adicionar áreas aos dados
        df_com_areas = df_filtrado.merge(df_areas, on='talhao', how='left')
        df_com_areas['area_ha'] = df_com_areas['area_ha'].fillna(25.0)

        status_text.text("Aplicando modelos hipsométricos...")
        progress_bar.progress(0.4)

        # Estimar alturas
        df_com_alturas = estimar_alturas_inventario(df_com_areas, melhor_hip)

        status_text.text("Aplicando modelos volumétricos...")
        progress_bar.progress(0.6)

        # Estimar volumes
        df_com_volumes = estimar_volumes_inventario(df_com_alturas, melhor_vol)

        status_text.text("Calculando métricas adicionais...")
        progress_bar.progress(0.7)

        # Calcular métricas adicionais
        df_completo = calcular_metricas_adicionais(df_com_volumes, parametros)

        status_text.text("Calculando estatísticas finais...")
        progress_bar.progress(0.9)

        # Calcular resumos
        resumo_parcelas = calcular_resumo_por_parcela(df_completo, parametros)
        resumo_talhoes = calcular_resumo_por_talhao(resumo_parcelas)
        estatisticas_gerais = calcular_estatisticas_gerais(resumo_parcelas, resumo_talhoes)

        progress_bar.progress(1.0)
        status_text.text("✅ Inventário processado com sucesso!")

        # Preparar resultados finais
        resultados = {
            'inventario_completo': df_completo,
            'resumo_parcelas': resumo_parcelas,
            'resumo_talhoes': resumo_talhoes,
            'estatisticas_gerais': estatisticas_gerais,
            'modelos_utilizados': {
                'hipsometrico': melhor_hip,
                'volumetrico': melhor_vol
            },
            'parametros_utilizados': parametros
        }

        # Salvar no session_state
        st.session_state.inventario_processado = resultados

        #st.success(f"🏆 Inventário processado com sucesso!")
        #st.info(f"📊 Modelos utilizados: {melhor_hip} (Hipsométrico) + {melhor_vol} (Volumétrico)")

        # Mostrar resultados
        mostrar_resultados_inventario(resultados)

    except Exception as e:
        st.error(f"❌ Erro no processamento do inventário: {e}")
        st.info("💡 Verifique os dados e configurações")
        with st.expander("🔍 Detalhes do erro"):
            st.code(traceback.format_exc())


def mostrar_resultados_inventario(resultados):
    """Mostra os resultados finais do inventário"""
    st.header("📊 Resultados Finais do Inventário")

    stats = resultados['estatisticas_gerais']

    # Métricas principais melhoradas com tooltips
    st.subheader("📈 Indicadores Principais")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("🌲 Parcelas", f"{stats['total_parcelas']:,}".replace(',', '.'),
                  help="**Total de Parcelas** - Número de unidades amostrais medidas no inventário florestal")
    with col2:
        st.metric("📏 Área Total", f"{formatar_brasileiro(stats['area_total_ha'], 1)} ha",
                  help="**Área Total** - Superfície total da floresta inventariada em hectares")
    with col3:
        st.metric("📊 Produtividade", f"{formatar_brasileiro(stats['vol_medio_ha'], 1)} m³/ha",
                  help="**Volume por Hectare** - Volume médio de madeira por unidade de área")
    with col4:
        st.metric("🌲 Estoque Total", formatar_numero_inteligente(stats['estoque_total_m3'], "m³"),
                  help="**Estoque Total** - Volume total de madeira em toda a floresta (Produtividade × Área Total)")
    with col5:
        ima_col1, ima_col2 = st.columns([3, 1])
        with ima_col1:
            st.metric("🚀 IMA Médio", f"{formatar_brasileiro(stats['ima_vol_medio'], 1)} m³/ha/ano",
                      help="**Incremento Médio Anual** - Crescimento médio anual em volume por hectare (Volume ÷ Idade)")
        with ima_col2:
            # Widget de ajuda para explicar o IMA
            with st.popover("ℹ️"):
                st.markdown("""
                **📈 Incremento Médio Anual (IMA)**

                Medida usada para indicar o crescimento médio anual em volume por hectare.

                **🧮 Fórmula:**
                ```
                IMA = Volume (m³/ha) ÷ Idade (anos)
                ```

                **📊 Interpretação (Eucalipto):**
                - **> 30 m³/ha/ano**: Alta produtividade
                - **20-30 m³/ha/ano**: Média produtividade  
                - **< 20 m³/ha/ano**: Baixa produtividade

                **💡 Uso Prático:**
                - Comparar diferentes talhões
                - Avaliar qualidade do sítio
                - Planejar rotação de corte
                - Calcular viabilidade econômica
                """, unsafe_allow_html=True)

    # Modelos utilizados
    st.subheader("🏆 Modelos Utilizados")
    col1, col2 = st.columns(2)

    with col1:
        hip_r2 = st.session_state.resultados_hipsometricos.get('resultados', {}).get(
            resultados['modelos_utilizados']['hipsometrico'], {}).get('r2g', 0)
        st.success(
            f"🌳 **Hipsométrico**: {resultados['modelos_utilizados']['hipsometrico']} (R² = {formatar_brasileiro(hip_r2, 3)})")
    with col2:
        vol_r2 = st.session_state.resultados_volumetricos.get('resultados', {}).get(
            resultados['modelos_utilizados']['volumetrico'], {}).get('r2', 0)
        st.success(
            f"📊 **Volumétrico**: {resultados['modelos_utilizados']['volumetrico']} (R² = {formatar_brasileiro(vol_r2, 3)})")

    # Abas com resultados detalhados
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Resumo Geral",
        "🌳 Por Talhão",
        "📈 Crescimento & IMA",
        "🌿 Estrutura & Densidade",
        "📋 Dados Completos",
        "💾 Downloads"
    ])

    with tab1:
        mostrar_aba_resumo_geral(stats)

    with tab2:
        mostrar_aba_talhao(resultados)

    with tab3:
        mostrar_aba_crescimento_ima(stats, resultados)

    with tab4:
        mostrar_aba_estrutura_densidade(stats, resultados)

    with tab5:
        mostrar_aba_dados_completos(resultados)

    with tab6:
        mostrar_aba_downloads(resultados)


def mostrar_aba_resumo_geral(stats):
    """Mostra aba com resumo geral melhorado"""

    # Métricas dendrométricas
    st.subheader("📊 Características Dendrométricas")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📏 DAP Médio", f"{formatar_brasileiro(stats['dap_medio'], 1)} cm",
                  help="**Diâmetro à Altura do Peito** - Diâmetro médio do tronco medido a 1,30m do solo")
        st.caption(
            f"Amplitude: {formatar_brasileiro(stats['dap_min'], 1)} - {formatar_brasileiro(stats['dap_max'], 1)} cm")

    with col2:
        st.metric("🌳 Altura Média", f"{formatar_brasileiro(stats['altura_media'], 1)} m",
                  help="**Altura Total** - Altura média das árvores do solo até o topo da copa")
        st.caption(
            f"Amplitude: {formatar_brasileiro(stats['altura_min'], 1)} - {formatar_brasileiro(stats['altura_max'], 1)} m")

    with col3:
        st.metric("📊 CV Volume", f"{formatar_brasileiro(stats['cv_volume'], 1)}%",
                  help="**Coeficiente de Variação** - Medida da variabilidade dos volumes entre parcelas (Desvio Padrão/Média × 100)")
        cv_qualif = "Baixo" if stats['cv_volume'] < 20 else "Médio" if stats['cv_volume'] < 40 else "Alto"
        st.caption(f"Variabilidade: {cv_qualif}")

    with col4:
        st.metric("📅 Idade Média", f"{formatar_brasileiro(stats['idade_media'], 1)} anos",
                  help="**Idade do Povoamento** - Tempo decorrido desde o plantio até a data da medição")

    # Classificação de produtividade
    st.subheader("📊 Classificação de Produtividade")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "🌲🌲🌲 Classe Alta",
            f"{stats['classe_alta']} parcelas",
            help=f"**Parcelas de Alta Produtividade** - Parcelas com volume ≥ {formatar_brasileiro(stats['q75_volume'], 1)} m³/ha (75º percentil)"
        )

    with col2:
        st.metric(
            "🌲🌲 Classe Média",
            f"{stats['classe_media']} parcelas",
            help=f"**Parcelas de Produtividade Média** - Parcelas com volume entre {formatar_brasileiro(stats['q25_volume'], 1)} e {formatar_brasileiro(stats['q75_volume'], 1)} m³/ha"
        )

    with col3:
        st.metric(
            "🌲 Classe Baixa",
            f"{stats['classe_baixa']} parcelas",
            help=f"**Parcelas de Baixa Produtividade** - Parcelas com volume < {formatar_brasileiro(stats['q25_volume'], 1)} m³/ha (25º percentil)"
        )

    # Métricas comerciais e ambientais
    st.subheader("💰 Potencial Comercial & Ambiental")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📦 Volume Comercial", f"{formatar_brasileiro(stats['vol_comercial_medio_ha'], 1)} m³/ha",
                  help="**Volume Comercial** - Volume de madeira aproveitável comercialmente (≈85% do volume total)")
        st.metric("📦 Estoque Comercial", formatar_numero_inteligente(stats['estoque_comercial_total_m3'], "m³"),
                  help="**Estoque Comercial Total** - Volume comercial total de toda a área (Volume Comercial × Área Total)")

    with col2:
        st.metric("🌿 Biomassa Total", formatar_numero_inteligente(stats['biomassa_total_ton'], "ton"),
                  help="**Biomassa Seca** - Peso da madeira seca total considerando densidade e fator de forma")
        st.metric("🌱 Carbono Estocado", formatar_numero_inteligente(stats['carbono_estimado_ton'], "ton CO₂"),
                  help="**Carbono Sequestrado** - Quantidade de CO₂ retirado da atmosfera e estocado na madeira (≈47% da biomassa)")

    with col3:
        st.metric("🏗️ Área Basal Média", f"{formatar_brasileiro(stats['area_basal_media_ha'], 1)} m²/ha",
                  help="**Área Basal** - Soma das áreas seccionais de todas as árvores por hectare (indica ocupação do terreno)")
        st.metric("🌲 Densidade Média", f"{formatar_brasileiro(stats['densidade_media_ha'], 0)} árv/ha",
                  help="**Densidade Atual** - Número de árvores vivas por hectare")

    with col4:
        st.metric("📈 Mortalidade", f"{formatar_brasileiro(stats['mortalidade_media'], 1)}%",
                  help="**Taxa de Mortalidade** - Percentual de árvores mortas desde o plantio")
        st.metric("🎯 Índice de Sítio", f"{formatar_brasileiro(stats['indice_sitio_medio'], 2)}",
                  help="**Qualidade do Sítio** - Capacidade produtiva do local (Altura Dominante/Idade)")

    # Projeções futuras
    st.subheader("🔮 Projeções de Colheita")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("📊 Volume Final Estimado", f"{formatar_brasileiro(stats['volume_final_estimado_ha'], 1)} m³/ha",
                  help="**Volume na Colheita** - Volume estimado ao final do ciclo de rotação (7 anos para eucalipto)")
    with col2:
        st.metric("🌲 Potencial de Colheita", formatar_numero_inteligente(stats['potencial_colheita_m3'], "m³"),
                  help="**Potencial Total de Colheita** - Volume total estimado para colheita em toda a área")
    with col3:
        ciclo_otimo = 7  # Assumindo ciclo típico de eucalipto
        anos_restantes = max(0, ciclo_otimo - stats['idade_media'])
        st.metric("⏰ Anos até Colheita", f"{formatar_brasileiro(anos_restantes, 1)} anos",
                  help="**Tempo para Colheita** - Anos restantes até atingir a idade ótima de corte (7 anos)")


def mostrar_aba_talhao(resultados):
    """Mostra aba com análise detalhada por talhão"""
    st.subheader("🌳 Análise Detalhada por Talhão")

    resumo_talhao = resultados['resumo_talhoes']

    # Verificar colunas disponíveis e selecionar as que existem
    colunas_base = ['talhao', 'area_ha', 'n_parcelas']
    colunas_opcionais = {
        'vol_ha': 'Volume (m³/ha)',
        'vol_medio_ha': 'Volume (m³/ha)',
        'ima_vol': 'IMA (m³/ha/ano)',
        'ima_vol_medio': 'IMA (m³/ha/ano)',
        'dap_medio': 'DAP (cm)',
        'altura_media': 'Altura (m)',
        'densidade_ha': 'Densidade (árv/ha)',
        'densidade_media_ha': 'Densidade (árv/ha)',
        'mortalidade_estimada': 'Mortalidade (%)',
        'mortalidade_media': 'Mortalidade (%)',
        'estoque_total_m3': 'Estoque (m³)',
        'classificacao_geral': 'Classificação'
    }

    # Montar lista de colunas para exibição
    colunas_exibir = colunas_base.copy()
    nomes_colunas = ['Talhão', 'Área (ha)', 'Parcelas']

    for col_original, nome_display in colunas_opcionais.items():
        if col_original in resumo_talhao.columns:
            colunas_exibir.append(col_original)
            nomes_colunas.append(nome_display)
            break  # Usar apenas a primeira versão encontrada para cada métrica

    # Preparar dados para exibição com formatação brasileira
    df_display = resumo_talhao[colunas_exibir].copy()
    df_display.columns = nomes_colunas

    # Formatar números usando a função brasileira
    colunas_numericas = [col for col in df_display.columns if col not in ['Talhão', 'Classificação']]
    if colunas_numericas:
        df_display = formatar_dataframe_brasileiro(df_display, colunas_numericas, decimais=1)

    # Colorir classificação se existir
    if 'Classificação' in df_display.columns:
        def colorir_classificacao(val):
            colors = {
                'Excelente': 'background-color: #90EE90',
                'Muito Bom': 'background-color: #87CEEB',
                'Bom': 'background-color: #98FB98',
                'Regular': 'background-color: #F0E68C',
                'Baixo': 'background-color: #FFA07A'
            }
            return colors.get(val, '')

        styled_df = df_display.style.applymap(colorir_classificacao, subset=['Classificação'])
        st.dataframe(styled_df, hide_index=True, use_container_width=True)
    else:
        st.dataframe(df_display, hide_index=True, use_container_width=True)

    # Destaques por talhão (apenas se colunas existirem)
    st.subheader("🏆 Destaques por Talhão")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Talhão mais produtivo
        col_volume = None
        for col in ['vol_ha', 'vol_medio_ha']:
            if col in resumo_talhao.columns:
                col_volume = col
                break

        if col_volume:
            idx_max_vol = resumo_talhao[col_volume].idxmax()
            talhao_max_vol = resumo_talhao.loc[idx_max_vol, 'talhao']
            vol_max = resumo_talhao.loc[idx_max_vol, col_volume]
            st.metric("🥇 Mais Produtivo", f"Talhão {talhao_max_vol}", f"{formatar_brasileiro(vol_max, 1)} m³/ha")
        else:
            st.info("Volume não disponível")

    with col2:
        # Maior IMA
        col_ima = None
        for col in ['ima_vol', 'ima_vol_medio']:
            if col in resumo_talhao.columns:
                col_ima = col
                break

        if col_ima:
            idx_max_ima = resumo_talhao[col_ima].idxmax()
            talhao_max_ima = resumo_talhao.loc[idx_max_ima, 'talhao']
            ima_max = resumo_talhao.loc[idx_max_ima, col_ima]
            st.metric("🚀 Maior IMA", f"Talhão {talhao_max_ima}", f"{formatar_brasileiro(ima_max, 1)} m³/ha/ano")
        else:
            st.info("IMA não disponível")

    with col3:
        # Maior área
        if 'area_ha' in resumo_talhao.columns:
            idx_max_area = resumo_talhao['area_ha'].idxmax()
            talhao_max_area = resumo_talhao.loc[idx_max_area, 'talhao']
            area_max = resumo_talhao.loc[idx_max_area, 'area_ha']
            st.metric("📏 Maior Área", f"Talhão {talhao_max_area}", f"{formatar_brasileiro(area_max, 1)} ha")
        else:
            st.info("Área não disponível")

    with col4:
        # Maior estoque
        if 'estoque_total_m3' in resumo_talhao.columns:
            idx_max_estoque = resumo_talhao['estoque_total_m3'].idxmax()
            talhao_max_estoque = resumo_talhao.loc[idx_max_estoque, 'talhao']
            estoque_max = resumo_talhao.loc[idx_max_estoque, 'estoque_total_m3']
            st.metric("🌲 Maior Estoque", f"Talhão {talhao_max_estoque}", formatar_numero_inteligente(estoque_max, "m³"))
        else:
            st.info("Estoque não disponível")


def mostrar_aba_crescimento_ima(stats, resultados):
    """Mostra aba focada em crescimento e IMA"""
    st.subheader("📈 Análise de Crescimento e IMA")

    # Classificação de IMA
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🌟 IMA Excelente", f"{stats['ima_excelente']} parcelas",
                  help="**IMA Excelente** - Parcelas com Incremento Médio Anual ≥ 25 m³/ha/ano (alta produtividade)")
    with col2:
        st.metric("📊 IMA Bom", f"{stats['ima_bom']} parcelas",
                  help="**IMA Bom** - Parcelas com IMA entre 15-25 m³/ha/ano (produtividade média-alta)")
    with col3:
        st.metric("📉 IMA Regular", f"{stats['ima_regular']} parcelas",
                  help="**IMA Regular** - Parcelas com IMA < 15 m³/ha/ano (produtividade baixa)")

    # Gráficos de crescimento
    st.subheader("📊 Gráficos de Crescimento")

    resumo_parcelas = resultados['resumo_parcelas']

    col1, col2 = st.columns(2)

    with col1:
        # Distribuição de IMA
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(resumo_parcelas['ima_vol'], bins=15, alpha=0.7, color='green', edgecolor='black')
        ax.axvline(stats['ima_vol_medio'], color='red', linestyle='--', linewidth=2,
                   label=f'Média: {formatar_brasileiro(stats["ima_vol_medio"], 1)} m³/ha/ano')
        ax.set_xlabel('IMA (m³/ha/ano)')
        ax.set_ylabel('Frequência')
        ax.set_title('Distribuição do IMA')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

    with col2:
        # Relação Volume vs IMA
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(resumo_parcelas['vol_ha'], resumo_parcelas['ima_vol'], alpha=0.6, color='darkgreen')
        ax.set_xlabel('Volume (m³/ha)')
        ax.set_ylabel('IMA (m³/ha/ano)')
        ax.set_title('Relação Volume vs IMA')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

    # Métricas de crescimento por componente
    st.subheader("🌱 Crescimento por Componente")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("📊 IMA Volume", f"{formatar_brasileiro(stats['ima_vol_medio'], 2)} m³/ha/ano",
                  help="**Incremento Médio Anual Volumétrico** - Crescimento médio anual em volume por hectare")
        st.metric("📈 IMA Área Basal", f"{formatar_brasileiro(stats['ima_area_basal_medio'], 2)} m²/ha/ano",
                  help="**IMA de Área Basal** - Crescimento médio anual da área basal por hectare")

    with col2:
        st.metric("🌿 IMA Biomassa", f"{formatar_numero_inteligente(stats['ima_biomassa_medio'], 'kg/ha/ano')}",
                  help="**IMA de Biomassa** - Crescimento médio anual da biomassa seca por hectare")
        st.metric("🎯 Índice de Sítio", f"{formatar_brasileiro(stats['indice_sitio_medio'], 2)}",
                  help="**Índice de Sítio** - Indicador da qualidade do local para crescimento florestal (altura/idade)")

    with col3:
        # Projeção de crescimento
        crescimento_anual = stats['ima_vol_medio']
        volume_5_anos = stats['vol_medio_ha'] + (crescimento_anual * 2)  # +2 anos
        st.metric("📊 Volume em 2 anos", f"{formatar_brasileiro(volume_5_anos, 1)} m³/ha",
                  help="**Projeção de Volume** - Volume estimado daqui a 2 anos baseado no IMA atual")
        st.caption("Projeção baseada no IMA atual")


def mostrar_aba_estrutura_densidade(stats, resultados):
    """Mostra aba com análise de estrutura e densidade"""
    st.subheader("🌿 Estrutura do Povoamento e Densidade")

    # Métricas de densidade
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("🌲 Densidade Atual", f"{formatar_brasileiro(stats['densidade_media_ha'], 0)} árv/ha",
                  help="**Densidade Atual** - Número de árvores vivas por hectare no momento da medição")
    with col2:
        densidade_inicial = resultados['parametros_utilizados'].get('densidade_plantio', 1667)
        st.metric("🌱 Densidade Inicial", f"{formatar_brasileiro(densidade_inicial, 0)} árv/ha",
                  help="**Densidade de Plantio** - Número de mudas plantadas inicialmente por hectare")
    with col3:
        st.metric("📉 Mortalidade", f"{formatar_brasileiro(stats['mortalidade_media'], 1)}%",
                  help="**Taxa de Mortalidade** - Percentual de árvores que morreram desde o plantio")
    with col4:
        sobrevivencia = 100 - stats['mortalidade_media']
        st.metric("✅ Sobrevivência", f"{formatar_brasileiro(sobrevivencia, 1)}%",
                  help="**Taxa de Sobrevivência** - Percentual de árvores que permaneceram vivas desde o plantio")

        #st.metric("✅ Sobrevivência", f"{sobrevivencia:.1f}%")

    # Distribuição diamétrica
    df_completo = resultados['inventario_completo']
    if 'classe_dap' in df_completo.columns:
        st.subheader("📊 Distribuição Diamétrica")

        dist_dap = df_completo['classe_dap'].value_counts().sort_index()

        col1, col2 = st.columns(2)

        with col1:
            # Gráfico de barras da distribuição
            fig, ax = plt.subplots(figsize=(8, 6))
            bars = ax.bar(range(len(dist_dap)), dist_dap.values, color='brown', alpha=0.7)
            ax.set_xlabel('Classe Diamétrica')
            ax.set_ylabel('Número de Árvores')
            ax.set_title('Distribuição Diamétrica')
            ax.set_xticks(range(len(dist_dap)))
            ax.set_xticklabels(dist_dap.index, rotation=45)
            ax.grid(True, alpha=0.3)

            # Adicionar valores nas barras com formatação brasileira
            for bar, val in zip(bars, dist_dap.values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(dist_dap.values) * 0.01,
                        f'{val}', ha='center', va='bottom')

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            # Tabela da distribuição com formatação brasileira
            st.write("**Distribuição por Classe:**")
            df_dist = pd.DataFrame({
                'Classe': dist_dap.index,
                'Árvores': dist_dap.values,
                'Percentual': (dist_dap.values / dist_dap.values.sum() * 100).round(1)
            })
            # Formatar a coluna de percentual
            df_dist['Percentual'] = df_dist['Percentual'].apply(lambda x: f"{formatar_brasileiro(x, 1)}%")
            st.dataframe(df_dist, hide_index=True)

    # Análise de biomassa e carbono
    st.subheader("🌿 Análise Ambiental")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🌿 Biomassa Total", formatar_numero_inteligente(stats['biomassa_total_ton'], "ton"),
                  help="**Biomassa Seca Total** - Peso total da madeira seca de toda a floresta")
        st.metric("🌱 Biomassa por Hectare",
                  f"{formatar_brasileiro(stats['biomassa_total_ton'] / stats['area_total_ha'], 1)} ton/ha",
                  help="**Biomassa por Hectare** - Peso médio da madeira seca por unidade de área")

    with col2:
        st.metric("🌱 Carbono Estocado", formatar_numero_inteligente(stats['carbono_estimado_ton'], "ton CO₂"),
                  help="**Carbono Sequestrado** - CO₂ retirado da atmosfera e fixado na madeira (≈47% da biomassa)")
        carbono_ha = stats['carbono_estimado_ton'] / stats['area_total_ha']
        st.metric("🌱 Carbono por Hectare", f"{formatar_brasileiro(carbono_ha, 1)} ton CO₂/ha",
                  help="**Sequestro de Carbono por Hectare** - Quantidade de CO₂ sequestrado por unidade de área")

    with col3:
        # Equivalente em carros retirados de circulação (assumindo 4.6 ton CO₂/ano por carro)
        carros_equivalente = stats['carbono_estimado_ton'] / 4.6
        st.metric("🚗 Equivalente em Carros", f"{formatar_numero_inteligente(carros_equivalente, 'carros/ano')}",
                  help="**Impacto Ambiental** - Número de carros que precisariam ser retirados de circulação por 1 ano para ter o mesmo efeito ambiental")
        st.caption("Emissão média anual por veículo")


def mostrar_aba_dados_completos(resultados):
    """Mostra aba com dados completos"""
    st.subheader("📋 Dados Completos")

    # Seletor de dataset
    datasets = {
        "Resumo por Parcela": resultados['resumo_parcelas'],
        "Resumo por Talhão": resultados['resumo_talhoes'],
        "Inventário Completo": resultados['inventario_completo'].head(1000)
    }

    dataset_selecionado = st.selectbox(
        "📊 Selecione o dataset:",
        options=list(datasets.keys()),
        key="dataset_selector_completo"
    )

    df_selecionado = datasets[dataset_selecionado]

    # Informações do dataset
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Registros", len(df_selecionado))
    with col2:
        st.metric("Colunas", len(df_selecionado.columns))
    with col3:
        if dataset_selecionado == "Inventário Completo" and len(resultados['inventario_completo']) > 1000:
            st.metric("Exibindo", "Primeiros 1.000")
        else:
            st.metric("Exibindo", "Todos")

    # Exibir dados
    st.dataframe(df_selecionado, hide_index=True, use_container_width=True)


def mostrar_aba_downloads(resultados):
    """Mostra aba com downloads melhorados"""
    st.subheader("💾 Downloads")

    # Seção de dados
    st.write("**📁 Arquivos de Dados:**")
    col1, col2, col3 = st.columns(3)

    with col1:
        csv_parcelas = resultados['resumo_parcelas'].to_csv(index=False)
        st.download_button(
            label="📊 Resumo por Parcela",
            data=csv_parcelas,
            file_name="resumo_parcelas_detalhado.csv",
            mime="text/csv",
            key="download_parcelas_detalhado"
        )

    with col2:
        csv_talhoes = resultados['resumo_talhoes'].to_csv(index=False)
        st.download_button(
            label="🌳 Resumo por Talhão",
            data=csv_talhoes,
            file_name="resumo_talhoes_detalhado.csv",
            mime="text/csv",
            key="download_talhoes_detalhado"
        )

    with col3:
        csv_completo = resultados['inventario_completo'].to_csv(index=False)
        st.download_button(
            label="📋 Inventário Completo",
            data=csv_completo,
            file_name="inventario_completo_detalhado.csv",
            mime="text/csv",
            key="download_completo_detalhado"
        )

    # Relatório executivo melhorado
    st.write("**📄 Relatórios:**")
    relatorio = gerar_relatorio_executivo_melhorado(resultados)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📄 Relatório Executivo Completo",
            data=relatorio,
            file_name="relatorio_inventario_completo.md",
            mime="text/markdown",
            key="download_relatorio_completo"
        )

    with col2:
        # Relatório resumido para gestão
        relatorio_gestao = gerar_relatorio_gestao(resultados)
        st.download_button(
            label="📋 Relatório Gerencial",
            data=relatorio_gestao,
            file_name="relatorio_gerencial.md",
            mime="text/markdown",
            key="download_relatorio_gestao"
        )


def gerar_relatorio_executivo_melhorado(resultados):
    """Gera relatório executivo completo melhorado com formatação brasileira"""
    stats = resultados['estatisticas_gerais']
    modelos = resultados['modelos_utilizados']

    relatorio = f"""# RELATÓRIO EXECUTIVO - INVENTÁRIO FLORESTAL COMPLETO

## 🏆 MODELOS SELECIONADOS
- **Hipsométrico**: {modelos['hipsometrico']}
- **Volumétrico**: {modelos['volumetrico']}

## 🌲 RESUMO EXECUTIVO
- **Parcelas avaliadas**: {stats['total_parcelas']}
- **Talhões**: {stats['total_talhoes']}
- **Área total**: {formatar_brasileiro(stats['area_total_ha'], 1)} ha
- **Estoque total**: {formatar_numero_inteligente(stats['estoque_total_m3'], 'm³')}
- **Estoque comercial**: {formatar_numero_inteligente(stats['estoque_comercial_total_m3'], 'm³')}
- **Produtividade média**: {formatar_brasileiro(stats['vol_medio_ha'], 1)} m³/ha
- **IMA médio**: {formatar_brasileiro(stats['ima_vol_medio'], 1)} m³/ha/ano

## 📊 CARACTERÍSTICAS DENDROMÉTRICAS
- **DAP médio**: {formatar_brasileiro(stats['dap_medio'], 1)} cm (amplitude: {formatar_brasileiro(stats['dap_min'], 1)} - {formatar_brasileiro(stats['dap_max'], 1)} cm)
- **Altura média**: {formatar_brasileiro(stats['altura_media'], 1)} m (amplitude: {formatar_brasileiro(stats['altura_min'], 1)} - {formatar_brasileiro(stats['altura_max'], 1)} m)
- **Densidade média**: {formatar_brasileiro(stats['densidade_media_ha'], 0)} árv/ha
- **Área basal média**: {formatar_brasileiro(stats['area_basal_media_ha'], 1)} m²/ha
- **Idade média**: {formatar_brasileiro(stats['idade_media'], 1)} anos

## 📈 ANÁLISE DE CRESCIMENTO
- **IMA Volume**: {formatar_brasileiro(stats['ima_vol_medio'], 2)} m³/ha/ano
- **IMA Área Basal**: {formatar_brasileiro(stats['ima_area_basal_medio'], 2)} m²/ha/ano
- **IMA Biomassa**: {formatar_brasileiro(stats['ima_biomassa_medio'], 0)} kg/ha/ano
- **Índice de Sítio**: {formatar_brasileiro(stats['indice_sitio_medio'], 2)}

## 🌿 ASPECTOS AMBIENTAIS
- **Biomassa total**: {formatar_numero_inteligente(stats['biomassa_total_ton'], 'toneladas')}
- **Carbono estocado**: {formatar_numero_inteligente(stats['carbono_estimado_ton'], 'toneladas CO₂')}
- **Mortalidade média**: {formatar_brasileiro(stats['mortalidade_media'], 1)}%

## 📊 CLASSIFICAÇÃO DE PRODUTIVIDADE
- **Classe Alta** (≥ {formatar_brasileiro(stats['q75_volume'], 1)} m³/ha): {stats['classe_alta']} parcelas
- **Classe Média** ({formatar_brasileiro(stats['q25_volume'], 1)} - {formatar_brasileiro(stats['q75_volume'], 1)} m³/ha): {stats['classe_media']} parcelas
- **Classe Baixa** (< {formatar_brasileiro(stats['q25_volume'], 1)} m³/ha): {stats['classe_baixa']} parcelas

## 📈 CLASSIFICAÇÃO DE IMA
- **IMA Excelente** (≥ 25 m³/ha/ano): {stats['ima_excelente']} parcelas
- **IMA Bom** (15-25 m³/ha/ano): {stats['ima_bom']} parcelas
- **IMA Regular** (< 15 m³/ha/ano): {stats['ima_regular']} parcelas

## 🔮 PROJEÇÕES DE COLHEITA
- **Volume final estimado**: {formatar_brasileiro(stats['volume_final_estimado_ha'], 1)} m³/ha
- **Potencial de colheita**: {formatar_numero_inteligente(stats['potencial_colheita_m3'], 'm³')}
- **Anos até colheita ótima**: {formatar_brasileiro(max(0, 7 - stats['idade_media']), 1)} anos

## 📈 VARIABILIDADE
- **CV produtividade**: {formatar_brasileiro(stats['cv_volume'], 1)}%
- **Amplitude volume**: {formatar_brasileiro(stats['vol_min_ha'], 1)} - {formatar_brasileiro(stats['vol_max_ha'], 1)} m³/ha

## 💰 ASPECTOS COMERCIAIS
- **Volume comercial médio**: {formatar_brasileiro(stats['vol_comercial_medio_ha'], 1)} m³/ha
- **Estoque comercial total**: {formatar_numero_inteligente(stats['estoque_comercial_total_m3'], 'm³')}
- **Percentual comercial**: {formatar_brasileiro((stats['vol_comercial_medio_ha'] / stats['vol_medio_ha'] * 100), 1)}%

## 🎯 RECOMENDAÇÕES TÉCNICAS
1. **Manejo**: Foco nos talhões de classe alta para maximizar produtividade
2. **Colheita**: Planejamento baseado no IMA e ciclo ótimo de 7 anos
3. **Silvicultura**: Atenção especial aos talhões com alta mortalidade
4. **Monitoramento**: Acompanhar evolução do IMA nas próximas medições

---
*Relatório gerado pelo Sistema Integrado de Inventário Florestal*
*Data: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}*
"""

    return relatorio


def gerar_relatorio_gestao(resultados):
    """Gera relatório resumido para gestão"""
    stats = resultados['estatisticas_gerais']
    resumo_talhoes = resultados['resumo_talhoes']

    # Encontrar melhores e piores talhões (verificar se colunas existem)
    col_volume = None
    for col in ['vol_ha', 'vol_medio_ha']:
        if col in resumo_talhoes.columns:
            col_volume = col
            break

    col_ima = None
    for col in ['ima_vol', 'ima_vol_medio']:
        if col in resumo_talhoes.columns:
            col_ima = col
            break

    if col_volume and len(resumo_talhoes) > 0:
        melhor_talhao = resumo_talhoes.loc[resumo_talhoes[col_volume].idxmax()]
        pior_talhao = resumo_talhoes.loc[resumo_talhoes[col_volume].idxmin()]

        melhor_vol = melhor_talhao[col_volume]
        pior_vol = pior_talhao[col_volume]
        melhor_ima = melhor_talhao.get(col_ima, 0) if col_ima else 0
        pior_ima = pior_talhao.get(col_ima, 0) if col_ima else 0
    else:
        # Valores padrão se não houver dados
        melhor_talhao = {'talhao': 'N/A'}
        pior_talhao = {'talhao': 'N/A'}
        melhor_vol = pior_vol = melhor_ima = pior_ima = 0

    relatorio = f"""# RELATÓRIO GERENCIAL - INVENTÁRIO FLORESTAL

## 📊 RESUMO EXECUTIVO
**Situação Atual do Patrimônio Florestal**

### 🎯 Indicadores Chave
- **Área Total**: {formatar_brasileiro(stats.get('area_total_ha', 0), 1)} hectares
- **Estoque Total**: {formatar_numero_inteligente(stats.get('estoque_total_m3', 0), 'm³')}
- **Valor Médio**: {formatar_brasileiro(stats.get('vol_medio_ha', 0), 1)} m³/ha
- **Produtividade**: {formatar_brasileiro(stats.get('ima_vol_medio', 0), 1)} m³/ha/ano
- **Idade Média**: {formatar_brasileiro(stats.get('idade_media', 0), 1)} anos

### 📈 PERFORMANCE POR TALHÃO

**🏆 Melhor Performance:**
- Talhão {melhor_talhao['talhao']}: {formatar_brasileiro(melhor_vol, 1)} m³/ha (IMA: {formatar_brasileiro(melhor_ima, 1)})

**⚠️ Requer Atenção:**
- Talhão {pior_talhao['talhao']}: {formatar_brasileiro(pior_vol, 1)} m³/ha (IMA: {formatar_brasileiro(pior_ima, 1)})

### 💰 POTENCIAL ECONÔMICO
- **Volume Comercial**: {formatar_numero_inteligente(stats.get('estoque_comercial_total_m3', 0), 'm³')}
- **Biomassa para Energia**: {formatar_numero_inteligente(stats.get('biomassa_total_ton', 0), 'toneladas')}
- **Créditos de Carbono**: {formatar_numero_inteligente(stats.get('carbono_estimado_ton', 0), 'ton CO₂')}

### 🎯 AÇÕES RECOMENDADAS

**Imediatas (0-6 meses):**
1. Intensificar manejo nos talhões de alta produtividade
2. Investigar causas da baixa performance em talhões críticos
3. Planejar colheita para talhões próximos ao ciclo ótimo

**Médio Prazo (6-18 meses):**
1. Reforma/replantio em áreas de baixa produtividade
2. Otimização do espaçamento para melhorar IMA
3. Implementação de práticas de manejo diferenciado

**Longo Prazo (2+ anos):**
1. Melhoramento genético baseado nos melhores materiais
2. Expansão para áreas com potencial similar aos melhores talhões
3. Certificação florestal para agregar valor

### 📊 CLASSIFICAÇÃO GERAL
- **{formatar_brasileiro(((stats.get('classe_alta', 0) / stats.get('total_parcelas', 1)) * 100), 1)}%** das parcelas em classe ALTA
- **{formatar_brasileiro(((stats.get('classe_media', 0) / stats.get('total_parcelas', 1)) * 100), 1)}%** das parcelas em classe MÉDIA  
- **{formatar_brasileiro(((stats.get('classe_baixa', 0) / stats.get('total_parcelas', 1)) * 100), 1)}%** das parcelas em classe BAIXA

---
**Próxima avaliação recomendada**: {(pd.Timestamp.now() + pd.DateOffset(years=1)).strftime('%m/%Y')}
"""

    return relatorio


def main():
    if not verificar_prerequisitos():
        return

    st.title("📈 Inventário Florestal")
    st.markdown("### Processamento Completo e Relatórios Finais")

    # Mostrar status das etapas anteriores
    mostrar_status_etapas()

    # Verificar se já foi processado
    if st.session_state.get('inventario_processado'):
        st.info("ℹ️ O inventário já foi processado. Resultados salvos abaixo.")

        # Botão para reprocessar
        if st.button("🔄 Reprocessar Inventário", key="btn_reprocessar_inv"):
            del st.session_state.inventario_processado
            st.rerun()

        # Mostrar resultados salvos
        mostrar_resultados_inventario(st.session_state.inventario_processado)
        return

    # Configurar áreas dos talhões
    config_areas = configurar_areas_talhoes()

    # Configurar parâmetros avançados
    parametros = configurar_parametros_avancados()

    # Resumo dos dados de entrada
    st.subheader("📋 Resumo dos Dados de Entrada")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Registros", len(st.session_state.dados_inventario))
    with col2:
        st.metric("Talhões", st.session_state.dados_inventario['talhao'].nunique())
    with col3:
        st.metric("Parcelas", st.session_state.dados_inventario['parcela'].nunique())
    with col4:
        cubagem_len = 0
        if hasattr(st.session_state, 'dados_cubagem') and st.session_state.dados_cubagem is not None:
            cubagem_len = len(st.session_state.dados_cubagem)
        st.metric("Árvores Cubadas", cubagem_len)

    # Preview das configurações
    with st.expander("👀 Preview das Configurações"):
        col1, col2 = st.columns(2)

        with col1:
            st.write("**📏 Configurações de Área:**")
            st.write(f"- Método: {config_areas['metodo']}")
            if config_areas['metodo'] == "Área fixa para todos":
                st.write(f"- Área por talhão: {config_areas['area_fixa']:.1f} ha")
            elif config_areas['metodo'] == "Simulação baseada em parcelas":
                st.write(f"- Fator de expansão: {config_areas.get('fator_expansao', 3.0):.1f} ha/parcela")

        with col2:
            st.write("**⚙️ Parâmetros Florestais:**")
            st.write(f"- Área da parcela: {parametros['area_parcela']} m²")
            st.write(f"- Densidade de plantio: {parametros['densidade_plantio']} árv/ha")
            st.write(f"- Taxa de sobrevivência: {parametros['sobrevivencia'] * 100:.0f}%")
            st.write(f"- Densidade da madeira: {parametros['densidade_madeira']} kg/m³")

    # Botão principal para executar
    if st.button("🚀 Executar Inventário Completo", type="primary", use_container_width=True):
        executar_inventario_completo(config_areas, parametros)


if __name__ == "__main__":
    main()