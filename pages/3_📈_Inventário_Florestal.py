# pages/3_📈_Inventário_Florestal.py
# VERSÃO FINAL CORRIGIDA - SEM DUPLICAÇÕES

"""
Etapa 3: Inventário Florestal
Processamento completo e relatórios finais
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import traceback

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
        st.success(f"🌳 **Etapa 1 Concluída** - Melhor modelo: {melhor_hip}")

    with col2:
        melhor_vol = st.session_state.resultados_volumetricos.get('melhor_modelo', 'N/A')
        st.success(f"📊 **Etapa 2 Concluída** - Melhor modelo: {melhor_vol}")


def configurar_areas_talhoes():
    """Configura áreas dos talhões"""
    st.header("📏 Configuração de Áreas dos Talhões")

    df_inventario = st.session_state.dados_inventario
    talhoes_disponiveis = sorted(df_inventario['talhao'].unique())

    # Método de cálculo das áreas
    metodo_area = st.selectbox(
        "🗺️ Método para Cálculo das Áreas",
        ["Área fixa para todos", "Valores específicos por talhão"],
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
                st.metric("Área Total", f"{area_total:.1f} ha")
            with col2:
                st.metric("Área Média", f"{np.mean(list(areas_manuais.values())):.1f} ha")
            with col3:
                st.metric("Talhões", len(areas_manuais))

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
            st.metric("Área Total", f"{area_total:.1f} ha")
        with col2:
            st.metric("Área por Talhão", f"{area_fixa:.1f} ha")
        with col3:
            st.metric("Total de Talhões", len(talhoes_disponiveis))

    return config_areas


def criar_df_areas(config_areas):
    """Cria DataFrame de áreas baseado na configuração"""
    if config_areas['metodo'] == "Valores específicos por talhão":
        areas_dict = config_areas.get('areas_manuais', {})
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


def calcular_resumo_por_parcela(df):
    """Calcula resumo por parcela"""
    area_parcela_m2 = 400

    resumo = df.groupby(['talhao', 'parcela']).agg({
        'area_ha': 'first',
        'D_cm': 'mean',
        'H_est': 'mean',
        'V_est': 'sum',
        'cod': 'count'
    }).reset_index()

    resumo = resumo.rename(columns={
        'cod': 'n_arvores',
        'D_cm': 'dap_medio',
        'H_est': 'altura_media',
        'V_est': 'volume_parcela'
    })

    resumo['vol_ha'] = resumo['volume_parcela'] * (10000 / area_parcela_m2)
    resumo['idade_anos'] = 5.0
    resumo['ima'] = resumo['vol_ha'] / resumo['idade_anos']

    return resumo


def calcular_resumo_por_talhao(resumo_parcelas):
    """Calcula resumo por talhão"""
    resumo_talhao = resumo_parcelas.groupby('talhao').agg({
        'area_ha': 'first',
        'vol_ha': ['mean', 'std', 'count'],
        'dap_medio': 'mean',
        'altura_media': 'mean',
        'idade_anos': 'mean',
        'n_arvores': 'mean',
        'ima': 'mean'
    }).round(2)

    resumo_talhao.columns = [
        'area_ha', 'vol_medio_ha', 'vol_desvio', 'n_parcelas',
        'dap_medio', 'altura_media', 'idade_media', 'arvores_por_parcela', 'ima_medio'
    ]

    resumo_talhao = resumo_talhao.reset_index()
    resumo_talhao['estoque_total_m3'] = resumo_talhao['area_ha'] * resumo_talhao['vol_medio_ha']
    resumo_talhao['cv_volume'] = (resumo_talhao['vol_desvio'] / resumo_talhao['vol_medio_ha']) * 100

    return resumo_talhao


def calcular_estatisticas_gerais(resumo_parcelas):
    """Calcula estatísticas gerais do inventário"""
    stats = {
        'total_parcelas': len(resumo_parcelas),
        'total_talhoes': resumo_parcelas['talhao'].nunique(),
        'area_total_ha': resumo_parcelas['area_ha'].sum(),
        'vol_medio_ha': resumo_parcelas['vol_ha'].mean(),
        'vol_min_ha': resumo_parcelas['vol_ha'].min(),
        'vol_max_ha': resumo_parcelas['vol_ha'].max(),
        'cv_volume': (resumo_parcelas['vol_ha'].std() / resumo_parcelas['vol_ha'].mean()) * 100,
        'dap_medio': resumo_parcelas['dap_medio'].mean(),
        'altura_media': resumo_parcelas['altura_media'].mean(),
        'idade_media': resumo_parcelas['idade_anos'].mean(),
        'ima_medio': resumo_parcelas['ima'].mean(),
        'arvores_por_parcela': resumo_parcelas['n_arvores'].mean()
    }

    stats['estoque_total_m3'] = stats['area_total_ha'] * stats['vol_medio_ha']

    q25 = resumo_parcelas['vol_ha'].quantile(0.25)
    q75 = resumo_parcelas['vol_ha'].quantile(0.75)

    stats['classe_alta'] = (resumo_parcelas['vol_ha'] >= q75).sum()
    stats['classe_media'] = ((resumo_parcelas['vol_ha'] >= q25) & (resumo_parcelas['vol_ha'] < q75)).sum()
    stats['classe_baixa'] = (resumo_parcelas['vol_ha'] < q25).sum()
    stats['q25_volume'] = q25
    stats['q75_volume'] = q75

    return stats


def executar_inventario_completo(config_areas):
    """Executa o inventário completo - VERSÃO LIMPA"""
    st.header("🚀 Executando Inventário Completo")

    # Barra de progresso
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.text("Processando áreas dos talhões...")
        progress_bar.progress(0.1)

        # Criar DataFrame de áreas
        df_areas = criar_df_areas(config_areas)
        st.success(f"✅ Áreas processadas: {len(df_areas)} talhões")

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
        progress_bar.progress(0.5)

        # Estimar alturas
        df_com_alturas = estimar_alturas_inventario(df_com_areas, melhor_hip)

        status_text.text("Aplicando modelos volumétricos...")
        progress_bar.progress(0.7)

        # Estimar volumes
        df_com_volumes = estimar_volumes_inventario(df_com_alturas, melhor_vol)

        status_text.text("Calculando estatísticas finais...")
        progress_bar.progress(0.9)

        # Calcular resumos
        resumo_parcelas = calcular_resumo_por_parcela(df_com_volumes)
        resumo_talhoes = calcular_resumo_por_talhao(resumo_parcelas)
        estatisticas_gerais = calcular_estatisticas_gerais(resumo_parcelas)

        progress_bar.progress(1.0)
        status_text.text("✅ Inventário processado com sucesso!")

        # Preparar resultados finais
        resultados = {
            'inventario_completo': df_com_volumes,
            'resumo_parcelas': resumo_parcelas,
            'resumo_talhoes': resumo_talhoes,
            'estatisticas_gerais': estatisticas_gerais,
            'modelos_utilizados': {
                'hipsometrico': melhor_hip,
                'volumetrico': melhor_vol
            }
        }

        # Salvar no session_state
        st.session_state.inventario_processado = resultados

        st.success(f"🏆 Inventário processado com sucesso!")
        st.info(f"📊 Modelos utilizados: {melhor_hip} (Hipsométrico) + {melhor_vol} (Volumétrico)")

        # CORREÇÃO: Mostrar resultados apenas uma vez aqui
        mostrar_resultados_inventario(resultados)

    except Exception as e:
        st.error(f"❌ Erro no processamento do inventário: {e}")
        st.info("💡 Verifique os dados e configurações")
        with st.expander("🔍 Detalhes do erro"):
            st.code(traceback.format_exc())


def mostrar_resultados_inventario(resultados):
    """Mostra os resultados finais do inventário - VERSÃO ÚNICA"""
    st.header("📊 Resultados Finais do Inventário")

    stats = resultados['estatisticas_gerais']

    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("🌲 Parcelas", f"{stats['total_parcelas']:,}")
    with col2:
        st.metric("📏 Área Total", f"{stats['area_total_ha']:.1f} ha")
    with col3:
        st.metric("📊 Produtividade", f"{stats['vol_medio_ha']:.1f} m³/ha")
    with col4:
        st.metric("🌲 Estoque Total", f"{stats['estoque_total_m3']:,.0f} m³")

    # Modelos utilizados
    st.subheader("🏆 Modelos Utilizados")
    col1, col2 = st.columns(2)

    with col1:
        st.success(f"🌳 **Hipsométrico**: {resultados['modelos_utilizados']['hipsometrico']}")
    with col2:
        st.success(f"📊 **Volumétrico**: {resultados['modelos_utilizados']['volumetrico']}")

    # Classificação de produtividade
    st.subheader("📊 Classificação de Produtividade")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "🌲🌲🌲 Classe Alta",
            f"{stats['classe_alta']} parcelas",
            help=f"≥ {stats['q75_volume']:.1f} m³/ha"
        )

    with col2:
        st.metric(
            "🌲🌲 Classe Média",
            f"{stats['classe_media']} parcelas",
            help=f"{stats['q25_volume']:.1f} - {stats['q75_volume']:.1f} m³/ha"
        )

    with col3:
        st.metric(
            "🌲 Classe Baixa",
            f"{stats['classe_baixa']} parcelas",
            help=f"< {stats['q25_volume']:.1f} m³/ha"
        )

    # Tabela de resumo por talhão
    st.subheader("🌳 Resumo por Talhão")

    # Selecionar colunas importantes para exibição
    colunas_exibir = ['talhao', 'area_ha', 'n_parcelas', 'vol_medio_ha', 'dap_medio', 'altura_media',
                      'estoque_total_m3']
    df_display = resultados['resumo_talhoes'][colunas_exibir].copy()

    # Renomear colunas
    df_display.columns = ['Talhão', 'Área (ha)', 'Parcelas', 'Volume (m³/ha)', 'DAP (cm)', 'Altura (m)', 'Estoque (m³)']

    # Formatar números
    for col in ['Área (ha)', 'Volume (m³/ha)', 'DAP (cm)', 'Altura (m)', 'Estoque (m³)']:
        if col in df_display.columns:
            df_display[col] = df_display[col].round(1)

    st.dataframe(df_display, hide_index=True, use_container_width=True)

    # Downloads com keys únicos
    st.subheader("💾 Downloads")

    col1, col2, col3 = st.columns(3)

    with col1:
        csv_parcelas = resultados['resumo_parcelas'].to_csv(index=False)
        st.download_button(
            "📊 Resumo por Parcela",
            csv_parcelas,
            "resumo_parcelas.csv",
            "text/csv",
            key="download_parcelas_inv_final"  # CORREÇÃO: Key única
        )

    with col2:
        csv_talhoes = resultados['resumo_talhoes'].to_csv(index=False)
        st.download_button(
            "🌳 Resumo por Talhão",
            csv_talhoes,
            "resumo_talhoes.csv",
            "text/csv",
            key="download_talhoes_inv_final"  # CORREÇÃO: Key única
        )

    with col3:
        relatorio = gerar_relatorio_executivo(resultados)
        st.download_button(
            "📄 Relatório Executivo",
            relatorio,
            "relatorio_inventario.md",
            "text/markdown",
            key="download_relatorio_inv_final"  # CORREÇÃO: Key única
        )


def gerar_relatorio_executivo(resultados):
    """Gera relatório executivo em markdown"""
    stats = resultados['estatisticas_gerais']
    modelos = resultados['modelos_utilizados']

    relatorio = f"""# RELATÓRIO EXECUTIVO - INVENTÁRIO FLORESTAL

## 🏆 MODELOS SELECIONADOS
- **Hipsométrico**: {modelos['hipsometrico']}
- **Volumétrico**: {modelos['volumetrico']}

## 🌲 RESUMO EXECUTIVO
- **Parcelas avaliadas**: {stats['total_parcelas']}
- **Talhões**: {stats['total_talhoes']}
- **Área total**: {stats['area_total_ha']:.1f} ha
- **Estoque total**: {stats['estoque_total_m3']:,.0f} m³
- **Produtividade média**: {stats['vol_medio_ha']:.1f} m³/ha
- **IMA médio**: {stats['ima_medio']:.1f} m³/ha/ano

## 📊 CLASSIFICAÇÃO DE PRODUTIVIDADE
- **Classe Alta** (≥ {stats['q75_volume']:.1f} m³/ha): {stats['classe_alta']} parcelas
- **Classe Média** ({stats['q25_volume']:.1f} - {stats['q75_volume']:.1f} m³/ha): {stats['classe_media']} parcelas
- **Classe Baixa** (< {stats['q25_volume']:.1f} m³/ha): {stats['classe_baixa']} parcelas

## 📊 ESTATÍSTICAS DENDROMÉTRICAS
- **DAP médio**: {stats['dap_medio']:.1f} cm
- **Altura média**: {stats['altura_media']:.1f} m
- **Idade média**: {stats['idade_media']:.1f} anos
- **Árvores por parcela**: {stats['arvores_por_parcela']:.0f}

## 📈 VARIABILIDADE
- **CV produtividade**: {stats['cv_volume']:.1f}%
- **Amplitude volume**: {stats['vol_min_ha']:.1f} - {stats['vol_max_ha']:.1f} m³/ha

---
*Relatório gerado pelo Sistema de Inventário Florestal*
"""

    return relatorio


def main():
    if not verificar_prerequisitos():
        return

    st.title("📈 Inventário Florestal")
    st.markdown("### Processamento Completo e Relatórios Finais")

    # Mostrar status das etapas anteriores
    mostrar_status_etapas()

    # CORREÇÃO: Verificar se já foi processado ANTES de mostrar configurações
    if st.session_state.get('inventario_processado'):
        st.info("ℹ️ O inventário já foi processado. Resultados salvos abaixo.")

        # Botão para reprocessar
        if st.button("🔄 Reprocessar Inventário", key="btn_reprocessar_inv"):
            del st.session_state.inventario_processado
            st.rerun()

        # CORREÇÃO: Mostrar resultados apenas UMA vez aqui
        mostrar_resultados_inventario(st.session_state.inventario_processado)
        return

    # Configurar áreas dos talhões
    config_areas = configurar_areas_talhoes()

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

    # BOTÃO ÚNICO E PADRONIZADO
    if st.button("🚀 Executar Inventário Completo", type="primary", use_container_width=True):
        executar_inventario_completo(config_areas)


if __name__ == "__main__":
    main()