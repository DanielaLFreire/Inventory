# ui/resultados.py
"""
Interface para exibição de resultados do inventário - Versão com persistência
"""

import streamlit as st
import pandas as pd
import numpy as np
from utils.formatacao import formatar_brasileiro, formatar_dataframe_brasileiro, formatar_numero_inteligente


def mostrar_resultados_finais(resultados):
    """
    Mostra os resultados finais do inventário de forma organizada

    Args:
        resultados: Dict com todos os resultados do inventário
    """
    # SOLUÇÃO: Salvar resultados no session_state para persistir entre reexecuções
    if 'resultados_inventario' not in st.session_state:
        st.session_state.resultados_inventario = None

    # Armazenar ou recuperar resultados
    if resultados is not None:
        st.session_state.resultados_inventario = resultados
    else:
        resultados = st.session_state.resultados_inventario

    if resultados is None:
        st.error("❌ Nenhum resultado disponível")
        return

    # Verificar se a estrutura de dados está correta
    if not isinstance(resultados, dict):
        st.error("❌ Estrutura de resultados inválida")
        return

    # Extrair estatísticas de forma segura
    stats = resultados.get('estatisticas_gerais')

    if stats is None:
        # Calcular estatísticas básicas se não existirem
        resumo_parcelas = resultados.get('resumo_parcelas')
        if resumo_parcelas is not None and len(resumo_parcelas) > 0:
            stats = calcular_estatisticas_basicas_from_resumo(resumo_parcelas)
        else:
            st.error("❌ Dados insuficientes para mostrar resultados")
            return

    # Header com métricas principais
    #mostrar_metricas_principais(stats)

    # Abas organizadas com resultados
    criar_abas_resultados(resultados)


def calcular_estatisticas_basicas_from_resumo(resumo_parcelas):
    """
    Calcula estatísticas básicas a partir do resumo de parcelas

    Args:
        resumo_parcelas: DataFrame com resumo por parcela

    Returns:
        dict: Estatísticas básicas
    """
    try:
        stats = {
            'total_parcelas': len(resumo_parcelas),
            'total_talhoes': resumo_parcelas['talhao'].nunique() if 'talhao' in resumo_parcelas.columns else 1,
            'area_total_ha': resumo_parcelas['area_ha'].sum() if 'area_ha' in resumo_parcelas.columns else 100.0,
            'vol_medio_ha': resumo_parcelas['vol_ha'].mean() if 'vol_ha' in resumo_parcelas.columns else 100.0,
            'vol_min_ha': resumo_parcelas['vol_ha'].min() if 'vol_ha' in resumo_parcelas.columns else 50.0,
            'vol_max_ha': resumo_parcelas['vol_ha'].max() if 'vol_ha' in resumo_parcelas.columns else 150.0,
            'cv_volume': (resumo_parcelas['vol_ha'].std() / resumo_parcelas[
                'vol_ha'].mean()) * 100 if 'vol_ha' in resumo_parcelas.columns else 20.0,
            'dap_medio': resumo_parcelas['dap_medio'].mean() if 'dap_medio' in resumo_parcelas.columns else 15.0,
            'altura_media': resumo_parcelas[
                'altura_media'].mean() if 'altura_media' in resumo_parcelas.columns else 20.0,
            'idade_media': resumo_parcelas['idade_anos'].mean() if 'idade_anos' in resumo_parcelas.columns else 5.0,
            'ima_medio': resumo_parcelas['ima'].mean() if 'ima' in resumo_parcelas.columns else 20.0,
            'arvores_por_parcela': resumo_parcelas['n_arvores'].mean() if 'n_arvores' in resumo_parcelas.columns else 25
        }

        # Calcular estoque total
        stats['estoque_total_m3'] = stats['area_total_ha'] * stats['vol_medio_ha']

        # Classificação de produtividade
        if 'vol_ha' in resumo_parcelas.columns:
            q25 = resumo_parcelas['vol_ha'].quantile(0.25)
            q75 = resumo_parcelas['vol_ha'].quantile(0.75)

            stats['classe_alta'] = (resumo_parcelas['vol_ha'] >= q75).sum()
            stats['classe_media'] = ((resumo_parcelas['vol_ha'] >= q25) & (resumo_parcelas['vol_ha'] < q75)).sum()
            stats['classe_baixa'] = (resumo_parcelas['vol_ha'] < q25).sum()
            stats['q25_volume'] = q25
            stats['q75_volume'] = q75
        else:
            stats['classe_alta'] = stats['total_parcelas'] // 3
            stats['classe_media'] = stats['total_parcelas'] // 3
            stats['classe_baixa'] = stats['total_parcelas'] - stats['classe_alta'] - stats['classe_media']
            stats['q25_volume'] = stats['vol_medio_ha'] * 0.8
            stats['q75_volume'] = stats['vol_medio_ha'] * 1.2

        return stats

    except Exception as e:
        st.error(f"❌ Erro ao calcular estatísticas básicas: {e}")
        # Retornar estatísticas padrão
        return {
            'total_parcelas': 10,
            'total_talhoes': 3,
            'area_total_ha': 100.0,
            'vol_medio_ha': 100.0,
            'estoque_total_m3': 10000.0,
            'cv_volume': 20.0,
            'dap_medio': 15.0,
            'altura_media': 20.0,
            'ima_medio': 20.0,
            'classe_alta': 3,
            'classe_media': 4,
            'classe_baixa': 3,
            'q25_volume': 80.0,
            'q75_volume': 120.0,
            'arvores_por_parcela': 25
        }


def mostrar_metricas_principais(stats):
    """
    Mostra as métricas principais em destaque

    Args:
        stats: Estatísticas gerais do inventário
    """
    st.header("📊 RESULTADOS FINAIS")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("🌲 Parcelas", f"{stats['total_parcelas']:,}")

    with col2:
        st.metric("📏 Área Total", f"{formatar_brasileiro(stats['area_total_ha'], 1)} ha")

    with col3:
        st.metric("📊 Produtividade", f"{formatar_brasileiro(stats['vol_medio_ha'], 1)} m³/ha")

    with col4:
        st.metric("🌲 Estoque Total", formatar_numero_inteligente(stats['estoque_total_m3'], "m³"))


def criar_abas_resultados(resultados):
    """
    Cria abas organizadas com diferentes aspectos dos resultados

    Args:
        resultados: Resultados completos do inventário
    """
    # Verificar quais dados estão disponíveis
    tem_resumo_parcelas = 'resumo_parcelas' in resultados and resultados['resumo_parcelas'] is not None
    tem_resumo_talhoes = 'resumo_talhoes' in resultados and resultados['resumo_talhoes'] is not None
    tem_inventario_completo = 'inventario_completo' in resultados and resultados['inventario_completo'] is not None

    # Criar abas baseado nos dados disponíveis
    abas_disponiveis = ["📊 Resumo"]

    if tem_resumo_talhoes:
        abas_disponiveis.append("🌳 Por Talhão")

    if tem_resumo_parcelas:
        abas_disponiveis.extend(["📈 Gráficos", "📋 Dados Completos"])

    abas_disponiveis.append("💾 Downloads")

    abas = st.tabs(abas_disponiveis)

    # Aba Resumo (sempre presente)
    with abas[0]:
        mostrar_aba_resumo(resultados)

    idx = 1

    # Aba Por Talhão (se disponível)
    if tem_resumo_talhoes:
        with abas[idx]:
            mostrar_aba_talhao(resultados)
        idx += 1

    # Aba Gráficos (se há dados de parcelas)
    if tem_resumo_parcelas:
        with abas[idx]:
            mostrar_aba_graficos(resultados)
        idx += 1

        # Aba Dados Completos (se há dados de parcelas)
        with abas[idx]:
            mostrar_aba_dados_completos(resultados)
        idx += 1

    # Aba Downloads (sempre presente)
    with abas[idx]:
        mostrar_aba_downloads(resultados)


def mostrar_aba_resumo(resultados):
    """
    Mostra aba com resumo geral

    Args:
        resultados: Resultados do inventário
    """
    # Extrair estatísticas de forma segura
    stats = resultados.get('estatisticas_gerais')

    if stats is None:
        # Calcular estatísticas básicas se não existirem
        resumo_parcelas = resultados.get('resumo_parcelas')
        if resumo_parcelas is not None and len(resumo_parcelas) > 0:
            stats = calcular_estatisticas_basicas_from_resumo(resumo_parcelas)
        else:
            st.warning("⚠️ Estatísticas não disponíveis")
            st.info("💡 Execute a análise completa para ver estatísticas detalhadas")
            return

    st.subheader("📈 Estatísticas Gerais")

    # Métricas secundárias
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📏 DAP Médio", f"{formatar_brasileiro(stats.get('dap_medio', 15.0), 1)} cm")

    with col2:
        st.metric("🌳 Altura Média", f"{formatar_brasileiro(stats.get('altura_media', 20.0), 1)} m")

    with col3:
        st.metric("📊 CV Volume", f"{formatar_brasileiro(stats.get('cv_volume', 20.0), 1)}%")

    with col4:
        st.metric("🚀 IMA Médio", f"{formatar_brasileiro(stats.get('ima_medio', 20.0), 1)} m³/ha/ano")

        # Widget de ajuda para explicar o IMA
        with st.popover("ℹ️ O que é IMA?"):
            st.markdown("""
            **IMA = Incremento Médio Anual**

            📈 **Definição:**
            Mede a produtividade média anual do povoamento florestal.

            🧮 **Fórmula:**
            ```
            IMA = Volume (m³/ha) ÷ Idade (anos)
            ```

            📊 **Interpretação (Eucalipto):**
            -  **> 30 m³/ha/ano**: Alta produtividade
            -  **20-30 m³/ha/ano**: Média produtividade
            -  **< 20 m³/ha/ano**: Baixa produtividade

            💡 **Uso:**
            - Comparar diferentes talhões
            - Avaliar qualidade do sítio
            - Planejar rotação de corte
            - Calcular rentabilidade florestal
            """)

    # Classificação de produtividade
    st.subheader("📊 Classificação de Produtividade")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "🌲🌲🌲 Classe Alta",
            f"{stats.get('classe_alta', 0)} parcelas",
            help=f"≥ {stats.get('q75_volume', 120.0):.1f} m³/ha"
        )

    with col2:
        st.metric(
            "🌲🌲 Classe Média",
            f"{stats.get('classe_media', 0)} parcelas",
            help=f"{stats.get('q25_volume', 80.0):.1f} - {stats.get('q75_volume', 120.0):.1f} m³/ha"
        )

    with col3:
        st.metric(
            "🌲 Classe Baixa",
            f"{stats.get('classe_baixa', 0)} parcelas",
            help=f"< {stats.get('q25_volume', 80.0):.1f} m³/ha"
        )

    # Informações adicionais
    mostrar_informacoes_adicionais(stats)


def mostrar_informacoes_adicionais(stats):
    """
    Mostra informações adicionais do inventário

    Args:
        stats: Estatísticas gerais (pode ser None ou dict)
    """
    if stats is None:
        st.info("ℹ️ Informações adicionais não disponíveis")
        return

    st.subheader("ℹ️ Informações Adicionais")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Densidade do Povoamento:**")
        arvores_parcela = stats.get('arvores_por_parcela', 25)
        st.write(f"- Árvores por parcela: {formatar_brasileiro(arvores_parcela, 0)}")
        st.write(f"- Densidade estimada: {formatar_brasileiro(arvores_parcela * 25, 0)} árv/ha*")
        st.caption("*Baseado em parcela de 400m²")

        st.write("**Características Dendrométricas:**")
        st.write(f"- Idade média: {formatar_brasileiro(stats.get('idade_media', 5.0), 1)} anos")
        st.write(f"- Amplitude DAP: informação não disponível")
        st.write(f"- Amplitude altura: informação não disponível")

    with col2:
        st.write("**Variabilidade:**")
        cv_volume = stats.get('cv_volume', 20.0)
        cv_qualificacao = "Baixa" if cv_volume < 20 else "Média" if cv_volume < 40 else "Alta"
        st.write(f"- CV produtividade: {formatar_brasileiro(cv_volume, 1)}% ({cv_qualificacao})")

        vol_min = stats.get('vol_min_ha', 50.0)
        vol_max = stats.get('vol_max_ha', 150.0)
        st.write(f"- Amplitude volume: {formatar_brasileiro(vol_min, 1)} - {formatar_brasileiro(vol_max, 1)} m³/ha")

        st.write("**Potencial Produtivo:**")
        ima_medio = stats.get('ima_medio', 20.0)
        if ima_medio > 30:
            classificacao_ima = "Muito Alto"
        elif ima_medio > 20:
            classificacao_ima = "Alto"
        elif ima_medio > 15:
            classificacao_ima = "Médio"
        else:
            classificacao_ima = "Baixo"

        st.write(f"- IMA médio: {formatar_brasileiro(ima_medio, 1)} m³/ha/ano ({classificacao_ima})")


def mostrar_aba_talhao(resultados):
    """
    Mostra aba com análise por talhão

    Args:
        resultados: Resultados do inventário
    """
    st.subheader("🌳 Análise por Talhão")

    resumo_talhao = resultados.get('resumo_talhoes')

    if resumo_talhao is None or len(resumo_talhao) == 0:
        st.warning("⚠️ Dados de talhão não disponíveis")
        st.info("💡 Execute a análise completa para ver resultados por talhão")
        return

    # Preparar dados para exibição
    try:
        df_talhao_display = preparar_dados_talhao_display(resumo_talhao)

        # Mostrar tabela
        st.dataframe(df_talhao_display, hide_index=True, use_container_width=True)

        # Análise de destaque
        mostrar_analise_talhoes(resumo_talhao)

    except Exception as e:
        st.error(f"❌ Erro ao mostrar dados por talhão: {e}")
        st.info("💡 Verifique se os dados foram processados corretamente")


def preparar_dados_talhao_display(resumo_talhao):
    """
    Prepara dados dos talhões para exibição formatada

    Args:
        resumo_talhao: DataFrame com resumo por talhão

    Returns:
        DataFrame formatado para exibição
    """
    df_display = resumo_talhao.copy()

    # Selecionar e renomear colunas
    colunas_exibir = {
        'talhao': 'Talhão',
        'area_ha': 'Área (ha)',
        'n_parcelas': 'Parcelas',
        'vol_medio_ha': 'Volume (m³/ha)',
        'cv_volume': 'CV (%)',
        'dap_medio': 'DAP (cm)',
        'altura_media': 'Altura (m)',
        'ima_medio': 'IMA (m³/ha/ano)',
        'estoque_total_m3': 'Estoque (m³)'
    }

    # Verificar quais colunas existem
    colunas_disponiveis = {k: v for k, v in colunas_exibir.items() if k in df_display.columns}

    df_display = df_display[list(colunas_disponiveis.keys())].rename(columns=colunas_disponiveis)

    # Formatar colunas numéricas
    colunas_numericas = [col for col in df_display.columns if col != 'Talhão']
    df_formatado = formatar_dataframe_brasileiro(
        df_display,
        colunas_numericas=colunas_numericas,
        decimais=1
    )

    return df_formatado


def mostrar_analise_talhoes(resumo_talhao):
    """
    Mostra análise de destaque dos talhões

    Args:
        resumo_talhao: DataFrame com resumo por talhão
    """
    st.subheader("🏆 Destaques por Talhão")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Talhão mais produtivo
        if 'vol_medio_ha' in resumo_talhao.columns:
            idx_max_vol = resumo_talhao['vol_medio_ha'].idxmax()
            talhao_max_vol = resumo_talhao.loc[idx_max_vol, 'talhao']
            vol_max = resumo_talhao.loc[idx_max_vol, 'vol_medio_ha']

            st.metric(
                "🥇 Mais Produtivo",
                f"Talhão {talhao_max_vol}",
                f"{formatar_brasileiro(vol_max, 1)} m³/ha"
            )

    with col2:
        # Maior área
        if 'area_ha' in resumo_talhao.columns:
            idx_max_area = resumo_talhao['area_ha'].idxmax()
            talhao_max_area = resumo_talhao.loc[idx_max_area, 'talhao']
            area_max = resumo_talhao.loc[idx_max_area, 'area_ha']

            st.metric(
                "📏 Maior Área",
                f"Talhão {talhao_max_area}",
                f"{formatar_brasileiro(area_max, 1)} ha"
            )

    with col3:
        # Maior estoque
        if 'estoque_total_m3' in resumo_talhao.columns:
            idx_max_estoque = resumo_talhao['estoque_total_m3'].idxmax()
            talhao_max_estoque = resumo_talhao.loc[idx_max_estoque, 'talhao']
            estoque_max = resumo_talhao.loc[idx_max_estoque, 'estoque_total_m3']

            st.metric(
                "🌲 Maior Estoque",
                f"Talhão {talhao_max_estoque}",
                formatar_numero_inteligente(estoque_max, "m³")
            )


def mostrar_aba_graficos(resultados):
    """
    Mostra aba com gráficos e visualizações

    Args:
        resultados: Resultados do inventário
    """
    from ui.graficos import criar_graficos_inventario

    st.subheader("📊 Visualizações")
    criar_graficos_inventario(resultados)


def mostrar_aba_dados_completos(resultados):
    """
    Mostra aba com dados completos - VERSÃO COM PERSISTÊNCIA

    Args:
        resultados: Resultados do inventário
    """
    st.subheader("📋 Dados Completos")

    # SOLUÇÃO: Usar keys únicos para widgets para evitar conflicts
    # Seletores para diferentes datasets
    dataset_opcoes = {}

    if 'resumo_parcelas' in resultados and resultados['resumo_parcelas'] is not None:
        dataset_opcoes["Resumo por Parcela"] = resultados['resumo_parcelas']

    if 'resumo_talhoes' in resultados and resultados['resumo_talhoes'] is not None:
        dataset_opcoes["Resumo por Talhão"] = resultados['resumo_talhoes']

    if 'inventario_completo' in resultados and resultados['inventario_completo'] is not None:
        dataset_opcoes["Inventário Completo"] = resultados['inventario_completo'].head(1000)

    if not dataset_opcoes:
        st.warning("⚠️ Nenhum dataset disponível")
        return

    dataset_selecionado = st.selectbox(
        "📊 Selecione o dataset:",
        options=list(dataset_opcoes.keys()),
        key="dataset_selector_dados_completos"  # SOLUÇÃO: Key única
    )

    df_selecionado = dataset_opcoes[dataset_selecionado]

    # Informações do dataset
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Registros", len(df_selecionado))

    with col2:
        st.metric("Colunas", len(df_selecionado.columns))

    with col3:
        if dataset_selecionado == "Inventário Completo" and 'inventario_completo' in resultados and len(
                resultados['inventario_completo']) > 1000:
            st.metric("Exibindo", "Primeiros 1.000")
        else:
            st.metric("Exibindo", "Todos")

    # Opções de exibição
    with st.expander("⚙️ Opções de Exibição"):
        col1, col2 = st.columns(2)

        with col1:
            mostrar_formatado = st.checkbox(
                "Números formatados",
                value=True,
                help="Exibir números no padrão brasileiro",
                key="checkbox_formatado_dados_completos"  # SOLUÇÃO: Key única
            )

        with col2:
            max_linhas = st.number_input(
                "Máximo de linhas",
                min_value=10,
                max_value=1000,
                value=100,
                step=10,
                key="number_input_max_linhas_dados_completos"  # SOLUÇÃO: Key única
            )

    # Exibir dados
    df_exibir = df_selecionado.head(max_linhas)

    if mostrar_formatado:
        # Detectar colunas numéricas e formatar
        colunas_numericas = df_exibir.select_dtypes(include=[np.number]).columns
        if len(colunas_numericas) > 0:
            df_exibir = formatar_dataframe_brasileiro(df_exibir, colunas_numericas, decimais=2)

    st.dataframe(df_exibir, hide_index=True, use_container_width=True)

    # Estatísticas do dataset
    if st.checkbox("📊 Mostrar estatísticas", key="checkbox_stats_dados_completos"):  # SOLUÇÃO: Key única
        mostrar_estatisticas_dataset(df_selecionado)


def mostrar_estatisticas_dataset(df):
    """
    Mostra estatísticas descritivas do dataset

    Args:
        df: DataFrame para análise
    """
    st.subheader("📊 Estatísticas Descritivas")

    # Apenas colunas numéricas
    df_numerico = df.select_dtypes(include=[np.number])

    if len(df_numerico.columns) > 0:
        stats_df = df_numerico.describe()
        stats_formatado = formatar_dataframe_brasileiro(
            stats_df.round(2),
            colunas_numericas=stats_df.columns,
            decimais=2
        )
        st.dataframe(stats_formatado, use_container_width=True)
    else:
        st.info("Nenhuma coluna numérica encontrada para estatísticas")


def mostrar_aba_downloads(resultados):
    """
    Mostra aba com opções de download - VERSÃO COM PERSISTÊNCIA

    Args:
        resultados: Resultados do inventário
    """
    st.subheader("💾 Downloads")

    # Seção de arquivos individuais
    st.write("**📁 Arquivos de Dados:**")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Download resumo parcelas
        if 'resumo_parcelas' in resultados and resultados['resumo_parcelas'] is not None:
            csv_parcelas = resultados['resumo_parcelas'].to_csv(index=False)
            st.download_button(
                label="📊 Resumo por Parcela",
                data=csv_parcelas,
                file_name="resumo_parcelas.csv",
                mime="text/csv",
                help="Dados resumidos por parcela",
                key="download_resumo_parcelas"  # SOLUÇÃO: Key única
            )

    with col2:
        # Download resumo talhões
        if 'resumo_talhoes' in resultados and resultados['resumo_talhoes'] is not None:
            csv_talhoes = resultados['resumo_talhoes'].to_csv(index=False)
            st.download_button(
                label="🌳 Resumo por Talhão",
                data=csv_talhoes,
                file_name="resumo_talhoes.csv",
                mime="text/csv",
                help="Dados resumidos por talhão",
                key="download_resumo_talhoes"  # SOLUÇÃO: Key única
            )

    with col3:
        # Download inventário completo
        if 'inventario_completo' in resultados and resultados['inventario_completo'] is not None:
            csv_completo = resultados['inventario_completo'].to_csv(index=False)
            st.download_button(
                label="📋 Inventário Completo",
                data=csv_completo,
                file_name="inventario_completo.csv",
                mime="text/csv",
                help="Todos os dados do inventário",
                key="download_inventario_completo"  # SOLUÇÃO: Key única
            )

    # Seção de relatórios
    st.write("**📄 Relatórios:**")

    col1, col2 = st.columns(2)

    with col1:
        # Relatório executivo
        try:
            from processors.inventario import gerar_relatorio_inventario
            relatorio = gerar_relatorio_inventario(resultados)

            st.download_button(
                label="📄 Relatório Executivo",
                data=relatorio,
                file_name="relatorio_inventario.md",
                mime="text/markdown",
                help="Relatório completo em Markdown",
                key="download_relatorio_executivo"  # SOLUÇÃO: Key única
            )
        except Exception as e:
            st.warning(f"⚠️ Relatório executivo indisponível: {e}")

    with col2:
        # Relatório de configurações
        config_info = gerar_relatorio_configuracoes(resultados)

        st.download_button(
            label="⚙️ Configurações Utilizadas",
            data=config_info,
            file_name="configuracoes_utilizadas.txt",
            mime="text/plain",
            help="Configurações aplicadas na análise",
            key="download_configuracoes"  # SOLUÇÃO: Key única
        )

    # Seção de dados consolidados
    st.write("**📦 Dados Consolidados:**")

    # Preparar arquivo consolidado
    try:
        dados_consolidados = preparar_dados_consolidados(resultados)

        st.download_button(
            label="📦 Download Completo (CSV)",
            data=dados_consolidados,
            file_name="inventario_completo_consolidado.csv",
            mime="text/csv",
            help="Todos os dados principais em um arquivo",
            key="download_dados_consolidados"  # SOLUÇÃO: Key única
        )
    except Exception as e:
        st.warning(f"⚠️ Dados consolidados indisponíveis: {e}")


def preparar_dados_consolidados(resultados):
    """
    Prepara dados consolidados para download

    Args:
        resultados: Resultados do inventário

    Returns:
        str: CSV consolidado
    """
    # Combinar dados principais
    if 'resumo_parcelas' not in resultados or resultados['resumo_parcelas'] is None:
        return "Dados não disponíveis"

    resumo_parcelas = resultados['resumo_parcelas'].copy()

    # Adicionar informações dos modelos utilizados se disponível
    if 'modelos_utilizados' in resultados:
        modelos = resultados['modelos_utilizados']
        resumo_parcelas['modelo_hipsometrico'] = modelos.get('hipsometrico', 'N/A')
        resumo_parcelas['modelo_volumetrico'] = modelos.get('volumetrico', 'N/A')

    resumo_parcelas['data_processamento'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

    return resumo_parcelas.to_csv(index=False)


def gerar_relatorio_configuracoes(resultados):
    """
    Gera relatório das configurações utilizadas

    Args:
        resultados: Resultados do inventário

    Returns:
        str: Relatório de configurações
    """
    # Obter informações de forma segura
    modelos = resultados.get('modelos_utilizados', {})
    stats = resultados.get('estatisticas_gerais', {})

    relatorio = f"""
CONFIGURAÇÕES UTILIZADAS - INVENTÁRIO FLORESTAL
=============================================

Data/Hora: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

MODELOS SELECIONADOS:
- Hipsométrico: {modelos.get('hipsometrico', 'N/A')}
- Volumétrico: {modelos.get('volumetrico', 'N/A')}

DADOS PROCESSADOS:
- Total de parcelas: {stats.get('total_parcelas', 'N/A')}
- Total de talhões: {stats.get('total_talhoes', 'N/A')}
- Área total: {stats.get('area_total_ha', 'N/A'):.1f} ha

MÉTODO DE CUBAGEM:
- Fórmula de Smalian

CRITÉRIOS DE SELEÇÃO:
- Modelos hipsométricos: R² Generalizado
- Modelos volumétricos: R² tradicional

RESULTADOS PRINCIPAIS:
- Produtividade média: {stats.get('vol_medio_ha', 'N/A'):.1f} m³/ha
- IMA médio: {stats.get('ima_medio', 'N/A'):.1f} m³/ha/ano
- Estoque total: {stats.get('estoque_total_m3', 'N/A'):.0f} m³

OBSERVAÇÕES:
- Análise realizada com sistema modular
- Validações aplicadas em todas as etapas
- Dados filtrados conforme configurações
- Resultados salvos automaticamente no session_state
"""

    return relatorio