# pages/4_🛩️_Dados_LiDAR.py - VERSÃO INTEGRADA COMPLETA
"""
Etapa 4: Integração com Dados LiDAR - VERSÃO COMPLETA
Página para integração e análise de dados LiDAR com inventário florestal
NOVA FUNCIONALIDADE: Processamento direto de arquivos LAS/LAZ
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import traceback

# Importar processadores LiDAR originais
from processors.lidar import (
    processar_dados_lidar,
    integrar_dados_lidar_inventario,
    comparar_alturas_campo_lidar,
    calibrar_modelo_hipsometrico_com_lidar,
    analisar_estrutura_florestal_lidar,
    gerar_alertas_automaticos_lidar,
    calcular_metricas_validacao_lidar,
    exportar_dados_integrados,
    gerar_relatorio_integracao_lidar
)

# Importar NOVO processador LAS integrado
from processors.las_processor_integrado import (
    ProcessadorLASIntegrado,
    processar_las_com_interface,
    mostrar_resultados_processamento_las,
    criar_interface_processamento_las,
    integrar_com_pagina_lidar
)

# Importar configurações centralizadas
from config.configuracoes_globais import (
    obter_configuracao_global,
    mostrar_status_configuracao_sidebar
)

from config.config import MENSAGENS_AJUDA_LIDAR, CORES_LIDAR

# Importar utilitários
from utils.formatacao import formatar_brasileiro, formatar_numero_inteligente

# Importar componentes de UI para manter identidade visual
from ui.components import (
    configurar_pagina_greenvista,
    criar_cabecalho_greenvista,
    criar_navegacao_rapida_botoes
)

# Configurar página com identidade visual
configurar_pagina_greenvista("Dados LiDAR", "🛩️")


def gerar_key_unica(base_key):
    """Gera uma key única para evitar conflitos"""
    timestamp = int(time.time() * 1000)
    return f"{base_key}_{timestamp}"


def verificar_prerequisitos():
    """Verifica se pré-requisitos estão atendidos"""
    problemas = []

    # Verificar dados básicos carregados
    if not hasattr(st.session_state, 'dados_inventario') or st.session_state.dados_inventario is None:
        problemas.append("Dados de inventário não disponíveis")

    # Verificar configuração global
    try:
        config_global = obter_configuracao_global()
        if not config_global.get('configurado', False):
            problemas.append("Sistema não configurado")
    except:
        problemas.append("Erro ao verificar configurações")

    if problemas:
        st.error("❌ Pré-requisitos não atendidos:")
        for problema in problemas:
            st.error(f"• {problema}")

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🏠 Página Principal", key="btn_principal_lidar"):
                st.switch_page("Principal.py")
        with col2:
            if st.button("⚙️ Configurações", key="btn_config_lidar"):
                st.switch_page("pages/0_⚙️_Configurações.py")
        with col3:
            if st.button("📈 Inventário", key="btn_inv_lidar"):
                st.switch_page("pages/3_📈_Inventário_Florestal.py")

        return False

    return True


def mostrar_introducao_lidar():
    """Mostra introdução sobre integração LiDAR"""
    st.markdown("### Análise Integrada: Campo + Sensoriamento Remoto")

    with st.expander("ℹ️ Sobre a Integração LiDAR"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **🛩️ O que é LiDAR:**
            - Light Detection and Ranging
            - Mapeamento 3D da floresta
            - Precisão centimétrica
            - Cobertura completa da área

            **📊 Métricas Principais:**
            - Altura média (zmean)
            - Altura máxima (zmax)
            - Variabilidade estrutural (zsd)
            - Cobertura do dossel
            """)

        with col2:
            st.markdown("""
            **🎯 Benefícios da Integração:**
            - ✅ Validação de modelos de campo
            - ✅ Calibração automática
            - ✅ Detecção de outliers
            - ✅ Mapeamento contínuo
            - ✅ Análise estrutural avançada

            **🔗 Fluxo de Trabalho:**
            1. Upload arquivo LAS/LAZ OU métricas CSV
            2. Processamento automático 
            3. Integração com inventário
            4. Análise comparativa
            """)


def mostrar_selecao_metodo_lidar():
    """Interface para seleção do método de processamento LiDAR"""
    st.header("📁 Dados LiDAR - Escolha o Método")

    # Verificar disponibilidade do processamento LAS
    processamento_las_disponivel = integrar_com_pagina_lidar()

    # Tabs para diferentes métodos
    if processamento_las_disponivel:
        tab1, tab2 = st.tabs([
            "🛩️ Processar Arquivo LAS/LAZ",
            "📊 Upload Métricas Processadas"
        ])
    else:
        tab1 = None
        tab2 = st.container()
        st.warning("⚠️ Processamento LAS não disponível - use métricas pré-processadas")

    return tab1, tab2, processamento_las_disponivel


def processar_metodo_las(tab_las):
    """Processamento direto de arquivos LAS/LAZ"""
    with tab_las:
        st.subheader("🛩️ Processamento Direto LAS/LAZ")

        # Interface de processamento
        criar_interface_processamento_las()

        # Upload do arquivo LAS
        arquivo_las = st.file_uploader(
            "📁 Arquivo LAS/LAZ",
            type=['las', 'laz'],
            help="Carregue arquivo LAS ou LAZ (máximo 500MB)",
            key="upload_las_file"
        )

        if arquivo_las is not None:
            st.success(f"✅ Arquivo carregado: {arquivo_las.name}")

            # Obter dados do inventário se disponível
            dados_inventario = getattr(st.session_state, 'dados_inventario', None)

            if dados_inventario is not None:
                st.info(f"📋 Inventário disponível: {len(dados_inventario)} registros")

                # Verificar se inventário tem coordenadas
                tem_coordenadas = 'x' in dados_inventario.columns and 'y' in dados_inventario.columns

                if tem_coordenadas:
                    st.success("📍 Inventário com coordenadas - processamento preciso")
                else:
                    st.warning("⚠️ Inventário sem coordenadas - estimativa automática")
            else:
                st.warning("⚠️ Sem dados de inventário - criação de grid automático")
                dados_inventario = None

            # Processar arquivo LAS
            resultado_las = processar_las_com_interface(arquivo_las, dados_inventario)

            if resultado_las is not None:
                # Salvar no session_state
                st.session_state.dados_lidar_las = {
                    'df_metricas': resultado_las,
                    'metodo': 'processamento_las',
                    'arquivo_original': arquivo_las.name,
                    'timestamp': pd.Timestamp.now()
                }

                # Mostrar resultados
                mostrar_resultados_processamento_las(resultado_las)

                return resultado_las

        return None


def processar_metodo_metricas(tab_metricas):
    """Upload de métricas LiDAR pré-processadas"""
    with tab_metricas:
        st.subheader("📊 Upload de Métricas LiDAR")

        # Upload do arquivo
        arquivo_lidar = st.file_uploader(
            "📊 Arquivo de Métricas LiDAR",
            type=['csv', 'xlsx', 'xls'],
            help=MENSAGENS_AJUDA_LIDAR['upload'],
            key="upload_lidar_metrics"
        )

        if arquivo_lidar is not None:
            return processar_e_integrar_lidar_metricas(arquivo_lidar)

        return None


def processar_e_integrar_lidar_metricas(arquivo_lidar):
    """Processa e integra dados LiDAR de métricas pré-processadas"""
    try:
        # Processar dados LiDAR
        with st.spinner("🔄 Processando dados LiDAR..."):
            df_lidar = processar_dados_lidar(arquivo_lidar)

        if df_lidar is None:
            return None, None, None

        # Mostrar preview dos dados LiDAR
        with st.expander("👀 Preview dos Dados LiDAR"):
            st.dataframe(df_lidar.head())

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Parcelas LiDAR", len(df_lidar))
            with col2:
                st.metric("Métricas", len([col for col in df_lidar.columns if col not in ['talhao', 'parcela']]))
            with col3:
                st.metric("Talhões", df_lidar['talhao'].nunique())

        # Integrar com dados do inventário
        with st.spinner("🔗 Integrando com dados de campo..."):
            df_integrado = integrar_dados_lidar_inventario(
                st.session_state.dados_inventario,
                df_lidar
            )

        # Comparar alturas campo vs LiDAR
        with st.spinner("📊 Comparando alturas campo vs LiDAR..."):
            stats_comparacao = comparar_alturas_campo_lidar(df_integrado)

        # Gerar alertas automáticos
        alertas = gerar_alertas_automaticos_lidar(df_integrado, stats_comparacao)

        # Salvar resultados no session_state
        st.session_state.dados_lidar = {
            'df_lidar': df_lidar,
            'df_integrado': df_integrado,
            'stats_comparacao': stats_comparacao,
            'alertas': alertas,
            'metodo': 'metricas_processadas',
            'timestamp': pd.Timestamp.now()
        }

        return df_integrado, stats_comparacao, alertas

    except Exception as e:
        st.error(f"❌ Erro no processamento: {e}")
        with st.expander("🔍 Debug Detalhado"):
            st.code(traceback.format_exc())
        return None, None, None


def mostrar_resultados_comparacao(stats_comparacao):
    """Mostra resultados da comparação campo vs LiDAR"""
    if stats_comparacao is None:
        st.warning("⚠️ Comparação não disponível - dados insuficientes")
        return

    st.header("📊 Comparação Campo vs LiDAR")

    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        correlacao = stats_comparacao['correlacao']
        st.metric(
            "🔗 Correlação",
            f"{correlacao:.3f}",
            help="Correlação entre alturas de campo e LiDAR"
        )

        if correlacao >= 0.8:
            st.success("Correlação excelente")
        elif correlacao >= 0.6:
            st.info("Correlação boa")
        else:
            st.warning("Correlação baixa")

    with col2:
        r2 = stats_comparacao['r2']
        st.metric(
            "📈 R²",
            f"{r2:.3f}",
            help="Coeficiente de determinação da regressão"
        )

    with col3:
        rmse = stats_comparacao['rmse']
        st.metric(
            "📏 RMSE",
            f"{rmse:.2f} m",
            help="Raiz do erro quadrático médio"
        )

    with col4:
        bias = stats_comparacao['diferenca_media']
        st.metric(
            "⚖️ Bias",
            f"{bias:+.2f} m",
            help="Diferença média (Campo - LiDAR)"
        )

    # Gráficos de comparação
    mostrar_graficos_comparacao(stats_comparacao)

    # Outliers detectados
    if len(stats_comparacao['outliers']) > 0:
        mostrar_outliers_detectados(stats_comparacao['outliers'])


def mostrar_graficos_comparacao(stats_comparacao):
    """Mostra gráficos da comparação campo vs LiDAR"""
    st.subheader("📊 Gráficos de Comparação")

    df_comp = stats_comparacao['dados_comparacao']

    col1, col2 = st.columns(2)

    with col1:
        # Gráfico de dispersão
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.scatter(df_comp['altura_media'], df_comp['H_m'],
                   alpha=0.6, color=CORES_LIDAR['campo'])

        # Linha 1:1
        min_val = min(df_comp['altura_media'].min(), df_comp['H_m'].min())
        max_val = max(df_comp['altura_media'].max(), df_comp['H_m'].max())
        ax.plot([min_val, max_val], [min_val, max_val],
                'r--', linewidth=2, label='Linha 1:1')

        # Linha de regressão
        x_reg = np.linspace(min_val, max_val, 100)
        y_reg = stats_comparacao['intercepto'] + stats_comparacao['coeficiente_angular'] * x_reg
        ax.plot(x_reg, y_reg, color=CORES_LIDAR['calibrado'],
                linewidth=2, label='Regressão')

        ax.set_xlabel('Altura LiDAR (m)')
        ax.set_ylabel('Altura Campo (m)')
        ax.set_title(f'Campo vs LiDAR (R² = {stats_comparacao["r2"]:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)
        plt.close(fig)

    with col2:
        # Histograma das diferenças
        fig, ax = plt.subplots(figsize=(8, 6))

        diferencas = df_comp['H_m'] - df_comp['altura_media']

        ax.hist(diferencas, bins=20, alpha=0.7,
                color=CORES_LIDAR['residuos'], edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
        ax.axvline(x=diferencas.mean(), color='blue', linestyle='-',
                   linewidth=2, label=f'Média: {diferencas.mean():.2f}m')

        ax.set_xlabel('Diferença Campo - LiDAR (m)')
        ax.set_ylabel('Frequência')
        ax.set_title('Distribuição das Diferenças')
        ax.legend()
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)
        plt.close(fig)


def mostrar_outliers_detectados(outliers):
    """Mostra outliers detectados na comparação"""
    st.subheader("⚠️ Outliers Detectados")

    st.info(f"📊 {len(outliers)} parcelas com grandes diferenças campo-LiDAR")

    # Preparar dados para exibição
    df_outliers = outliers[['talhao', 'parcela', 'H_m', 'altura_media', 'diferenca_absoluta']].copy()
    df_outliers['diferenca_relativa'] = ((df_outliers['H_m'] - df_outliers['altura_media']) / df_outliers[
        'altura_media']) * 100

    # Renomear colunas
    df_outliers = df_outliers.rename(columns={
        'H_m': 'Altura Campo (m)',
        'altura_media': 'Altura LiDAR (m)',
        'diferenca_absoluta': 'Diferença Abs (m)',
        'diferenca_relativa': 'Diferença Rel (%)'
    })

    st.dataframe(df_outliers, hide_index=True, use_container_width=True)

    if st.button("📥 Download Outliers", key="download_outliers"):
        csv_outliers = df_outliers.to_csv(index=False)
        st.download_button(
            "📄 Baixar Lista de Outliers",
            csv_outliers,
            "outliers_campo_lidar.csv",
            "text/csv",
            key=gerar_key_unica("btn_download_outliers")
        )


def mostrar_analise_estrutural(df_integrado):
    """Mostra análise estrutural baseada em LiDAR - VERSÃO CORRIGIDA"""
    st.header("🌲 Análise Estrutural Florestal")

    if 'altura_media' not in df_integrado.columns:
        st.warning("⚠️ Dados LiDAR não disponíveis para análise estrutural")
        return

    # Métricas estruturais gerais
    dados_estruturais = df_integrado.dropna(subset=['altura_media'])

    if len(dados_estruturais) == 0:
        st.error("❌ Nenhum dado estrutural válido encontrado")
        return

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        altura_media = dados_estruturais['altura_media'].mean()
        st.metric("🌳 Altura Média", f"{altura_media:.1f} m")

    with col2:
        if 'desvio_altura' in dados_estruturais.columns:
            variabilidade_media = dados_estruturais['desvio_altura'].mean()
            st.metric("📊 Variabilidade", f"{variabilidade_media:.1f} m")
        else:
            variabilidade_calc = dados_estruturais['altura_media'].std()
            st.metric("📊 Desvio Padrão", f"{variabilidade_calc:.1f} m")

    with col3:
        if 'cobertura' in dados_estruturais.columns:
            cobertura_media = dados_estruturais['cobertura'].mean()
            st.metric("🍃 Cobertura", f"{cobertura_media:.1f}%")
        else:
            st.metric("🍃 Cobertura", "N/A")

    with col4:
        if 'complexidade' in dados_estruturais.columns:
            complexidade_media = dados_estruturais['complexidade'].mean()
            st.metric("🔀 Complexidade", f"{complexidade_media:.2f}")
        else:
            st.metric("🔀 Complexidade", "N/A")

    # Análise por talhão
    mostrar_analise_por_talhao(dados_estruturais)

    # Gráficos estruturais - usar a função corrigida
    mostrar_graficos_estruturais(dados_estruturais)


def mostrar_analise_por_talhao(dados_estruturais):
    """Mostra análise estrutural por talhão - VERSÃO CORRIGIDA"""
    st.subheader("📊 Análise por Talhão")

    try:
        # Verificar se há dados suficientes
        if 'talhao' not in dados_estruturais.columns:
            st.warning("⚠️ Coluna 'talhao' não encontrada")
            return

        # Calcular estatísticas por talhão
        agg_dict = {
            'altura_media': ['mean', 'std', 'count']
        }

        # Adicionar outras métricas se disponíveis
        if 'desvio_altura' in dados_estruturais.columns:
            agg_dict['desvio_altura'] = 'mean'
        if 'cobertura' in dados_estruturais.columns:
            agg_dict['cobertura'] = 'mean'

        stats_talhao = dados_estruturais.groupby('talhao').agg(agg_dict).round(2)

        # Achatar colunas multi-nível
        if hasattr(stats_talhao.columns, 'levels'):
            stats_talhao.columns = [f"{col[0]}_{col[1]}" if len(col) > 1 and col[1] != '' else col[0]
                                    for col in stats_talhao.columns]

        stats_talhao = stats_talhao.reset_index()

        # Renomear para exibição
        rename_map = {
            'altura_media_mean': 'Altura Média (m)',
            'altura_media_std': 'Desvio Padrão (m)',
            'altura_media_count': 'N° Parcelas'
        }

        if 'desvio_altura' in stats_talhao.columns:
            rename_map['desvio_altura'] = 'Variabilidade (m)'
        if 'cobertura' in stats_talhao.columns:
            rename_map['cobertura'] = 'Cobertura (%)'

        stats_display = stats_talhao.rename(columns=rename_map)

        # Filtrar apenas colunas existentes
        colunas_existentes = [col for col in rename_map.values() if col in stats_display.columns]
        colunas_mostrar = ['talhao'] + colunas_existentes

        st.dataframe(stats_display[colunas_mostrar], hide_index=True, use_container_width=True)

        # Mostrar resumo estatístico
        st.info(
            f"📊 **Resumo:** {len(stats_display)} talhões analisados com {dados_estruturais['altura_media'].count()} parcelas no total")

    except Exception as e:
        st.error(f"❌ Erro na análise por talhão: {str(e)}")
        st.info("Mostrando dados brutos disponíveis:")
        st.dataframe(dados_estruturais.head(), use_container_width=True)


def mostrar_graficos_estruturais(dados_estruturais):
    """Mostra gráficos da análise estrutural - VERSÃO CORRIGIDA"""
    st.subheader("📊 Distribuições Estruturais")

    # Verificar se há dados suficientes
    if len(dados_estruturais) == 0 or 'altura_media' not in dados_estruturais.columns:
        st.warning("⚠️ Dados insuficientes para gráficos estruturais")
        return

    # Filtrar dados válidos
    dados_estruturais = dados_estruturais.dropna(subset=['altura_media'])
    if len(dados_estruturais) == 0:
        st.warning("⚠️ Nenhum dado de altura válido encontrado")
        return

    col1, col2 = st.columns(2)

    with col1:
        # Distribuição de alturas por talhão - CORRIGIDO
        fig, ax = plt.subplots(figsize=(10, 6))

        try:
            talhoes = sorted(dados_estruturais['talhao'].unique())

            # Preparar dados do boxplot - filtrar apenas talhões com dados
            dados_boxplot = []
            labels_validos = []

            for t in talhoes:
                dados_talhao = dados_estruturais[dados_estruturais['talhao'] == t]['altura_media'].dropna()
                if len(dados_talhao) > 0:  # Só incluir se houver dados
                    dados_boxplot.append(dados_talhao.values)
                    labels_validos.append(f'T{t}')

            if len(dados_boxplot) > 0:
                bp = ax.boxplot(dados_boxplot, labels=labels_validos, patch_artist=True)

                # Colorir boxes
                for patch in bp['boxes']:
                    patch.set_facecolor('#2E8B57')  # SeaGreen
                    patch.set_alpha(0.7)

                ax.set_xlabel('Talhão')
                ax.set_ylabel('Altura LiDAR (m)')
                ax.set_title('Distribuição de Alturas por Talhão')
                ax.grid(True, alpha=0.3)

                # Rotacionar labels se necessário
                if len(labels_validos) > 5:
                    plt.xticks(rotation=45)
            else:
                ax.text(0.5, 0.5, 'Dados insuficientes\npara boxplot',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Distribuição de Alturas por Talhão')

        except Exception as e:
            ax.text(0.5, 0.5, f'Erro no gráfico:\n{str(e)}',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Distribuição de Alturas por Talhão - Erro')

        st.pyplot(fig)
        plt.close(fig)

    with col2:
        # Histograma geral de alturas - CORRIGIDO
        fig, ax = plt.subplots(figsize=(10, 6))

        try:
            alturas_validas = dados_estruturais['altura_media'].dropna()

            if len(alturas_validas) > 0:
                # Determinar número de bins apropriado
                n_bins = min(20, max(5, len(alturas_validas) // 3))

                ax.hist(alturas_validas, bins=n_bins, alpha=0.7,
                        color='#4682B4', edgecolor='black')  # SteelBlue

                # Linha da média
                media = alturas_validas.mean()
                ax.axvline(media, color='red', linestyle='--', linewidth=2,
                           label=f'Média: {media:.1f}m')

                ax.set_xlabel('Altura LiDAR (m)')
                ax.set_ylabel('Frequência')
                ax.set_title('Distribuição Geral de Alturas')
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Adicionar estatísticas no gráfico
                ax.text(0.02, 0.98,
                        f'N: {len(alturas_validas)}\n'
                        f'Média: {media:.1f}m\n'
                        f'Std: {alturas_validas.std():.1f}m\n'
                        f'Min: {alturas_validas.min():.1f}m\n'
                        f'Max: {alturas_validas.max():.1f}m',
                        transform=ax.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            else:
                ax.text(0.5, 0.5, 'Dados insuficientes\npara histograma',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Distribuição Geral de Alturas - Sem Dados')

        except Exception as e:
            ax.text(0.5, 0.5, f'Erro no gráfico:\n{str(e)}',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Distribuição Geral de Alturas - Erro')

        st.pyplot(fig)
        plt.close(fig)


def mostrar_calibracao_modelos(df_integrado):
    """Mostra interface para calibração de modelos com LiDAR"""
    st.header("🔧 Calibração de Modelos com LiDAR")

    # Verificar se há modelos hipsométricos disponíveis
    if not hasattr(st.session_state, 'resultados_hipsometricos'):
        st.warning("⚠️ Execute primeiro os modelos hipsométricos (Etapa 1)")
        return

    resultados_hip = st.session_state.resultados_hipsometricos
    melhor_modelo = resultados_hip.get('melhor_modelo')

    if not melhor_modelo:
        st.error("❌ Modelo hipsométrico não disponível")
        return

    st.info(f"🏆 Calibrando modelo: **{melhor_modelo}**")

    # Botão para executar calibração
    if st.button("🚀 Executar Calibração com LiDAR", type="primary"):
        with st.spinner("🔧 Calibrando modelo..."):
            resultado_calibracao = calibrar_modelo_hipsometrico_com_lidar(
                df_integrado,
                resultados_hip['resultados'][melhor_modelo]['modelo']
            )

        if resultado_calibracao:
            # Salvar resultado da calibração
            st.session_state.calibracao_lidar = resultado_calibracao

            # Mostrar resultados da calibração
            mostrar_resultados_calibracao(resultado_calibracao, melhor_modelo)
        else:
            st.error("❌ Falha na calibração")


def mostrar_resultados_calibracao(resultado_calibracao, nome_modelo):
    """Mostra resultados da calibração"""
    st.subheader("🎯 Resultados da Calibração")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        r2_original = st.session_state.resultados_hipsometricos['resultados'][nome_modelo]['r2g']
        st.metric("📊 R² Original", f"{r2_original:.4f}")

    with col2:
        r2_calibrado = resultado_calibracao['r2_calibrado']
        delta_r2 = r2_calibrado - r2_original
        st.metric("🎯 R² Calibrado", f"{r2_calibrado:.4f}", delta=f"{delta_r2:+.4f}")

    with col3:
        rmse_calibrado = resultado_calibracao['rmse_calibrado']
        st.metric("📏 RMSE Calibrado", f"{rmse_calibrado:.2f} m")

    with col4:
        cv_score = resultado_calibracao.get('cv_score')
        if cv_score:
            st.metric("✅ CV Score", f"{cv_score:.4f}")
        else:
            st.metric("✅ CV Score", "N/A")

    # Mostrar equação calibrada
    st.subheader("📐 Equação Calibrada")
    coefs = resultado_calibracao['coeficientes']
    st.latex(f"H = {coefs['intercepto']:.3f} + {coefs['slope']:.3f} \\times D")

    # Gráfico de comparação
    mostrar_grafico_calibracao(resultado_calibracao, nome_modelo)


def mostrar_grafico_calibracao(resultado_calibracao, nome_modelo):
    """Mostra gráfico comparando modelo original vs calibrado"""
    st.subheader("📊 Comparação: Original vs Calibrado")

    df_calib = resultado_calibracao['dados_calibracao']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Gráfico 1: Modelo original vs LiDAR
    ax1.scatter(df_calib['D_cm'], df_calib['altura_media'],
                alpha=0.6, color=CORES_LIDAR['lidar'], label='LiDAR (referência)')
    ax1.scatter(df_calib['D_cm'], df_calib['H_m'],
                alpha=0.6, color=CORES_LIDAR['campo'], label='Campo (original)')

    ax1.set_xlabel('DAP (cm)')
    ax1.set_ylabel('Altura (m)')
    ax1.set_title(f'Modelo Original: {nome_modelo}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Gráfico 2: Modelo calibrado vs LiDAR
    ax2.scatter(df_calib['D_cm'], df_calib['altura_media'],
                alpha=0.6, color=CORES_LIDAR['lidar'], label='LiDAR (referência)')
    ax2.scatter(df_calib['D_cm'], resultado_calibracao['predicoes_calibradas'],
                alpha=0.6, color=CORES_LIDAR['calibrado'], label='Modelo calibrado')

    ax2.set_xlabel('DAP (cm)')
    ax2.set_ylabel('Altura (m)')
    ax2.set_title(f'Modelo Calibrado (R² = {resultado_calibracao["r2_calibrado"]:.3f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def mostrar_alertas_lidar(alertas):
    """Mostra alertas automáticos gerados"""
    if not alertas:
        st.success("✅ Nenhum alerta detectado - dados consistentes")
        return

    st.header("⚠️ Alertas Automáticos")

    for alerta in alertas:
        if "🔴" in alerta:
            st.error(alerta)
        elif "⚠️" in alerta:
            st.warning(alerta)
        else:
            st.info(alerta)


def mostrar_downloads_lidar(df_integrado, stats_comparacao, alertas):
    """Mostra opções de download para dados LiDAR"""
    st.header("💾 Downloads")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Download dados integrados
        if st.button("📊 Dados Integrados", key="btn_download_integrados"):
            csv_integrados = exportar_dados_integrados(df_integrado)
            st.download_button(
                "📥 Baixar Dados Integrados",
                csv_integrados,
                "dados_campo_lidar_integrados.csv",
                "text/csv",
                key=gerar_key_unica("download_integrados")
            )

    with col2:
        # Download relatório de comparação
        if stats_comparacao:
            relatorio = gerar_relatorio_integracao_lidar(df_integrado, stats_comparacao, alertas)
            st.download_button(
                "📄 Relatório LiDAR",
                relatorio,
                "relatorio_integracao_lidar.md",
                "text/markdown",
                key=gerar_key_unica("download_relatorio_lidar")
            )

    with col3:
        # Download métricas de validação
        if st.button("📈 Métricas Validação", key="btn_download_validacao"):
            metricas = calcular_metricas_validacao_lidar(df_integrado)
            import json
            json_metricas = json.dumps(metricas, indent=2, ensure_ascii=False)
            st.download_button(
                "📥 Baixar Métricas",
                json_metricas,
                "metricas_validacao_lidar.json",
                "application/json",
                key=gerar_key_unica("download_validacao")
            )


def mostrar_resumo_integracao():
    """Mostra resumo da integração LiDAR"""
    dados_lidar = getattr(st.session_state, 'dados_lidar', None)
    dados_las = getattr(st.session_state, 'dados_lidar_las', None)

    if dados_lidar is None and dados_las is None:
        return

    st.header("📋 Resumo da Integração LiDAR")

    # Determinar fonte dos dados
    if dados_las is not None:
        # Dados de processamento LAS
        df_metricas = dados_las['df_metricas']
        metodo = "Processamento LAS/LAZ"

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("📊 Total Parcelas", len(df_metricas))
        with col2:
            pontos_total = df_metricas['n_pontos'].sum()
            st.metric("🛩️ Total Pontos", f"{pontos_total:,}")
        with col3:
            altura_media = df_metricas['altura_media'].mean()
            st.metric("🌳 Altura Média", f"{altura_media:.1f} m")
        with col4:
            cobertura_media = df_metricas['cobertura'].mean()
            st.metric("🍃 Cobertura", f"{cobertura_media:.1f}%")

        st.info(f"📁 **Método:** {metodo} - {dados_las['arquivo_original']}")

    elif dados_lidar is not None:
        # Dados de métricas processadas
        dados = dados_lidar
        df_integrado = dados['df_integrado']
        stats_comparacao = dados['stats_comparacao']

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_parcelas = len(df_integrado)
            st.metric("📊 Total Parcelas", total_parcelas)

        with col2:
            parcelas_lidar = df_integrado['altura_media'].notna().sum() if 'altura_media' in df_integrado.columns else 0
            cobertura = (parcelas_lidar / total_parcelas) * 100 if total_parcelas > 0 else 0
            st.metric("🛩️ Cobertura LiDAR", f"{cobertura:.1f}%")

        with col3:
            if stats_comparacao:
                correlacao = stats_comparacao['correlacao']
                st.metric("🔗 Correlação", f"{correlacao:.3f}")
            else:
                st.metric("🔗 Correlação", "N/A")

        with col4:
            n_alertas = len(dados['alertas'])
            st.metric("⚠️ Alertas", n_alertas)

        st.info(f"📁 **Método:** Métricas pré-processadas")


def mostrar_dados_salvos_lidar():
    """Mostra dados LiDAR salvos e permite reprocessamento"""
    dados_lidar = getattr(st.session_state, 'dados_lidar', None)
    dados_las = getattr(st.session_state, 'dados_lidar_las', None)

    if dados_lidar is None and dados_las is None:
        return False

    st.markdown("---")
    st.subheader("📂 Dados LiDAR Salvos")

    # Mostrar informações dos dados salvos
    if dados_las is not None:
        st.info(f"🛩️ **Processamento LAS:** {dados_las['arquivo_original']}")
        timestamp = dados_las['timestamp'].strftime('%d/%m/%Y %H:%M:%S')
        st.caption(f"📅 Processado em: {timestamp}")

        if st.checkbox("👀 Mostrar Dados LAS Processados", key="mostrar_las_salvo"):
            mostrar_resultados_processamento_las(dados_las['df_metricas'])

    if dados_lidar is not None:
        st.info(f"📊 **Métricas Processadas:** Análise completa disponível")
        timestamp = dados_lidar['timestamp'].strftime('%d/%m/%Y %H:%M:%S')
        st.caption(f"📅 Processado em: {timestamp}")

        if st.checkbox("👀 Mostrar Análise Completa", key="mostrar_lidar_salvo"):
            df_integrado = dados_lidar['df_integrado']
            stats_comparacao = dados_lidar['stats_comparacao']
            alertas = dados_lidar['alertas']

            # Recriar abas com dados salvos
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Comparação",
                "🌲 Estrutural",
                "🔧 Calibração",
                "⚠️ Alertas",
                "💾 Downloads"
            ])

            with tab1:
                mostrar_resultados_comparacao(stats_comparacao)

            with tab2:
                mostrar_analise_estrutural(df_integrado)

            with tab3:
                mostrar_calibracao_modelos(df_integrado)

            with tab4:
                mostrar_alertas_lidar(alertas)

            with tab5:
                mostrar_downloads_lidar(df_integrado, stats_comparacao, alertas)

    return True


def limpar_dados_lidar():
    """Limpa dados LiDAR salvos"""
    keys_para_limpar = ['dados_lidar', 'dados_lidar_las', 'calibracao_lidar']

    for key in keys_para_limpar:
        if hasattr(st.session_state, key):
            delattr(st.session_state, key)

    st.success("🗑️ Dados LiDAR limpos!")
    st.rerun()


def main():
    """Função principal da página LiDAR integrada"""
    # Criar cabeçalho com identidade visual
    criar_cabecalho_greenvista("Dados LiDAR")

    # Verificar pré-requisitos
    if not verificar_prerequisitos():
        return

    # Mostrar status da configuração na sidebar
    mostrar_status_configuracao_sidebar()

    # Introdução
    mostrar_introducao_lidar()

    # Mostrar resumo se há dados salvos
    mostrar_resumo_integracao()

    # Seleção do método de processamento
    tab_las, tab_metricas, processamento_las_disponivel = mostrar_selecao_metodo_lidar()

    # Variáveis para controlar fluxo
    resultado_las = None
    resultado_metricas = None

    # Processamento LAS (se disponível)
    if processamento_las_disponivel and tab_las is not None:
        resultado_las = processar_metodo_las(tab_las)

    # Processamento de métricas
    if tab_metricas is not None:
        resultado_metricas = processar_metodo_metricas(tab_metricas)

    # Se há dados processados em métricas, mostrar análise completa
    if resultado_metricas is not None:
        df_integrado, stats_comparacao, alertas = resultado_metricas

        if df_integrado is not None:
            st.markdown("---")
            st.subheader("📊 Análise Completa LiDAR")

            # Criar abas para diferentes análises
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Comparação",
                "🌲 Estrutural",
                "🔧 Calibração",
                "⚠️ Alertas",
                "💾 Downloads"
            ])

            with tab1:
                mostrar_resultados_comparacao(stats_comparacao)

            with tab2:
                mostrar_analise_estrutural(df_integrado)

            with tab3:
                mostrar_calibracao_modelos(df_integrado)

            with tab4:
                mostrar_alertas_lidar(alertas)

            with tab5:
                mostrar_downloads_lidar(df_integrado, stats_comparacao, alertas)

    # Mostrar dados salvos se existirem e não há processamento ativo
    elif not mostrar_dados_salvos_lidar():
        # Nenhum dado disponível - mostrar instruções
        st.markdown("---")
        st.info("📋 **Próximos Passos:**")

        if processamento_las_disponivel:
            st.markdown("""
            1. **🛩️ Processamento LAS/LAZ:** Carregue arquivo LAS/LAZ para processamento direto
            2. **📊 Métricas Processadas:** Upload de arquivo CSV/Excel com métricas já calculadas

            **💡 Recomendação:** Use processamento LAS/LAZ para máxima precisão e controle
            """)
        else:
            st.markdown("""
            1. **📦 Instalar Dependências:** Execute `pip install laspy[lazrs,laszip] geopandas` para habilitar processamento LAS
            2. **📊 Métricas Processadas:** Use arquivo CSV/Excel com métricas já calculadas

            **💡 Alternativa:** Processe arquivos LAS no R e faça upload das métricas
            """)

    # Botão para limpar dados
    if hasattr(st.session_state, 'dados_lidar') or hasattr(st.session_state, 'dados_lidar_las'):
        st.markdown("---")
        if st.button("🗑️ Limpar Dados LiDAR", type="secondary", help="Remove todos os dados LiDAR salvos"):
            limpar_dados_lidar()

    # Navegação rápida
    st.markdown("---")
    criar_navegacao_rapida_botoes()


if __name__ == "__main__":
    main()