# pages/3_📈_Inventário_Florestal.py
"""
Etapa 3: Inventário Florestal
Processamento completo e relatórios finais
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
from processors.inventario import processar_inventario_completo, gerar_relatorio_inventario
from processors.areas import processar_areas_por_metodo, validar_areas_processadas
from ui.configuracoes import criar_configuracoes_areas
from ui.resultados import mostrar_resultados_finais
from ui.graficos import criar_graficos_inventario
from utils.formatacao import formatar_brasileiro, formatar_numero_inteligente

st.set_page_config(
    page_title="Inventário Florestal",
    page_icon="📈",
    layout="wide"
)


def gerar_id_unico(base=""):
    """Gera ID único baseado em timestamp"""
    return f"{base}_{int(time.time() * 1000)}"


def verificar_prerequisitos():
    """Verifica se as etapas anteriores foram concluídas"""
    problemas = []

    if not hasattr(st.session_state, 'arquivos_carregados') or not st.session_state.arquivos_carregados:
        problemas.append("Arquivos não carregados")

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
            if st.button("🏠 Página Principal", key=f"btn_principal_{gerar_id_unico()}"):
                st.switch_page("app.py")

        with col2:
            if st.button("🌳 Hipsométricos", key=f"btn_hip_{gerar_id_unico()}"):
                st.switch_page("pages/1_🌳_Modelos_Hipsométricos.py")

        with col3:
            if st.button("📊 Volumétricos", key=f"btn_vol_{gerar_id_unico()}"):
                st.switch_page("pages/2_📊_Modelos_Volumétricos.py")

        return False

    return True


def mostrar_status_etapas():
    """Mostra status das etapas anteriores"""
    st.subheader("✅ Status das Etapas Anteriores")

    col1, col2 = st.columns(2)

    with col1:
        # Etapa 1 - Hipsométricos
        if st.session_state.get('resultados_hipsometricos'):
            melhor_hip = st.session_state.resultados_hipsometricos.get('melhor_modelo', 'N/A')
            st.success(f"🌳 **Etapa 1 Concluída** - Melhor modelo: {melhor_hip}")
        else:
            st.error("🌳 Etapa 1 não concluída")

    with col2:
        # Etapa 2 - Volumétricos
        if st.session_state.get('resultados_volumetricos'):
            melhor_vol = st.session_state.resultados_volumetricos.get('melhor_modelo', 'N/A')
            st.success(f"📊 **Etapa 2 Concluída** - Melhor modelo: {melhor_vol}")
        else:
            st.error("📊 Etapa 2 não concluída")


def configurar_areas_talhoes():
    """Configura áreas dos talhões"""
    st.header("📏 Configuração de Áreas dos Talhões")

    df_inventario = st.session_state.dados_inventario
    talhoes_disponiveis = sorted(df_inventario['talhao'].unique())

    # Método de cálculo das áreas
    id_selectbox = gerar_id_unico("selectbox_metodo")
    metodo_area = st.selectbox(
        "🗺️ Método para Cálculo das Áreas",
        [
            "Simular automaticamente",
            "Valores informados manualmente",
            "Upload shapefile (se disponível)",
            "Coordenadas das parcelas (se disponível)"
        ],
        help="Como definir as áreas dos talhões",
        key=id_selectbox
    )

    config_areas = {'metodo': metodo_area}

    if metodo_area == "Valores informados manualmente":
        st.write("**📝 Informe as áreas por talhão (hectares):**")

        areas_manuais = {}
        n_colunas = min(4, len(talhoes_disponiveis))
        colunas = st.columns(n_colunas)

        for i, talhao in enumerate(talhoes_disponiveis):
            col_idx = i % n_colunas
            with colunas[col_idx]:
                id_area = gerar_id_unico(f"area_talhao_{talhao}")
                areas_manuais[talhao] = st.number_input(
                    f"Talhão {talhao}",
                    min_value=0.1,
                    max_value=1000.0,
                    value=25.0,
                    step=0.1,
                    key=id_area
                )

        config_areas['areas_manuais'] = areas_manuais

        # Mostrar resumo
        if areas_manuais:
            area_total = sum(areas_manuais.values())
            area_media = np.mean(list(areas_manuais.values()))

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Área Total", f"{area_total:.1f} ha")
            with col2:
                st.metric("Área Média", f"{area_media:.1f} ha")
            with col3:
                st.metric("Talhões", len(areas_manuais))

    elif metodo_area == "Simular automaticamente":
        st.info("🎲 **Simulação Automática de Áreas**")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Método:**")
            st.write("• Baseado no número de parcelas por talhão")
            st.write("• Cada parcela representa 2-5 hectares")
            st.write("• Variação aleatória realística aplicada")

        with col2:
            id_fator_min = gerar_id_unico("slider_fator_min")
            id_fator_max = gerar_id_unico("slider_fator_max")
            id_variacao = gerar_id_unico("slider_variacao")

            fator_min = st.slider("Fator mínimo (ha/parcela)", 1.0, 10.0, 2.5, 0.5, key=id_fator_min)
            fator_max = st.slider("Fator máximo (ha/parcela)", fator_min, 15.0, 4.0, 0.5, key=id_fator_max)
            variacao = st.slider("Variação (%)", 0, 50, 20, 5, key=id_variacao)

        config_areas.update({
            'simulacao_fator_min': fator_min,
            'simulacao_fator_max': fator_max,
            'simulacao_variacao': variacao / 100
        })

    else:
        st.info(f"💡 Método {metodo_area} será processado automaticamente se arquivos estiverem disponíveis")

    return config_areas


def processar_areas(config_areas):
    """Processa áreas dos talhões"""
    df_inventario = st.session_state.dados_inventario

    try:
        if config_areas['metodo'] == "Simular automaticamente":
            df_areas = processar_areas_por_metodo('simulacao',
                                                  df_inventario=df_inventario,
                                                  config=config_areas)

        elif config_areas['metodo'] == "Valores informados manualmente":
            areas_dict = config_areas.get('areas_manuais', {})
            talhoes_lista = df_inventario['talhao'].unique()
            df_areas = processar_areas_por_metodo('manual',
                                                  areas_dict=areas_dict,
                                                  talhoes=talhoes_lista)

        else:
            # Fallback para simulação
            df_areas = processar_areas_por_metodo('simulacao',
                                                  df_inventario=df_inventario,
                                                  config={'simulacao_fator_min': 2.5,
                                                          'simulacao_fator_max': 4.0,
                                                          'simulacao_variacao': 0.2})

        # Validar áreas
        if df_areas is not None:
            validacao = validar_areas_processadas(df_areas, df_inventario)

            if validacao['valido']:
                st.success("✅ Áreas processadas com sucesso!")

                # Mostrar resumo
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Talhões", len(df_areas))
                with col2:
                    st.metric("Área Total", f"{df_areas['area_ha'].sum():.1f} ha")
                with col3:
                    st.metric("Área Média", f"{df_areas['area_ha'].mean():.1f} ha")
                with col4:
                    cv_areas = (df_areas['area_ha'].std() / df_areas['area_ha'].mean()) * 100
                    st.metric("CV Áreas", f"{cv_areas:.1f}%")

                # Mostrar alertas se houver
                if validacao['alertas']:
                    with st.expander("⚠️ Alertas"):
                        for alerta in validacao['alertas']:
                            st.warning(alerta)

                return df_areas
            else:
                st.error("❌ Problemas na validação das áreas:")
                for erro in validacao['erros']:
                    st.error(f"• {erro}")
                return None

        return None

    except Exception as e:
        st.error(f"❌ Erro ao processar áreas: {e}")
        return None


def executar_inventario_completo(config_areas):
    """Executa o processamento completo do inventário"""
    st.header("🚀 Processamento do Inventário Completo")

    # Processar áreas
    df_areas = processar_areas(config_areas)

    if df_areas is None:
        st.error("❌ Não foi possível processar as áreas dos talhões")
        return

    # Preparar configuração completa
    melhor_hip = st.session_state.resultados_hipsometricos['melhor_modelo']
    melhor_vol = st.session_state.resultados_volumetricos['melhor_modelo']

    config_completa = {
        'areas_talhoes': df_areas,
        'melhor_modelo_hip': melhor_hip,
        'melhor_modelo_vol': melhor_vol,
        **config_areas
    }

    # Barra de progresso
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.text("Preparando dados do inventário...")
        progress_bar.progress(0.1)

        status_text.text("Aplicando modelos hipsométricos...")
        progress_bar.progress(0.3)

        status_text.text("Aplicando modelos volumétricos...")
        progress_bar.progress(0.6)

        status_text.text("Calculando estatísticas finais...")
        progress_bar.progress(0.8)

        # Usar função modular existente
        resultados = processar_inventario_completo(
            st.session_state.dados_inventario,
            config_completa,
            melhor_hip,
            melhor_vol
        )

        progress_bar.progress(1.0)
        status_text.text("✅ Inventário processado com sucesso!")

        # Salvar no session_state
        st.session_state.inventario_processado = resultados

        # Mostrar resultados
        mostrar_resultados_inventario(resultados)

    except Exception as e:
        st.error(f"❌ Erro no processamento do inventário: {e}")
        st.info("💡 Verifique os dados e configurações")


def mostrar_resultados_inventario(resultados):
    """Mostra os resultados finais do inventário"""
    st.header("📊 Resultados Finais do Inventário")

    # Usar função modular existente
    mostrar_resultados_finais(resultados)

    # Informações dos modelos utilizados
    st.subheader("🏆 Modelos Utilizados")

    col1, col2 = st.columns(2)

    with col1:
        melhor_hip = resultados['modelos_utilizados']['hipsometrico']
        st.success(f"🌳 **Hipsométrico**: {melhor_hip}")

        # Estatísticas do modelo hipsométrico
        if st.session_state.get('resultados_hipsometricos'):
            hip_stats = st.session_state.resultados_hipsometricos['resultados'][melhor_hip]
            st.write(f"• R² Generalizado: {hip_stats['r2g']:.4f}")
            st.write(f"• RMSE: {hip_stats['rmse']:.4f}")

    with col2:
        melhor_vol = resultados['modelos_utilizados']['volumetrico']
        st.success(f"📊 **Volumétrico**: {melhor_vol}")

        # Estatísticas do modelo volumétrico
        if st.session_state.get('resultados_volumetricos'):
            vol_stats = st.session_state.resultados_volumetricos['resultados'][melhor_vol]
            st.write(f"• R²: {vol_stats['r2']:.4f}")
            st.write(f"• RMSE: {vol_stats['rmse']:.4f}")


def criar_relatorio_executivo(resultados, contexto="atual"):
    """Cria relatório executivo do inventário"""
    st.subheader("📄 Relatório Executivo")

    # Usar função modular existente
    relatorio = gerar_relatorio_inventario(resultados)

    # Mostrar no streamlit
    st.markdown(relatorio)

    # Botão de download com ID único
    id_download = gerar_id_unico(f"download_relatorio_executivo_{contexto}")
    st.download_button(
        label="📥 Download Relatório Executivo",
        data=relatorio,
        file_name="relatorio_executivo_inventario.md",
        mime="text/markdown",
        help="Relatório completo em formato Markdown",
        key=id_download
    )


def mostrar_downloads_avancados(resultados, contexto="atual"):
    """Mostra opções avançadas de download"""
    st.subheader("💾 Downloads Avançados")

    # Gerar IDs únicos para esta seção
    id_base = gerar_id_unico(contexto)

    col1, col2, col3 = st.columns(3)

    with col1:
        # Dados consolidados
        if 'resumo_parcelas' in resultados and resultados['resumo_parcelas'] is not None:
            csv_parcelas = resultados['resumo_parcelas'].to_csv(index=False)
            st.download_button(
                "📊 Resumo por Parcela",
                csv_parcelas,
                "resumo_parcelas.csv",
                "text/csv",
                key=f"download_parcelas_{id_base}"
            )

    with col2:
        # Dados por talhão
        if 'resumo_talhoes' in resultados and resultados['resumo_talhoes'] is not None:
            csv_talhoes = resultados['resumo_talhoes'].to_csv(index=False)
            st.download_button(
                "🌳 Resumo por Talhão",
                csv_talhoes,
                "resumo_talhoes.csv",
                "text/csv",
                key=f"download_talhoes_{id_base}"
            )

    with col3:
        # Inventário completo
        if 'inventario_completo' in resultados and resultados['inventario_completo'] is not None:
            # Limitar tamanho para download
            df_download = resultados['inventario_completo'].head(5000)
            csv_completo = df_download.to_csv(index=False)

            st.download_button(
                "📋 Inventário Completo",
                csv_completo,
                "inventario_completo.csv",
                "text/csv",
                help="Primeiros 5.000 registros",
                key=f"download_completo_{id_base}"
            )


def mostrar_comparacao_modelos():
    """Mostra comparação entre os modelos das etapas anteriores"""
    with st.expander("📊 Comparação dos Modelos Selecionados"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🌳 Modelos Hipsométricos")

            if st.session_state.get('resultados_hipsometricos'):
                hip_results = st.session_state.resultados_hipsometricos['resultados']

                # Criar ranking
                hip_ranking = []
                for modelo, resultado in hip_results.items():
                    hip_ranking.append({
                        'Modelo': modelo,
                        'R² Gen.': f"{resultado['r2g']:.4f}",
                        'RMSE': f"{resultado['rmse']:.4f}"
                    })

                df_hip = pd.DataFrame(hip_ranking)
                df_hip = df_hip.sort_values('R² Gen.', ascending=False)

                # Destacar o melhor
                melhor_hip = st.session_state.resultados_hipsometricos['melhor_modelo']

                for i, row in df_hip.iterrows():
                    if row['Modelo'] == melhor_hip:
                        st.success(f"🏆 **{row['Modelo']}** - R²: {row['R² Gen.']} - RMSE: {row['RMSE']}")
                    else:
                        st.write(f"• {row['Modelo']} - R²: {row['R² Gen.']} - RMSE: {row['RMSE']}")

        with col2:
            st.subheader("📊 Modelos Volumétricos")

            if st.session_state.get('resultados_volumetricos'):
                vol_results = st.session_state.resultados_volumetricos['resultados']

                # Criar ranking
                vol_ranking = []
                for modelo, resultado in vol_results.items():
                    vol_ranking.append({
                        'Modelo': modelo,
                        'R²': f"{resultado['r2']:.4f}",
                        'RMSE': f"{resultado['rmse']:.4f}"
                    })

                df_vol = pd.DataFrame(vol_ranking)
                df_vol = df_vol.sort_values('R²', ascending=False)

                # Destacar o melhor
                melhor_vol = st.session_state.resultados_volumetricos['melhor_modelo']

                for i, row in df_vol.iterrows():
                    if row['Modelo'] == melhor_vol:
                        st.success(f"🏆 **{row['Modelo']}** - R²: {row['R²']} - RMSE: {row['RMSE']}")
                    else:
                        st.write(f"• {row['Modelo']} - R²: {row['R²']} - RMSE: {row['RMSE']}")


def main():
    if not verificar_prerequisitos():
        return

    st.title("📈 Inventário Florestal")
    st.markdown("### Processamento Completo e Relatórios Finais")

    # Mostrar status das etapas anteriores
    mostrar_status_etapas()

    # Mostrar comparação dos modelos
    mostrar_comparacao_modelos()

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
        st.metric("Árvores Cubadas",
                  len(st.session_state.dados_cubagem) if st.session_state.dados_cubagem is not None else 0)

    # Botão para executar inventário completo
    id_botao_executar = gerar_id_unico("btn_executar_inventario")
    if st.button("🚀 Executar Inventário Completo", type="primary", use_container_width=True, key=id_botao_executar):
        executar_inventario_completo(config_areas)

    # Mostrar resultados salvos se existirem
    if st.session_state.get('inventario_processado'):
        resultados_salvos = st.session_state.inventario_processado

        # Abas para organizar os resultados
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Resultados", "📄 Relatório", "📈 Gráficos", "💾 Downloads"])

        with tab1:
            mostrar_resultados_inventario(resultados_salvos)

        with tab2:
            criar_relatorio_executivo(resultados_salvos, "tab_relatorio")

        with tab3:
            st.subheader("📈 Visualizações do Inventário")
            criar_graficos_inventario(resultados_salvos)

        with tab4:
            mostrar_downloads_avancados(resultados_salvos, "tab_downloads")

    # Rodapé com informações
    st.markdown("---")
    st.markdown("""
    ### 💡 Sobre esta Etapa

    O **Inventário Florestal** aplica os melhores modelos selecionados nas etapas anteriores para:

    - **🌳 Estimar alturas** usando o melhor modelo hipsométrico
    - **📊 Calcular volumes** usando o melhor modelo volumétrico  
    - **📏 Processar áreas** dos talhões conforme método selecionado
    - **📈 Gerar estatísticas** completas por parcela e talhão
    - **📄 Criar relatórios** executivos e técnicos

    **Saídas principais:**
    - Resumo por parcela (produtividade, características dendrométricas)
    - Resumo por talhão (estoque, área, IMA)
    - Inventário completo (dados árvore por árvore)
    - Relatórios executivos e gráficos
    """)


if __name__ == "__main__":
    main()