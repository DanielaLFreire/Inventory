# pages/1_🌳_Modelos_Hipsométricos.py
"""
Etapa 1: Modelos Hipsométricos
Análise completa da relação altura-diâmetro
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from models.hipsometrico import ajustar_todos_modelos_hipsometricos
from ui.configuracoes import criar_configuracoes_principais
from ui.graficos import criar_graficos_modelos
from utils.formatacao import formatar_brasileiro, classificar_qualidade_modelo

st.set_page_config(
    page_title="Modelos Hipsométricos",
    page_icon="🌳",
    layout="wide"
)


def gerar_id_unico(base=""):
    """Gera ID único baseado em timestamp"""
    return f"{base}_{int(time.time() * 1000)}"


def verificar_dados():
    """Verifica se dados estão disponíveis"""
    if not hasattr(st.session_state, 'arquivos_carregados') or not st.session_state.arquivos_carregados:
        st.error("❌ Dados não carregados. Volte à página principal.")
        if st.button("🏠 Voltar à Página Principal", key="btn_voltar_hip"):
            st.switch_page("app.py")
        return False
    return True


def obter_info_modelos():
    """Informações dos modelos hipsométricos"""
    return {
        "Curtis": {
            "equacao": r"\ln(H) = \beta_0 + \beta_1 \cdot \frac{1}{D}",
            "tipo": "Linear",
            "descricao": "Modelo logarítmico simples baseado no inverso do diâmetro"
        },
        "Campos": {
            "equacao": r"\ln(H) = \beta_0 + \beta_1 \cdot \frac{1}{D} + \beta_2 \cdot \ln(H_{dom})",
            "tipo": "Linear",
            "descricao": "Extensão do Curtis incluindo altura dominante"
        },
        "Henri": {
            "equacao": r"H = \beta_0 + \beta_1 \cdot \ln(D)",
            "tipo": "Linear",
            "descricao": "Modelo linear com logaritmo do diâmetro"
        },
        "Prodan": {
            "equacao": r"\frac{D^2}{H-1.3} = \beta_0 + \beta_1 \cdot D + \beta_2 \cdot D^2 + \beta_3 \cdot D \cdot Idade",
            "tipo": "Linear",
            "descricao": "Modelo baseado no quociente de forma, inclui idade"
        },
        "Chapman": {
            "equacao": r"H = b_0 \cdot (1 - e^{-b_1 \cdot D})^{b_2}",
            "tipo": "Não-Linear",
            "descricao": "Modelo sigmoidal com crescimento assintótico"
        },
        "Weibull": {
            "equacao": r"H = a \cdot (1 - e^{-b \cdot D^c})",
            "tipo": "Não-Linear",
            "descricao": "Modelo baseado na distribuição de Weibull"
        },
        "Mononuclear": {
            "equacao": r"H = a \cdot (1 - b \cdot e^{-c \cdot D})",
            "tipo": "Não-Linear",
            "descricao": "Modelo exponencial mononuclear"
        }
    }


def mostrar_configuracoes():
    """Interface de configurações"""
    st.header("⚙️ Configurações dos Modelos Hipsométricos")

    if st.session_state.dados_inventario is None:
        st.error("❌ Dados de inventário não disponíveis")
        return None

    # Usar função modular existente
    config = criar_configuracoes_principais(st.session_state.dados_inventario)

    # Configurações específicas para hipsométricos
    with st.expander("🔧 Configurações Avançadas"):
        col1, col2 = st.columns(2)

        with col1:
            incluir_nao_lineares = st.checkbox(
                "Incluir modelos não-lineares",
                value=True,
                help="Chapman, Weibull, Mononuclear (mais demorados)",
                key="checkbox_nao_lineares_hip"
            )

        with col2:
            max_iteracoes = st.number_input(
                "Máximo de iterações (não-lineares)",
                min_value=1000,
                max_value=10000,
                value=5000,
                help="Para convergência dos modelos não-lineares",
                key="number_input_max_iter_hip"
            )

    config.update({
        'incluir_nao_lineares': incluir_nao_lineares,
        'max_iteracoes': max_iteracoes
    })

    return config


def mostrar_info_modelos():
    """Mostra informações dos modelos"""
    with st.expander("📚 Informações dos Modelos Hipsométricos"):
        modelos_info = obter_info_modelos()

        col1, col2 = st.columns(2)

        for i, (modelo, info) in enumerate(modelos_info.items()):
            with col1 if i % 2 == 0 else col2:
                with st.container():
                    st.subheader(f"{modelo} - {info['tipo']}")
                    st.latex(info['equacao'])
                    st.write(f"**Descrição:** {info['descricao']}")
                    st.markdown("---")


def executar_analise_hipsometrica(config):
    """Executa análise dos modelos hipsométricos"""
    st.header("🚀 Executando Análise Hipsométrica")

    # Barra de progresso
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.text("Preparando dados...")
        progress_bar.progress(0.1)

        # Filtrar dados conforme configuração
        df_inventario = st.session_state.dados_inventario

        # Aplicar filtros básicos
        df_filtrado = df_inventario.copy()

        if config.get('talhoes_excluir'):
            df_filtrado = df_filtrado[~df_filtrado['talhao'].isin(config['talhoes_excluir'])]

        if config.get('diametro_min'):
            df_filtrado = df_filtrado[df_filtrado['D_cm'] >= config['diametro_min']]

        if config.get('codigos_excluir'):
            df_filtrado = df_filtrado[~df_filtrado['cod'].isin(config['codigos_excluir'])]

        # Remover dados inválidos
        df_filtrado = df_filtrado[
            (df_filtrado['D_cm'].notna()) &
            (df_filtrado['H_m'].notna()) &
            (df_filtrado['D_cm'] > 0) &
            (df_filtrado['H_m'] > 1.3)
            ]

        if len(df_filtrado) < 20:
            st.error("❌ Poucos dados válidos após filtros. Ajuste os critérios.")
            return

        progress_bar.progress(0.3)
        status_text.text("Ajustando modelos...")

        # Ajustar todos os modelos usando função modular
        resultados, predicoes, melhor_modelo = ajustar_todos_modelos_hipsometricos(df_filtrado)

        progress_bar.progress(1.0)
        status_text.text("✅ Análise concluída!")

        if not resultados:
            st.error("❌ Nenhum modelo foi ajustado com sucesso")
            return

        # Salvar no session_state
        st.session_state.resultados_hipsometricos = {
            'resultados': resultados,
            'predicoes': predicoes,
            'melhor_modelo': melhor_modelo,
            'dados': df_filtrado,
            'config': config
        }

        st.success(f"🏆 Melhor modelo: **{melhor_modelo}**")

        # Mostrar resultados
        mostrar_resultados_hipsometricos(resultados, predicoes, df_filtrado, "execucao")

    except Exception as e:
        st.error(f"❌ Erro na análise: {e}")
        st.info("💡 Verifique os dados e tente novamente")


def mostrar_resultados_hipsometricos(resultados, predicoes, df_dados, contexto="atual"):
    """Mostra resultados dos modelos hipsométricos"""
    st.header("📊 Resultados dos Modelos Hipsométricos")

    # Gerar IDs únicos para esta execução
    id_base = gerar_id_unico(contexto)

    # Tabs para organizar resultados
    tab1, tab2, tab3, tab4 = st.tabs(["🏆 Ranking", "📊 Gráficos", "🔢 Coeficientes", "💾 Downloads"])

    with tab1:
        # Ranking dos modelos
        st.subheader("🏆 Ranking dos Modelos")

        ranking_data = []
        for modelo, resultado in resultados.items():
            r2g = resultado['r2g']
            rmse = resultado['rmse']
            qualidade = classificar_qualidade_modelo(r2g)

            ranking_data.append({
                'Ranking': 0,  # Será preenchido após ordenação
                'Modelo': modelo,
                'R² Generalizado': f"{r2g:.4f}",
                'RMSE': f"{rmse:.4f}",
                'Qualidade': qualidade
            })

        # Ordenar por R²
        df_ranking = pd.DataFrame(ranking_data)
        df_ranking = df_ranking.sort_values('R² Generalizado', ascending=False)
        df_ranking['Ranking'] = range(1, len(df_ranking) + 1)

        st.dataframe(df_ranking, hide_index=True, use_container_width=True)

        # Destaque do melhor
        melhor = df_ranking.iloc[0]
        st.success(f"🏆 **Melhor modelo**: {melhor['Modelo']} (R² = {melhor['R² Generalizado']})")

    with tab2:
        # Gráficos usando função modular
        criar_graficos_modelos(df_dados, resultados, predicoes, 'hipsometrico')

    with tab3:
        # Coeficientes detalhados
        st.subheader("🔢 Coeficientes dos Modelos")

        for modelo, resultado in resultados.items():
            with st.expander(f"📊 {modelo} - Coeficientes"):
                modelo_obj = resultado.get('modelo')
                if modelo_obj and hasattr(modelo_obj, 'modelo'):
                    # Para modelos lineares
                    try:
                        coefs = modelo_obj.modelo.coef_
                        intercept = modelo_obj.modelo.intercept_

                        st.write(f"**Intercepto**: {intercept:.6f}")
                        for i, coef in enumerate(coefs):
                            st.write(f"**Coeficiente {i + 1}**: {coef:.6f}")

                    except:
                        st.info("Coeficientes não disponíveis para este modelo")

                elif 'parametros' in resultado:
                    # Para modelos não-lineares
                    params = resultado['parametros']
                    param_names = resultado.get('param_names', [f'Parâmetro {i + 1}' for i in range(len(params))])

                    for name, param in zip(param_names, params):
                        st.write(f"**{name}**: {param:.6f}")

                # Estatísticas do modelo
                st.write(f"**R² Generalizado**: {resultado['r2g']:.4f}")
                st.write(f"**RMSE**: {resultado['rmse']:.4f}")

    with tab4:
        # Downloads com IDs únicos
        st.subheader("💾 Downloads")

        col1, col2, col3 = st.columns(3)

        with col1:
            # Ranking dos modelos
            csv_ranking = df_ranking.to_csv(index=False)
            st.download_button(
                "📊 Ranking dos Modelos",
                csv_ranking,
                "ranking_modelos_hipsometricos.csv",
                "text/csv",
                key=f"download_ranking_{id_base}"
            )

        with col2:
            # Dados com predições
            df_resultado = df_dados.copy()
            for modelo, pred in predicoes.items():
                df_resultado[f'pred_{modelo}'] = pred
                df_resultado[f'residuo_{modelo}'] = df_dados['H_m'] - pred

            csv_dados = df_resultado.to_csv(index=False)
            st.download_button(
                "📄 Dados com Predições",
                csv_dados,
                "dados_predicoes_hipsometricas.csv",
                "text/csv",
                key=f"download_dados_{id_base}"
            )

        with col3:
            # Relatório resumido
            relatorio = gerar_relatorio_hipsometrico(resultados, df_ranking)
            st.download_button(
                "📄 Relatório Resumido",
                relatorio,
                "relatorio_hipsometricos.md",
                "text/markdown",
                key=f"download_relatorio_{id_base}"
            )


def gerar_relatorio_hipsometrico(resultados, df_ranking):
    """Gera relatório dos modelos hipsométricos"""
    melhor = df_ranking.iloc[0]

    relatorio = f"""# RELATÓRIO - MODELOS HIPSOMÉTRICOS

## 🏆 MELHOR MODELO
**{melhor['Modelo']}** - {melhor['Qualidade']}
- R² Generalizado: {melhor['R² Generalizado']}
- RMSE: {melhor['RMSE']}

## 📊 RANKING COMPLETO
"""

    for _, row in df_ranking.iterrows():
        relatorio += f"\n{row['Ranking']}. **{row['Modelo']}** - {row['Qualidade']}"
        relatorio += f"\n   - R²: {row['R² Generalizado']}, RMSE: {row['RMSE']}\n"

    relatorio += f"""
## 📈 RESUMO DA ANÁLISE
- Total de modelos avaliados: {len(resultados)}
- Modelos lineares: {len([m for m in resultados.keys() if m in ['Curtis', 'Campos', 'Henri', 'Prodan']])}
- Modelos não-lineares: {len([m for m in resultados.keys() if m in ['Chapman', 'Weibull', 'Mononuclear']])}

## 🎯 RECOMENDAÇÃO
Use o modelo **{melhor['Modelo']}** para estimativas de altura neste povoamento.

---
*Relatório gerado pelo Sistema de Inventário Florestal*
"""

    return relatorio


def main():
    if not verificar_dados():
        return

    st.title("🌳 Modelos Hipsométricos")
    st.markdown("### Análise da Relação Altura-Diâmetro")

    # Mostrar informações dos modelos
    mostrar_info_modelos()

    # Configurações
    config = mostrar_configuracoes()

    if config is None:
        return

    # Mostrar resumo dos dados
    df_inventario = st.session_state.dados_inventario
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Registros", len(df_inventario))
    with col2:
        st.metric("Talhões", df_inventario['talhao'].nunique())
    with col3:
        st.metric("DAP Médio", f"{df_inventario['D_cm'].mean():.1f} cm")
    with col4:
        st.metric("Altura Média", f"{df_inventario['H_m'].mean():.1f} m")

    # Botão para executar análise
    if st.button("🚀 Executar Análise Hipsométrica", type="primary", use_container_width=True):
        executar_analise_hipsometrica(config)

    # Mostrar resultados salvos se existirem
    if st.session_state.resultados_hipsometricos:
        resultados_salvos = st.session_state.resultados_hipsometricos
        mostrar_resultados_hipsometricos(
            resultados_salvos['resultados'],
            resultados_salvos['predicoes'],
            resultados_salvos['dados'],
            "salvos"
        )


if __name__ == "__main__":
    main()