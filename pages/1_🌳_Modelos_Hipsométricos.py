# pages/1_🌳_Modelos_Hipsométricos.py - VERSÃO PADRONIZADA
"""
Etapa 1: Modelos Hipsométricos - COM IDENTIDADE VISUAL PADRONIZADA
Análise completa da relação altura-diâmetro com filtros globais e parâmetros configurados
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import uuid
from models.hipsometrico import ajustar_todos_modelos_hipsometricos, validar_parametros_configuracao
from ui.graficos import criar_graficos_modelos
from utils.formatacao import formatar_brasileiro, classificar_qualidade_modelo

# Importar configurações centralizadas
from config.configuracoes_globais import (
    obter_configuracao_global,
    aplicar_filtros_configuracao_global,
    mostrar_status_configuracao_sidebar
)

# Importar componentes visuais padronizados
from ui.components import (
    configurar_pagina_greenvista,
    criar_cabecalho_greenvista,
    criar_navegacao_rapida_botoes,
    mostrar_alertas_sistema
)
from ui.sidebar import criar_sidebar_melhorada

# Configurar página com identidade visual
configurar_pagina_greenvista("Modelos Hipsométricos", "🌳")


def gerar_key_unica(base_key):
    """Gera uma key única para evitar conflitos"""
    timestamp = int(time.time() * 1000)
    return f"{base_key}_{timestamp}"


def verificar_prerequisitos():
    """Verifica se pré-requisitos estão atendidos"""
    problemas = []

    # Verificar dados carregados
    if not hasattr(st.session_state, 'arquivos_carregados') or not st.session_state.arquivos_carregados:
        problemas.append("Dados não carregados")

    # Verificar configuração global
    config_global = obter_configuracao_global()
    if not config_global.get('configurado', False):
        problemas.append("Sistema não configurado")

    if problemas:
        st.error("❌ Pré-requisitos não atendidos:")
        for problema in problemas:
            st.error(f"• {problema}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🏠 Página Principal", key="btn_principal_hip"):
                st.switch_page("Principal.py")
        with col2:
            if st.button("⚙️ Configurações", key="btn_config_hip"):
                st.switch_page("pages/0_⚙️_Configurações.py")

        return False

    return True


def mostrar_configuracao_aplicada():
    """Mostra as configurações que serão aplicadas nesta etapa"""
    config = obter_configuracao_global()

    with st.expander("⚙️ Configurações Aplicadas nesta Etapa"):
        col1, col2 = st.columns(2)

        with col1:
            st.write("**🔍 Filtros de Dados:**")
            st.write(f"• Diâmetro mínimo: {config.get('diametro_min', 4.0)} cm")

            talhoes_excluir = config.get('talhoes_excluir', [])
            if talhoes_excluir:
                st.write(f"• Talhões excluídos: {talhoes_excluir}")
            else:
                st.write("• Talhões excluídos: Nenhum")

            codigos_excluir = config.get('codigos_excluir', [])
            if codigos_excluir:
                st.write(f"• Códigos excluídos: {codigos_excluir}")
            else:
                st.write("• Códigos excluídos: Nenhum")

        with col2:
            st.write("**🧮 Configurações de Modelos:**")
            st.write(
                f"• Modelos não-lineares: {'Incluídos' if config.get('incluir_nao_lineares', True) else 'Excluídos'}")
            st.write(f"• Máximo iterações: {config.get('max_iteracoes', 5000)}")
            st.write(f"• Tolerância: {config.get('tolerancia_ajuste', 0.01)}")

    # Mostrar parâmetros dos modelos não-lineares
    if config.get('incluir_nao_lineares', True):
        mostrar_parametros_nao_lineares(config)

    # Botão para ajustar configurações
    if st.button("🔧 Ajustar Configurações", key="btn_ajustar_config_hip"):
        st.switch_page("pages/0_⚙️_Configurações.py")


def mostrar_parametros_nao_lineares(config):
    """Mostra parâmetros dos modelos não-lineares configurados"""
    with st.expander("🔧 Parâmetros Iniciais dos Modelos Não-Lineares"):
        st.info("💡 Estes são os parâmetros iniciais configurados para os modelos não-lineares")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("**Chapman (H = b₀(1-e^(-b₁D))^b₂):**")
            chapman = config.get('parametros_chapman', {})
            st.write(f"• b₀ = {chapman.get('b0', 42.12)}")
            st.write(f"• b₁ = {chapman.get('b1', 0.01)}")
            st.write(f"• b₂ = {chapman.get('b2', 1.00)}")

        with col2:
            st.write("**Weibull (H = a(1-e^(-bD^c))):**")
            weibull = config.get('parametros_weibull', {})
            st.write(f"• a = {weibull.get('a', 42.12)}")
            st.write(f"• b = {weibull.get('b', 0.01)}")
            st.write(f"• c = {weibull.get('c', 1.00)}")

        with col3:
            st.write("**Mononuclear (H = a(1-be^(-cD))):**")
            mono = config.get('parametros_mononuclear', {})
            st.write(f"• a = {mono.get('a', 42.12)}")
            st.write(f"• b = {mono.get('b', 1.00)}")
            st.write(f"• c = {mono.get('c', 0.10)}")

        # Validar parâmetros e mostrar avisos
        validacao = validar_parametros_configuracao(config)
        if validacao['avisos']:
            st.warning("⚠️ **Avisos sobre os parâmetros configurados:**")
            for aviso in validacao['avisos']:
                st.warning(f"• {aviso}")

        if validacao['erros']:
            st.error("❌ **Erros nos parâmetros:**")
            for erro in validacao['erros']:
                st.error(f"• {erro}")
        else:
            st.success("✅ Parâmetros válidos para os modelos não-lineares")


def aplicar_preview_dados():
    """Aplica filtros e mostra preview dos dados"""
    st.subheader("📊 Preview dos Dados Filtrados")

    # Usar filtros centralizados
    df_original = st.session_state.dados_inventario
    df_filtrado = aplicar_filtros_configuracao_global(df_original)

    # Mostrar estatísticas
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        delta_registros = len(df_filtrado) - len(df_original)
        st.metric("Registros", f"{len(df_filtrado):,}", delta=f"{delta_registros:+,}")

    with col2:
        delta_talhoes = df_filtrado['talhao'].nunique() - df_original['talhao'].nunique()
        st.metric("Talhões", df_filtrado['talhao'].nunique(), delta=f"{delta_talhoes:+}")

    with col3:
        if len(df_original) > 0:
            percentual = (len(df_filtrado) / len(df_original)) * 100
            st.metric("Dados Mantidos", f"{percentual:.1f}%")
        else:
            st.metric("Dados Mantidos", "0%")

    with col4:
        if len(df_filtrado) < 20:
            st.error("⚠️ Poucos dados!")
        elif len(df_filtrado) < 50:
            st.warning("⚠️ Dados limitados")
        else:
            st.success("✅ Dados suficientes")

    # Verificar se há dados suficientes
    if len(df_filtrado) < 20:
        st.error("❌ Dados insuficientes após aplicar filtros")
        st.info("💡 Ajuste as configurações para incluir mais dados")
        return None

    # Preview dos dados
    with st.expander("👀 Preview dos Dados"):
        st.write("**Primeiros registros após filtros:**")
        st.dataframe(df_filtrado.head(10))

        st.write("**Distribuição por talhão:**")
        dist_talhao = df_filtrado['talhao'].value_counts().sort_index()
        st.dataframe(dist_talhao.reset_index().rename(columns={'index': 'Talhão', 'talhao': 'Registros'}))

    return df_filtrado


def executar_analise_hipsometrica():
    """Executa análise hipsométrica com configurações centralizadas"""
    st.header("🚀 Executando Análise Hipsométrica")

    # Aplicar filtros centralizados
    df_filtrado = aplicar_filtros_configuracao_global(st.session_state.dados_inventario)

    if len(df_filtrado) < 20:
        st.error("❌ Dados insuficientes após filtros. Ajuste as configurações.")
        return

    # Obter configurações globais completas
    config = obter_configuracao_global()

    # Validar parâmetros antes de executar
    validacao = validar_parametros_configuracao(config)
    if validacao['erros']:
        st.error("❌ Erro nas configurações dos parâmetros:")
        for erro in validacao['erros']:
            st.error(f"• {erro}")
        return

    if validacao['avisos']:
        st.warning("⚠️ Avisos sobre as configurações:")
        for aviso in validacao['avisos']:
            st.warning(f"• {aviso}")

    # Barra de progresso
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.text("🔄 Preparando dados...")
        progress_bar.progress(0.1)

        status_text.text("🧮 Ajustando modelos com parâmetros configurados...")
        progress_bar.progress(0.3)

        # Chamar função COM configurações completas (incluindo parâmetros)
        resultados, predicoes, melhor_modelo = ajustar_todos_modelos_hipsometricos(df_filtrado, config)

        progress_bar.progress(1.0)
        status_text.text("✅ Análise concluída!")

        if not resultados:
            st.error("❌ Nenhum modelo foi ajustado com sucesso")
            return

        # Salvar resultados com configuração aplicada
        st.session_state.resultados_hipsometricos = {
            'resultados': resultados,
            'predicoes': predicoes,
            'melhor_modelo': melhor_modelo,
            'dados': df_filtrado,
            'config_aplicada': config,
            'timestamp': pd.Timestamp.now()
        }

        st.success(f"🏆 Melhor modelo: **{melhor_modelo}**")

        # Mostrar informações sobre parâmetros utilizados
        mostrar_info_parametros_utilizados(config, resultados)

        # Mostrar resultados
        mostrar_resultados_hipsometricos(resultados, predicoes, df_filtrado, contexto="novo")

    except Exception as e:
        st.error(f"❌ Erro na análise: {e}")
        st.info("💡 Verifique as configurações e tente novamente")

        # Debug detalhado
        with st.expander("🔍 Debug Detalhado"):
            st.write(f"**Tipo do erro**: {type(e).__name__}")
            st.write(f"**Mensagem**: {str(e)}")
            st.write(f"**Dados filtrados**: {len(df_filtrado)} registros")
            st.write(f"**Configurações**: {config}")


def mostrar_info_parametros_utilizados(config, resultados):
    """Mostra informações sobre parâmetros utilizados nos modelos"""
    with st.expander("🔧 Parâmetros Utilizados nos Modelos"):
        st.info("💡 Informações sobre como as configurações foram aplicadas nos modelos")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**⚙️ Configurações Aplicadas:**")
            st.write(f"• Incluir não-lineares: {'Sim' if config.get('incluir_nao_lineares', True) else 'Não'}")
            st.write(f"• Máx. iterações: {config.get('max_iteracoes', 5000)}")
            st.write(f"• Tolerância: {config.get('tolerancia_ajuste', 0.01)}")

            if config.get('incluir_nao_lineares', True):
                st.write("**🔧 Parâmetros Iniciais Usados:**")
                chapman = config.get('parametros_chapman', {})
                st.write(
                    f"• Chapman: b₀={chapman.get('b0', 42.12)}, b₁={chapman.get('b1', 0.01)}, b₂={chapman.get('b2', 1.00)}")

                weibull = config.get('parametros_weibull', {})
                st.write(
                    f"• Weibull: a={weibull.get('a', 42.12)}, b={weibull.get('b', 0.01)}, c={weibull.get('c', 1.00)}")

                mono = config.get('parametros_mononuclear', {})
                st.write(f"• Mononuclear: a={mono.get('a', 42.12)}, b={mono.get('b', 1.00)}, c={mono.get('c', 0.10)}")

        with col2:
            st.write("**📊 Resultados Obtidos:**")
            modelos_lineares = [m for m in resultados.keys() if m in ['Curtis', 'Campos', 'Henri', 'Prodan']]
            modelos_nao_lineares = [m for m in resultados.keys() if m in ['Chapman', 'Weibull', 'Mononuclear']]

            st.write(f"• Modelos lineares ajustados: {len(modelos_lineares)}")
            st.write(f"• Modelos não-lineares ajustados: {len(modelos_nao_lineares)}")
            st.write(f"• Total de modelos: {len(resultados)}")

            if modelos_nao_lineares:
                st.write("**🏆 Performance dos Não-Lineares:**")
                for modelo in modelos_nao_lineares:
                    if modelo in resultados:
                        r2 = resultados[modelo]['r2g']
                        st.write(f"• {modelo}: R² = {r2:.4f}")

        # Mostrar aviso se parâmetros podem não ter convergido bem
        if config.get('incluir_nao_lineares', True):
            modelos_nao_lineares_resultado = [m for m in resultados.keys() if
                                              m in ['Chapman', 'Weibull', 'Mononuclear']]
            if len(modelos_nao_lineares_resultado) < 3:
                st.warning(
                    "⚠️ Alguns modelos não-lineares falharam. Considere ajustar os parâmetros iniciais nas configurações.")


def mostrar_resultados_hipsometricos(resultados, predicoes, df_dados, contexto="novo"):
    """Mostra resultados dos modelos hipsométricos"""
    st.header("📊 Resultados dos Modelos Hipsométricos")

    # Adicionar identificador do contexto
    if contexto == "salvo":
        st.info("ℹ️ Exibindo resultados salvos da execução anterior")

    # Criar sufixo único baseado no contexto e timestamp
    sufixo = f"_{contexto}_{int(time.time())}"

    # Tabs para organizar resultados
    tab1, tab2, tab3, tab4 = st.tabs(["🏆 Ranking", "📊 Gráficos", "🔢 Coeficientes", "💾 Downloads"])

    with tab1:
        mostrar_ranking_modelos(resultados)

    with tab2:
        try:
            criar_graficos_modelos(df_dados, resultados, predicoes, 'hipsometrico')
        except Exception as e:
            st.error(f"Erro ao criar gráficos: {e}")
            st.info("Recarregue a página se o erro persistir")

    with tab3:
        mostrar_coeficientes_modelos(resultados)

    with tab4:
        mostrar_downloads_hipsometricos(resultados, predicoes, df_dados, sufixo)


def mostrar_ranking_modelos(resultados):
    """Mostra ranking dos modelos"""
    st.subheader("🏆 Ranking dos Modelos")

    ranking_data = []
    for modelo, resultado in resultados.items():
        r2g = resultado['r2g']
        rmse = resultado['rmse']
        qualidade = classificar_qualidade_modelo(r2g)

        ranking_data.append({
            'Ranking': 0,
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


def mostrar_coeficientes_modelos(resultados):
    """Mostra coeficientes detalhados dos modelos"""
    st.subheader("🔢 Coeficientes dos Modelos")

    for modelo, resultado in resultados.items():
        with st.expander(f"📊 {modelo} - Coeficientes"):
            modelo_obj = resultado.get('modelo')

            if modelo_obj and hasattr(modelo_obj, 'modelo'):
                try:
                    if hasattr(modelo_obj.modelo, 'coef_'):
                        # Modelo linear
                        coefs = modelo_obj.modelo.coef_
                        intercept = modelo_obj.modelo.intercept_

                        st.write(f"**Intercepto**: {intercept:.6f}")
                        for i, coef in enumerate(coefs):
                            st.write(f"**Coeficiente {i + 1}**: {coef:.6f}")

                    elif hasattr(modelo_obj, 'parametros'):
                        # Modelo não-linear
                        params = modelo_obj.parametros

                        # Mostrar parâmetros iniciais vs finais
                        params_iniciais = modelo_obj.params_iniciais

                        st.write("**Parâmetros Iniciais vs Finais:**")
                        for i, (inicial, final) in enumerate(zip(params_iniciais, params)):
                            param_name = get_parameter_name(modelo, i)
                            st.write(f"**{param_name}**: {inicial:.6f} → {final:.6f}")

                        st.write("**Convergência:**")
                        convergiu = all(abs(f - i) < 100 for i, f in zip(params_iniciais, params))
                        if convergiu:
                            st.success("✅ Modelo convergiu adequadamente")
                        else:
                            st.warning("⚠️ Grandes mudanças nos parâmetros - verificar convergência")

                    else:
                        st.info("Coeficientes não disponíveis para este modelo")

                except Exception as e:
                    st.warning(f"Erro ao exibir coeficientes: {e}")

            # Estatísticas do modelo
            st.write(f"**R² Generalizado**: {resultado['r2g']:.4f}")
            st.write(f"**RMSE**: {resultado['rmse']:.4f}")


def get_parameter_name(modelo, index):
    """Retorna nomes dos parâmetros para cada modelo"""
    param_names = {
        'Chapman': ['b₀', 'b₁', 'b₂'],
        'Weibull': ['a', 'b', 'c'],
        'Mononuclear': ['a', 'b', 'c']
    }

    if modelo in param_names and index < len(param_names[modelo]):
        return param_names[modelo][index]
    else:
        return f"Parâmetro {index + 1}"


def mostrar_downloads_hipsometricos(resultados, predicoes, df_dados, sufixo=""):
    """Mostra opções de download"""
    st.subheader("💾 Downloads")

    # Ranking dos modelos
    ranking_data = []
    for modelo, resultado in resultados.items():
        ranking_data.append({
            'Modelo': modelo,
            'R2_Generalizado': resultado['r2g'],
            'RMSE': resultado['rmse'],
            'Qualidade': classificar_qualidade_modelo(resultado['r2g'])
        })

    df_ranking = pd.DataFrame(ranking_data).sort_values('R2_Generalizado', ascending=False)

    col1, col2, col3 = st.columns(3)

    with col1:
        csv_ranking = df_ranking.to_csv(index=False)
        st.download_button(
            "📊 Ranking dos Modelos",
            csv_ranking,
            "ranking_modelos_hipsometricos.csv",
            "text/csv",
            key=gerar_key_unica(f"download_ranking_hip{sufixo}")
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
            key=gerar_key_unica(f"download_dados_hip{sufixo}")
        )

    with col3:
        # Relatório com parâmetros utilizados
        relatorio = gerar_relatorio_hipsometrico_com_parametros(resultados, df_ranking)
        st.download_button(
            "📄 Relatório com Parâmetros",
            relatorio,
            "relatorio_hipsometricos_parametros.md",
            "text/markdown",
            key=gerar_key_unica(f"download_relatorio_hip{sufixo}")
        )


def gerar_relatorio_hipsometrico_com_parametros(resultados, df_ranking):
    """Gera relatório incluindo parâmetros utilizados"""
    config = obter_configuracao_global()
    melhor = df_ranking.iloc[0]

    relatorio = f"""# RELATÓRIO - MODELOS HIPSOMÉTRICOS COM PARÂMETROS

## 🏆 MELHOR MODELO
**{melhor['Modelo']}** - {melhor['Qualidade']}
- R² Generalizado: {melhor['R2_Generalizado']:.4f}
- RMSE: {melhor['RMSE']:.4f}

## ⚙️ CONFIGURAÇÕES APLICADAS
### Filtros de Dados:
- Diâmetro mínimo: {config.get('diametro_min', 4.0)} cm
- Talhões excluídos: {config.get('talhoes_excluir', [])}
- Códigos excluídos: {config.get('codigos_excluir', [])}

### Configurações de Modelos:
- Modelos não-lineares: {'Incluídos' if config.get('incluir_nao_lineares', True) else 'Excluídos'}
- Máximo iterações: {config.get('max_iteracoes', 5000)}
- Tolerância: {config.get('tolerancia_ajuste', 0.01)}

## 🔧 PARÂMETROS INICIAIS DOS MODELOS NÃO-LINEARES
"""

    if config.get('incluir_nao_lineares', True):
        # Chapman
        chapman = config.get('parametros_chapman', {})
        relatorio += f"""
### Chapman (H = b₀(1-e^(-b₁D))^b₂)
- b₀ (altura assintótica): {chapman.get('b0', 42.12)}
- b₁ (taxa de crescimento): {chapman.get('b1', 0.01)}
- b₂ (parâmetro de forma): {chapman.get('b2', 1.00)}
"""

        # Weibull
        weibull = config.get('parametros_weibull', {})
        relatorio += f"""
### Weibull (H = a(1-e^(-bD^c)))
- a (altura assintótica): {weibull.get('a', 42.12)}
- b (parâmetro de escala): {weibull.get('b', 0.01)}
- c (parâmetro de forma): {weibull.get('c', 1.00)}
"""

        # Mononuclear
        mono = config.get('parametros_mononuclear', {})
        relatorio += f"""
### Mononuclear (H = a(1-be^(-cD)))
- a (altura assintótica): {mono.get('a', 42.12)}
- b (parâmetro de intercepto): {mono.get('b', 1.00)}
- c (taxa de decaimento): {mono.get('c', 0.10)}
"""

    relatorio += f"""
## 📊 RANKING COMPLETO
"""

    for i, (_, row) in enumerate(df_ranking.iterrows(), 1):
        relatorio += f"\n{i}. **{row['Modelo']}** - {row['Qualidade']}"
        relatorio += f"\n   - R²: {row['R2_Generalizado']:.4f}, RMSE: {row['RMSE']:.4f}\n"

    relatorio += f"""
## 📈 RESUMO DA ANÁLISE
- Total de modelos avaliados: {len(resultados)}
- Configuração centralizada aplicada: ✅
- Parâmetros iniciais utilizados: ✅
- Timestamp: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 🎯 RECOMENDAÇÃO
Use o modelo **{melhor['Modelo']}** para estimativas de altura neste povoamento.

---
*Relatório gerado pelo GreenVista - Sistema de Inventário Florestal*
*Parâmetros iniciais dos modelos não-lineares aplicados conforme configuração global*
"""

    return relatorio


def main():
    # Verificar pré-requisitos
    if not verificar_prerequisitos():
        return

    # Criar cabeçalho padronizado
    criar_cabecalho_greenvista("Modelos Hipsométricos")

    # Criar sidebar padronizada
    criar_sidebar_melhorada()

    # Mostrar status da configuração na sidebar
    mostrar_status_configuracao_sidebar()

    # Botão para limpar resultados anteriores (evita conflitos de keys)
    if st.button("🗑️ Limpar Resultados Anteriores", key="limpar_resultados_hip"):
        if 'resultados_hipsometricos' in st.session_state:
            del st.session_state.resultados_hipsometricos
            st.success("✅ Resultados limpos!")
            st.rerun()

    # Mostrar configurações aplicadas
    mostrar_configuracao_aplicada()

    # Preview com filtros aplicados
    df_filtrado = aplicar_preview_dados()
    if df_filtrado is None:
        return

    # Informações dos modelos
    with st.expander("📚 Informações dos Modelos Hipsométricos"):
        st.markdown("""
        ### 🧮 Modelos Disponíveis

        **Lineares:**
        - **Curtis**: ln(H) = β₀ + β₁ × (1/D)
        - **Campos**: ln(H) = β₀ + β₁ × (1/D) + β₂ × ln(H_dom)
        - **Henri**: H = β₀ + β₁ × ln(D)
        - **Prodan**: D²/(H-1.3) = β₀ + β₁×D + β₂×D² + β₃×D×Idade

        **Não-lineares (com parâmetros configuráveis):**
        - **Chapman**: H = b₀ × (1 - e^(-b₁×D))^b₂
        - **Weibull**: H = a × (1 - e^(-b×D^c))
        - **Mononuclear**: H = a × (1 - b × e^(-c×D))

        💡 **Novidade**: Os parâmetros iniciais dos modelos não-lineares são configurados 
        na Etapa 0 e aplicados automaticamente aqui!
        """)

    # Botão para executar análise
    if st.button("🚀 Executar Análise Hipsométrica", type="primary", use_container_width=True):
        executar_analise_hipsometrica()

    # Mostrar resultados salvos se existirem
    if hasattr(st.session_state, 'resultados_hipsometricos') and st.session_state.resultados_hipsometricos:
        st.markdown("---")
        st.subheader("📂 Resultados Salvos")

        resultados_salvos = st.session_state.resultados_hipsometricos

        # Verificar se configuração mudou desde a última execução
        config_atual = obter_configuracao_global()
        config_salva = resultados_salvos.get('config_aplicada', {})

        if config_atual != config_salva:
            st.warning("""
            ⚠️ **Configurações Alteradas**

            As configurações globais foram modificadas desde a última execução.
            Os resultados abaixo podem não refletir as configurações atuais.

            **Recomendação**: Execute a análise novamente para aplicar as novas configurações.
            """)

        # Checkbox para controlar exibição e evitar conflitos
        if st.checkbox("👀 Mostrar Resultados Salvos", key="mostrar_resultados_salvos_hip"):
            mostrar_resultados_hipsometricos(
                resultados_salvos['resultados'],
                resultados_salvos['predicoes'],
                resultados_salvos['dados'],
                "salvo"  # CONTEXTO DIFERENTE PARA EVITAR CONFLITO DE KEYS
            )

    # Navegação rápida final
    st.markdown("---")
    criar_navegacao_rapida_botoes()

    # Mostrar alertas do sistema
    mostrar_alertas_sistema()


if __name__ == "__main__":
    main()