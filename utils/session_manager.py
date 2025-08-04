# utils/session_manager.py - VERSÃO COMPLETA E FUNCIONAL
"""
Gerenciador completo do session_state - Versão sem erros de import
"""

import streamlit as st
import pandas as pd
from typing import Optional, Tuple, List, Dict, Any


def verificar_dados_inventario() -> Tuple[bool, Optional[pd.DataFrame], List[str]]:
    """
    Verifica se dados de inventário estão disponíveis

    Returns:
        tuple: (dados_encontrados, dataframe, diagnosticos)
    """
    diagnosticos = []

    # Estratégia 1: Verificar dados_inventario diretamente
    if hasattr(st.session_state, 'dados_inventario'):
        dados = st.session_state.dados_inventario
        if dados is not None and isinstance(dados, pd.DataFrame) and len(dados) > 0:
            # Verificar colunas necessárias
            colunas_necessarias = ['D_cm', 'H_m', 'talhao']
            colunas_presentes = [col for col in colunas_necessarias if col in dados.columns]

            if len(colunas_presentes) >= 3:
                diagnosticos.append(f"✅ Dados encontrados em 'dados_inventario': {len(dados)} registros")
                return True, dados, diagnosticos
            else:
                diagnosticos.append(
                    f"❌ dados_inventario existe mas faltam colunas: {set(colunas_necessarias) - set(colunas_presentes)}")
        else:
            diagnosticos.append("❌ dados_inventario existe mas está vazio ou inválido")
    else:
        diagnosticos.append("❌ dados_inventario não encontrado")

    # Estratégia 2: Buscar em outras keys
    diagnosticos.append("🔍 Buscando dados em outras keys...")

    keys_candidatas = []
    for key in st.session_state.keys():
        if any(termo in key.lower() for termo in ['inventario', 'dados', 'df']):
            valor = st.session_state[key]
            if isinstance(valor, pd.DataFrame) and len(valor) > 0:
                keys_candidatas.append((key, valor))

    if keys_candidatas:
        diagnosticos.append(f"Encontradas {len(keys_candidatas)} keys candidatas")

        for key, dados in keys_candidatas:
            colunas_necessarias = ['D_cm', 'H_m', 'talhao']
            colunas_presentes = [col for col in colunas_necessarias if col in dados.columns]

            if len(colunas_presentes) >= 2:  # Critério mais flexível
                diagnosticos.append(f"✅ Dados válidos encontrados em '{key}': {len(dados)} registros")

                # Atualizar referência principal
                st.session_state.dados_inventario = dados
                diagnosticos.append("🔄 Referência 'dados_inventario' atualizada")

                return True, dados, diagnosticos

    diagnosticos.append("❌ Nenhum dado de inventário válido encontrado")
    return False, None, diagnosticos


def verificar_configuracao() -> Tuple[bool, Optional[Dict], List[str]]:
    """
    Verifica se configuração está disponível

    Returns:
        tuple: (configurado, config_dict, diagnosticos)
    """
    diagnosticos = []

    if not hasattr(st.session_state, 'config_global'):
        diagnosticos.append("❌ config_global não encontrada")
        return False, None, diagnosticos

    config = st.session_state.config_global

    if not isinstance(config, dict):
        diagnosticos.append("❌ config_global não é um dicionário")
        return False, None, diagnosticos

    diagnosticos.append(f"✅ config_global encontrada com {len(config)} keys")

    # Verificar se está marcada como configurada
    if config.get('configurado', False):
        diagnosticos.append("✅ Marcada como configurada")
        return True, config, diagnosticos

    # Verificar se tem conteúdo que indica configuração
    keys_importantes = ['diametro_min', 'metodo_area', 'incluir_nao_lineares']
    keys_presentes = [key for key in keys_importantes if key in config]

    if len(keys_presentes) >= 2:
        diagnosticos.append(f"⚠️ Parece configurada ({len(keys_presentes)}/3 keys importantes) mas não marcada")

        # Auto-corrigir
        try:
            config['configurado'] = True
            st.session_state.config_global = config
            diagnosticos.append("🔄 Auto-corrigido flag 'configurado'")
            return True, config, diagnosticos
        except Exception as e:
            diagnosticos.append(f"❌ Erro ao auto-corrigir: {e}")
            return False, config, diagnosticos
    else:
        diagnosticos.append(f"❌ Configuração incompleta (apenas {len(keys_presentes)}/3 keys importantes)")
        return False, config, diagnosticos


def sincronizar_flags() -> List[str]:
    """
    Sincroniza flags do session_state

    Returns:
        list: Correções aplicadas
    """
    correcoes = []

    try:
        # Verificar dados
        dados_ok, dados, _ = verificar_dados_inventario()

        if dados_ok:
            if not hasattr(st.session_state, 'arquivos_carregados') or not st.session_state.arquivos_carregados:
                st.session_state.arquivos_carregados = True
                correcoes.append("Flag 'arquivos_carregados' setada")
        else:
            if hasattr(st.session_state, 'arquivos_carregados') and st.session_state.arquivos_carregados:
                st.session_state.arquivos_carregados = False
                correcoes.append("Flag 'arquivos_carregados' limpa")

        # Verificar configuração (já faz auto-correção interna)
        config_ok, _, _ = verificar_configuracao()

    except Exception as e:
        correcoes.append(f"Erro na sincronização: {e}")

    return correcoes


def inicializar_pagina_sistema() -> bool:
    """
    Função principal para inicializar qualquer página

    Returns:
        bool: True se pré-requisitos estão OK
    """
    try:
        # Sincronizar flags
        correcoes = sincronizar_flags()

        # Verificar status final
        dados_ok, _, _ = verificar_dados_inventario()
        config_ok, _, _ = verificar_configuracao()

        return dados_ok and config_ok

    except Exception as e:
        st.error(f"Erro na inicialização do sistema: {e}")
        return False


def mostrar_debug_sidebar():
    """Mostra debug na sidebar"""
    if st.sidebar.checkbox("🔍 Debug Session State", key="debug_session_state"):
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔧 Debug")

        # Status dos dados
        dados_ok, dados, diag_dados = verificar_dados_inventario()

        if dados_ok:
            st.sidebar.success(f"✅ Dados: {len(dados)} registros")
            st.sidebar.write(f"Talhões: {dados['talhao'].nunique()}")
        else:
            st.sidebar.error("❌ Dados não encontrados")

        # Status da configuração
        config_ok, config, diag_config = verificar_configuracao()

        if config_ok:
            st.sidebar.success("✅ Configuração OK")
            if config:
                st.sidebar.write(f"Keys: {len(config)}")
        else:
            st.sidebar.error("❌ Não configurado")

        # Diagnósticos detalhados
        if st.sidebar.checkbox("📋 Diagnóstico Detalhado"):
            st.sidebar.write("**Dados:**")
            for diag in diag_dados[-3:]:
                st.sidebar.caption(diag)

            st.sidebar.write("**Config:**")
            for diag in diag_config[-3:]:
                st.sidebar.caption(diag)

        # Botão para sincronizar
        if st.sidebar.button("🔄 Sincronizar"):
            correcoes = sincronizar_flags()
            if correcoes:
                for correcao in correcoes:
                    st.sidebar.success(f"✅ {correcao}")
                st.rerun()
            else:
                st.sidebar.info("ℹ️ Já sincronizado")


def mostrar_pagina_erro():
    """Mostra página de erro padrão"""
    st.error("❌ Pré-requisitos não atendidos")

    # Status visual
    dados_ok, dados, diag_dados = verificar_dados_inventario()
    config_ok, config, diag_config = verificar_configuracao()

    col1, col2 = st.columns(2)

    with col1:
        if dados_ok:
            st.success(f"✅ Dados: {len(dados)} registros")
        else:
            st.error("❌ Dados não carregados")

    with col2:
        if config_ok:
            st.success("✅ Sistema configurado")
        else:
            st.error("❌ Sistema não configurado")

    # Diagnóstico expandido
    with st.expander("🔍 Diagnóstico Detalhado"):
        if not dados_ok:
            st.write("**❌ Problema com dados:**")
            for linha in diag_dados:
                st.write(f"• {linha}")

            st.info("""
            **💡 Como resolver:**
            1. Volte à página principal
            2. Faça upload dos arquivos de inventário
            3. Certifique-se de que o upload foi concluído
            """)

        if not config_ok:
            st.write("**❌ Problema com configuração:**")
            for linha in diag_config:
                st.write(f"• {linha}")

            st.info("""
            **💡 Como resolver:**
            1. Vá para a página de Configurações
            2. Configure todos os parâmetros
            3. Clique em 'Salvar Configurações'
            """)

    # Botões de navegação
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🏠 Página Principal", key="nav_principal"):
            st.switch_page("Principal.py")

    with col2:
        if st.button("⚙️ Configurações", key="nav_config"):
            st.switch_page("pages/0_⚙️_Configurações.py")

    with col3:
        if st.button("🔄 Recarregar", key="nav_reload"):
            st.rerun()


# FUNÇÕES DE CONVENIÊNCIA

def obter_dados_inventario() -> Optional[pd.DataFrame]:
    """Obtém dados de inventário de forma segura"""
    dados_ok, dados, _ = verificar_dados_inventario()
    return dados if dados_ok else None


def obter_configuracao() -> Optional[Dict]:
    """Obtém configuração de forma segura"""
    config_ok, config, _ = verificar_configuracao()
    return config if config_ok else None


def prerequisitos_ok() -> bool:
    """Verifica se pré-requisitos estão OK"""
    return inicializar_pagina_sistema()


def mostrar_status_prerequisitos():
    """Mostra status visual dos pré-requisitos"""
    dados_ok, dados, _ = verificar_dados_inventario()
    config_ok, config, _ = verificar_configuracao()

    col1, col2 = st.columns(2)

    with col1:
        if dados_ok:
            st.success(f"✅ Dados carregados ({len(dados)} registros)")
        else:
            st.error("❌ Dados não carregados")

    with col2:
        if config_ok:
            st.success("✅ Sistema configurado")
        else:
            st.error("❌ Sistema não configurado")

    return dados_ok and config_ok


# FUNÇÃO PARA CORREÇÃO MANUAL DE EMERGÊNCIA

def correcao_manual_emergencia():
    """Função de emergência para corrigir problemas manualmente"""
    if st.sidebar.button("🚨 Correção de Emergência"):
        try:
            # Buscar qualquer DataFrame que pareça dados de inventário
            for key in st.session_state.keys():
                valor = st.session_state[key]
                if isinstance(valor, pd.DataFrame) and len(valor) > 0:
                    # Verificar se tem pelo menos algumas colunas que indicam inventário
                    colunas_inventario = ['D_cm', 'H_m', 'DAP', 'altura', 'diametro', 'talhao', 'parcela']
                    colunas_encontradas = [col for col in colunas_inventario if col in valor.columns]

                    if len(colunas_encontradas) >= 2:
                        st.session_state.dados_inventario = valor
                        st.session_state.arquivos_carregados = True
                        st.sidebar.success(f"✅ Dados corrigidos de '{key}'")
                        st.sidebar.write(f"Colunas encontradas: {colunas_encontradas}")
                        break

            # Tentar corrigir configuração
            if hasattr(st.session_state, 'config_global'):
                config = st.session_state.config_global
                if isinstance(config, dict) and len(config) > 0:
                    config['configurado'] = True
                    st.session_state.config_global = config
                    st.sidebar.success("✅ Configuração corrigida")

            st.rerun()

        except Exception as e:
            st.sidebar.error(f"❌ Erro na correção: {e}")


# EXEMPLO DE USO COMPLETO

def exemplo_uso_completo():
    """
    EXEMPLO COMPLETO de como usar na sua página:

    # No topo da sua página:
    from utils.session_manager import (
        inicializar_pagina_sistema,
        mostrar_pagina_erro,
        mostrar_debug_sidebar,
        correcao_manual_emergencia
    )

    def main():
        st.title("🌳 Minha Página")

        # Debug e correção de emergência na sidebar
        mostrar_debug_sidebar()
        correcao_manual_emergencia()

        # Verificar pré-requisitos
        if not inicializar_pagina_sistema():
            mostrar_pagina_erro()
            return

        # Se chegou aqui, tudo OK!
        st.success("✅ Sistema pronto")

        # Usar dados com segurança
        dados = st.session_state.dados_inventario
        config = st.session_state.config_global

        # Sua lógica continua...
        st.write(f"Dados: {len(dados)} registros")
        st.write(f"Configuração: {len(config)} parâmetros")
    """
    pass