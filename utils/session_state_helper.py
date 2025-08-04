# utils/session_manager.py
"""
Gerenciador centralizado do session_state para resolver problemas de persistência
Este arquivo deve ser importado no início de cada página para garantir consistência
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional, Tuple


class SessionStateManager:
    """Gerenciador centralizado do session_state"""

    @staticmethod
    def verificar_dados_inventario() -> Dict[str, Any]:
        """
        Verifica status dos dados de inventário de forma robusta

        Returns:
            dict: Status completo dos dados
        """
        resultado = {
            'encontrado': False,
            'dados': None,
            'fonte': None,
            'diagnostico': [],
            'colunas_obrigatorias': ['D_cm', 'H_m', 'talhao', 'parcela'],
            'colunas_encontradas': []
        }

        # Lista de possíveis keys onde os dados podem estar
        keys_candidatas = [
            'dados_inventario',
            'df_inventario',
            'inventario_dados',
            'dados_florestal',
            'arquivo_dados_inventario'
        ]

        # Verificar cada key candidata
        for key in keys_candidatas:
            if hasattr(st.session_state, key):
                valor = getattr(st.session_state, key)
                resultado['diagnostico'].append(f"✓ Encontrada key '{key}'")

                if isinstance(valor, pd.DataFrame) and len(valor) > 0:
                    resultado['diagnostico'].append(f"  - É DataFrame com {len(valor)} registros")

                    # Verificar colunas
                    colunas_presentes = []
                    for col_obrigatoria in resultado['colunas_obrigatorias']:
                        if col_obrigatoria in valor.columns:
                            colunas_presentes.append(col_obrigatoria)

                    resultado['colunas_encontradas'] = colunas_presentes

                    # Se tem pelo menos 3 das 4 colunas obrigatórias, considerar válido
                    if len(colunas_presentes) >= 3:
                        resultado['encontrado'] = True
                        resultado['dados'] = valor
                        resultado['fonte'] = key
                        resultado['diagnostico'].append(f"  ✅ Dados válidos encontrados em '{key}'")

                        # Padronizar referência principal
                        if key != 'dados_inventario':
                            st.session_state.dados_inventario = valor
                            resultado['diagnostico'].append("  🔄 Referência padronizada para 'dados_inventario'")

                        return resultado
                    else:
                        resultado['diagnostico'].append(
                            f"  ❌ Faltam colunas obrigatórias: {set(resultado['colunas_obrigatorias']) - set(colunas_presentes)}")

                else:
                    resultado['diagnostico'].append(f"  ❌ '{key}' não é DataFrame válido")
            else:
                resultado['diagnostico'].append(f"✗ Key '{key}' não encontrada")

        # Se nenhuma key padrão funcionou, buscar em todas as keys
        if not resultado['encontrado']:
            resultado['diagnostico'].append("🔍 Buscando em todas as keys do session_state...")

            for key in st.session_state.keys():
                if any(termo in key.lower() for termo in ['dados', 'inventario', 'df', 'florestal']):
                    valor = getattr(st.session_state, key)
                    if isinstance(valor, pd.DataFrame) and len(valor) > 0:
                        # Verificar se parece com dados de inventário
                        colunas_valor = valor.columns.tolist()
                        colunas_presentes = [col for col in resultado['colunas_obrigatorias'] if col in colunas_valor]

                        if len(colunas_presentes) >= 2:  # Critério mais flexível na busca geral
                            resultado['encontrado'] = True
                            resultado['dados'] = valor
                            resultado['fonte'] = key
                            resultado['colunas_encontradas'] = colunas_presentes
                            resultado['diagnostico'].append(f"  ✅ Dados encontrados em '{key}' (busca geral)")

                            # Padronizar referência
                            st.session_state.dados_inventario = valor
                            resultado['diagnostico'].append("  🔄 Referência criada como 'dados_inventario'")
                            break

        return resultado

    @staticmethod
    def verificar_configuracao() -> Dict[str, Any]:
        """
        Verifica status da configuração global

        Returns:
            dict: Status da configuração
        """
        resultado = {
            'existe': False,
            'configurado': False,
            'config': None,
            'diagnostico': [],
            'keys_importantes': ['diametro_min', 'metodo_area', 'incluir_nao_lineares']
        }

        if hasattr(st.session_state, 'config_global'):
            config = st.session_state.config_global
            resultado['existe'] = True
            resultado['config'] = config
            resultado['diagnostico'].append("✓ config_global encontrada")

            if isinstance(config, dict):
                # Verificar se está marcada como configurada
                if config.get('configurado', False):
                    resultado['configurado'] = True
                    resultado['diagnostico'].append("✅ Marcada como configurada")
                else:
                    # Verificar se tem conteúdo que indica configuração real
                    keys_presentes = [key for key in resultado['keys_importantes'] if key in config]
                    if len(keys_presentes) >= 2:
                        resultado['diagnostico'].append("⚠️ Parece configurada mas não marcada como tal")
                        # Auto-corrigir
                        config['configurado'] = True
                        st.session_state.config_global = config
                        resultado['configurado'] = True
                        resultado['diagnostico'].append("🔄 Auto-corrigido flag 'configurado'")
                    else:
                        resultado['diagnostico'].append("❌ Configuração incompleta")

                resultado['diagnostico'].append(f"  - Keys presentes: {len(config)}")
                resultado['diagnostico'].append(f"  - Keys importantes: {keys_presentes}")
            else:
                resultado['diagnostico'].append("❌ config_global não é dict")
        else:
            resultado['diagnostico'].append("✗ config_global não encontrada")

        return resultado

    @staticmethod
    def sincronizar_flags() -> list:
        """
        Sincroniza flags do session_state com estado real

        Returns:
            list: Lista de correções aplicadas
        """
        correcoes = []

        # Verificar dados
        status_dados = SessionStateManager.verificar_dados_inventario()
        if status_dados['encontrado']:
            if not hasattr(st.session_state, 'arquivos_carregados') or not st.session_state.arquivos_carregados:
                st.session_state.arquivos_carregados = True
                correcoes.append(f"Flag 'arquivos_carregados' setada (dados em '{status_dados['fonte']}')")
        else:
            if hasattr(st.session_state, 'arquivos_carregados') and st.session_state.arquivos_carregados:
                st.session_state.arquivos_carregados = False
                correcoes.append("Flag 'arquivos_carregados' limpa (sem dados válidos)")

        # Verificar configuração
        status_config = SessionStateManager.verificar_configuracao()
        if status_config['configurado']:
            # Já está correto ou foi auto-corrigido
            pass

        return correcoes

    @staticmethod
    def diagnostico_completo() -> Dict[str, Any]:
        """
        Diagnóstico completo do session_state

        Returns:
            dict: Diagnóstico detalhado
        """
        diagnostico = {
            'dados': SessionStateManager.verificar_dados_inventario(),
            'configuracao': SessionStateManager.verificar_configuracao(),
            'session_state_info': {
                'total_keys': len(st.session_state.keys()),
                'keys_relevantes': []
            },
            'flags': {
                'arquivos_carregados': getattr(st.session_state, 'arquivos_carregados', False)
            }
        }

        # Listar keys relevantes
        for key in st.session_state.keys():
            if any(termo in key.lower() for termo in ['dados', 'config', 'arquivo', 'resultado', 'inventario']):
                valor = getattr(st.session_state, key)
                info_key = {
                    'key': key,
                    'tipo': type(valor).__name__,
                    'tamanho': None
                }

                if hasattr(valor, '__len__'):
                    try:
                        info_key['tamanho'] = len(valor)
                    except:
                        pass

                diagnostico['session_state_info']['keys_relevantes'].append(info_key)

        return diagnostico

    @staticmethod
    def inicializar_pagina() -> Dict[str, Any]:
        """
        Função principal para inicializar qualquer página
        Deve ser chamada no início de cada página

        Returns:
            dict: Status da inicialização
        """
        resultado = {
            'dados_ok': False,
            'config_ok': False,
            'correcoes_aplicadas': [],
            'problemas': [],
            'diagnostico': None
        }

        # Aplicar correções automáticas
        correcoes = SessionStateManager.sincronizar_flags()
        resultado['correcoes_aplicadas'] = correcoes

        # Verificar status final
        status_dados = SessionStateManager.verificar_dados_inventario()
        status_config = SessionStateManager.verificar_configuracao()

        resultado['dados_ok'] = status_dados['encontrado']
        resultado['config_ok'] = status_config['configurado']

        if not resultado['dados_ok']:
            resultado['problemas'].append("Dados não carregados")

        if not resultado['config_ok']:
            resultado['problemas'].append("Sistema não configurado")

        # Gerar diagnóstico se há problemas
        if resultado['problemas']:
            resultado['diagnostico'] = SessionStateManager.diagnostico_completo()

        return resultado

    @staticmethod
    def mostrar_debug_sidebar():
        """Mostra informações de debug na sidebar"""
        if st.sidebar.checkbox("🔍 Debug Session State", key="debug_session_manager"):
            st.sidebar.markdown("---")
            st.sidebar.subheader("🔧 Session Manager")

            # Verificar e mostrar status
            status_dados = SessionStateManager.verificar_dados_inventario()
            status_config = SessionStateManager.verificar_configuracao()

            # Status dos dados
            if status_dados['encontrado']:
                st.sidebar.success(f"✅ Dados: {len(status_dados['dados'])} registros")
                st.sidebar.caption(f"Fonte: {status_dados['fonte']}")
            else:
                st.sidebar.error("❌ Dados não encontrados")

            # Status da configuração
            if status_config['configurado']:
                st.sidebar.success("✅ Configuração OK")
            else:
                st.sidebar.error("❌ Não configurado")

            # Botão para forçar sincronização
            if st.sidebar.button("🔄 Sincronizar Estado"):
                correcoes = SessionStateManager.sincronizar_flags()
                if correcoes:
                    for correcao in correcoes:
                        st.sidebar.success(f"✅ {correcao}")
                    st.rerun()
                else:
                    st.sidebar.info("ℹ️ Estado já sincronizado")

            # Mostrar diagnóstico detalhado
            if st.sidebar.checkbox("📋 Diagnóstico Detalhado"):
                diagnostico = SessionStateManager.diagnostico_completo()

                st.sidebar.write("**📊 Resumo:**")
                st.sidebar.write(f"- Total keys: {diagnostico['session_state_info']['total_keys']}")
                st.sidebar.write(f"- Keys relevantes: {len(diagnostico['session_state_info']['keys_relevantes'])}")

                st.sidebar.write("**🔍 Dados:**")
                for diag in diagnostico['dados']['diagnostico'][-3:]:  # Últimas 3 linhas
                    st.sidebar.caption(diag)

                st.sidebar.write("**⚙️ Config:**")
                for diag in diagnostico['configuracao']['diagnostico'][-2:]:  # Últimas 2 linhas
                    st.sidebar.caption(diag)


def inicializar_pagina_sistema():
    """
    Função para ser importada e usada no início de cada página

    Returns:
        bool: True se pré-requisitos estão OK, False caso contrário
    """
    # Criar instância do manager
    manager = SessionStateManager()

    # Inicializar
    status = manager.inicializar_pagina()

    # Mostrar debug se solicitado
    manager.mostrar_debug_sidebar()

    # Retornar se está tudo OK
    return len(status['problemas']) == 0


def verificar_e_corrigir_dados():
    """
    Função específica para verificar e corrigir dados

    Returns:
        tuple: (dados_encontrados: bool, dataframe: pd.DataFrame ou None, diagnostico: list)
    """
    manager = SessionStateManager()
    status = manager.verificar_dados_inventario()

    return status['encontrado'], status['dados'], status['diagnostico']


def verificar_e_corrigir_config():
    """
    Função específica para verificar e corrigir configuração

    Returns:
        tuple: (configurado: bool, config: dict ou None, diagnostico: list)
    """
    manager = SessionStateManager()
    status = manager.verificar_configuracao()

    return status['configurado'], status['config'], status['diagnostico']


# Função de conveniência para uso nas páginas
def mostrar_status_prerequisitos():
    """Mostra status dos pré-requisitos de forma visual"""
    dados_ok, dados, _ = verificar_e_corrigir_dados()
    config_ok, config, _ = verificar_e_corrigir_config()

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


# Exemplo de uso nas páginas:
def exemplo_uso_na_pagina():
    """
    Exemplo de como usar o SessionStateManager nas páginas
    """
    # No início da página (depois dos imports):
    from utils.session_manager import inicializar_pagina_sistema, mostrar_status_prerequisitos

    # Inicializar sistema
    prerequisitos_ok = inicializar_pagina_sistema()

    if not prerequisitos_ok:
        st.error("❌ Pré-requisitos não atendidos")
        mostrar_status_prerequisitos()

        # Botões para navegar
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🏠 Página Principal"):
                st.switch_page("Principal.py")
        with col2:
            if st.button("⚙️ Configurações"):
                st.switch_page("pages/0_⚙️_Configurações.py")

        return  # Para a execução da página

    # Continuar com o resto da página...
    st.title("Minha Página")
    # resto do código...


# IMPLEMENTAÇÃO COMPLETA PARA PÁGINAS ESPECÍFICAS

def integrar_em_modelos_hipsometricos():
    """
    Exemplo completo para a página de Modelos Hipsométricos
    """
    # No início do arquivo, depois dos imports:
    import streamlit as st
    import pandas as pd
    # ... outros imports

    # Importar o session manager
    from utils.session_manager import (
        inicializar_pagina_sistema,
        mostrar_status_prerequisitos,
        verificar_e_corrigir_dados,
        verificar_e_corrigir_config
    )

    def main():
        st.set_page_config(
            page_title="Modelos Hipsométricos",
            page_icon="🌳",
            layout="wide"
        )

        # PASSO 1: Inicializar sistema e verificar pré-requisitos
        prerequisitos_ok = inicializar_pagina_sistema()

        if not prerequisitos_ok:
            st.title("🌳 Modelos Hipsométricos")
            st.error("❌ Pré-requisitos não atendidos")

            # Mostrar status detalhado
            dados_ok, dados, diag_dados = verificar_e_corrigir_dados()
            config_ok, config, diag_config = verificar_e_corrigir_config()

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

            # Diagnóstico detalhado
            with st.expander("🔍 Diagnóstico Detalhado"):
                if not dados_ok:
                    st.write("**Problema com dados:**")
                    for linha in diag_dados[-5:]:  # Últimas 5 linhas
                        st.write(f"• {linha}")

                if not config_ok:
                    st.write("**Problema com configuração:**")
                    for linha in diag_config[-3:]:  # Últimas 3 linhas
                        st.write(f"• {linha}")

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

            return  # Para aqui se pré-requisitos não OK

        # PASSO 2: Se chegou aqui, pré-requisitos estão OK
        st.title("🌳 Modelos Hipsométricos")
        st.markdown("### Análise da Relação Altura-Diâmetro")

        # Dados estão garantidamente disponíveis
        df_inventario = st.session_state.dados_inventario
        config_global = st.session_state.config_global

        # PASSO 3: Continuar com a lógica específica da página
        mostrar_configuracao_aplicada(config_global)

        # Preview dos dados
        df_filtrado = aplicar_preview_dados(df_inventario, config_global)
        if df_filtrado is None:
            return

        # Botão para executar análise
        if st.button("🚀 Executar Análise Hipsométrica", type="primary"):
            executar_analise_hipsometrica(df_filtrado, config_global)

        # Mostrar resultados se existirem
        if 'resultados_hipsometricos' in st.session_state:
            mostrar_resultados_salvos()


def integrar_em_modelos_volumetricos():
    """
    Exemplo para página de Modelos Volumétricos
    """
    from utils.session_manager import inicializar_pagina_sistema

    def main():
        st.set_page_config(
            page_title="Modelos Volumétricos",
            page_icon="📊",
            layout="wide"
        )

        # Verificar pré-requisitos
        if not inicializar_pagina_sistema():
            st.title("📊 Modelos Volumétricos")
            st.error("❌ Pré-requisitos não atendidos")

            # Verificar se modelos hipsométricos foram executados
            if 'resultados_hipsometricos' not in st.session_state:
                st.warning("⚠️ Execute primeiro os Modelos Hipsométricos")
                if st.button("🌳 Ir para Modelos Hipsométricos"):
                    st.switch_page("pages/1_🌳_Modelos_Hipsométricos.py")

            return

        # Verificar pré-requisito específico desta página
        if 'resultados_hipsometricos' not in st.session_state:
            st.title("📊 Modelos Volumétricos")
            st.warning("⚠️ Modelos hipsométricos são pré-requisito para esta etapa")

            if st.button("🌳 Executar Modelos Hipsométricos Primeiro"):
                st.switch_page("pages/1_🌳_Modelos_Hipsométricos.py")
            return

        # Continuar com lógica específica...
        st.title("📊 Modelos Volumétricos")
        # resto da implementação...


def integrar_em_inventario_completo():
    """
    Exemplo para página de Inventário Completo
    """
    from utils.session_manager import inicializar_pagina_sistema

    def main():
        st.set_page_config(
            page_title="Inventário Completo",
            page_icon="🌲",
            layout="wide"
        )

        # Verificar pré-requisitos básicos
        if not inicializar_pagina_sistema():
            st.title("🌲 Inventário Completo")
            st.error("❌ Pré-requisitos básicos não atendidos")
            return

        # Verificar pré-requisitos específicos desta página
        prerequisitos_especificos = [
            ('resultados_hipsometricos', 'Modelos Hipsométricos'),
            ('resultados_volumetricos', 'Modelos Volumétricos')
        ]

        faltando = []
        for key, nome in prerequisitos_especificos:
            if key not in st.session_state:
                faltando.append(nome)

        if faltando:
            st.title("🌲 Inventário Completo")
            st.warning(f"⚠️ Pré-requisitos faltando: {', '.join(faltando)}")

            st.info("💡 Execute as etapas anteriores primeiro:")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🌳 Modelos Hipsométricos"):
                    st.switch_page("pages/1_🌳_Modelos_Hipsométricos.py")
            with col2:
                if st.button("📊 Modelos Volumétricos"):
                    st.switch_page("pages/2_📊_Modelos_Volumétricos.py")

            return

        # Se chegou aqui, todos os pré-requisitos estão OK
        st.title("🌲 Inventário Completo")
        st.success("✅ Todos os pré-requisitos atendidos")

        # Continuar com a implementação...
        # Aqui você tem garantia de que todos os dados e resultados estão disponíveis


# TEMPLATE GENÉRICO PARA QUALQUER PÁGINA
def template_pagina_generica():
    """
    Template genérico que pode ser adaptado para qualquer página
    """
    import streamlit as st
    from utils.session_manager import inicializar_pagina_sistema

    def main():
        # 1. Configurar página
        st.set_page_config(
            page_title="Nome da Página",
            page_icon="🔶",
            layout="wide"
        )

        # 2. Verificar pré-requisitos básicos
        if not inicializar_pagina_sistema():
            st.title("🔶 Nome da Página")
            st.error("❌ Pré-requisitos básicos não atendidos")

            # Botões de navegação padrão
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🏠 Página Principal"):
                    st.switch_page("Principal.py")
            with col2:
                if st.button("⚙️ Configurações"):
                    st.switch_page("pages/0_⚙️_Configurações.py")

            return

        # 3. Verificar pré-requisitos específicos (se necessário)
        # prerequisitos_especificos = verificar_prerequisitos_especificos()
        # if not prerequisitos_especificos:
        #     return

        # 4. Continuar com a lógica da página
        st.title("🔶 Nome da Página")
        st.success("✅ Pré-requisitos atendidos")

        # Dados estão disponíveis em:
        # - st.session_state.dados_inventario
        # - st.session_state.config_global

        # Implementar resto da página...

        # 5. Salvar resultados no session_state se necessário
        # st.session_state.resultados_minha_pagina = resultados

    if __name__ == "__main__":
        main()


# FUNÇÕES AUXILIARES PARA USO COMUM

def verificar_prerequisitos_especificos(prerequisitos_lista):
    """
    Verifica pré-requisitos específicos de uma página

    Args:
        prerequisitos_lista: Lista de tuplas (key_session_state, nome_amigavel)

    Returns:
        bool: True se todos os pré-requisitos estão OK
    """
    faltando = []

    for key, nome in prerequisitos_lista:
        if key not in st.session_state or st.session_state[key] is None:
            faltando.append(nome)

    if faltando:
        st.warning(f"⚠️ Pré-requisitos faltando: {', '.join(faltando)}")
        return False

    return True


def botoes_navegacao_padrao():
    """Botões de navegação padrão para quando há problemas"""
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🏠 Página Principal", key="nav_principal_padrao"):
            st.switch_page("Principal.py")

    with col2:
        if st.button("⚙️ Configurações", key="nav_config_padrao"):
            st.switch_page("pages/0_⚙️_Configurações.py")

    with col3:
        if st.button("🔄 Recarregar", key="nav_reload_padrao"):
            st.rerun()


def mostrar_resumo_dados_disponiveis():
    """Mostra resumo dos dados disponíveis na sessão"""
    if st.checkbox("📊 Resumo dos Dados Disponíveis"):
        col1, col2 = st.columns(2)

        with col1:
            st.write("**📋 Dados Básicos:**")
            if 'dados_inventario' in st.session_state:
                dados = st.session_state.dados_inventario
                st.write(f"• Registros: {len(dados)}")
                st.write(f"• Talhões: {dados['talhao'].nunique()}")
                st.write(f"• Colunas: {len(dados.columns)}")
            else:
                st.write("• Não disponível")

        with col2:
            st.write("**🎯 Resultados:**")
            resultados_keys = [k for k in st.session_state.keys() if 'resultado' in k.lower()]
            if resultados_keys:
                for key in resultados_keys:
                    st.write(f"• {key}: ✅")
            else:
                st.write("• Nenhum resultado salvo ainda")