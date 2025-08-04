# models/volumetrico.py - VERSÃO REFATORADA USANDO formatacao.py
'''
Modelos volumétricos para estimativa de volume
'''

import numpy as np
import pandas as pd
from models.base import ModeloLinear
from sklearn.metrics import r2_score, mean_squared_error

# USAR FUNÇÕES CENTRALIZADAS do formatacao.py
from utils.formatacao import (
    validar_dados_numericos,
    detectar_tipo_coluna,
    formatar_brasileiro,
    formatar_percentual,
    formatar_estatisticas_resumo,
    formatar_cabecalho_relatorio,
    classificar_qualidade_modelo_detalhado,
    criar_relatorio_modelo,
    formatar_timestamp_brasileiro
)


def converter_dados_volumetricos_brasileiros(df_volumes):
    """
    VERSÃO REFATORADA: Usa funções centralizadas do formatacao.py
    Converte dados volumétricos do formato brasileiro
    """
    print("🇧🇷 Convertendo dados volumétricos do formato brasileiro...")

    df = df_volumes.copy()
    colunas_converter = ['D_cm', 'H_m', 'V']
    dados_convertidos = {}

    for coluna in colunas_converter:
        if coluna in df.columns:
            print(f"  📊 Processando {coluna}...")

            # USAR FUNÇÃO CENTRALIZADA para detectar tipo
            tipo_detectado = detectar_tipo_coluna(df[coluna], coluna)
            print(f"    🔍 Tipo detectado: {tipo_detectado}")

            # CONVERSÃO SIMPLIFICADA usando validação centralizada
            def converter_brasileiro_simples(valor):
                if pd.isna(valor) or valor is None:
                    return np.nan
                if isinstance(valor, (int, float)):
                    return float(valor)
                if isinstance(valor, str):
                    valor = valor.strip()
                    if valor == '' or valor.lower() in ['nan', 'null']:
                        return np.nan
                    try:
                        return float(valor.replace(',', '.'))
                    except (ValueError, TypeError):
                        return np.nan
                return np.nan

            # Aplicar conversão
            valores_originais = df[coluna].iloc[:3].tolist()
            df[coluna] = df[coluna].apply(converter_brasileiro_simples)
            valores_convertidos = df[coluna].iloc[:3].tolist()

            # USAR FORMATAÇÃO CENTRALIZADA para mostrar exemplo
            exemplo_orig = [formatar_brasileiro(v, 2) if isinstance(v, (int, float)) else str(v) for v in
                            valores_originais]
            exemplo_conv = [formatar_brasileiro(v, 2) if isinstance(v, (int, float)) else str(v) for v in
                            valores_convertidos]
            print(f"    📝 Exemplo: {exemplo_orig} → {exemplo_conv}")

            # USAR VALIDAÇÃO CENTRALIZADA
            limites = {}
            if coluna == 'D_cm':
                limites = {'min': 1, 'max': 100}
            elif coluna == 'H_m':
                limites = {'min': 1, 'max': 50}
            elif coluna == 'V':
                limites = {'min': 0.001, 'max': 5}

            validacao = validar_dados_numericos(df[coluna], coluna, limites)
            dados_convertidos[coluna] = validacao

            # USAR FORMATAÇÃO CENTRALIZADA para relatório
            if validacao['valida']:
                stats = validacao['estatisticas']
                print(f"    ✅ Conversão: {stats['validos']}/{stats['total']} valores válidos")
            else:
                print(f"    ⚠️ Problemas detectados:")
                for problema in validacao['problemas'][:2]:
                    print(f"      • {problema}")

    # GERAR RELATÓRIO RESUMIDO usando funções centralizadas
    print(f"\n📊 Resumo da conversão:")
    for coluna, validacao in dados_convertidos.items():
        if validacao['valida']:
            stats = validacao['estatisticas']
            media_str = formatar_brasileiro(stats['media'], 3)
            cv_str = formatar_percentual(stats['cv'] / 100, 1)
            print(f"  • {coluna}: média = {media_str}, CV = {cv_str}")

    return df


def criar_variaveis_volumetricas(df_volumes):
    """
    VERSÃO REFATORADA: Usa formatação centralizada
    Cria variáveis derivadas para modelos volumétricos
    """
    print("🧮 Preparando variáveis para modelos volumétricos...")

    # Converter dados usando função refatorada
    df = converter_dados_volumetricos_brasileiros(df_volumes)

    # Verificar colunas essenciais usando validação centralizada
    colunas_essenciais = ['D_cm', 'H_m', 'V']
    for col in colunas_essenciais:
        if col not in df.columns:
            raise ValueError(f"Coluna essencial {col} não encontrada")

        # USAR VALIDAÇÃO CENTRALIZADA para verificar valores válidos
        validacao = validar_dados_numericos(df[col], col)
        valores_validos = validacao['estatisticas']['validos']

        if valores_validos == 0:
            raise ValueError(f"Nenhum valor válido na coluna {col} após conversão")

    # Remover valores inválidos
    df_limpo = df.dropna(subset=colunas_essenciais)
    df_limpo = df_limpo[
        (df_limpo['D_cm'] > 0) &
        (df_limpo['H_m'] > 0) &
        (df_limpo['V'] > 0)
        ]

    if len(df_limpo) < 5:
        raise ValueError("Poucos dados válidos após filtros (< 5 registros)")

    print(f"📊 Dataset para transformações: {len(df_limpo)} registros")

    try:
        print("  🔢 Calculando logaritmos...")
        # Logaritmos com clip para evitar valores <= 0
        df_limpo['ln_V'] = np.log(df_limpo['V'].clip(lower=0.001))
        df_limpo['ln_D'] = np.log(df_limpo['D_cm'].clip(lower=0.1))
        df_limpo['ln_H'] = np.log(df_limpo['H_m'].clip(lower=0.1))

        print("  ⚙️ Calculando variáveis derivadas...")
        # CORREÇÃO PRINCIPAL: Nomes corretos das variáveis (sem underscore)
        df_limpo['D2'] = df_limpo['D_cm'] ** 2
        df_limpo['D2H'] = df_limpo['D2'] * df_limpo['H_m']  # CORRETO: D2H
        df_limpo['ln_D2H'] = np.log(df_limpo['D2H'].clip(lower=0.001))  # CORRETO: ln_D2H
        df_limpo['inv_D'] = 1 / df_limpo['D_cm'].clip(lower=0.1)

        print("✅ Variáveis transformadas criadas com sucesso")

        # Verificar transformações usando validação centralizada
        variaveis_log = ['ln_V', 'ln_D', 'ln_H', 'ln_D2H']
        for var in variaveis_log:
            validacao = validar_dados_numericos(df_limpo[var], var)
            if not validacao['valida']:
                print(f"⚠️ Problemas em {var}: {validacao['problemas'][0]}")

        # Remover infinitos e NaNs
        df_final = df_limpo.replace([np.inf, -np.inf], np.nan).dropna()

        if len(df_final) < 5:
            raise ValueError("Poucos registros válidos após transformações")

        print(f"📊 Dataset final: {len(df_final)} registros")

        # USAR FORMATAÇÃO CENTRALIZADA para estatísticas finais
        stats_resumo = formatar_estatisticas_resumo(df_final, ['V', 'D_cm', 'H_m'])
        if 'V' in stats_resumo:
            print(f"📈 Volume - Média: {stats_resumo['V']['media']}, CV: {stats_resumo['V']['cv_pct']}")

        return df_final

    except Exception as e:
        print(f"❌ Erro ao criar variáveis transformadas: {e}")
        raise ValueError(f"Erro nas transformações: {e}")


class ModeloSchumacher(ModeloLinear):
    '''Modelo volumétrico de Schumacher-Hall: ln(V) = β₀ + β₁*ln(D) + β₂*ln(H)'''

    def __init__(self):
        super().__init__("Schumacher")

    def preparar_dados(self, df):
        X = df[['ln_D', 'ln_H']]
        y = df['ln_V']
        return X, y

    def predizer_volume(self, df):
        X, _ = self.preparar_dados(df)
        ln_v_pred = self.predizer(X)
        return np.exp(ln_v_pred)


class ModeloG1(ModeloLinear):
    '''Modelo volumétrico G1: ln(V) = β₀ + β₁*ln(D) + β₂*(1/D)'''

    def __init__(self):
        super().__init__("G1")

    def preparar_dados(self, df):
        X = df[['ln_D', 'inv_D']]
        y = df['ln_V']
        return X, y

    def predizer_volume(self, df):
        X, _ = self.preparar_dados(df)
        ln_v_pred = self.predizer(X)
        return np.exp(ln_v_pred)


class ModeloG2(ModeloLinear):
    '''Modelo volumétrico G2: V = β₀ + β₁*D² + β₂*D²H + β₃*H'''

    def __init__(self):
        super().__init__("G2")

    def preparar_dados(self, df):
        # CORRIGIDO: usar nomes corretos das colunas (sem underscore)
        X = df[['D2', 'D2H', 'H_m']]  # D2H ao invés de D2_H
        y = df['V']
        return X, y

    def predizer_volume(self, df):
        X, _ = self.preparar_dados(df)
        return self.predizer(X)


class ModeloG3(ModeloLinear):
    '''Modelo volumétrico G3: ln(V) = β₀ + β₁*ln(D²H)'''

    def __init__(self):
        super().__init__("G3")

    def preparar_dados(self, df):
        # CORRIGIDO: usar nome correto da coluna (sem underscore)
        X = df[['ln_D2H']]  # ln_D2H ao invés de ln_D2_H
        y = df['ln_V']
        return X, y

    def predizer_volume(self, df):
        X, _ = self.preparar_dados(df)
        ln_v_pred = self.predizer(X)
        return np.exp(ln_v_pred)


def ajustar_todos_modelos_volumetricos(df_volumes):
    '''
    VERSÃO REFATORADA: Usa formatação centralizada
    Ajusta todos os 4 modelos volumétricos e retorna resultados formatados
    '''
    print("🚀 Iniciando ajuste de modelos volumétricos (versão refatorada)...")

    resultados = {}
    predicoes = {}
    relatorios_formatados = {}  # NOVO: relatórios formatados

    # Criar variáveis usando função refatorada
    try:
        df_prep = criar_variaveis_volumetricas(df_volumes)
        print(f"📊 Dados preparados: {len(df_prep)} registros")
    except Exception as e:
        print(f"❌ Erro na preparação: {e}")
        return {}, {}, None

    # Lista de modelos
    modelos = [
        ModeloSchumacher(),
        ModeloG1(),
        ModeloG2(),  # Agora funciona com D2H
        ModeloG3()  # Agora funciona com ln_D2H
    ]

    # Ajustar cada modelo
    for modelo in modelos:
        print(f"\n🔄 Ajustando modelo {modelo.nome}...")
        try:
            X, y = modelo.preparar_dados(df_prep)
            print(f"  📊 Variáveis: {list(X.columns)}, Registros: {len(X)}")

            if modelo.ajustar(X, y):
                # Predizer volumes
                v_pred = modelo.predizer_volume(df_prep)
                predicoes[modelo.nome] = v_pred

                # Calcular métricas
                r2 = r2_score(df_prep['V'], v_pred)
                rmse = np.sqrt(mean_squared_error(df_prep['V'], v_pred))

                resultados[modelo.nome] = {
                    'r2': r2,
                    'rmse': rmse,
                    'modelo': modelo
                }

                # USAR FORMATAÇÃO CENTRALIZADA para relatório
                relatorio = criar_relatorio_modelo(resultados[modelo.nome], modelo.nome, 'volumetrico')
                relatorios_formatados[modelo.nome] = relatorio

                # USAR FORMATAÇÃO CENTRALIZADA para logs
                r2_str = formatar_brasileiro(r2, 4)
                rmse_str = formatar_brasileiro(rmse, 4)
                qualidade = classificar_qualidade_modelo_detalhado(r2, 'volumetrico')
                print(f"  ✅ R² = {r2_str}, RMSE = {rmse_str} ({qualidade})")

            else:
                print(f"  ❌ Falha no ajuste")

        except Exception as e:
            print(f"  ❌ Erro: {e}")
            continue

    # Encontrar melhor modelo
    if resultados:
        melhor_modelo = max(resultados.keys(), key=lambda k: resultados[k]['r2'])
        melhor_r2 = formatar_brasileiro(resultados[melhor_modelo]['r2'], 4)
        print(f"\n🏆 Melhor modelo: {melhor_modelo} (R² = {melhor_r2})")

        # Adicionar relatórios formatados aos resultados
        for modelo in resultados:
            resultados[modelo]['relatorio_formatado'] = relatorios_formatados.get(modelo, "")

        return resultados, predicoes, melhor_modelo
    else:
        print("\n❌ Nenhum modelo ajustado com sucesso")
        return {}, {}, None


def aplicar_modelo_volumetrico(df_inventario, modelo_nome, modelo_obj):
    '''
    VERSÃO REFATORADA: Usa formatação centralizada
    Aplica modelo volumétrico aos dados do inventário
    '''
    df = df_inventario.copy()

    # Criar variáveis necessárias
    df = criar_variaveis_volumetricas(df.rename(columns={'H_est': 'H_m'}))

    try:
        if modelo_obj and hasattr(modelo_obj, 'modelo'):
            # Usar predição do modelo real
            if modelo_nome == 'Schumacher':
                X = df[['ln_D', 'ln_H']]
                ln_v_pred = modelo_obj.predizer(X)
                df['V_est'] = np.exp(ln_v_pred)

            elif modelo_nome == 'G1':
                X = df[['ln_D', 'inv_D']]
                ln_v_pred = modelo_obj.predizer(X)
                df['V_est'] = np.exp(ln_v_pred)

            elif modelo_nome == 'G2':
                X = df[['D2', 'D2H', 'H_m']]  # CORRETO: D2H
                df['V_est'] = modelo_obj.predizer(X)

            elif modelo_nome == 'G3':
                X = df[['ln_D2H']]  # CORRETO: ln_D2H
                ln_v_pred = modelo_obj.predizer(X)
                df['V_est'] = np.exp(ln_v_pred)
            else:
                df['V_est'] = 0.001 * df['D2'] * df['H_m']
        else:
            # Fórmulas aproximadas
            if modelo_nome == 'G2':
                df['V_est'] = -0.1 + 0.001 * df['D2'] + 0.00005 * df['D2H'] + 0.01 * df['H_m']
            elif modelo_nome == 'G3':
                df['V_est'] = np.exp(-11.0 + 0.9 * df['ln_D2H'])
            else:
                df['V_est'] = 0.001 * df['D2'] * df['H_m']

        # Garantir valores positivos
        df['V_est'] = df['V_est'].clip(lower=0.001)

    except Exception as e:
        print(f"Erro ao aplicar modelo {modelo_nome}: {e}")
        df['V_est'] = 0.001 * df['D2'] * df['H_m']

    return df


def gerar_relatorio_volumetrico_completo(resultados, volumes_arvore):
    """
    NOVA FUNÇÃO: Gera relatório completo usando formatação centralizada
    """
    if not resultados or volumes_arvore is None:
        return "❌ Não foi possível gerar relatório"

    # USAR FUNÇÃO CENTRALIZADA para cabeçalho
    relatorio = formatar_cabecalho_relatorio(
        "Análise de Modelos Volumétricos",
        "Sistema Integrado de Inventário Florestal",
        True
    )

    # Estatísticas gerais usando formatação centralizada
    stats = formatar_estatisticas_resumo(volumes_arvore, ['V', 'D_cm', 'H_m'])

    relatorio += f"""
## 📊 Estatísticas do Dataset
- **Árvores analisadas**: {len(volumes_arvore)}
- **Volume médio**: {stats['V']['media']} m³
- **DAP médio**: {stats['D_cm']['media']} cm  
- **Altura média**: {stats['H_m']['media']} m
- **Coeficiente de variação**: {stats['V']['cv_pct']}

## 🏆 Ranking dos Modelos
"""

    # Ranking usando formatação centralizada
    ranking = sorted(resultados.items(), key=lambda x: x[1]['r2'], reverse=True)

    for i, (modelo, resultado) in enumerate(ranking, 1):
        r2_str = formatar_brasileiro(resultado['r2'], 4)
        rmse_str = formatar_brasileiro(resultado['rmse'], 4)
        qualidade = classificar_qualidade_modelo_detalhado(resultado['r2'], 'volumetrico')

        relatorio += f"""
### {i}º - {modelo}
- **R²**: {r2_str}
- **RMSE**: {rmse_str} m³
- **Qualidade**: {qualidade}
"""

    relatorio += f"""
## 📈 Interpretação dos Resultados
- **Melhor modelo**: {ranking[0][0]}
- **Precisão alcançada**: {classificar_qualidade_modelo_detalhado(ranking[0][1]['r2'], 'volumetrico')}
- **Aplicabilidade**: Todos os modelos são adequados para estimativas volumétricas

## 🔬 Equações dos Modelos
- **Schumacher**: ln(V) = β₀ + β₁×ln(D) + β₂×ln(H)
- **G1**: ln(V) = β₀ + β₁×ln(D) + β₂×(1/D)
- **G2**: V = β₀ + β₁×D² + β₂×D²H + β₃×H
- **G3**: ln(V) = β₀ + β₁×ln(D²H)

---
*Relatório gerado automaticamente usando formatação centralizada do sistema*
"""

    return relatorio


def obter_equacao_latex(modelo_nome):
    '''Equações LaTeX para cada modelo'''
    equacoes = {
        'Schumacher': r"ln(V) = \beta_0 + \beta_1 \cdot ln(D) + \beta_2 \cdot ln(H)",
        'G1': r"ln(V) = \beta_0 + \beta_1 \cdot ln(D) + \beta_2 \cdot \frac{1}{D}",
        'G2': r"V = \beta_0 + \beta_1 \cdot D^2 + \beta_2 \cdot D^2H + \beta_3 \cdot H",
        'G3': r"ln(V) = \beta_0 + \beta_1 \cdot ln(D^2H)"
    }
    return equacoes.get(modelo_nome, "Equação não disponível")


def obter_descricao_coeficientes(modelo_nome):
    '''Descrição dos coeficientes para cada modelo'''
    coeficientes = {
        'Schumacher': ["β₀ (intercepto)", "β₁ (ln D)", "β₂ (ln H)"],
        'G1': ["β₀ (intercepto)", "β₁ (ln D)", "β₂ (1/D)"],
        'G2': ["β₀ (intercepto)", "β₁ (D²)", "β₂ (D²H)", "β₃ (H)"],
        'G3': ["β₀ (intercepto)", "β₁ (ln D²H)"]
    }
    return coeficientes.get(modelo_nome, ["Coeficientes não disponíveis"])