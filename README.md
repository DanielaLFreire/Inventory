# 🌲 Sistema Integrado de Inventário Florestal - GreenVista

## 📋 Visão Geral

O GreenVista é um sistema completo para análise de inventário florestal que integra modelos matemáticos tradicionais com dados de sensoriamento remoto LiDAR. O sistema utiliza inteligência artificial para democratizar análises complexas através de uma interface web intuitiva.

## 🎯 Características Principais

- **🤖 IA Democratizada**: Tecnologia avançada em interface simples
- **⚙️ Configuração Centralizada**: Configure uma vez, use em todas as etapas
- **🔬 Cientificamente Robusto**: 7 modelos hipsométricos + 4 volumétricos
- **📊 Relatórios Profissionais**: Exportação em múltiplos formatos
- **🛩️ Integração LiDAR**: Calibração automática com dados de sensoriamento remoto
- **🌐 Acessível**: Funciona em qualquer navegador, sem instalação

## 🚀 Fluxo de Trabalho

### 1. 📁 Upload de Dados
- **Inventário**: DAP, altura, talhão, parcela, código
- **Cubagem**: Medições detalhadas para modelos volumétricos
- **Opcionais**: Shapefile de áreas, coordenadas GPS, dados LiDAR

### 2. ⚙️ Etapa 0 - Configurações
- Configure **uma vez**, use em **todas as etapas**
- Filtros globais (diâmetro mínimo, talhões excluídos)
- Parâmetros dos modelos não-lineares
- Método de cálculo de áreas

### 3. 🌳 Etapa 1 - Modelos Hipsométricos
- 7 modelos disponíveis (lineares e não-lineares)
- Seleção automática do melhor modelo
- Análise de significância estatística

### 4. 📊 Etapa 2 - Modelos Volumétricos
- Método de Smalian para cubagem
- 4 modelos volumétricos especializados
- Análise de resíduos e qualidade

### 5. 📈 Etapa 3 - Inventário Final
- Aplicação automática dos melhores modelos
- Relatórios executivos detalhados
- Análise por talhão e classificação de produtividade

### 6. 🛩️ Etapa 4 - Integração LiDAR (Opcional)
- Comparação automática campo vs LiDAR
- Calibração de modelos hipsométricos
- Análise estrutural avançada
- Detecção automática de outliers

## 📂 Estrutura dos Arquivos de Entrada

### 📋 Arquivo de Inventário (Obrigatório)
```csv
D_cm;H_m;talhao;parcela;cod;idade_anos
15.5;18.2;1;1;D;5
12.3;16.8;1;1;D;5
...
```

**Colunas obrigatórias:**
- `D_cm`: Diâmetro em centímetros
- `H_m`: Altura em metros
- `talhao`: Identificador do talhão
- `parcela`: Identificador da parcela
- `cod`: Código da árvore (D=Dominante, N=Normal, C=Cortada, I=Invasora)

**Colunas opcionais:**
- `idade_anos`: Idade do povoamento

### 📏 Arquivo de Cubagem (Obrigatório)
```csv
arv;talhao;d_cm;h_m;D_cm;H_m
1;1;15.2;0.1;15.5;18.2
1;1;14.8;2.0;15.5;18.2
1;1;14.1;4.0;15.5;18.2
...
```

**Colunas obrigatórias:**
- `arv`: ID da árvore
- `talhao`: ID do talhão
- `d_cm`: Diâmetro da seção
- `h_m`: Altura da seção
- `D_cm`: DAP da árvore
- `H_m`: Altura total

### 📍 Arquivo de Coordenadas (Opcional)
```csv
talhao;parcela;x;y
1;1;-45.123456;-23.654321
1;2;-45.123789;-23.654654
...
```

### 🛩️ Arquivo de Métricas LiDAR (Opcional)
```csv
talhao;parcela;zmean;zmax;zsd;pzabove2
1;1;18.5;22.3;2.1;85.4
1;2;17.8;21.1;1.9;82.7
...
```

**Métricas principais:**
- `zmean`: Altura média (m)
- `zmax`: Altura máxima (m)
- `zsd`: Desvio padrão da altura
- `pzabove2`: % de cobertura acima de 2m

## 🔧 Instalação e Configuração

### Dependências Principais
```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn scipy openpyxl xlrd
```

### Dependências Opcionais (para LiDAR e Shapefile)
```bash
pip install geopandas pyxlsb chardet
```

### Executar o Sistema
```bash
streamlit run Principal.py
```

## 🎓 Modelos Disponíveis

### 🌳 Modelos Hipsométricos (Altura-Diâmetro)

**Lineares:**
1. **Curtis**: ln(H) = β₀ + β₁ × (1/D)
2. **Campos**: ln(H) = β₀ + β₁ × (1/D) + β₂ × ln(H_dom)
3. **Henri**: H = β₀ + β₁ × ln(D)
4. **Prodan**: D²/(H-1.3) = β₀ + β₁ × D + β₂ × D²

**Não-lineares:**
5. **Chapman**: H = b₀ × (1 - exp(-b₁ × D))^b₂
6. **Weibull**: H = a × (1 - exp(-b × D^c))
7. **Mononuclear**: H = a × (1 - b × exp(-c × D))

### 📊 Modelos Volumétricos (Cubagem)

1. **Schumacher-Hall**: ln(V) = β₀ + β₁ × ln(D) + β₂ × ln(H)
2. **G1 (Goulding)**: ln(V) = β₀ + β₁ × ln(D) + β₂ × (1/D)
3. **G2**: V = β₀ + β₁ × D² + β₂ × D²H + β₃ × H
4. **G3 (Spurr)**: ln(V) = β₀ + β₁ × ln(D²H)

## 📈 Métricas de Qualidade

### R² (Coeficiente de Determinação)
- **Hipsométricos**: R² ≥ 0.90 = Excelente
- **Volumétricos**: R² ≥ 0.95 = Excelente

### Classificação de Produtividade
- **Eucalipto**: 
  - Excelente: ≥250 m³/ha
  - Muito Boa: 200-249 m³/ha
  - Boa: 150-199 m³/ha
  - Regular: 100-149 m³/ha

## 🛩️ Integração LiDAR

### Benefícios
- ✅ Validação de modelos de campo
- ✅ Calibração automática de estimativas
- ✅ Detecção de outliers
- ✅ Mapeamento contínuo da floresta
- ✅ Análise estrutural avançada

### Fluxo de Trabalho LiDAR
1. **Processamento**: Extrair métricas das nuvens de pontos
2. **Upload no Sistema**: Carregar arquivo CSV com métricas
3. **Integração Automática**: Sistema faz merge com dados de campo
4. **Análise Comparativa**: Validação automática campo vs LiDAR
5. **Calibração**: Ajuste de modelos baseado em LiDAR

## 📊 Relatórios Gerados

### Relatórios por Etapa
- **Hipsométricos**: Comparação de modelos, equações, gráficos de ajuste
- **Volumétricos**: Análise de resíduos, validação de cubagem
- **Inventário**: Resumo por talhão, classificação de produtividade
- **LiDAR**: Comparação campo-LiDAR, calibração de modelos

### Formatos de Export
- **CSV**: Dados tabulares (separador brasileiro)
- **Excel**: Planilhas organizadas por aba
- **PDF**: Relatórios executivos (via impressão do navegador)
- **Markdown**: Relatórios técnicos detalhados

## ⚠️ Troubleshooting

### Problemas Comuns

**1. Arquivo não carrega**
- Verificar formato (CSV, Excel)
- Verificar encoding (UTF-8 recomendado)
- Verificar separadores (; para CSV brasileiro)

**2. Dados insuficientes para modelos**
- Mínimo 30 observações para modelos lineares
- Mínimo 50 observações para modelos não-lineares
- Verificar filtros de diâmetro mínimo

**3. Modelos não-lineares não convergem**
- Ajustar parâmetros iniciais na Etapa 0
- Aumentar número máximo de iterações
- Verificar qualidade dos dados de entrada

**4. Integração LiDAR falha**
- Verificar compatibilidade de talhão/parcela
- Validar métricas LiDAR (valores realísticos)
- Verificar formato do arquivo (CSV com separador ;)

### Limites do Sistema
- **Tamanho máximo**: 200MB por arquivo
- **Memória**: Recomendado 8GB RAM para datasets grandes
- **Modelos não-lineares**: Podem demorar vários minutos

## 🤝 Suporte e Contribuição

### Suporte
- Interface intuitiva com instruções contextuais
- Validação automática detecta e reporta problemas
- Preview de dados antes de processar
- Status em tempo real na sidebar

### Estrutura do Código

```
├── Principal.py                 # Página principal
├── requirements.txt          # Dependências
├── README.md                # Documentação
│
├── pages/
│   ├── 0_⚙️_Configurações.py   # Configurações globais
│   ├── 1_🌳_Modelos_Hipsométricos.py
│   ├── 2_📊_Modelos_Volumétricos.py
│   ├── 3_📈_Inventário_Florestal.py
│   └── 4_🛩️_Dados_LiDAR.py     # Integração LiDAR
├── config/
│   ├── config.py               # Configurações gerais
│   └── configuracoes_globais.py # Sistema de configuração
├── processors/
│   ├── lidar.py                # Processamento LiDAR
│   ├── cubagem.py           # Processamento de cubagem
│   ├── inventario.py        # Processamento do inventário
│   └── areas.py             # Processamento de áreas
├── models/
│   ├── base.py              # Classes base
│   ├── hipsometrico.py      # Modelos hipsométricos
│   └── volumetrico.py       # Modelos volumétricos
├── ui/
│   ├── sidebar.py              # Interface da barra lateral
│   ├── configuracoes.py     # Configurações
│   ├── components.py           # Componentes visuais
│   └── graficos.py             # Gráficos e visualizações
└── utils/
    ├── arquivo_handler.py      # Manipulação de arquivos
    ├── validacao.py         # Validação de dados
    └── formatacao.py           # Formatação brasileira
```

## 📚 Referências Científicas

### Modelos Hipsométricos
- Curtis, R.O. (1967). Height-diameter and height-diameter-age equations
- Campos, J.C.C. et al. (1984). Tabelas de volume para eucalipto
- Prodan, M. (1965). Forest biometrics

### Modelos Volumétricos
- Schumacher, F.X. & Hall, F.S. (1933). Logarithmic expression of timber-tree volume
- Goulding, C.J. (1979). Validation of growth models used in forest management

### Integração LiDAR
- Næsset, E. (2002). Predicting forest stand characteristics with airborne scanning laser
- Maltamo, M. et al. (2014). Forestry applications of airborne laser scanning

## 🔄 Atualizações e Versões

### Versão Atual: 4.0 (LiDAR Integration)
- ✅ Integração completa com dados LiDAR
- ✅ Calibração automática de modelos
- ✅ Sistema de configurações centralizadas
- ✅ Processamento automático na sidebar
- ✅ Validação robusta de dados

### Próximas Funcionalidades
- 🔮 Integração com imagens de satélite
- 🔮 Modelos de crescimento e produção
- 🔮 API para integração com outros sistemas
- 🔮 Análise de biomassa e carbono

## 📊 Exemplos de Uso

### Caso 1: Inventário Tradicional
1. Upload inventário + cubagem
2. Configurar filtros (DAP ≥ 5cm)
3. Executar modelos hipsométricos
4. Executar modelos volumétricos
5. Gerar inventário final

### Caso 2: Inventário com LiDAR
1. Upload inventário + cubagem + LiDAR
2. Configurar sistema
3. Executar etapas 1-3
4. Integrar dados LiDAR
5. Calibrar modelos com LiDAR
6. Gerar relatório comparativo

### Caso 3: Análise de Produtividade
1. Upload dados com múltiplos talhões
2. Configurar áreas específicas por talhão
3. Executar análise completa
4. Classificar produtividade por talhão
5. Exportar ranking de talhões

## 🎯 Casos de Uso Específicos

### Para Empresas Florestais
- **Inventário operacional**: Estimativas precisas de volume por talhão
- **Planejamento de colheita**: Priorização baseada em produtividade
- **Controle de qualidade**: Validação com dados LiDAR

### Para Consultores
- **Avaliação de ativos**: Relatórios técnicos profissionais
- **Auditoria florestal**: Validação independente de inventários
- **Estudos de viabilidade**: Análise de potencial produtivo

### Para Pesquisadores
- **Desenvolvimento de modelos**: Comparação de diferentes equações
- **Validação científica**: Integração campo-sensoriamento remoto
- **Análise estrutural**: Métricas avançadas de complexidade florestal

## ⚡ Performance e Otimização

### Recomendações de Hardware
- **Mínimo**: 4GB RAM, processador dual-core
- **Recomendado**: 8GB RAM, processador quad-core
- **Ideal**: 16GB RAM, SSD, processador octa-core

### Otimização de Dados
- Remover colunas desnecessárias antes do upload
- Usar formato .xlsx para datasets grandes
- Filtrar dados obviamente incorretos antes do upload

### Dicas de Performance
- Modelos lineares: ~1-2 segundos
- Modelos não-lineares: ~30-60 segundos
- Integração LiDAR: ~10-30 segundos
- Inventário final: ~5-15 segundos

## 🔐 Segurança e Privacidade

### Dados Locais
- Todos os dados ficam no navegador do usuário
- Nenhum upload para servidores externos
- Processamento 100% local

### Recomendações
- Fazer backup dos dados originais
- Exportar configurações importantes
- Não fechar o navegador durante processamentos longos

## 📖 Tutorial Passo a Passo

### Primeira Utilização

**1. Preparação dos Dados**
```
- Organize seus dados em planilhas Excel ou CSV
- Verifique se as colunas obrigatórias estão presentes
- Remova linhas completamente vazias
- Salve em formato UTF-8 se possível
```

**2. Configuração Inicial**
```
- Abra o sistema no navegador
- Faça upload do inventário e cubagem na sidebar
- Vá para Etapa 0 (Configurações)
- Defina filtros apropriados (DAP mínimo, talhões a excluir)
- Configure método de área
- Ajuste parâmetros dos modelos se necessário
- Salve as configurações
```

**3. Execução das Análises**
```
- Etapa 1: Modelos Hipsométricos
  * Execute todos os modelos
  * Analise a qualidade dos ajustes
  * Anote o melhor modelo selecionado

- Etapa 2: Modelos Volumétricos
  * Execute a cubagem com método Smalian
  * Execute os modelos volumétricos
  * Verifique análise de resíduos

- Etapa 3: Inventário Final
  * Gere estimativas por parcela
  * Analise resumo por talhão
  * Exporte relatórios necessários
```

**4. Integração LiDAR (Opcional)**
```
- Processe dados LiDAR no R (use script fornecido)
- Faça upload do arquivo de métricas LiDAR
- Execute comparação campo vs LiDAR
- Calibre modelos se necessário
- Gere relatório integrado
```

### Comandos R para Processamento LiDAR

```r
# Exemplo de script R para extrair métricas LiDAR
library(lidR)
library(sf)

# Carregar nuvem de pontos
las <- readLAS("arquivo.las")

# Definir parcelas (exemplo com coordenadas)
parcelas <- st_read("parcelas.shp")

# Extrair métricas por parcela
metricas <- grid_metrics(las, ~list(
  zmean = mean(Z),
  zmax = max(Z),
  zsd = sd(Z),
  pzabove2 = mean(Z > 2) * 100
), 20) # resolução 20m

# Exportar para CSV
write.csv(metricas, "metricas_lidar.csv", row.names = FALSE)
```

## 🧪 Testes e Validação

### Dados de Teste
O sistema inclui validação automática que verifica:
- Consistência dos dados (DAP vs altura)
- Valores dentro de limites realísticos
- Completude das observações
- Qualidade dos ajustes de modelos

### Benchmarks de Qualidade
- **R² Hipsométrico**: Eucalipto > 0.85, Pinus > 0.80
- **R² Volumétrico**: Geral > 0.95
- **RMSE Relativo**: < 15% para altura, < 10% para volume
- **Correlação LiDAR**: > 0.80 entre campo e LiDAR

## 🎓 Treinamento e Capacitação

### Nível Básico (2 horas)
- Interface do sistema
- Upload e validação de dados
- Configurações essenciais
- Execução das etapas principais
- Interpretação de resultados básicos

### Nível Intermediário (4 horas)
- Configurações avançadas
- Análise de qualidade de modelos
- Interpretação de resíduos
- Customização de parâmetros
- Resolução de problemas comuns

### Nível Avançado (8 horas)
- Integração com dados LiDAR
- Desenvolvimento de modelos customizados
- Análise estatística avançada
- Integração com outros sistemas
- Automação de workflows

## 📞 Suporte Técnico

### Auto-diagnóstico
1. Verificar console do navegador (F12)
2. Testar com dados de exemplo menores
3. Limpar cache do navegador
4. Verificar versão do navegador (Chrome/Firefox recomendados)

### Logs Úteis
- Status de carregamento na sidebar
- Mensagens de erro nas etapas
- Relatórios de validação de dados
- Métricas de qualidade dos modelos

### Solução de Problemas Específicos

**Problema**: Modelos não-lineares não convergem
**Solução**: 
- Verificar qualidade dos dados (R² linear > 0.7)
- Ajustar parâmetros iniciais na configuração
- Aumentar número máximo de iterações
- Reduzir tolerância de convergência

**Problema**: Integração LiDAR falha
**Solução**:
- Verificar formato de talhão/parcela (números inteiros)
- Validar valores das métricas LiDAR (> 0)
- Confirmar que existem parcelas em comum
- Verificar encoding do arquivo (UTF-8)

**Problema**: Relatórios vazios
**Solução**:
- Verificar se todas as etapas foram executadas
- Confirmar que dados passaram na validação
- Verificar filtros de configuração (muito restritivos?)
- Tentar com subset menor dos dados

## 🌟 Boas Práticas

### Preparação de Dados
- **Sempre** fazer backup dos dados originais
- Verificar coordenadas se usar GPS (formato decimal)
- Padronizar códigos de árvores (D, N, C, I)
- Incluir idade quando disponível

### Configuração
- Começar com filtros conservadores
- Testar com subset pequeno primeiro
- Documentar configurações utilizadas
- Exportar configurações para reutilização

### Análise
- Sempre verificar R² e RMSE dos modelos
- Analisar gráficos de resíduos
- Comparar resultados entre modelos
- Validar resultados com conhecimento local

### Relatórios
- Incluir informações de configuração
- Documentar método de coleta dos dados
- Especificar modelos utilizados
- Adicionar data e responsável técnico

---

**Desenvolvido com ❤️ pela Neural Tec para a comunidade florestal brasileira**

*Sistema GreenVista - Democratizando a análise de inventário florestal através da inteligência artificial*