# Dashboard Eleitoral com Modelos de Regressão e Visualizações Avançadas

## Objetivo Geral
O dashboard tem como objetivo fornecer uma análise abrangente e interativa sobre dados eleitorais, permitindo a exploração de diferentes aspectos relacionados ao perfil dos candidatos, suas características e variáveis associadas. Ele também integra ferramentas de aprendizado de máquina para criar modelos de regressão, possibilitando previsões sobre as taxas de gênero feminino com base em variáveis explicativas. A interface foi projetada para ser intuitiva, com visualizações gráficas e opções de personalização, permitindo ao usuário explorar os dados de forma flexível.

## Funcionalidades Principais

### 1. **Carregamento de Dados**
O dashboard permite o carregamento automático de dados no formato CSV, que são então exibidos diretamente na barra lateral. Isso facilita a visualização completa das informações sem precisar navegar por outros elementos da interface.

### 2. **Download de Dados**
O usuário tem a opção de baixar o conjunto de dados completo diretamente em formato CSV. Essa funcionalidade é útil para quem deseja uma cópia local dos dados analisados ou quem quer realizar análises externas adicionais.

### 3. **Visualizações Personalizadas**
A interface oferece uma ampla gama de visualizações gráficas, permitindo que o usuário selecione qual aspecto dos dados deseja explorar. As opções incluem:

- **Distribuição de Gênero (Feminino):** Exibe a distribuição média de candidatas do gênero feminino por partido.
- **Distribuição de Cor/Raça Preta:** Mostra a média de candidatos autodeclarados pretos em cada partido.
- **Distribuição de Bens Totais:** Visualiza a soma dos bens declarados por partido.
- **Média de Bens por Partido:** Fornece a média dos bens declarados por partido.
- **Distribuição de Idade Média:** Mostra a idade média dos candidatos de cada partido.
- **Taxa de Candidatos Casados e Solteiros:** Exibe as taxas de estado civil dos candidatos por partido.
- **Heatmap de Correlação:** Permite a visualização de correlações entre variáveis numéricas selecionadas, facilitando a identificação de possíveis relações entre as variáveis.
- **Scatter Plot (Gráfico de Dispersão):** Oferece uma análise visual da relação entre duas variáveis numéricas selecionadas.
- **Séries Temporais:** Exibe a evolução de uma variável numérica ao longo do tempo.

### 4. **Treinamento de Modelos de Regressão**
O dashboard também oferece uma seção dedicada ao treinamento de modelos de regressão com diferentes algoritmos de aprendizado de máquina. Os modelos disponíveis incluem:

- **Random Forest Regressor**
- **Linear Regression**
- **Decision Tree Regressor**

O usuário pode selecionar o algoritmo desejado, ajustar seus hiperparâmetros (como profundidade de árvores e número de estimadores), e definir as variáveis explicativas e a variável alvo. Essa funcionalidade permite prever a taxa de candidatas do gênero feminino com base em variáveis como idade média, estado civil, cor/raça, e bens declarados.

### 5. **Remoção de Outliers**
Para melhorar a precisão dos modelos, o dashboard oferece uma funcionalidade de remoção de outliers utilizando o algoritmo Isolation Forest. O nível de contaminação pode ser ajustado pelo usuário, o que possibilita controlar o impacto dos valores extremos nos resultados.

### 6. **Avaliação do Modelo**
Após o treinamento, os modelos são avaliados com base em duas métricas de erro:

- **MSE (Mean Squared Error)**
- **MAE (Mean Absolute Error)**

Esses erros são visualizados em um gráfico de barras, comparando as duas métricas para o modelo escolhido. Além disso, o dashboard exibe uma tabela com as previsões versus os valores reais da variável alvo, facilitando a análise do desempenho do modelo.

## Possíveis Insights

- **Distribuições por Partido:** A análise da distribuição de gênero, cor/raça e estado civil pode fornecer insights importantes sobre a diversidade dos candidatos em cada partido, permitindo uma compreensão mais profunda das características demográficas dos candidatos em diferentes contextos políticos.
  
- **Análise de Bens e Idade:** Comparar as distribuições de bens totais e idade média pode revelar correlações entre o patrimônio declarado e o perfil de idade dos candidatos, possibilitando insights sobre o impacto socioeconômico nas candidaturas.

- **Modelos de Regressão:** A aplicação de modelos de regressão para prever a taxa de candidatas femininas com base em variáveis como idade, estado civil e patrimônio pode revelar padrões e tendências em relação à representatividade feminina nas eleições.

- **Correlação de Variáveis:** O heatmap de correlação permite identificar variáveis fortemente correlacionadas, oferecendo uma visão clara das inter-relações entre diferentes características dos candidatos.

## Conclusão
Este dashboard serve como uma ferramenta poderosa para a análise exploratória de dados eleitorais, fornecendo uma interface amigável para visualizações gráficas e treinamento de modelos de regressão. Com suas funcionalidades interativas, o usuário pode obter insights significativos sobre o perfil dos candidatos e usar modelos preditivos para avaliar diferentes variáveis com eficiência.

## Acesse o Dashboard
Você pode acessar o dashboard através do seguinte link: [Dashboard Eleitoral](https://tse-dashboard.streamlit.app/)

## Participantes:
1 - João Renan Lopes
2 - Carlos Egger

## Agradecimentos
Agradeço ao professor Pedro Girotto pelo incentivo na matéria de estatística, o que foi fundamental para a criação deste projeto.
