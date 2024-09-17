import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import IsolationForest

# Função para carregar os dados do CSV
def carregar_dados(caminho_arquivo: str) -> pd.DataFrame:
    try:
        dados = pd.read_csv(caminho_arquivo)
        return dados
    except FileNotFoundError:
        st.error(f"Erro: Arquivo {caminho_arquivo} não encontrado.")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        st.error(f"O arquivo {caminho_arquivo} está vazio.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Ocorreu um erro ao carregar os dados: {e}")
        return pd.DataFrame()

# Função para normalizar os dados
def normalizar_dados(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# Função para remover outliers usando Isolation Forest
def remover_outliers(X, y, contamination=0.05):
    isolation_forest = IsolationForest(contamination=contamination, random_state=42)
    outliers = isolation_forest.fit_predict(X)
    mask = outliers != -1
    return X[mask], y[mask]

# Função para permitir o download do dataframe como CSV
def baixar_dados(df):
    csv = df.to_csv(index=False)
    st.sidebar.download_button(
        label="Baixar CSV completo",
        data=csv,
        file_name='dados_completos.csv',
        mime='text/csv'
    )

# Função principal do Streamlit
def main():
    st.title("Dashboard Eleitoral com Modelos de Regressão e Visualizações Avançadas")

    # Carregar o arquivo automaticamente
    caminho_arquivo = 'tse-analises.csv'
    data = carregar_dados(caminho_arquivo)

    if not data.empty:
        # Sidebar para opções de visualização
        st.sidebar.header("Opções de Visualização")

        # Exibir o DataFrame completo na Sidebar
        st.sidebar.subheader("Dados completos")
        st.sidebar.write(data)

        # Opção para baixar o DataFrame completo
        baixar_dados(data)

        # Menu para seleção de visualizações específicas
        visualizacao_opcao = st.sidebar.selectbox(
            "Selecione a visualização:",
            (
                "Distribuição de Gênero (Feminino)",
                "Distribuição de Cor/Raça Preta",
                "Distribuição de Bens Totais",
                "Média de Bens por Partido",
                "Distribuição de Idade Média",
                "Taxa de Candidatos Casados",
                "Taxa de Candidatos Solteiros",
                "Heatmap de Correlação",
                "Scatter Plot",
                "Séries",
                "Treinamento de Regressão"
            )
        )

        # Visualizações simples para análise de dados
        if visualizacao_opcao == "Distribuição de Gênero (Feminino)":
            st.subheader("Distribuição de Gênero (Feminino)")
            st.bar_chart(data.groupby('SG_PARTIDO')['txGenFeminino'].mean())

        elif visualizacao_opcao == "Distribuição de Cor/Raça Preta":
            st.subheader("Distribuição de Cor/Raça Preta por Partido")
            st.bar_chart(data.groupby('SG_PARTIDO')['txCorRacaPreta'].mean())

        elif visualizacao_opcao == "Distribuição de Bens Totais":
            st.subheader("Distribuição de Bens Totais por Partido")
            st.bar_chart(data.groupby('SG_PARTIDO')['totalBens'].sum())

        elif visualizacao_opcao == "Média de Bens por Partido":
            st.subheader("Média de Bens por Partido")
            st.bar_chart(data.groupby('SG_PARTIDO')['avgBens'].mean())

        elif visualizacao_opcao == "Distribuição de Idade Média":
            st.subheader("Distribuição da Idade Média por Partido")
            st.bar_chart(data.groupby('SG_PARTIDO')['avgIdade'].mean())

        elif visualizacao_opcao == "Taxa de Candidatos Casados":
            st.subheader("Taxa de Candidatos Casados por Partido")
            st.bar_chart(data.groupby('SG_PARTIDO')['txEstadoCivilCasado'].mean())

        elif visualizacao_opcao == "Taxa de Candidatos Solteiros":
            st.subheader("Taxa de Candidatos Solteiros por Partido")
            st.bar_chart(data.groupby('SG_PARTIDO')['txEstadoCivilSolteiro'].mean())

        elif visualizacao_opcao == "Heatmap de Correlação":
            st.subheader("Heatmap de Correlação")
            data_numerica = data.select_dtypes(include=['float64', 'int64'])
            colunas_selecionadas = st.multiselect(
                "Selecione as variáveis para o Heatmap", data_numerica.columns.tolist(), default=data_numerica.columns.tolist()
            )
            if colunas_selecionadas:
                corr = data_numerica[colunas_selecionadas].corr()
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
                st.pyplot(fig)
            else:
                st.write("Selecione ao menos uma variável para exibir o Heatmap.")

        elif visualizacao_opcao == "Scatter Plot":
            st.subheader("Scatter Plot (Gráfico de Dispersão)")
            colunas_numericas = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
            if len(colunas_numericas) >= 2:
                x_axis = st.selectbox("Selecione a variável para o eixo X", colunas_numericas)
                y_axis = st.selectbox("Selecione a variável para o eixo Y", colunas_numericas)
                fig, ax = plt.subplots()
                ax.scatter(data[x_axis], data[y_axis])
                ax.set_xlabel(x_axis)
                ax.set_ylabel(y_axis)
                st.pyplot(fig)
            else:
                st.write("Não há colunas numéricas suficientes para criar um scatter plot.")

        elif visualizacao_opcao == "Séries":
            st.subheader("Séries")
            colunas_numericas = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
            if colunas_numericas:
                coluna_serie = st.selectbox("Selecione a variável para a série temporal", colunas_numericas)
                st.line_chart(data[coluna_serie])
            else:
                st.write("Não há colunas numéricas suficientes para exibir uma série temporal.")

        # Adicionando Machine Learning na aba de Treinamento de IA (apenas modelos de regressão)
        elif visualizacao_opcao == "Treinamento de Regressão":
            st.subheader("Treinamento de Modelos de Regressão")

            # Escolher o algoritmo de regressão
            algoritmo_escolhido = st.selectbox("Escolha o algoritmo:",
                                               ["Random Forest Regressor", "Linear Regression", "Decision Tree Regressor"])

            # Definir as features e o target predefinidos
            target_column = 'txGenFeminino'  # Agora, a variável alvo é a taxa de Gênero Feminino
            features_columns = ['avgIdade', 'txEstadoCivilCasado', 'txCorRacaNaoBranca', 'totalBens', 'SG_PARTIDO']

            # Exibir features e target predefinidos para ajustes
            target_column = st.selectbox("Selecione a coluna alvo (target)", data.columns, index=data.columns.get_loc(target_column))
            features_columns = st.multiselect("Selecione as features (variáveis explicativas)", data.columns, default=features_columns)

            # Controle de hiperparâmetros para os modelos
            if algoritmo_escolhido == "Random Forest Regressor":
                n_estimators = st.slider("Número de Estimadores (árvores)", 10, 200, 100)
                max_depth = st.slider("Profundidade Máxima das Árvores", 2, 20, 10)
            elif algoritmo_escolhido == "Decision Tree Regressor":
                max_depth = st.slider("Profundidade Máxima da Árvore", 2, 20, 10)
            contamination = st.slider("Remoção de Outliers (percentual)", 0.01, 0.1, 0.05)

            if st.button("Treinar Modelo"):
                # Separar as features (X) e a target (y)
                X = data[features_columns]
                y = data[target_column]

                # Converter variáveis categóricas em numéricas usando One-Hot Encoding
                categorical_cols = X.select_dtypes(include=['object']).columns
                X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

                # Remover outliers
                X, y = remover_outliers(X, y, contamination=contamination)

                # Dividir os dados em treino e teste
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Normalizar os dados
                X_train_scaled, X_test_scaled = normalizar_dados(X_train, X_test)

                # Treinamento do Modelo de Regressão
                if algoritmo_escolhido == "Random Forest Regressor":
                    modelo = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                elif algoritmo_escolhido == "Linear Regression":
                    modelo = LinearRegression()
                elif algoritmo_escolhido == "Decision Tree Regressor":
                    modelo = DecisionTreeRegressor(max_depth=max_depth, random_state=42)

                modelo.fit(X_train_scaled, y_train)
                y_pred = modelo.predict(X_test_scaled)

                # Avaliação
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                st.write(f"Modelo {algoritmo_escolhido} - MSE: {mse:.2f}, MAE: {mae:.2f}")

                # Gráfico dos erros MSE e MAE em um único gráfico de barras
                st.subheader(f"Comparação de MSE e MAE - {algoritmo_escolhido}")
                fig, ax = plt.subplots()
                ax.bar(["MSE", "MAE"], [mse, mae], color=['blue', 'green'])
                ax.set_ylabel('Erro')
                st.pyplot(fig)

                # Exibir Previsões vs Valores Reais
                st.subheader(f"Previsões vs Valores Reais - {algoritmo_escolhido}")
                resultados = pd.DataFrame({"Valor Real": y_test, "Previsão": y_pred})
                st.write(resultados)

        # Estatísticas descritivas gerais
        st.subheader("Estatísticas descritivas gerais")
        st.write(data.describe())

    else:
        st.error("Nenhum dado foi carregado. Verifique o arquivo e tente novamente.")

# Rodar a função principal
if __name__ == "__main__":
    main()
