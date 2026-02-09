"""---
## 1. Importando bibliotecas

"""

# Importação das bibliotecas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import io


import warnings
warnings.filterwarnings('ignore')

# Configurações de visualização
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
sns.set_palette('husl')

# Vamos definir o seed para que os resultados possam ser reproduzidos
np.random.seed(42)


"""---
## 2. Leitura do Dataset

"""

# Ler arquivo .csv
df = pd.read_csv('Custo_medico2.csv')


"""---
## 3. Análise Exploratória de Dados

"""

# Estrutura dos dados

relatorio_estrutura = ''
relatorio_estrutura += 'ANÁLISE EXPLORATÓRIA INICIAL'
relatorio_estrutura += '\nDimensões do dataset: {}'.format(df.shape)
relatorio_estrutura += '\n' + '-'*90

print(11)

# Mudando o Output de print para string, para guardar um unico output com todos os dados
buffer = io.StringIO()
df.info(buf=buffer)
s = buffer.getvalue()
print(s)

relatorio_estrutura += '\nInfo df:\n{}'.format(s)
relatorio_estrutura += '\n' + '-'*90

relatorio_estrutura += '\nValores ausentes por coluna:\n{}'.format(df.isnull().sum())
relatorio_estrutura += '\n' + '-'*90


# Estatísticas descritivas (equivalente a summary() no R)
relatorio_stats = 'ESTATÍSTICAS DESCRITIVAS\n'


# Descritivas completas
# Incluindo IQR, cv

descritivas = df.describe().T
descritivas['median'] = df.median(numeric_only=True)
descritivas['IQR'] = descritivas['75%'] - descritivas['25%']
descritivas['cv'] = (descritivas['std'] / descritivas['mean'] * 100).round(2)

# Renomear colunas para português
descritivas = descritivas.rename(columns={
    'count': 'N',
    'mean': 'Média',
    'std': 'DP',
    'min': 'Mín',
    'max': 'Máx',
    'median': 'Mediana',
    'cv': 'CV%'
})

# Arredondar todos os valores para 2 casas decimais
descritivas = descritivas.map(lambda x: round(x, 2))


# Evitando a compressão de colunas
with pd.option_context(
    'display.max_rows', None,
    'display.max_columns', None,
    'display.width', 1000,
    'display.max_colwidth', None
):
    relatorio_stats += '\nEstatisticas descritivas variáveis quantitativas:\n{}'.format(descritivas)
    print(descritivas)
relatorio_stats += '\n' + '='*120

"""Desses resultados chama a atenção a diferença de ao menos duas
ordens de grandeza entre os custos medicos e as demais variáveis"""

# Avaliar variaveis qualitativas presentes e seu balanceamento na população
df0 = df.copy() #Facilitar execução com df virgem
descricao_qualitativas = '\nBalanceamento de grupos\n'

descricao_qualitativas += '\nSexo:\n{}'.format(
    df0.groupby(['sexo']).size().reset_index(name='Contagem'))
descricao_qualitativas += '\n' + '-'*90

descricao_qualitativas += '\nFumantes:\n{}'.format(
    df0.groupby(['fumante']).size().reset_index(name='Contagem'))
descricao_qualitativas += '\n' + '-'*90

descricao_qualitativas += '\nRegioes:\n{}'.format(
    df0.groupby(['regiao']).size().reset_index(name='Contagem'))
descricao_qualitativas += '\n' + '='*90

relatorio_stats += descricao_qualitativas

# Tratar as variáveis qualitativas presentes, para que possam ser usadas no modelo
# Cada possivel valor das colunas qualitativas sera transformado em uma coluna binaria, que dira se o observavel apresenta o valor ou nao
# Uma coluna sera eliminada, para evitar covariancia. Ex: no caso de dois possiveis valor, nao ser um deles, implica que será o outro
# Ex pratico: se for homem, não será mulher
# n-1 colunas descrevem os valores adequadamente, no caso estamos eliminando a primeira coluna drop_first = True
# E estamos transformando o output True ou False em 1 ou 0
df1 = pd.get_dummies(df[['sexo','fumante','regiao']], drop_first=True).map(lambda x: 1 if x else 0)

# Uma vez criadas as colunas de substituicao das variaveis quantitativas podemos eliminar as originais
df.drop(columns=['sexo','fumante','regiao'], inplace=True)

# Juntamos agora as novas colunas que representam as qualitativas quantitativamente
df = pd.concat([df,df1],axis=1)

# Funcao para teste de normalidade Shpario Wilk
def teste_normalidade(dados, nome_var):
    stat_sw, p_sw = stats.shapiro(dados)
    alpha = 0.05
    normal_sw = 'Normal' if p_sw > alpha else 'Não Normal'
    return {
        'Variável': nome_var,
        'Shapiro-Wilk (stat)': round(stat_sw, 4),
        'p-valor': round(p_sw, 4),
        'Conclusão': normal_sw
    }

# Vamos rodar o teste de normalidade em todas as colunas, para verificar a hipotese de aleatoriedade nos dados H0
resultados_norm = []
for var in list(df.columns):
    resultados_norm.append(teste_normalidade(df[var], var))

df_normalidade = pd.DataFrame(resultados_norm)
print('Testes de Normalidade (alfa = 0.05):')

relatorio_stats += '\nTeste de normalidade das variáveis, tentando rejeitar H0:\n{}'.format(df_normalidade)
relatorio_stats += '\n' + '-'*90

# Matriz de correlação entre as variáveis
dfc = df.corr()
# Evitando a compressão de colunas
with pd.option_context(
    'display.max_rows', None,
    'display.max_columns', None,
    'display.width', 1000,
    'display.max_colwidth', None
):
    relatorio_stats += '\nTeste de correlação das variáveis, tentando ver candidatos para o modelo preditivo:\n{}'.format(dfc)
relatorio_stats += '\n' + '-'*90

"""---
## 4. Selecionando variáveis para descrever o valor de gasto

"""

# Funcionalizado para reaproveitar o codigo, assim como o teste de normalidade
def testa_OLS():
    # Usar Statsmodels para treinar um modelo linear baseado em mínimos quadrados nos dados selecionados
    x_const = sm.add_constant(x)
    regress = sm.OLS(y, x_const)
    modelo = regress.fit()
    # Guardar as métricas de resultados do moodelo em uma string
    return modelo.summary().as_text()

# Dados completos, o alvo é o custo, os demais são os parametros para explicar o custo
y = df['custos_medicos']
x = df[[
         'idade',
         'IMC',
         'filhos',
         'doencas_cronicas',
         'consultas_ano',
         'atividade_fisica',
         'dieta_saudavel',
         'uso_medicamentos',
         'sexo_masculino',
         'fumante_sim',
         'regiao_noroeste',
         'regiao_sudeste',
         'regiao_sudoeste'
         ]]

# Testando resultados com todas as variaveis
r_ols_todas_variaveis = testa_OLS()


# Testando resultados com todas as variaveis, y dividido por 100
y = y / 100
r_ols_todas_variaveis_yPor100 = testa_OLS()

# Dividr por 100 reduz o erro, mas nao muda pvalor, R2 ou estatisticaF

# Vamos selecionar agora apenas as variaveis com p valor 0.0000....
y = df['custos_medicos']
x = df[[
         'idade',
         'IMC',
         'filhos',
         #'doencas_cronicas',
         #'consultas_ano',
         #'atividade_fisica',
         #'dieta_saudavel',
         #'uso_medicamentos',
         
         #'sexo_masculino',
         'fumante_sim',
         #'regiao_noroeste',
         #'regiao_sudeste',
         #'regiao_sudoeste'
         ]]

# E dividir y por 1000, colocando ele proximo as ordens de grandeza das demais variaveis
y = y / 1000
r_ols_variaveis_pvalor0_yPor1000 = testa_OLS()

relatorio_variaveis = 'ANÁLISE VARIÁVEIS'
relatorio_variaveis += '\n\n\nTestando resultados com todas as variaveis:\n{}'.format(r_ols_todas_variaveis)
relatorio_variaveis += '\n\n\nTestando resultados com todas as variaveis y (custos medicos) dividido por 100:\n{}'.format(r_ols_todas_variaveis_yPor100)
relatorio_variaveis += '\n\n\nTestando resultados com variaveis pvalor 0 e dividido por 1000 (uniformizando ordens de grandeza):\n{}'.format(r_ols_variaveis_pvalor0_yPor1000)

"""Esta configuração manteve o R2 proximo a 0.75, reduziu o erro para aproximadamente 1/3
da primeira tentativa"""



# A partir deste achado dividimos a base em treino e teste, para finalmente
# Comparar modelos preditivos para esta base
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


"""---
## 5. Tentativa Regressão Linear Sklearn

"""

from sklearn.linear_model import LinearRegression

modelo = LinearRegression()
modelo.fit(X_train, y_train)
#treinamento do modelo

#modelo.score(X_train, y_train) #== r2
#quanto dos dados de treinamento o modelo explica

y_pred = modelo.predict(X_test)
y_pred_train = modelo.predict(X_train)

#metricas
from sklearn import metrics
r2_train = metrics.r2_score(y_train, y_pred_train)
r2_test = metrics.r2_score(y_test, y_pred)
print('R2 treino:', r2_train)
print('R2 teste:', r2_test)
mae_train = metrics.mean_absolute_error(y_train, y_pred_train)
mae_test = metrics.mean_absolute_error(y_test, y_pred)
print('MAE treino:', mae_train)
print('MAE teste:', mae_test)
mse_train = metrics.mean_squared_error(y_train, y_pred_train)
mse_test = metrics.mean_squared_error(y_test, y_pred)
print('MSE treino:', mse_train)
print('MSE teste:', mse_test)
rmse_train = metrics.root_mean_squared_error(y_train, y_pred_train)
rmse_test = metrics.root_mean_squared_error(y_test, y_pred)
print('RMSE treino:', rmse_train)
print('RMSE teste:', rmse_test)

# Reescalando os dados (custo dividido por 1000)

#grafico dos erros do modelo
plt.scatter(y_test, y_pred, label='Linear')
plt.legend(loc='upper left')
plt.xlabel("Custo real (x1.000)")
plt.ylabel("Custo previsto p/ modelo (x1.000)")
plt.title("Desempenho modelos Regressão Linear x Árvore de Decisão p/ Custo Médico")
plt.plot(y_test, y_test, color='red')
#plt

#modelo com arvore de decisão
from sklearn.tree import DecisionTreeRegressor
modelo = DecisionTreeRegressor(max_depth=6)
modelo.fit(X_train, y_train)
y_pred_ar = modelo.predict(X_test)
y_pred_ar_train = modelo.predict(X_train)
r2_test = metrics.r2_score(y_test, y_pred_ar)
r2_train = metrics.r2_score(y_train, y_pred_ar_train)
print('R2 treino:', r2_train)
print('R2 teste:', r2_test)


#grafico dos erros do modelo
plt.scatter(y_test, y_pred_ar, label='Árvore (depth=6)')
plt.legend(loc='upper left')
plt.plot(y_test, y_test, color='red')
plt
