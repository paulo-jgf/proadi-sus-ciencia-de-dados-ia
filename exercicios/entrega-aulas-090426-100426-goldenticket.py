# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 08:56:50 2026

@author: PAULO.GFERREIRA
"""

"""
Comentários iniciais

Inicialmente lamentamos não entregar a versão jupiter notebook deste trabalho.
Em um primeiro momento uma das colegas foi construindo em paralelo o notebook,
porém o VSCODE travou, e notamos que não haveria tempo para conclusão da tarefa
com o grupo reunido durante aula, sendo assim optamos por seguir o trabalho neste
script, executado na IDE spyder, para maior facilidade de visualização das variáveis.

Para que esse script rode é necessário que as bases de dados estejam em uma subpasta com
o nome base aula090426, sem a presença de quaisquer outros arquivos além dos dose meses de dados
segregados

"""


# Aqui fizemos a importação inicial de pacotes
import pandas as pd
import statsmodels.api as sm


"""
Começamos importando o arquivo referente ao primeiro mes de 2024, o que o que parecia ser
isso, para ver seu conteudo.

Abrimos também, paralelamente, o arquivo de dicionário fornecido, Convenções SIH RD,
para viabilizar a melhor exploração inicial dos dados.

Utiliamos o parametro low_memory False, conforme dica do professor, e para evitar
um erro de tipo de dados que se manifestou.

Os códigos gerados por IA no script abaixo serão indicados no comentário imediatamente
anterior. Os demais foram escritos pelo próprio grupo com consulta aos mecanismo
de busca.

"""

base = pd.read_csv(r'base aula090426/RD202401.csv', delimiter=";", low_memory=False)

"""Apos a carga, usamos este comando abaixo para ver quais colunas existem
na base, e que valores unicos aparecem em cada uma"""

explora_colunas = ['{} : {}'.format(col, base[col].unique()) for col in list(base.columns)]

"""Nesse passo verificamos que a variavel alvo, MORTE, apresenta apenas os valores 0 e 1,
não havendo necessidade de descartar registros por erros neste valor.

Também, a partir da análise do dicionário de dados Convenções SIH RD, decidimos focar
nos casos de alta complexidade (COMPLEX = 3), e eliminar registros para os quais não
se pudesse determinar a idade do paciente (COD_IDADE = 0) e seu sexo (0 e 9)"""

base = base[(base['COMPLEX'] == 3) 
            & (base['COD_IDADE'] != 0) 
            & (base['SEXO'] != 0) 
            & (base['SEXO'] != 9)]

"""Fizemos uma rapida avaliação de mortes ou não mortes por Especialidade,
e deixamos aqui para registro, mas não avançamos nessa direção."""

dfi = base[['ESPEC','MORTE']].copy()
quantitativo_espec_morte1 = dfi[dfi['MORTE'] == 1].groupby('ESPEC').count()
quantitativo_espec_morte0 = dfi[dfi['MORTE'] == 0].groupby('ESPEC').count()

"""Na exploração inicial verificamos que o valor de IDADE é apresentado em diferentes
grandezas, e criamos a seguinte função para transformar todas as idades para a grandeza
anos."""

def transformaIdadeEmFaixa(X):
    
    idade = X['IDADE']
    cod_idade = X['COD_IDADE']
    
    if cod_idade == 2:
        idade = idade / 365
    if cod_idade == 3:
        idade = idade / 12
           
    if idade < 0: #erro
        return 0
    if idade < 1: #bebe
        return 1
    elif idade < 12: #infatil
        return 2
    elif idade < 18: #adolescente
        return 3
    elif idade < 60: #aduto
        return 4
    else:            #idoso
        return 5

"""Aplicamos então a função na base, para testar o resultado."""

base['FAIXA_ETARIA'] = base.apply(transformaIdadeEmFaixa, axis=1)

"""Verificamos também que havia dois valores para sexo feminino, e sem mais informações
sobre essa ocorrência, decidimos transformar todas elas em um mesmo valor, 2.
Na sequencia aplicamos a operação a base."""

def corrigeSexo(x):
    if x == 3:
        return 2
    else:
        return x

base['SEXO'] = base['SEXO'].apply(lambda x: corrigeSexo(x))

"""Avaliando o dicionário, concluimos que as seguintes colunas provalvemente não teriam
influência no resultado de morte.

Durante o trabalho voltamos atrás e testamos o parametro RACA_COR, que de fato não
se mostrou relevante no recorte que escolhemos, que será descrito a seguir. Dado o
tempo disponível algumas partes do código foram reproveitadas, e acabamos não mantendo
o registro desta exploração do parametro RACA_COR."""

base = base.drop(['COBRANCA', 'GESTAO', 'HOMONIMO',
                  'INSTRU', 'VINCPREV', 'FINANC', 'RACA_COR'], axis=1)

"""A função abaixo foi gerada por IA, para contar os dados de morte ou não porte
por CID, que na nossa avaliação e pesquisa na internet, está apresentado na coluna
DIAG_PRINC. Posteriormente a função foi modificada pelo grupo para generalizar 
sua aplicabilidade, permitindo o uso para avaliação de morte e não morte por outras colunas.
Para tal foi incluido o parametro de função coluna, com valor default DIAG_PRINC."""

def summarize_mortality_by(df, coluna='DIAG_PRINC'):
    """
    Groups data by DIAG_PRINC and counts occurrences of MORTE (0 and 1).
    Returns a formatted DataFrame with total counts and mortality rates.
    """
    
    # 1. Using crosstab to create the frequency table
    # This automatically handles the columns for 0 and 1
    summary = pd.crosstab(df[coluna], df['MORTE'])
    
    # 2. Renaming columns for better clarity
    # Handling cases where 0 or 1 might be missing in small datasets
    summary.columns = [f'MORTE_{col}' for col in summary.columns]
    
    # 3. Adding a 'Total' column for context
    summary['TOTAL_CASOS'] = summary.sum(axis=1)
    
    # 4. Professional Suggestion: Calculating the Mortality Rate (%)
    # This provides more insight than raw numbers alone
    if 'MORTE_1' in summary.columns:
        summary['TAXA_MORTALIDADE_%'] = (summary['MORTE_1'] / summary['TOTAL_CASOS'] * 100).round(2)
    
    # Sorting by the highest number of cases to highlight critical areas
    return summary.sort_values(by='TOTAL_CASOS', ascending=False)

"""A funçao acima foi aplicada para avaliação de outras colunas, mas decidimos
a partir de seu resultados, seguir com um filtro baseado mesmo no DIAG_PRINC.

Reparamos que os CIDs S06x apresentavam incidência considerável de desfechos de morte,
e, considerando que este curso pode atingir a cabeça dos alunos, resolvemos seguir a
análise apenas para estes casos, que são associados ao traumatismo craniano.

A expectativa era também de enfrentar menos problemas com o desbalanceamento de grupos,
considerando a escassez de tempo para desenvolver a tarefa."""

df_morte_CID_princ = summarize_mortality_by(base, coluna='DIAG_PRINC')

# Pancada na cabeça!! S062, S065, S068, S069
base = base[base['DIAG_PRINC'].isin(['S062', 'S065', 'S066', 'S068', 'S069'])]

"""Ao observar que a filtragem reduziu o conjunto de dados para a ordem de mil
registros, resolvemos ampliar o escopo para todos os meses de dados disponíveis.
Para tal utilizamos o pacote glob que varre os diretórios e forma lista de endereços de
arquivos conforme parametro de busca utilizado.

Cada arquivo foi importado e filtrado, na iteração, de modo a não atingir o
limite de memoria dos computadores dos colegas do grupo"""

from glob import glob

arquivos = glob(r'base aula090426/*')

# o loop a seguir acrescenta a lista os dataframes de cada arquivo da base após filtragens
dfs = []
for arq in arquivos:
    
    base = pd.read_csv(arq, delimiter=";", low_memory=False)
    
    base = base[(base['COMPLEX'] == 3) 
                & (base['COD_IDADE'] != 0) 
                & (base['SEXO'] != 0) 
                & (base['SEXO'] != 9)]
    
    base = base.drop(['COBRANCA', 'GESTAO', 'HOMONIMO',
                      'INSTRU', 'VINCPREV', 'FINANC'], axis=1)

    base = base[base['DIAG_PRINC'].isin(['S062', 'S065', 'S066', 'S068', 'S069'])]
    
    dfs.append(base.copy())

# Por fim, todos os elementos da base foram unidos em um mesmo dataframe
base = pd.concat(dfs, ignore_index=True)

# Essa linha foi incluida para evitar termos de repetir o processo acima, que demora aqui 5 minutos
bk = base.copy()

"""A seguir aplciamos as mesmas correções desenvolvidas para a base de apenas um mês
com base nas observações do dicionário. Repetimos a avaliação de valores unicos,
mas não reparamos nenhuma surpresa no tempo disponivel para avaliação."""

base['FAIXA_ETARIA'] = base.apply(transformaIdadeEmFaixa, axis=1)
base['SEXO'] = base['SEXO'].apply(lambda x: corrigeSexo(x))
explora_colunas = ['{} : {}'.format(col, base[col].unique()) for col in list(base.columns)]

"""Nos passos abaixo, até a criação e testes dos modelos, testamos várias hipóteses,
que acabaram não guardadas para registro inclusive por travamento do VSCODE no computador
que estava replicando as opeções no notebook.

Inicialmente testamos utilizar apenas a variável MUNIC_MOV, imaginando que seria o
Município onde o paciente fora tratado. No subconjunto são aproximadamente 220 códigos
de muicipios, que foram devidamente transformados em dummy variable com a função própria
do pandas (que aparecerá em passos mais abaixo), alguns deles se mostraram potencialmente
importantes com p-valor ~ 0, porém mais tarde descobrimos que tal parametro se refere ao 
município responsável por questões administrativas ou financeiras quanto ao atendimento,
o que nos levou a descartar integralmente o parametro.

Aqui, como já referido, testamos tambem RACA_COR, que nao se mostrou relevante neste
recorte.

O trecho abaixo de codigo foi o que ficamos re-executando para tentar entender e melhorar 
os resultados. Apenas a seleçao logo abaixo foi mantida, a última versão por hora."""


df = base[['MORTE','FAIXA_ETARIA', 'MARCA_UTI', 'QT_DIARIAS']]

"""Tentamos usar aqui a matriz de correlaçao do pandas que foi impossivel avalair com 
todas as variáveis. Mas após a seleção acima, já faria algum sentido na nossa opinião,
embora falte a dummificação."""

dfc = df.corr()

"""Ao avaliarmos os maus resultados da regressão linear, que será mostrada abaixo,
e comparativamente os melhores resultados da arvore de decisão, resolvemos transformar
a variável QT_DIARIAS, que intuitivamente nos pareceu uma boa variável para prever a
variável de interesse, em uma variável categorica.

O grupo tem duvidas de como tratar adequadamente as variáveis categoricas em regressoes
polinomiais, e de modo inverso, como tratar adequadamente as variáveis claramente
numéricas em modelos de arvore de decisão ou de categorização.

Por hora, esta foi nossa melhor solução, sem tempo para pesquisa adicional."""

def transformaDiariasEmFaixa(X):
    
    qtd = X['QT_DIARIAS']
           
    if qtd == 0: #sem internacao
        return 'SemIternacao'
    if qtd < 7: #ate 1 semana
        return 'Ate1Semana'
    elif qtd < 30: #ate 1 mes
        return 'Ate1Mes'
    else:            #mais de 1 mes
        return 'Mais1Mes'

"""Aplicamos então a transformação em faixas etárias, e preparamos os tipos de variáveis
a serem dummificadas em string, o que pareceu um requisito para o método funcionar."""

df['QT_DIARIAS'] = df.apply(transformaDiariasEmFaixa, axis=1)

df[['FAIXA_ETARIA', 'MARCA_UTI']] = df[
    ['FAIXA_ETARIA', 'MARCA_UTI']].map(lambda x: str(x))

df = pd.concat([df[['MORTE']], 
               pd.get_dummies(df[['FAIXA_ETARIA','MARCA_UTI','QT_DIARIAS']], drop_first=True).map(lambda x: 1 if x else 0)],
               axis=1)
              

"""Daqui para baixo adaptamos codigos já desenvolvidos para a atividade anterior,
e reexecutamos várias vezes incluindo e removendo variáveis, bem como ajustando a
profundidade da arvore.

Todos os resultados com regressão linear ficaram entre 4% e 13% de R2.

Já os resultados com via arvore ficaram entre 15% e 29%."""


"""Funcionalizado para reaproveitar o codigo, e avaliar as métricas da regressão linear"""
def testa_OLS():
    # Usar Statsmodels para treinar um modelo linear baseado em mínimos quadrados nos dados selecionados
    x_const = sm.add_constant(x)
    regress = sm.OLS(y, x_const)
    modelo = regress.fit()
    # Guardar as métricas de resultados do moodelo em uma string
    return modelo.summary().as_text()

# Dados completos, o Y é morte, os demais são os parametros para explicar o alvo
y = df['MORTE']

# Optamos por simplesmente remover tudo que não é morte, para maior efiência nos testes
x = df.drop(['MORTE'], axis=1)

# Testando resultados com todas as variaveis, o resultado foi muito ruim
r_ols_todas_variaveis = testa_OLS()

# A partir deste achado dividimos a base em treino e teste, para finalmente
# Comparar modelos preditivos para esta base. 42 mantido mochileiro!!!
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#modelo com arvore de decisão
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics

modelo = DecisionTreeRegressor(max_depth=10)
modelo.fit(X_train, y_train)
y_pred_ar = modelo.predict(X_test)
y_pred_ar_train = modelo.predict(X_train)
r2_test = metrics.r2_score(y_test, y_pred_ar)
r2_train = metrics.r2_score(y_train, y_pred_ar_train)
print('R2 treino:', r2_train)
print('R2 teste:', r2_test)


"""Os resultados R2 da Arvores de decisão nesta versão final conforme configuração presente
foram:
    R2 treino: 0.24484691383758728
    R2 teste: 0.24763053190466422
    
Ainda temos dúvidas de como conciliar as variáveis numpericas e categoricas nas diferentes
abordagens. Além disso não foi possível compreender e avançar na abordagem probabilistica
para prever o Y, de modo que fora do exercício academico este trabalho ainda não tem
qualquer valor."""


















