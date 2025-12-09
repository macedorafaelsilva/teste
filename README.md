 import plotly.express as px  # para gerar gráfico dinâmico
import os  # Comando para mudar o diretório padrão do Python e buscar no diretório desejado
import matplotlib.pyplot as plt
import seaborn as sns  # visualização de gráficos
import numpy as np
import pandas as pd

pip install pandas
pip install numpy
pip install seaborn
pip install matplotlib
pip install plotly.express
pip install scikit-learn

%matplotlib inline #modo interativo para gráficos
%matplotlib notebook #para gráficos interativos


# -----------------------Base de dados de Crédito------------------------------
# Fonte: https://www.kaggle.com/laotse/credit-risk-dataset

os.chdir("G:\Meu Drive\CURSOS\MACHINE LEARNING E DATA SCIENCE COM PYTHON\Bases de dados")

# Exploração dos dados

base_credit = pd.read_csv("credit_data.csv")
print(base_credit.head)

base_credit.head(10)
base_credit.tail(10)  # tail de últimos registros

base_credit.describe()

base_credit[base_credit['income'] >= 69995.685578]

base_credit[base_credit['loan'] <= 1.377630]

# Visualização dos dados

# retorna quantos 0 e 1.
print(np.unique(base_credit['default'], return_counts=True))

sns.countplot(x=base_credit['default'])  # sns é para geração de gráficos

plt.hist(x=base_credit['age'])  # hist gera um histograma

plt.hist(x=base_credit['income'])

plt.hist(x=base_credit['loan'])

grafico = px.scatter_matrix(base_credit, dimensions=[
                            'age', 'income', 'loan'], color='default')
grafico.show()

# Tratamento de valores inconsistentes

base_credit.loc[base_credit['age'] < 0] #Loc é para localizar algum valor

base_credit[base_credit['age'] < 0]

base_credit2 = base_credit.drop('age', axis = 1) # Apagar a coluna inteira (de todos os registros da base de dados)
base_credit2

base_credit.index

base_credit[base_credit['age']<0].index

base_credit3 = base_credit.drop(base_credit[base_credit['age']<0].index) 
base_credit3
            # Apagar somente os registros com valores inconsistentes
            #tem que colocar o .index para que identifique quais indices são menor que zero.

base_credit3.loc[base_credit3['age']<0]

            # Porém, preencher os valores inconsistente manualmente - essa técnica é a mais recomendável

    #Preencher com a média

base_credit.mean()
base_credit['age'].mean()
base_credit['age'][base_credit['age'] > 0].mean() #pegou a média das idades maiores que zero para não pegar os outliers
base_credit.loc[base_credit['age'] < 0,'age'] = 40.92
base_credit.loc[base_credit['age'] < 0]
base_credit.head(27)


# Tratamento de valores faltantes

base_credit.isnull()
base_credit.isnull().sum()
base_credit.loc[pd.isnull(base_credit['age'])]

base_credit['age'].fillna(base_credit['age'].mean(), inplace = True) #para fazer a inserção da média direto nos valores ausentes

base_credit.loc[pd.isnull(base_credit['age'])]

base_credit.loc[(base_credit['clientid'] == 29) | (base_credit['clientid'] == 31) | (base_credit['clientid'] == 32)]

base_credit.loc[base_credit['clientid'].isin([29, 31, 32])]


# Divisão entre Previsores e Classes

type(base_credit)

x_credit = base_credit.iloc[:,1:4].values #Na variável (eixo) X será apenas os previsores Na varável (eixo) Y terá a classe que serve apenas para armazenar as respostas
            # 1:4 significa que irá buscar nas coluna de 1 até a 3.
x_credit
type(x_credit)

y_credit = base_credit.iloc[:,4].values #vai buscar apenas a coluna 4
y_credit
type(y_credit)

# Escalonamento de valores (para deixar tudo na mesma escala)

x_credit

x_credit[:,0].min(), x_credit[:,1].min(), x_credit[:,2].min()
x_credit[:,0].max(), x_credit[:,1].max(), x_credit[:,2].max()

""" 
Para deixar os atributos na mesma escala, já que eles estão em valores muito
distantantes um do outro, é necessário fazer as seguintes fórmulas:

    Padronização (Standardization): X = x - média(x) / desvio padrão(x)
mais indicada quando tem outliers.

    Normalização (Normalization): X = x[que é o próprio valor] - mínimo(x) / máximo(x) - mínimo(x)
    
"""

from sklearn.preprocessing import StandardScaler # sklearn é a Biblioteca para aprendizagem de máquina no Python
scaler_credit = StandardScaler()
x_credit = scaler_credit.fit_transform(x_credit)

x_credit[:,0].min(), x_credit[:,1].min(), x_credit[:,2].min()
x_credit[:,0].max(), x_credit[:,1].max(), x_credit[:,2].max()

x_credit


#-------------------------Base de Dados do censo------------------------------- 

# Fonte: https://archive.ics.uci.edu/ml/datasets/adult

os.chdir("G:\Meu Drive\CURSOS\MACHINE LEARNING E DATA SCIENCE COM PYTHON\Bases de dados")
base_census = pd.read_csv("census.csv")

base_census #tem atributos numéricos e categóricos, como ocupação, estado civil.

base_census.describe() #Essas métricas são importantes para verificar se os dados estão inconsistentes.

base_census.isnull().sum()

# Visualização dos dados

np.unique(base_census['income'],return_counts=True)
sns.countplot(x = base_census['income']);

plt.hist(x = base_census['age']);
plt.hist(x = base_census['education-num']);
plt.hist(x = base_census['hour-per-week']);

grafico = px.treemap(base_census,path=['workclass','age'])
grafico.show(renderer="browser")

grafico = px.treemap(base_census,path=['occupation', 'relationship','age'])
grafico.show(renderer="browser")

grafico = px.parallel_categories(base_census, dimensions=['occupation', 'relationship'])
grafico.show(renderer="browser")

grafico = px.parallel_categories(base_census, dimensions=['workclass', 'occupation', 'relationship','income'])
grafico.show(renderer="browser")

grafico = px.parallel_categories(base_census, dimensions=['occupation', 'income'])
grafico.show(renderer="browser")

grafico = px.parallel_categories(base_census, dimensions=['education','income'])
grafico.show(renderer="browser")

# Divisão entre Previsores e Classe

base_census.columns
x_census = base_census.iloc[:, 0:14].values # Por padrão os previsão são X e as classes são Y.
x_census #para comprovar que a seleção da coluna 0 a 13 foi carregada na variável
x_census[0]

y_census = base_census.iloc[:, 14].values
y_census

# Tratamento de atributos categóricos
    #Label Enconder - Usado para transformar Strings em valores numéricos
    
from sklearn.preprocessing import LabelEncoder

label_encoder_teste = LabelEncoder()
x_census[:,1]
teste = label_encoder_teste.fit_transform(x_census[:,1])
teste

x_census[0]

'''
Nas variáveis abaixo foi necessário fazer o LabelEncoder para cada atributo (coluna)
da tabela Census. E assim, transformou-se as Strings em valores numéricos.

'''

label_encoder_workclass = LabelEncoder()
label_encoder_education = LabelEncoder()
label_encoder_marital = LabelEncoder()
label_encoder_occupation = LabelEncoder()
label_encoder_relationship = LabelEncoder()
label_encoder_race = LabelEncoder()
label_encoder_sex = LabelEncoder()
label_encoder_country = LabelEncoder()

x_census[:, 1] = label_encoder_workclass.fit_transform(x_census[:,1])
x_census[:, 3] = label_encoder_education.fit_transform(x_census[:,3])
x_census[:, 5] = label_encoder_marital.fit_transform(x_census[:,5])
x_census[:, 6] = label_encoder_occupation.fit_transform(x_census[:,6])
x_census[:, 7] = label_encoder_relationship.fit_transform(x_census[:,7])
x_census[:, 8] = label_encoder_race.fit_transform(x_census[:,8])
x_census[:, 9] = label_encoder_sex.fit_transform(x_census[:,9])
x_census[:, 13] = label_encoder_country.fit_transform(x_census[:,13])

x_census[0]
x_census

# One Hot Enconder 

'''
Usado para balancear os valores de cada atributo da tabela, por exemplo: Tem-se
3 modelos de carros:
    Gol Pálio Uno
    1    2     3
    
Para que o Uno não fique mais importante que os demais é necessário aplicar o 
One Hot Encoder, em que o algoritimo irá fazer o seguinte.

Gol     1 0 0
Pálio   0 1 0
Uno     0 0 1

Assim, cada atributo terá o mesmo peso que as outras categorias. 

'''

len(np.unique(base_census['workclass'])) # para esse atributo, por exemplo, serão criados 9 novas colunas.

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

onehotencoder_census = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [1,3,5,6,7,8,9,13])], remainder='passthrough')
    # O remainder do código acima serve para não apagar os atributos originais da tabela e fazer apenas a transformação.

x_census = onehotencoder_census.fit_transform(x_census).toarray()

x_census
x_census[0]
x_census.shape # A tabela agora passou a ter 108 colunas (atributos)

'''
A literatura fala que essa é a melhor forma para trabalhar com atributos categóricos.
Pode, então, primeiro fazer o Label Enconder: transformar de atributos categóricos para 
numéricos e na sequência fazer o One Hot Encoder.
'''

# Escalonamento dos valores - para deixar todos os atributos na mesam escala

from sklearn.preprocessing import StandardScaler # Que é o desvio padrão
scaler_census = StandardScaler()
x_census = scaler_census.fit_transform(x_census)

x_census[0]

'''
Para saber os acertos do algorítimo, é só dividir a quantidade de acertos
pela quantidade de registros.
Por exemplo, 3 acertos e 1 erro. Então o acerto é de 75%.
'''

# ------------------Disão das bases em Treinamento e Teste---------------------

from sklearn.model_selection import train_test_split # é uma biblioteca para separar as bases

x_credit_treinamento, x_credit_teste, y_credit_treinamento, y_credit_teste = train_test_split(x_credit, y_credit, test_size = 0.25, random_state = 0)
# O X são os previsores e o Y são as classes 
# O random_state serve para que toda vez que executar o código, os mesmos valores companham a base de teste e a de treinamento

x_credit_treinamento.shape
y_credit_treinamento.shape
x_credit_teste.shape, y_credit_teste.shape

x_census_treinamento, x_census_teste, y_census_treinamento, y_census_teste = train_test_split(x_census, y_census, test_size = 0.15, random_state = 0)
# O percentual de teste é menor, porque a base é bem maior.

x_census_treinamento.shape, y_census_treinamento.shape
x_census_teste.shape, y_census_teste.shape


# ----------------- Salvar as variáveis das bases tratadas acima---------------

import pickle

''' 
mode = wb é para escrever a variável. Tem que importar o pickle, porque é nessa 
biblioteca que é possível salvar as variáveis.
O f é a variável definda para usar com as variáveis das bases. As variáveis 
precisam ser lançadas em formato de lista [].

'''

with open('credit.pkl', mode='wb') as f:
    pickle.dump([x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste], f)


with open('census.pkl', mode='wb') as f:
    pickle.dump([x_census_treinamento, y_census_treinamento, x_census_teste, y_census_teste], f)

'''
Com os dois with open devidamente gerados, é possível trabalhar com os algoritmos
de aprendizagem de máquina sem ter que refazer todo o pré-processamento desenvolvidos
desde o início da página.
'''
