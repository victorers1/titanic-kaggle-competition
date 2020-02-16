# -*- coding: utf-8 -*-
"""
Spyder Editor
Este é um arquivo de script temporário.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow.keras.callbacks as kc
import tensorflow.keras.layers as kl
import tensorflow.keras.models as km
import tensorflow.keras.utils as ut

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split    

#Lendo arquivos de treino e teste
train = pd.read_csv('train.csv', index_col='PassengerId', encoding='utf-8')
test = pd.read_csv('test.csv',index_col='PassengerId', encoding='utf-8')

#Exlcuindo colunas que não serão usadas
train = train.drop(columns=['Unnamed'])

print("Tabela de treino")
print(train.head(10)) # Degustação dos dados iniciais

#Adaptando colunas de treino para tipo numeŕico
train['sex'] = (train['sex']=='man').astype(int) # sex agora: 'male'==1, 'female'==0
train['survived'] = (train['survived']=='yes').astype(int) # survived agora: yes==1, no==0
train.loc[train['class'] == '1st class', 'class'] = 0 # 
train.loc[train['class'] == '2nd class', 'class'] = 1 # 
train.loc[train['class'] == '3rd class', 'class'] = 2 # 
train.loc[train['age'] == 'child', 'age'] = 0 # age agora: child==0, adults==1
train.loc[train['age'] == 'adults', 'age'] = 1
train.loc[train['survived'] == 'yes', 'age'] = 0

#Novos dados de treino 
print("Nova tabela de treino")
print(train.head(10))

#Aplicando mesmas mudanças nos dados de teste
test = test.drop(columns=['Unnamed'])
test['sex'] = (test['sex']=='man').astype(int) # sex agora: 'male'==1, 'female'==0
test.loc[test['class'] == '1st class', 'class'] = 0 # 
test.loc[test['class'] == '2nd class', 'class'] = 1 # 
test.loc[test['class'] == '3rd class', 'class'] = 2 # 
test.loc[test['age'] == 'child', 'age'] = 0 # age agora: child==0, adults==1
test.loc[test['age'] == 'adults', 'age'] = 1

#Criando tabela de entrada para a rede
dataInput = train[['age', 'sex', 'class']].dropna()
print("Entrada da rede:")
print(dataInput)

#Criando tabela gabarito da rede
dataOutput = train['survived'].dropna()

# Separação de dados de treino e teste
#x_train, x_test, y_train, y_test = train_test_split(dataInput, dataOutput, test_size=0.0)

#Abordagem por rede neural: criação do modelo
model = km.Sequential()
model.add(kl.Dense(20, input_dim=3, activation='relu'))
model.add(kl.Dense(20, activation='relu'))
model.add(kl.Dense(20, activation='relu'))
model.add(kl.Dense(1, activation='sigmoid'))
model.compile(loss='msle', optimizer='adamax', metrics=['accuracy'])  # categorical_crossentropy é o recomenndado para classificações excludentes
monitor = kc.EarlyStopping(monitor='loss', min_delta=1e-2, patience=2, verbose=0, mode='auto')  # Condição de parada do treinamento

H = model.fit(dataInput[['age','sex','class']], dataOutput, epochs=300)

#val_perd, val_prec = model.evaluate(x_test, y_test)
#print('Perda: {:2.2f}%'.format(val_perd*100))
#print('Precisão: {:2.2f}%'.format(val_prec*100)+' das amostras foram rotulados corretamente.')

plt.plot(H.history['loss'])
plt.title('Custo por época')
plt.show()

predictions = model.predict(test[['age', 'sex', 'class']]) # prediction apresenta chance de ter sobrevivido
predictions = (predictions>=0.5).astype(int) # adotamos que se chance acima de 0.5, significa sobrevivência

submission = pd.DataFrame(predictions, columns=['survived'], index=test.index)
submission = submission.rename_axis('id')
submission.loc[submission['survived'] == 0, 'survived'] = 'no'
submission.loc[submission['survived'] == 1, 'survived'] = 'yes'

#submission = submission.sort_index(axis=0)
submission.to_csv('submission.csv')

submission.info()
#for idPass, classe, age, sex in test.iterrows():
#    predictions = model.predict([])