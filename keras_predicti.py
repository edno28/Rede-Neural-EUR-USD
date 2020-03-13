import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_squared_error
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import model_from_json

#-----//////---------/////---------#-----//////---------/////---------#-----//////---------/////---------
new_model=0
nome_model="epoc=250;tick=90;time=5;dados:1900.json"
nome_peso="epoc=250;tick=90;time=5;dados:1900.h5"
detalhamento="epoc=250;tick=105;time=5;dados:1900"

q_banco=10000
tick_treino=90
div_dataset=0.9#porcentagem do banco usado para treinos
coluna_fechamento=3
n_entradas=4
epocas=250
#-----//////---------/////---------#-----//////---------/////---------#-----//////---------/////---------
dataset = pd.read_csv(os.path.join('EUR/USD',r"C:\Users\Edno2\OneDrive\Área de Trabalho\NovasIAs\EURUSD_INDICADORESTECNICOS_tick5_2019.csv"),usecols=[0,1,2,3,4])

data=dataset.iloc[:,:]

X= data.iloc[:q_banco,1:].values
y= data.iloc[:,4:5].values


scaler_x = StandardScaler()
X = scaler_x.fit_transform(X)
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)

X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y,
                                                                test_size =div_dataset,
                                                                random_state = 0)
#-----//////---------/////---------#-----//////---------/////---------#-----//////---------/////---------
X_test = []
y_treinamento=y_treinamento[tick_treino:]
y_teste=y_teste[tick_treino:]

for i in range(tick_treino,len(X_teste)):
    X_test.append(X_teste[i-tick_treino:i,])    

X_test= np.array(X_test)
X_test = np.reshape(X_test,(len(X_test),tick_treino, n_entradas))



if new_model==1:
    X_train = []
    
    y_treinamento=y_treinamento[tick_treino:]
    y_teste=y_teste[tick_treino:]
    
    for i in range(tick_treino,len(X_treinamento)):
        X_train.append(X_treinamento[i-tick_treino:i,])
        
    X_train= np.array(X_train)
    X_train =  np.reshape(X_train,(len(X_train),tick_treino,n_entradas))
    
    
    
#-----//////---------/////---------#-----//////---------/////---------#-----//////---------/////---------
    def regressor():
      regressor = Sequential()
    
      #CAMADA 1 activation="tahn"
      regressor.add(LSTM(units = 33,  return_sequences = True, input_shape = (tick_treino, n_entradas)))
      regressor.add(Dropout(0.2))
    
      
      regressor.add(LSTM(units = 33,return_sequences=False))
      regressor.add(Dropout(0.2))
    
      
      regressor.add(Dense(95,activation="selu"))
      
      regressor.add(Dense(1))
    
     
      regressor.compile(optimizer = "adam", loss ='mean_squared_error')
      
      return regressor
    
    
    regressor = regressor()
    print(regressor.summary())
    
    history = regressor.fit(X_train, y_treinamento, epochs =epocas, batch_size =150, shuffle=True)
 
#-----//////---------/////---------#-----//////---------/////---------#-----//////---------/////---------
    # serialize model to JSON
    regressor_json = regressor.to_json()
    with open("{}.json".format(detalhamento), "w") as json_file:
        json_file.write(regressor_json)
    # serialize weights to HDF5
    regressor.save_weights("{}.h5".format(detalhamento))
    print("Saved model to disk")

#-----//////---------/////---------#-----//////---------/////---------#-----//////---------/////---------
    previsoes = regressor.predict(X_test) 


if new_model==0:
    # load json and create model
    json_file = open(nome_model, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(nome_peso)
    print("Loaded model from disk")
    detalhamento=nome_peso
    previsoes= loaded_model.predict(X_test)
#-----//////---------/////---------#-----//////---------/////---------#-----//////---------/////---------
y_teste = scaler_y.inverse_transform(y_teste)
#previsoes = scaler_y.inverse_transform(previsoes)
#-----//////---------/////---------#-----//////---------/////---------#-----//////---------/////---------
mae = mean_absolute_error(y_teste, previsoes)
rmse = math.sqrt(mean_squared_error(y_teste, previsoes))

def mape(real,predict):
    mape=0
    for c in range(0,len(real)):
        mape+=(((real[c]-predict[c])/len(real))*100)/len(real)
    return mape
        
        
mape=mape(y_teste, previsoes)
mape=mape[0]
#-----//////---------/////---------#-----//////---------/////---------#-----//////---------/////---------
#Filtro
def binario(valor):
    bi=[]
    for c in range(1,len(valor)):
        if valor[c]>valor[c-1]:
            bi.append(1)
        elif valor[c]<valor[c-1]:
            bi.append(0)
        elif valor[c]==valor[c-1]:
            bi.append(-1)
    return bi

def filtro(real,predito):   
    d=[]
    g=0
    for c in range(0,len(predito)):
        if predito[c]==-1 or real[c]==-1:
            d.append(c-g)
            g+=1
    for c in range(0,len(d)):
        del(real[d[c]])
        del(predito[d[c]])
            
    return real,predito

x2=binario(previsoes)
x1=binario(y_teste)
real,predito=filtro(x1,x2) 
#-----//////---------/////---------#-----//////---------/////---------#-----//////---------/////---------
#tp e tn são os corretos
tn, fp, fn, tp=confusion_matrix(real,predito).ravel()
acuracia=((tp+tn)/(tp+tn+fp+fn))*100
print("-----------------------\nVerdadeiro Positivo:{}\nVerdadeiro Negativo:{}\n-----------------------\nFalso Positivo:{}\nFalso Negativo:{}\n-----------------------".format(tp,tn,fp,fn))
print("Acurracia:{:.2f}\nMAE:{:.5f}\nRMSE:{:.5f}\nMAPE:{:.5f}\n-----------------------".format(acuracia,mae,rmse,mape))
#-----//////---------/////---------#-----//////---------/////---------#-----//////---------/////---------

