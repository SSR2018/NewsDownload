import keras
from keras import Sequential
from keras.layers import LSTM,Bidirectional,Dropout,BatchNormalization,Flatten,Dense,Activation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
print(keras.__version__)

time_step = 3
df = pd.read_csv('601318.csv')
y = df.close[:-4].values
dff =df.copy(deep=True)
dff = dff.drop(['close'],axis = 1)
x = []
for i in range(1,len(dff)-3):
    x.append(dff.iloc[i:i+time_step,2:].values)
x = np.array(x)
print('the shape of x is ',x.shape,'the shape of y is ',y.shape)

batch = 16

model = Sequential()
model.add(LSTM(30,batch_input_shape=(batch,x.shape[1],x.shape[2]),stateful=True,return_sequences=True))
model.add(BatchNormalization())
# model.add(Activation('relu'))
model.add(Bidirectional(LSTM(10,stateful=True,return_sequences= True)))
model.add(Dropout(0.4))
# model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(1,activation='relu'))

print(model.summary())
model.compile(loss='mean_squared_error', optimizer='adam')
def generate(l):
    while 1:
        for i in range(l//batch):
           yield (x[i*batch:i*batch+batch],y[i*batch:i*batch+batch])
model.fit_generator(generate(x.shape[0]),steps_per_epoch=x.shape[0] // batch,nb_epoch=50,verbose=2)
pre = model.predict_on_batch(x[-5:,:,:])
print(pre,end='\n')
print(y[-5:],end='\n')
# plt.plot(np.range(10),y[-10:],'r')
# plt.plot(np.range(10),pre,'b')
# plt.show()
