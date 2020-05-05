import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#setup neural network
model = Sequential()
model.add(Dense(units=12, activation='relu', input_dim=5))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=1, activation='softmax'))
model.compile(loss='mean_squared_error',optimizer='sgd',metrics=['accuracy'])

X = [[0,0,0,0,0],[1,1,1,1,1]]
y = [[0],[1]]

X = np.array(X)
y = np.array(y)
print (y.shape)
#X = X.reshape(X.shape[0], 1,5)
model.train_on_batch(X,y)

test = [[0,0,0,0,0]]
test = np.array(test)
model.predict(test)
