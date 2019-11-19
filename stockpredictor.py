#importing libraries
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 

#importing train and test set
training_set=pd.read_csv("Google_Stock_Price_Train.csv")
training_set=training_set.iloc[:,1:2].values
test_set=pd.read_csv("Google_Stock_Price_Test.csv")
real_stock_price=test_set.iloc[:,1:2].values

#print(training_set)


#rescaling and reshaping the train set
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
training_set=sc.fit_transform(training_set)
#print(training_set)
X_train=training_set[0:1257]
Y_train=training_set[1:1258]
print(X_train)
print(Y_train)
X_train=np.reshape(X_train,(1257,1,1))


#Adding the Neural Network layers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


regressor=Sequential()
regressor.add(LSTM(units=4,activation='sigmoid',input_shape=(None,1)))
regressor.add(Dense(units=1))
regressor.compile(optimizer='Nadam',loss='mean_squared_error')
regressor.fit(X_train,Y_train,batch_size=32,epochs=200)


#predicting the output
inputs=real_stock_price
inputs=sc.transform(inputs)
inputs=np.reshape(inputs,(20,1,1))
predicted_stock_price=regressor.predict(inputs)
predicted_stock_price=sc.inverse_transform(predicted_stock_price)

#plotting the results
plt.plot(real_stock_price,color='red',label='Real Google Stock price')
plt.plot(predicted_stock_price,color='blue',label='Predicted Google stock price')
plt.title('Google stock price prediction')
plt.xlabel('Time')
plt.ylabel('Stock price')
plt.legend()
plt.show()
