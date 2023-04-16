import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import yfinance as yf
from pandas_datareader import data as pdr
yf.pdr_override()
start='2010-01-01'
end='2023-01-01'

user_input=st.text_input('Enter Stock Picker','TSLA')

df=pdr.DataReader(user_input, start, end)

#describing the data
st.subheader('Data from 2010-2022')
st.write(df.describe())


#visualisations
st.subheader('Closing Price vs Time Chart')
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart with 100 moving average')
ma100=df.Close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))  
plt.plot(df.Close)
plt.plot(ma100)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100 and 200 moving average')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))  
plt.plot(df.Close,'b',label='Original Price')
plt.plot(ma100,'r',label='Moving Avg. of past 100 days')
plt.plot(ma200,'g',label='Moving Avg. of last 200 days')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)

#splitting the data into Training and Testing
data_training=(df['Close'][0:int(len(df)*0.70)])
data_testing=(df['Close'][int(len(df)*0.70):int(len(df))])
data_training=data_training.to_frame()
data_testing=data_testing.to_frame()

scaler=MinMaxScaler(feature_range=(0,1))
data_training_array=scaler.fit_transform(data_training)

x_train=[]
y_train=[]

for i in range(100,data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i,0])
    
x_train,y_train=np.array(x_train), np.array(y_train)

#Loading my model
model=load_model('keras_model.h5')

#Testing Part
past_100_days=data_training.tail(100)
final_df=past_100_days.append(data_testing,ignore_index=True)
input_data=scaler.fit_transform(final_df)

x_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test,y_test = np.array(x_test), np.array(y_test)

#predictions
y_predicted=model.predict(x_test)  
scaler=scaler.scale_

scale_factor=1/scaler[0]    
y_predicted=y_predicted*scale_factor
y_test=y_test*scale_factor


#Final graph
st.subheader('Predictions vs Original')
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predicted,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)