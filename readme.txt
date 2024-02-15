Project Title: Share Market Analysis 
Contributors: Jeet Mehta.

 
This project has 3 components:
Analysis, Model Training and an app to predict prices.

Motivation for the Project:
To Analyse and Predict stock prices using Moving Average as technical indicator. 
Traders find the Analysis and Prediction useful and need the Analysis for their trading decisions.

There's a pdf file that has the code with outputs: Project_File.pdf.
Read the file to get an read the code and it's Output.
To run the code  Follow the instructions.

To run our Data Analytics And Visualisaiton Project on Share Market Prediction,
You need to first install these libraries using pip:

pandas_datareader
numpy
pandas
matplotlib
sklearn
keras
yfinance
datetime
os
streamlit

You need an internet connection for downloading data sets form the internet using yfinance library,
Open Jupyter Notebook,
Open the Project,
If you need to change directory of Jupyter Notebook,
Run this command on terminal,
jupyter notebook --notebook-dir="dir"

Open the jupyter notebook files and use the project.

Share_Market_Analysis.ipynb File shows the technical analysis of 3 stocks : Tesla, Ford and GM for Comparision.
We take the data from yfinance library for stock data sets.
The agenda of this project is to answer various questions that a stock trader or investor asks for analysing the stocks in order to make the business decisions.
This project answers questions on the stock data.
It focuses on using Moving Averages as a technical analysis for stock.
When a Short Term Moving Average crosses Long Term Moving Average:
In upward direction, signifying an upward trend, Stock most probably starts increasing.
In upward direction, signifying a downward trend, Stock most probably starts decreasing.   
We computed and plotted 50 days, 100 days and 200 days (MA)Moving Average of GM, Ford and Tesla.
We then proved how Our Analsis is supported by the data.

Now As we proved in Share_Market_Analysis Project how Moving Averages are an excellent indicators for predicting stock projection,
We made a Stock Prediction Model using Keras Library.
Model_Training.ipynb is our jupyter file that trained our Share Market Prediction Model.
We took data set of Tesla from 2010 to 2019 to train our model.
We used 70% of the data for training and 30% for testing.
Used Long Short Term Memory Neural Network which is a type of RNN Neural Net.
LSTM Neural Net is used when you need to predict or to make projections.
Used MinMaxScaler to transform features by scaling each feature to a given range.
Scaled the prices data set to be between 0 and 1 using MinMaxScaler.
Used RELU Activation Function.
After the 
Trained our Model and Saved it as 'keras_model.h5'.

Now time to predict stock prices.

App.py is python Script that uses our trained model ('keras_model.h5') to predict prices of 3 stocks GM, Ford and Tesla.
It uses Streamlit to run the application.
Open the app.py program
Run one of the commands in terminal to run the app:
streamlit run app.py
streamlit run dir\app.py  //here dir is your directory


You will get link to open the App on the terminal,
Click on link and open the App,
Enter Stock Ticker, 
TSLA,  GM and F  for Tesla, GM and Ford respectively.
Wait for the processing,
In less than minute my App will plot the Predictions vs Original Price Chart.



