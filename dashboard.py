# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 16:33:12 2021

@author: Supriya
"""
import math
import numpy as np
import streamlit as st
#import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
from plotly import graph_objs as go
import plotly.express as px
import nltk
nltk.download('wordnet')
import snscrape.modules.twitter as snstwitter
import warnings
from nltk.corpus import stopwords, wordnet
nltk.download('stopwords')
import re
#from rake_nltk import Rake
import pickle
from nltk.stem import WordNetLemmatizer

# Warnings ignore 
warnings.filterwarnings(action='ignore')

st.set_option('deprecation.showPyplotGlobalUse', False)


st.title('Stock Market Analysis Dashboard')

# Sidebar options
option = st.sidebar.selectbox('Navigation',['Fundamental Analysis','Technical Analysis'])
st.set_option('deprecation.showfileUploaderEncoding', False)
if option == "Fundamental Analysis":
    start=st.date_input('Start',value=pd.to_datetime('2020-10-01'))
    end=st.date_input('End',value=pd.to_datetime('today'))



#tickers=('INFY','SBIN.NS','RELIANCE.NS','TCS.NS','IDEA.NS','MSFT')
    user_input=st.text_input('Enter stock Ticker','SBIN.NS')
#dropdown=st.multiselect('Pick your assets',tickers)
    df=data.DataReader(user_input,'yahoo',start,end)
    df.reset_index(inplace=True)

#Describing data
    st.subheader('Data from {}-to-{}'.format(start, end))
    st.write(df.tail())
    df2 = pd.DataFrame(df,columns=['Date','High','Low','Close','Open'])
    st.write('Previous day Stock price details',df2.iloc[-1:])
#Visualizations
    st.subheader('Closing price vs Time Chart')

    fig1 = px.line(df, x=df['Date'], y=df['Close'])
    st.plotly_chart(fig1)

    st.subheader('Opening price vs Time Chart')
    fig2 = px.line(df, x=df['Date'], y=df['Open'])
    st.plotly_chart(fig2)

    st.subheader('Short term Gain Chart')
    df['gain']=df.Close.pct_change(periods = 1)
    fig3 = px.line(df, x=df['Date'], y=df['gain'])
    st.plotly_chart(fig3)


    st.subheader('Closing Price vs Time chart with 10MA,5MA')
    MA5 = df.Close.rolling(5).mean()
    MA10 = df.Close.rolling(10).mean()
    fig=plt.figure(figsize=(12,6))
    plt.plot(MA5)
    plt.plot(MA10)
    plt.plot(df.Close)

    st.pyplot(fig)

# Create a new dataframe with only the 'Close column 
    data = df.filter(['Close'])
# Convert the dataframe to a numpy array
    dataset = data.values

# Get the number of rows to train the model on
    training_data_len = int(np.ceil( len(dataset) * .95 ))



    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

# Create the training data set 
# Create the scaled training data set
    train_data = scaled_data[0:int(training_data_len), :]
# Split the data into x_train and y_train data sets
    x_train = []
    y_train = []

    for i in range(100, len(train_data)):
       x_train.append(train_data[i-100:i, 0])
       y_train.append(train_data[i, 0])
    #if i<= 100:
     #   print(x_train)
      #  print(y_train)
       # print()
        
# Convert the x_train and y_train to numpy arrays 
    x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
# x_train.shape

#load model
    model=load_model('lstm_model.h5')

# Create the testing data set
# Create a new array containing scaled values  
    test_data = scaled_data[training_data_len - 100: , :]
# Create the data sets x_test and y_test
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(100, len(test_data)):
       x_test.append(test_data[i-100:i, 0])
    
# Convert the data to a numpy array
    x_test = np.array(x_test)

# Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

# Get the models predicted price values 
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

# Get the root mean squared error (RMSE)
#rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
#rmse

# Plot the data
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions

# Visualize the data
    st.subheader('Predictions on Test data using LSTM model')
    fig10=plt.figure(figsize=(16,6))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.show()
    st.pyplot(fig10)


#Forcasting
    user_input=st.text_input('Enter no of days you want to forcast',5)
    n=int(user_input)
    i=0
    X_input = df.iloc[-100:].Close.values   # getting last 100 rows and converting to array
    while i<=n-1:
       X_input=np.array(X_input[-100:])
       X_input1 = scaler.transform(X_input.reshape(-1,1))      # converting to 2D array and scaling
       X_input2 = np.reshape(X_input1, (1,100,1))
       predictions = model.predict(X_input2)
       predictions = scaler.inverse_transform(predictions)
       X_input=[*X_input, *predictions[-1]]
  #print('Close price prediction after', n ,'days',predictions[0,0])
       a=np.ravel(predictions) #convert 2D in to 1D
  
       st.write('Close price prediction after', i+1 ,'days',predictions[0,0])
       i=i+1
       
elif option == "Technical Analysis":
    st.subheader("Technical Analysis,includes reading the charts and using statistical figures to identify the trends in the stock market")
    user_input=st.text_input('Enter stock Ticker','SBIN.NS')
    start=st.date_input('Start',value=pd.to_datetime('2021-10-11'))
    end=st.date_input('End',value=pd.to_datetime('today'))

# Creating list to append tweet data to
    tweets_list2 = []

# Using TwitterSearchScraper to scrape data and append tweets to list
# Using enumerate to get the tweet and the index (to break at certain nu7mber of tweets)
    for i, tweet in enumerate(snstwitter.TwitterHashtagScraper(user_input).get_items()):
        if i > 500:
           break

    # Save the required details like content, date in list
        tweets_list2.append([tweet.date,tweet.content])

# Creating a dataframe from the tweets list above
    tweets_df2 = pd.DataFrame(tweets_list2,
                          columns=['Datetime','Text'])

    st.write(tweets_df2)

    input_text = ' '.join(tweets_df2['Text'])
    if st.button("Predict sentiment"):
        
        #st.write("Number of words in Review:", len(input_text.split()))
        wordnet=WordNetLemmatizer()
        text=re.sub('[^A-za-z0-9]',' ',input_text)
        text=text.lower()
        text=text.split(' ')
        text = [wordnet.lemmatize(word) for word in text if word not in (stopwords.words('english'))]
        text = ' '.join(text)
        pickle_in = open('model.pkl', 'rb') 
        model = pickle.load(pickle_in)
        pickle_in = open('vectorizer.pkl', 'rb') 
        vectorizer = pickle.load(pickle_in)
        transformed_input = vectorizer.transform([text])
        
        if model.predict(transformed_input) == -1:
           st.write("Input TWEETs has Negative Sentiment.:Stock price may go down:")
        elif model.predict(transformed_input) == 1:
           st.write("Input TWEETs has Positive Sentiment.:smile:Stock price may go up")
        else:
          st.write(" Input TWEETs has Neutral Sentiment: Can not say about Stock price may go down/up")       
