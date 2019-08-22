# ToxicCommentClassification

This project is an implementation of toxic comment classification challenge in Kaggle and it is hosted [here](https://gseshadri-toxic.herokuapp.com/) as a consumable application. 


This project has three parts

1. Preprocessing

2. Modeling

3. Dash Application

## Preprocessing:

Removed stopwords, punctuations, blank lines and some urls, hyperlinks and IPs from the input texts. Used WordNetLemmatizer to lemmatize the words and used glove 100d word vectors as embeddings.

## Models:

Built three different models using Keras. It includes a CNN, a RNN and a Naive Bayes SVM. These model outputs are stacked to get the final output.

## Dash Application:

A consumable UI is created using dash and is hosted in heroku.


