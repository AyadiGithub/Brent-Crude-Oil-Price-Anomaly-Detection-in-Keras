# Brent-Crude-Oil-Price-Anomaly-Detection-in-Keras

## Brent crude oil price Anomaly Detection in Time Series Data with Keras

## Project Objective:
- Design and Build an LSTM Autoencoder in Keras
- Detect anomalies (sudden price changes) in Brent Crude Oil Price
- Data range: 1988 - 2020

The Dataset is a time series with daily Price changes for Brent Crude Oil from 1988 to 2020. 
Brent Crude is the International Benchmark for Crude oil. 

Source of Dataset: Yahoo Finance

## The dataset is a CSV format with: 
1. Daily timestamp
2. Daily Open and Close Price
3. Daily High and Low Price
4. Volume
5. 24h Change %

## The Project will follow the below steps:
1. Project Overview and Import Libraries
2. Load and Inspect Brent Crude Oil Price History
3. Data Preprocessing
4. Temporalize Data and Create Training and Test Splits
5. Build a CuDNNLSTM Autoencoder for a CuDNN capable GPU and/or LSTM for CPU
6. Train the Autoencoder (GPU or CPU)
7. Plot Metrics and Evaluate the Model
8. Detect Anomalies in the Brent Crude Oil Price History
9. Visualize the Anomalies based on threshold

## The required libraries:
- Numpy
- Tensorflow 2.0+
- Keras
- Pandas
- Seaborn
- Matplotlib
- Plotly
- Scikit Learn

## Other requirements
- CUDA Capable GPU (For CuDNN LSTM)
- CuDNN (For CuDNN LSTM)
- CUDA 10.1
- Tensorflow-gpu 
- Keras-gpu
