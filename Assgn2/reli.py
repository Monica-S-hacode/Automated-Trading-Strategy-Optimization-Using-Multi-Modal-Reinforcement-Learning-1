import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
import csv

# Fetch 1 year of daily stock data for Reliance
ticker = "RELIANCE.NS"
data = yf.download(ticker, period="1y")
data.to_csv("reliance_1y_data.csv")

# View first few rows
print(data.head())
