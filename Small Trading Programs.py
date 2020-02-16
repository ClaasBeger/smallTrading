#Module für die kommenden Programme:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from pandas_datareader import data as web
import datetime as dt
from matplotlib import style
import pyEX as p
import matplotlib.dates as mdates
from mpl_finance import candlestick_ohlc

##Beispiel für eine Funktion
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0,10,100)
y = np.sin(x)
plt.plot(y)
plt.show()

#Beispiel für ein Diagramm
mu, sigma = 172, 4   #Mittelwert 172 Standartabweichung 4
x = mu + sigma *np.random.randn(10000)  #Erzeugen von Zufallswerten
plt.hist(x, 50, normed=1, facecolor='darkblue')

plt.xlabel('Groesse')
plt.ylabel('Wahrscheinlichkeit')
plt.title('Groesse von Schuelern')
plt.text(160, .125, r'$\mu=172, \sigma=4$')
plt.axis([155, 190, 0, 0.15])   #Wertebereich der Achsen
plt.grid(True)   #Gitter oder nicht
plt.savefig('diagramm.png')
plt.show()

#Boxplot
mu, sigma = 172, 4   #Mittelwert 172 Standartabweichung 4
x = mu + sigma *np.random.randn(10000)  #Erzeugen von Zufallswerten
plt.boxplot(x)
plt.grid(True)
plt.show()

Regressionsfunktion
x = 5* np.random.random(25)   #Zufallswerte
y = 5* np.random.random(25)

reg = np.polyfit(x,y,1)  
regf = np.poly1d(reg)  #Umwandeln in Polynomfunktion 1.Grades

plt.plot(x,y,'bo',x,regf(x), 'r--')
plt.show()

#Kreisdiagramm
label = 'Pizza', 'Pommes', 'Salat', 'Burger'
groessen = [215, 130, 245, 210]
farben = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
explode = (0.1, 0, 0, 0)

plt.pie(groessen, explode=explode, labels=label,colors = farben, autopct= '%1.1f%%', shadow=True, startangle = 140)

plt.axis('equal')
plt.show()

#Tabelle mit Externer Speicherung
daten = {'Name' : ['Alex', "Benni", 'Claudia'],
         'Alter' : [19,18,21],
         'Groesse' : [183,178, 163]}
df = pd.DataFrame(daten, columns=['Name','Alter', 'Groesse'])
df = df.sort_values("Name")
df.to_csv('daten.csv')
x = pd.read_csv('daten.csv')
print(x)

#Aktienkurse mit Regressionsgerade      alle Aktien von S&P 500
start = dt.datetime(2016,1,1)
end = dt.date.today()

SAP = web.DataReader("SAP", "yahoo",start, end)
#apple["Adj Close"].plot(grid = True)

x= list(range(0, len(SAP.index.tolist()), 1))
closing = SAP["Adj Close"]

date_x = SAP.index

reg = np.polyfit(x, closing,1)
regf = np.poly1d(reg)

plt.grid = True
plt.plot(date_x, closing, 'b', date_x, regf(x), 'r')
plt.show
style.use('fivethirtyeight') #style

#Daten in Dateien schreiben und lesen
df = web.DataReader('AAPL', 'yahoo',)
print(df.head())
df['Adj Close'].plot()
df = df['Adj Close'].rolling(window = 100, min_periods = 0).mean()
df.dropna(inplace = True)
print(df)
#plt.show()
df.to_csv("apple.csv", sep=';')
x.to_html('apple.html')    #Schreiben in HTML oder CSV Datei oder JSON
df.to_json('apple.jason')
df = pd.read_html('apple.html')# reading from any listed data


#100ma, and EMA (Print vor Plot um Werte zu bekommen)
ticker = 'AAPL'                                #standard plot
timeframe = '1y'
df = p.chartDF(ticker,timeframe)
df = df[['close']]
df.reset_index(level=0, inplace = True)
df.columns=['ds','y']
plt.plot(df.ds, df.y)    # pyEX as p
plt.show

rolling_mean = df.y.rolling(window=100).mean()    #day moving average
plt.plot(df.ds, df.y, label='AAPL')
plt.plot(df.ds, rolling_mean,label='AAPL 100 Day SMA', color='orange')
plt.legend(loc='upper left')
plt.show()

exp = df.y.ewm(span=100, adjust=False).mean()  #exponential moving average (EMA)
plt.plot(df.ds, df.y, label= 'AAPL')
plt.plot(df.ds, exp, label='AAPL 100 Day EMA')
plt.legend(loc='upper left')
plt.show()

#100 Day MA inkl. Volumen
start = datetime.datetime(2016,1,1)
end = datetime.date.today()

df = web.DataReader("AAPL", "yahoo",start, end)

print(df.head())
df['Adj Close'].plot()
plt.show()
df['100ma'] = df['Adj Close'].rolling(window = 100, min_periods = 0).mean()
df.dropna(inplace = True)
print(df.head())

ax1= plt.subplot2grid((6,1),(0,0),rowspan=5,colspan=1)
ax2= plt.subplot2grid((6,1),(5,0),rowspan=1,colspan=1, sharex = ax1)
ax1.plot(df.index, df['Adj Close'])
ax1.plot(df.index, df['100ma'])
ax2.bar(df.index, df['Volume'])


#10 Tage mit farblicher Hervorhebung und Handelsvolumen Axe 1 im Candlestick Format
start = dt.datetime(2016,1,1)
end = dt.date.today()

df = web.DataReader("AAPL", "yahoo",start, end)


#print(df.head())
df['Adj Close'].plot()

df['100ma'] = df['Adj Close'].rolling(window = 100, min_periods = 0).mean()
df.dropna(inplace = True)

df_ohlc = df['Adj Close'].resample('10D').ohlc()
df_volume = df['Volume'].resample('10D').sum()
df_ohlc.reset_index(inplace=True)
df_ohlc['Date']= df_ohlc['Date'].map(mdates.date2num)
ax1 = plt.subplot2grid((6,1),(0,0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((6,1),(5,0), rowspan=1, colspan=1, sharex=ax1)
ax1.xaxis_date()

candlestick_ohlc(ax1, df_ohlc.values, width=5, colorup='g', colordown='r')
ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values)
ax1.grid(True, color='black')
plt.show()
#print(df.head())

#load into pickle datei
import bs4 as bs
import requests
import pickle
def lade_sp500_ticker():
  Response = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
  Soup = bs.BeautifulSoup(Response.text,'lxml')
  Tabelle = Soup.find('table', {'class':'wikitable sortable'})
  tickers = []
  for reihe in Tabelle.findAll ('tr') [1:]:
      ticker = reihe.findAll ('td') [1].text
      tickers.append(ticker)
  with open("sp500tickers.pickle", 'wb') as f:
      pickle.dump(tickers, f)
  print(tickers)
  
  return(tickers)
lade_sp500_ticker()

#Speichern von S und P 500 in Pickle
import bs4 as bs
import datetime as dt
import os
import yfinance as yf
from pandas_datareader import data as pdr
import pickle
import requests

yf.pdr_override()

def save_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[1].text
        ticker = str(ticker).replace('.','-')
        tickers.append(ticker)
    
    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)
    
    print(tickers)
    return tickers


# save_sp500_tickers()
def get_data_from_yahoo(reload_sp500=False):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)
    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    start = dt.datetime(2010, 1, 1)
    end = dt.datetime.now()
    for ticker in tickers:
        # just in case your connection breaks, we'd like to save our progress!
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            df = pdr.get_data_yahoo(ticker, start, end)
            df.reset_index(inplace=True)
            df.set_index("Date", inplace=True)
            df = df.drop('Symbol', axis = 1)
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
        else:
            print('Already have {}'.format(ticker))


get_data_from_yahoo()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
from matplotlib import style
import pyEX as p
import matplotlib.dates as mdates
from mpl_finance import candlestick_ohlc
import bs4 as bs
import requests
import pickle
import os

yf.pdr_override()
##web.get_data_google('spy').tail()
def lade_sp500_ticker():
  Response = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
  Soup = bs.BeautifulSoup(Response.text,'lxml')
  Tabelle = Soup.find('table', {'class':'wikitable sortable'})
  tickers = []
  for reihe in Tabelle.findAll ('tr') [1:]:
      ticker = reihe.findAll ('td') [0].text
      tickers.append(ticker)
  with open("sp500tickers.pickle", 'wb') as f:
      pickle.dump(tickers, f)
  print(tickers)
  
  return(tickers)
lade_sp500_ticker()
def lade_preise_von_yahoo(ticker_neuladen = False):
    
    if ticker_neuladen:
        tickers = lade_sp500_ticker()
    else:
        with open("sp500tickers.pickle", 'rb')as f:
            tickers = pickle.load(f)
    if not os.path.exists('kursdaten'):
        os.makedirs('kursdaten')
    
    start = dt.datetime(2010, 1, 1)
    end = dt.datetime(2019,12,31)
    
    for ticker in tickers:
        if not os.path.exists('kursdaten/{}.csv'.format(ticker)):
            print("{} wird geladen...".format(ticker))
            df = web.DataReader(ticker, 'yahoo', start, end)
            df.to_csv('kursdaten/{}.csv'.format(ticker))    
        else:
           print("{} bereits vorhanden!".format(ticker))
    
def daten_kompilieren():
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)
    main_df = pd.DataFrame()
    print("Daten werden kompiliert...")
    for ticker in tickers:
        df = pd.read_csv("kursdaten/{}.csv".format(ticker))
        df.set_index("Date", inplace = True)
        
        df.rename(columns = {"Adj Close":ticker}, inplace = True)
        df.drop(["Open","High","Low","Close","Volume"],1, inplace = True)
        
        if main_df.empty:
            main_df= df
        else:
            main_df = main_df.join(df, how='outer')
        main_df.to_csv('sp500_daten.csv')
        print("Daten kompiliert")
lade_preise_von_yahoo(ticker_neuladen=True)
#daten_kompilieren()