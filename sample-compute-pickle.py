import numpy as np
import pandas as pd
from datetime import datetime
import yfinance as yf

#
# calibrate timing
#
_begin = '2021-03-11'
_end = datetime.now()
_td = pd.Timestamp(_begin) - pd.Timestamp(_end)
_period = str(-_td.days)+'d'
_interval = '1h'
print(_period)
#
# get btc and eth median spot prices
#
data_eth = yf.download(tickers='ETH-USD',period=_period,interval=_interval)
data_btc = yf.download(tickers='BTC-USD',period=_period,interval=_interval)
_btc = pd.Series(data=data_btc["Close"])
_eth = pd.Series(data=data_eth["Close"])
#
_btc_resampled = _btc.resample('6H').median()
_eth_resampled = _eth.resample('6H').median()
# get dict
btc_spot_prices = _btc_resampled.to_dict()
eth_spot_prices = _eth_resampled.to_dict()

#
# define methods
#
def _asset(sym):
    tokens = sym.split("-")
    asset = tokens[0]
    return asset

def _spot_price(asset,T):
    if asset == "BTC":
        return btc_spot_prices(T)
    elif asset == "ETH":
        return eth_spot_prices(T)
    else: raise ValueError("asset type invalid, must be ETH or BTC")

def _exercised(sym):
    tokens = sym.split('-')
    asset = tokens[0]
    T = datetime.datetime.strptime(tokens[1],'%d%b%y')
    K = int(tokens[2])
    Y = int(tokens[3] == 'C')
    if Y == 1:
        return max(spot_price(asset,T)-K,0) > 0
    elif Y == 0:
        return max(K-spot_price(asset,T),0) > 0
    else:
        raise ValueError("Y invalid, must be 1 or 0")

def _type(sym):
    tokens = sym.split('-')
    return int(tokens[3] == 'C')

def _strike(sym):
    tokens = sym.split('-')
    return int(tokens[2])

def _maturity(sym):
    tokens = sym.split('-')
    return datetime.datetime.strptime(tokens[1],'%d%b%y')
#
#
#
_A = np.vectorize(_asset)
_Y = np.vectorize(_type)
_K = np.vectorize(_strike)
_T = np.vectorize(_maturity)
_chi = np.vectorize(_exercised)
#
# need to apply lambda here
#
def _S(syms,t):
    A = _A(syms)
    S = _spot_price(A,t)
    return np.column_stack((S,np.ones(len(A))))

def _rmin_loss(syms):
    Y = _Y(syms)
    T = _T(syms)
    K = _K(syms)
    Rmin = np.column_stack((Y,np.multiply(1-Y,K)))
    chi = np.outer(_chi(syms),np.ones(2))
    rhs = np.column_stack((np.ones(len(syms)),K)) - 2*Rmin
    return Rmin, np.multiply(chi,rhs)

def _gain(p):
    return np.column_stack((np.zeros(len(p)),p))

def _Rprime(p,syms):
    G = _gain(p)
    Rmin,L = _rmin_loss(syms)
    Rprime = Rmin + L + G
    return Rprime

def instantaneous_return(syms,p,t):
    R,_ = _rmin_loss(syms)
    Rprime = _Rprime(p,syms)
    S = _S(syms,t)
    TVR = np.matmul(S.transpose(),R).trace()
    TVRp = np.matmul(S.transpose(),Rprime).trace()
    return TVRp/TVR

#
# need to reset data if beginning of new day
# 
#
nrows = 8437280
directory = "../../options_csv/"
day = '2021-03-11'
filename = "deribit_quotes_"+day+"_OPTIONS.csv"
data = pd.read_csv(directory+ filename,nrows=nrows)
data = data.dropna()
data["datetime"] = pd.to_datetime(data['local_timestamp']*1e3)
data = data.set_index("datetime")
#
#
#
_tg = _btc_resampled.index
print("min",_tg.min())
print("max",_tg.max())
for i,t in enumerate(_tg[:80]):
    # get timestamp objects
    ts_start=t.floor('h')
    str_start = ts_start.strftime('%H:%M')
    ts_end=_tg[i+1].floor('h')
    str_end = ts_end.strftime('%H:%M')
    print("median btc spot price at time ",ts_start," is ",btc_spot_prices[ts_start])
    print("median eth spot price at time ",ts_start," is ",eth_spot_prices[ts_start])
    #
    #
    #
    deribit_sample = data.between_time(str_start,str_end)
    option_sample = deribit_sample.groupby(['symbol']).median()
    syms = option_sample.index.values
    p = sample.ask_price.values
    R = instantaneous_return(syms,p,t)
    print(R)
    #
    # pass to methods, compute instantaneous_return
