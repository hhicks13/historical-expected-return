import numpy as np
import pandas as pd
from datetime import datetime
import yfinance as yf
import sys

#
# calibrate timing
#
_begin = '2021-03-11'
_end = datetime.now()
#_td = pd.Timestamp(_begin) - pd.Timestamp(_end)
#_period = str(-_td.days)+'d'
#_interval = '1h'

#
# get btc and eth median spot prices
#
data_eth = yf.download(tickers='ETH-USD',start=pd.Timestamp(_begin),end=pd.Timestamp(_end),interval='1h')
data_btc = yf.download(tickers='BTC-USD',start=pd.Timestamp(_begin),end=pd.Timestamp(_end),interval='1h')
_btc = pd.Series(data=data_btc["Close"])
_eth = pd.Series(data=data_eth["Close"])
#
_btc_resampled = _btc.resample('6H').mean()
_eth_resampled = _eth.resample('6H').mean()
# get dict
_bsp0 = _btc_resampled.to_dict()
_esp0 = _eth_resampled.to_dict()

    
#
# clean timestamp keys
#
btc_spot_prices = {}
eth_spot_prices = {}
for _key in _bsp0.keys():
    val_btc = _bsp0[_key]
    val_eth = _esp0[_key]
    print(val_btc)
    print(val_eth)
    
    _key = _key.replace(second=0,microsecond=0,minute=0)
    key = pd.Timestamp(year=_key.year,month=_key.month,day=_key.day,hour=_key.hour,minute=0,second=0)
    
    btc_spot_prices[key] = val_btc
    eth_spot_prices[key] = val_eth
    
#
# define methods
#
def _asset(sym):
    tokens = sym.split("-")
    asset = tokens[0]
    return asset

def _spot_price(asset,T):
    print(type(T))
    print(T)
    if asset == "BTC":
        val = btc_spot_prices[T]
        return val
    elif asset == "ETH":
        val = eth_spot_prices[T]
        return val
    else: raise ValueError("asset type invalid, must be ETH or BTC")

def _exercised(sym):
    tokens = sym.split('-')
    asset = tokens[0]
    T = pd.Timestamp(datetime.strptime(tokens[1],'%d%b%y'))
    if T < datetime.now():return 0
    print("type going into exercised",type(T))
    K = int(tokens[2])
    Y = int(tokens[3] == 'C')
    S = lambda asset,T : btc_spot_prices[T] if asset == 'BTC' else eth_spot_prices[T]
    if Y == 1:
        return int(max(S(asset,T)-K,0) > 0)
    elif Y == 0:
        return int(max(K-S(asset,T),0) > 0)
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
    return datetime.strptime(tokens[1],'%d%b%y')
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
    spot = lambda asset : btc_spot_prices[t] if asset == 'BTC' else eth_spot_prices[t]
    _spot = np.vectorize(spot)
    S = _spot(A)
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
    # need to return vector of spot prices
    S = _S(syms,t)
    TVR = np.matmul(S.transpose(),R).trace()
    TVRp = np.matmul(S.transpose(),Rprime).trace()
    return TVRp/TVR

#
# need to reset data if beginning of new day
# 
#
nrows = 843
directory = "../options_csv/"
day = '2021-03-11'
filename = "deribit_quotes_"+day+"_OPTIONS.csv"
data = pd.read_csv(directory+ filename,nrows=nrows)
data = data.dropna()
data["datetime"] = pd.to_datetime(data['local_timestamp']*1e3)
data = data.set_index("datetime")
#
#
#

_tg = np.array([key for key in btc_spot_prices.keys()],dtype=object)
for ts in _tg:
    print(ts,btc_spot_prices[ts])

for i,_ts in enumerate(_tg[:80]):

    #
    ts_start = _ts
    ts_end = _tg[i+1]
    # convert to string
    str_start = ts_start.strftime('%H:%M')
    str_end = ts_end.strftime('%H:%M')
    print("type going into main loop",type(ts_start))
    print("median btc spot price at time ",ts_start," is ",btc_spot_prices[ts_start])
    print("median eth spot price at time ",ts_start," is ",eth_spot_prices[ts_start])
    #
    #
    #
    deribit_sample = data.between_time(str_start,str_end)
    option_sample = deribit_sample.groupby(['symbol']).median()
    
    syms = option_sample.index.values
    p = option_sample.ask_price.values

    # remove rows with future expiration (missing data)
    print("before")
    print(syms.shape)
    print(p.shape)
    T = _T(syms)
    mask = np.array([t < datetime.now() for t in T],dtype=bool)
    p = p[mask,...]
    syms= syms[mask,...]
    print("after")
    print(syms.shape)
    print(p.shape)

    # if empty skip, else compute return
    if syms.shape[0] == 0:
        continue
    else:
        ir = instantaneous_return(syms,p,ts_start)
        print(ts_start,ir)
    #
    # if it passes test, then save as pickle object
