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

# outer loop
directory = "../options_csv/"
_timegrid = np.array([key for key in btc_spot_prices.keys()],dtype=object)
# outer loop over days ##########
for i in range(1):
    #
    # need to reset data if beginning of new day
    # 
    #
    #nrows = 8437280
    nrows = 10
    day = '2021-03-11'
    filename = "deribit_quotes_"+day+"_OPTIONS.csv"
    data = pd.read_csv(directory+ filename,nrows=nrows)
    data = data.dropna()
    data["datetime"] = pd.to_datetime(data['local_timestamp']*1e3)
    data = data.set_index("datetime")
    ts_day0 = pd.Timestamp(day)
    ts_day0 = ts_day0.replace(hour=0,minute=0,second=0)
    ts_day1 = ts_day0.replace(day=ts_day0.day+1)
    _timegrid_mask = np.array([t >= ts_day0 and t < ts_day1 for t in _timegrid],dtype=bool)
    sys.exit()
    # inner loop #####################################
    for j,_ts in enumerate(_timegrid[_timegrid_mask,...]):
        if _ts.hour == 18:
            print("finished sampling day")
            break
        #
        ts_start = _ts
        ts_end = _timegrid[j+1]
        # convert to string
        str_start = ts_start.strftime('%H:%M')
        str_end = ts_end.strftime('%H:%M')
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
