import logging
from pathlib import Path
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
data_eth = yf.download(tickers='ETH-USD',start=pd.Timestamp(_begin),end=pd.Timestamp(_end),interval='1h',progress=False)
data_btc = yf.download(tickers='BTC-USD',start=pd.Timestamp(_begin),end=pd.Timestamp(_end),interval='1h',progress=False)
_btc = pd.Series(data=data_btc["Close"])
_btc_hi = pd.Series(data=data_btc["High"])
_btc_low = pd.Series(data=data_btc["Low"])

_eth = pd.Series(data=data_eth["Close"])
_eth_hi = pd.Series(data=data_eth["High"])
_eth_low = pd.Series(data=data_eth["Low"])

#
_btc_resampled = _btc.resample('6H').mean()
_eth_resampled = _eth.resample('6H').mean()

_btc_hi = _btc_hi.resample('1D').max()
_eth_hi = _eth_hi.resample('1D').max()
_btc_low = _btc_low.resample('1D').min()
_eth_low = _eth_low.resample('1D').min()

# get dict
_bsp0 = _btc_resampled.to_dict()
_esp0 = _eth_resampled.to_dict()

_bh0 = _btc_hi.to_dict()
_bl0 = _btc_low.to_dict()
_eh0 = _eth_hi.to_dict()
_el0 = _eth_low.to_dict()

#
#
# clean timestamp keys
#
btc_spot_prices = {}
eth_spot_prices = {}
btc_lh = {}
eth_lh = {}
for _key in _bh0.keys():
    val_bh = _bh0[_key]
    val_bl = _bl0[_key]
    val_eh = _eh0[_key]
    val_el = _el0[_key]
    key = pd.Timestamp(year=_key.year,month=_key.month,day=_key.day,hour=_key.hour,minute=0,second=0,microsecond=0)
    btc_lh[key] = (val_bl,val_bh)
    eth_lh[key] = (val_el,val_eh)

for _key in _bsp0.keys():
    val_btc = _bsp0[_key]
    val_eth = _esp0[_key]
    
    key = pd.Timestamp(year=_key.year,month=_key.month,day=_key.day,hour=_key.hour,minute=0,second=0,microsecond=0)
    day = key.replace(hour=0)

    val_bhl = btc_lh[day]
    val_ehl = eth_lh[day]
    
    btc_spot_prices[key] = val_btc
    eth_spot_prices[key] = val_eth

    btc_lh[key] = val_bhl
    eth_lh[key] = val_ehl

#
# define methods
#
def _asset(sym):
    tokens = sym.split("-")
    asset = tokens[0]
    return asset

def _exercised(sym):
    tokens = sym.split('-')
    asset = tokens[0]
    T = pd.Timestamp(datetime.strptime(tokens[1],'%d%b%y'))
    K = int(tokens[2])
    Y = int(tokens[3] == 'C')
    S_max = lambda asset,T : btc_lh[T][1] if asset == 'BTC' else eth_lh[T][1]
    S_min = lambda asset,T : btc_lh[T][0] if asset == 'BTC' else eth_lh[T][0]
    if Y == 1:
        return int(max(S_max(asset,T)-K,0) > 0)
    elif Y == 0:
        return int(max(K-S_min(asset,T),0) > 0)
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
def btc2usd_S(p,t):
    spot = lambda premium : btc_spot_prices[t]*premium
    _spot = np.vectorize(spot)
    return _spot(p)
    

def _Smatrix(syms,t):
    A = _A(syms)
    spot = lambda asset : btc_spot_prices[t] if asset == 'BTC' else eth_spot_prices[t]
    _spot = np.vectorize(spot)
    S = _spot(A)
    #left column S, right column 1s
    return np.column_stack((S,np.ones(len(A))))

def _rmin_loss(syms):
    Y = _Y(syms)
    K = _K(syms)
    chi = _chi(syms)
    Rmin = np.column_stack((Y,np.multiply(1-Y,K)))
    chi2 = np.outer(chi,np.ones(2))
    rhs = np.column_stack((np.ones(len(syms)),K)) - 2*Rmin
    return Rmin, np.multiply(chi2,rhs)

def usd_gain(p,t):
    # convert btc to usd and then reshape matrix.
    _p = btc2usd_S(p,t)
    assert _p.shape == p.shape
    return np.column_stack((np.zeros(len(_p)),_p))

def _Rprime(p,syms,t):
    G = usd_gain(p,t)
    Rmin,L = _rmin_loss(syms)
    Rprime = Rmin + L + G
    return Rprime

def growth_ratio(syms,p,t):
    R,L = _rmin_loss(syms)
    Rprime = _Rprime(p,syms,t)
    # need to return matrix of spot prices
    S = _Smatrix(syms,t)
    TVR = np.matmul(S.transpose(),R).trace()
    TVRp = np.matmul(S.transpose(),Rprime).trace()
    return TVRp/TVR

# outer loop
directory = "../options_csv/"
_timegrid = np.array([key for key in btc_spot_prices.keys()],dtype=object)
# outer loop over days ##########
data_begin = _begin
data_end = '2021-05-29'
trading_days = pd.date_range(start=pd.Timestamp(data_begin),end=pd.Timestamp(data_end),freq='D')
for _day in trading_days:
    #
    # need to reset data if beginning of new day
    # 
    #
    day = str(_day.year) + '-' + str(_day.month).zfill(2) + '-' + str(_day.day).zfill(2)
    nrows = 8437280
    
    filename = "deribit_quotes_"+day+"_OPTIONS.csv"
    path = directory + filename
    datafile = Path(path)
    if not datafile.is_file():continue
    
    data = pd.read_csv(path,nrows=nrows)
    data = data.dropna()
    data["datetime"] = pd.to_datetime(data['local_timestamp']*1e3)
    data = data.set_index("datetime")
    ts_day0 = pd.Timestamp(day)
    ts_day0 = ts_day0.replace(hour=0,minute=0,second=0)
    ts_day1 = ts_day0.replace(day=ts_day0.day+1)
    _timegrid_mask = np.array([t >= ts_day0 and t < ts_day1 for t in _timegrid],dtype=bool)
    # inner loop #####################################
    for j,_ts in enumerate(_timegrid[_timegrid_mask,...]):
        if _ts.hour == 18:
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
        # only invest in short term options (1-month out)
        T = _T(syms)
        month_mask = np.array([t.month <= day.month+1 for t in T],dtype=bool)
        p_month = p[month_mask,...]
        syms_month= syms[month_mask,...]
        market_share_month = syms_month.shape[0]/syms.shape[0]
        # invest in as many as data will allow
        #max_mask = np.array([t <= datetime.now() for t in T],dtype=bool)
        #p_max = p[max_mask,...]
        #syms_max = syms[max_mask,...]
        #market_share_max = syms_max.shape[0]/syms.shape[0]
        T_month = _T(syms_month)
        max_T_month = max(T_month)
        

        
        # if empty skip, else compute return
        


        
        if syms.shape[0] == 0:
            continue
        else:
            ratio = instantaneous_return(syms_month,p_month,ts_start)
            monthly_return = 1-ratio
            apr_month = ((monthly_return/30)*365)*100
            # time, instant-return, exercise-rate,net-gain,discard-rate
            print(ts_start,",",apr_month,",",ave_T_month,",",min_T_month,",",max_T_month,",",market_share_month)
            #
            # if it passes test, then save as pickle object
