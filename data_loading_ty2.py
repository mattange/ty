# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 16:39:05 2019

@author: mangelon
"""

import numpy as np
import pandas as pd
import quantfin_options as qfo
import quantfin_bonds as qfb

import calendar

filename = 'FutOpt.xlsx'
SHEET = 'ty2'
start_date = '2003-10-01'
end_date = '2019-10-01'
DATES_TO_KILL = pd.DatetimeIndex(['2008-03-06','2008-03-07', '2008-03-10', 
                                '2008-03-11','2008-03-12', 
                                '2008-05-27', '2008-05-28', '2008-05-29', '2008-05-30',
                                '2008-08-06', '2008-08-12', '2008-08-13', '2008-08-15', 
                                '2008-08-22', '2008-11-24', '2008-11-25', 
                                '2008-11-26', '2008-11-27', '2009-01-28', 
                                '2009-01-29','2009-02-03',
                                '2009-02-23', '2009-02-24', '2009-02-25', '2009-02-26',
                                '2009-02-27', '2009-03-02', '2009-03-03', '2009-03-04',
                                '2009-03-05'])

DATES_CALLS_TO_PUTS = pd.DatetimeIndex(['2003-10-22', '2003-10-23', '2003-10-24', 
                                        '2003-10-27', '2003-10-28', '2003-10-29', 
                                        '2008-03-13', '2008-03-14', '2008-03-17', 
                                        '2008-03-18', '2008-03-19', '2008-03-20',
                                        '2008-03-21', '2008-03-24', '2008-03-25',
                                        '2008-08-25', '2008-08-26', '2008-08-27',
                                        '2008-08-28', '2008-08-29', '2008-09-01',
                                        '2008-09-02', '2008-09-03', '2008-09-04',
                                        '2008-09-05', '2008-09-08', '2008-09-09',
                                        '2008-09-10', '2008-09-11', '2008-11-28',
                                        '2008-12-01', '2008-12-02', '2008-12-03',
                                        '2008-12-04', '2008-12-05', '2008-12-08',
                                        '2008-12-09', '2008-12-10', '2008-12-11',
                                        '2008-12-12', '2008-12-15', '2008-12-16',
                                        '2009-03-06', '2009-03-09', '2009-03-10',
                                        '2009-03-11', '2009-03-12', '2009-03-13', 
                                        '2009-03-16', '2009-03-17', '2009-03-18'])

k_minmax = 12.0
k_step = 0.5
FACTOR = 64

sheet_name = [SHEET, 'under_fut']
header = 0
index_col = 0
skiprows = range(1, 13)
data = pd.read_excel(filename, sheet_name=sheet_name, index_col=index_col, 
                     header=header, skiprows=skiprows)

ty = data[SHEET].loc[start_date:end_date]
ty.index = pd.to_datetime(ty.index)
under_fut = data['under_fut']
#fill in the calendar of the expiry dates in Python
c = calendar.Calendar(firstweekday=calendar.SATURDAY)
s_exp_dt = pd.Series(index=under_fut.index, name='OPT_EXP_DT')
for row in under_fut.itertuples():
    contract = row.Index
    del_dt = row.FUT_DLV_DT_FIRST
    monthcal = c.monthdatescalendar(del_dt.year, del_dt.month-1)
    s_exp_dt.loc[contract] = monthcal[3][-1]  #fourth friday of the month
under_fut[s_exp_dt.name] = pd.to_datetime(s_exp_dt)

ty = ty.merge(under_fut, right_on='OPT_EXP_DT',left_on='ticker',right_index=True)

#assign the time to expiry
ty['opt_tau_act365'] = (ty['OPT_EXP_DT'] - ty.index) / pd.Timedelta('365 days')
ty.loc[ty['OPT_EXP_DT'] < ty.index, 'opt_tau_act365'] = 0

# do some data correction: remove some days where the vols are negative and unusable
ty.drop(labels=DATES_TO_KILL, errors='ignore', inplace=True)

put_ivols = ['put_10d', 'put_25d', 'put_40d', 'put_50d', 'put_60d', 'put_75d', 'put_90d', 'hist_put_ivol']
call_ivols = ['call_90d', 'call_75d', 'call_60d', 'call_50d', 'call_40d', 'call_25d', 'call_10d', 'hist_call_ivol']
dt = ty.index.intersection(DATES_CALLS_TO_PUTS)
ty.loc[dt,put_ivols] = ty.loc[dt,call_ivols].values


# summarise the volatilities:
# note that the average does not work as the in the money option implied
# vols are some times wrong....
ty['atm_ivol'] = (ty['put_50d'] + ty['call_50d']) / 2 / 100
ty['10dp_ivol'] = ty['put_10d'] / 100
ty['25dp_ivol'] = ty['put_25d'] / 100
ty['40dp_ivol'] = ty['put_40d'] / 100
ty['40dc_ivol'] = ty['call_40d'] / 100
ty['25dc_ivol'] = ty['call_25d'] / 100
ty['10dc_ivol'] = ty['call_10d'] / 100


#only act on info where expiry is greater than 2 days
msk = ty['opt_tau_act365'] > 2/365
skew_vols = ['10dp_ivol', '25dp_ivol', '40dp_ivol', '40dc_ivol', '25dc_ivol', '10dc_ivol']
k_cols = ['last','opt_tau_act365', 'atm_ivol'] + skew_vols
k_data = ty.loc[msk, k_cols]
all_data = ty.loc[msk,:]


# find strikes corresponding to delta % options
px = k_data['last'].values
tau = k_data['opt_tau_act365'].values
atm_vol = k_data['atm_ivol'].values
k_res_cols = ['atm_k','10dp_k','25dp_k','40dp_k', '40dc_k', '25dc_k', '10dc_k']
k_deltas = [-0.1, -0.25, -0.4, 0.4, 0.25, 0.1]
k_res = pd.DataFrame(index=k_data.index, columns=k_res_cols, dtype=np.float)
k_res['atm_k'] = k_data['last']
for i in range(len(k_res_cols[1:])):
    if k_res_cols[i+1][3] == 'p':
        opt_type = np.array(['p' for i in range(px.size)])
    else:
        opt_type = np.array(['c' for i in range(px.size)])
    k_res[k_res_cols[i+1]] = qfo.black_scholes.impl_strike(
            k_deltas[i]*np.ones(shape=px.shape),
            px, tau, k_data[skew_vols[i]].values, 
            opt_type, px, value_type='d', model='l')

# now that i have all strikes corresponding to the delta%,
# calculate also the yields looping through dates
y_res_cols = ['atm_y','10dp_y', '25dp_y', '40dp_y', 
              '40dc_y', '25dc_y', '10dc_y']
y_res = pd.DataFrame(index=k_data.index, columns=y_res_cols, dtype=np.float)
for row in y_res.itertuples():
    rowIdx = row.Index
    ppp = k_res.loc[rowIdx].values*all_data['FUT_CNVS_FACTOR'].loc[rowIdx]
    y_res.loc[rowIdx][y_res_cols] = qfb.bonds.ytm(ppp, 100.0, 
                                          all_data['CPN'].loc[rowIdx]/100.0, 
                                          all_data['CPN_FREQ'].loc[rowIdx],
                                          np.datetime64(all_data['FUT_CTD_MTY'].loc[rowIdx]),
                                          np.datetime64(all_data['FUT_DLV_DT_LAST'].loc[rowIdx]),
                                          dcc="30/360 US", 
                                          eom=False)*100

p_res_cols = ['atms_P', '10dp_P','25dp_P','40dp_P', '40dc_P', '25dc_P', '10dc_P']
p_res = pd.DataFrame(index=k_data.index, columns=p_res_cols, dtype=np.float)
for col in p_res.iteritems():
    colName = col[0]
    colIdx = p_res_cols.index(colName)
    opt_type = p_res_cols[colIdx][3]
    p_res[colName] = qfo.black_scholes.fwd_value(px, k_res[k_res_cols[colIdx]].values, \
         tau, k_data[k_cols[colIdx+2]], opt_type=opt_type)
    
dv01_res_cols = ['atm_dv01','10dp_dv01','25dp_dv01','40dp_dv01',
                 '40dc_dv01', '25dc_dv01', '10dc_dv01']
dv01_res = pd.DataFrame(index=k_data.index, columns=dv01_res_cols, dtype=np.float)
for row in dv01_res.itertuples():
    rowIdx = row.Index
    dv01_res.loc[rowIdx][dv01_res_cols] = qfb.bonds.dv01(y_res.loc[rowIdx].values / 100.0, 
                                          100.0 / all_data['FUT_CNVS_FACTOR'].loc[rowIdx], 
                                          all_data['CPN'].loc[rowIdx]/100.0, 
                                          all_data['CPN_FREQ'].loc[rowIdx],
                                          np.datetime64(all_data['FUT_CTD_MTY'].loc[rowIdx]),
                                          np.datetime64(all_data['FUT_DLV_DT_LAST'].loc[rowIdx]),
                                          dcc="30/360 US", eom=False, x_is_yield=True, step=0.5)


yield_p_res_cols = ['atms_syP','10dp_cyP','25dp_cyP','40dp_cyP', 
                    '40dc_pyP', '25dc_pyP', '10dc_pyP']
yield_p_res = pd.DataFrame(-p_res.values / dv01_res[dv01_res_cols].values / 100,
                           index=k_data.index, columns=yield_p_res_cols, dtype=np.float)


normvol_res_cols = ['atms_sy_iNvol','10dp_cy_iNvol','25dp_cy_iNvol', '40dp_cy_iNvol', 
                    '40dc_py_iNvol', '25dc_py_iNvol', '10dc_py_iNvol']
normvol_res = pd.DataFrame(index=k_data.index, columns=normvol_res_cols, dtype=np.float)
for col in normvol_res.iteritems():
    colName = col[0]
    colIdx = normvol_res_cols.index(colName)
    opt_type = np.array([normvol_res_cols[colIdx][5] for i in range(normvol_res.shape[0])])
    sigma_guess = np.ones_like(opt_type, dtype=np.float)*0.75
    normvol_res[colName] = qfo.black_scholes.impl_vol(yield_p_res[yield_p_res_cols[colIdx]].values,
               y_res[y_res_cols[0]].values, y_res[y_res_cols[colIdx]].values, tau,
               opt_type, sigma_guess, model='n')


vols_data = pd.concat([k_res, y_res, dv01_res, p_res, yield_p_res, normvol_res], axis=1)


# SABR PARAMS
# SABR PARAMS PRICE
# find SABR parameters for given combinations of strikes fwd and vols
sabr_res_cols = ['alpha_price','beta_price','rho_price','nu_price']
sabr_res = pd.DataFrame(index=k_data.index, columns=sabr_res_cols, dtype=np.float)
sabr_res['beta_price'] = 1.0 #assume lognormal behaviour for the price
skew_weights = np.ones(shape=(len(skew_vols),),dtype=np.float)
#now estimate all parameters based on the beta
#a single estimation appears not to be working unless the regression is on very long periods
# so try with the loop
for i in range(sabr_res.shape[0]):
    try:       
        sabr_params = qfo.sabr.calibrate(px[i], tau[i], atm_vol[i], 
                                         k_res[k_res_cols[1:]].iloc[i].values, 
                                         k_data[skew_vols].iloc[i].values, 
                                         beta=sabr_res['beta_price'].iloc[i],
                                         skew_weights=skew_weights,
                                         skew_k_relative=False,
                                         skew_sigma_relative=False,
                                         fit_as_sequence=True,model='l')
        sabr_res['alpha_price'].iloc[i] = sabr_params[0]
        sabr_res['rho_price'].iloc[i] = sabr_params[2]
        sabr_res['nu_price'].iloc[i] = sabr_params[3]
    except RuntimeError as e:
        continue
#pad dates where we do not have observations for lack of convergence with prior values
sabr_res.fillna(method='ffill', inplace=True)

# SABR PARAMS YIELDS
sabr_res_y_cols = ['alpha_yield','beta_yield','rho_yield','nu_yield']
sabr_res_y = pd.DataFrame(index=k_data.index, columns=sabr_res_y_cols, dtype=np.float)
sabr_res_y['beta_yield'] = 0.0 #assume fully normal behaviour for the yield

# find SABR parameters for given combinations of strikes fwd and vols
for i in range(sabr_res_y.shape[0]):
    try:
        sabr_params = qfo.sabr.calibrate(y_res['atm_y'].iloc[i], tau[i], 
                                         normvol_res['atms_sy_iNvol'].iloc[i], 
                                         y_res[y_res_cols[1:]].iloc[i].values,
                                         normvol_res[normvol_res_cols[1:]].iloc[i].values,
                                         beta=sabr_res_y['beta_yield'].iloc[i],
                                         skew_weights=skew_weights,
                                         skew_k_relative=False,
                                         skew_sigma_relative=False,
                                         fit_as_sequence=True,model='n')
        sabr_res_y['alpha_yield'].iloc[i] = sabr_params[0]
        sabr_res_y['rho_yield'].iloc[i] = sabr_params[2]
        sabr_res_y['nu_yield'].iloc[i] = sabr_params[3]
    except RuntimeError as e:
        continue
#pad dates where we do not have observations with prior values
sabr_res_y.fillna(method='ffill', inplace=True)


# now create the prices based on a central "nearest ATM" price,
# the prices of all calls and puts 
k_minmax = 12.0
k_step = 0.5
strk = np.arange(-k_minmax,k_minmax + k_step, k_step)
cols_CALLS = ['C {:+.2f}'.format(n, 1) for n in strk]
cols_PUTS = ['P {:+.2f}'.format(n, 1) for n in strk]
cols = ['base_k'] + cols_PUTS + cols_CALLS
price_detail = pd.DataFrame(index=k_data.index, columns=cols, dtype=np.float)
price_detail['base_k'] = np.round(px*2)/2


for rowIdx in range(price_detail.shape[0]):
    KK = price_detail['base_k'].iloc[rowIdx] + strk
    sabr_vol = qfo.sabr.volatility(px[rowIdx], KK, tau[rowIdx], 
                              sabr_res['alpha_price'].iloc[rowIdx],
                              sabr_res['beta_price'].iloc[rowIdx],
                              sabr_res['rho_price'].iloc[rowIdx],
                              sabr_res['nu_price'].iloc[rowIdx],
                              model='l')
    p_CALL = qfo.black_scholes.fwd_value(px[rowIdx], KK, tau[rowIdx],
                                      sabr_vol, opt_type='c')
    p_PUT = qfo.black_scholes.fwd_value(px[rowIdx], KK, tau[rowIdx],
                                      sabr_vol, opt_type='p')
    p = np.concatenate((p_PUT, p_CALL))
    p_round = np.round(p * FACTOR) / FACTOR
    price_detail.iloc[rowIdx][cols[1:]] = p_round
    
    

# now create the output in full with all information, drop the double time column
output = pd.concat([ty, vols_data, sabr_res, sabr_res_y, price_detail],axis=1)
writer = pd.ExcelWriter(SHEET + '_analysis.xlsx', 
                        date_format = 'dd/mmm/yyyy',
                        datetime_format = 'dd/mmm/yyyy')

output.to_excel(writer, sheet_name=SHEET, na_rep='na')
writer.close()

