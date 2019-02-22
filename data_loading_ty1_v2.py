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
SHEET = 'ty1'
start_date = '2003-10-01'
end_date = '2019-10-01'
DATES_TO_KILL = pd.DatetimeIndex(['2008-02-22','2008-05-23', '2008-05-26', 
                                '2008-08-22', '2008-11-21','2009-02-20',
                                '2009-05-22', '2009-05-25', '2009-08-21'])

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
ivols_res_cols = ['atm_ivol'] + skew_vols
k_cols = ['last','opt_tau_act365', 'atm_ivol'] + skew_vols
k_data = ty.loc[msk, k_cols]
all_data = ty.loc[msk,:]


#################################################################################
# Strike calculations
#################################################################################
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

#################################################################################
# Yield strike calculations
#################################################################################
# now that i have all strikes corresponding to the delta%,
# calculate also the yields looping through dates
y_res_cols = ['atm_y','10dp_y', '25dp_y', '40dp_y', '40dc_y', '25dc_y', '10dc_y']
y_res = pd.DataFrame(index=k_data.index, columns=y_res_cols, dtype=np.float)
for rowIdx in k_data.index:
    ppp = k_res.loc[rowIdx].values*all_data['FUT_CNVS_FACTOR'].loc[rowIdx]
    y_res.loc[rowIdx][y_res_cols] = qfb.bonds.ytm(ppp, 100.0, 
                                          all_data['CPN'].loc[rowIdx]/100.0, 
                                          all_data['CPN_FREQ'].loc[rowIdx],
                                          np.datetime64(all_data['FUT_CTD_MTY'].loc[rowIdx]),
                                          np.datetime64(all_data['FUT_DLV_DT_LAST'].loc[rowIdx]),
                                          dcc="30/360 US", 
                                          eom=False)*100


#################################################################################
# Price calculations
#################################################################################
p_res_cols = ['atms_P', '10dp_P','25dp_P','40dp_P', '40dc_P', '25dc_P', '10dc_P']
p_res = pd.DataFrame(index=k_data.index, columns=p_res_cols, dtype=np.float)
for colName in p_res_cols:
    colIdx = p_res_cols.index(colName)
    opt_type = p_res_cols[colIdx][3]
    p_res[colName] = qfo.black_scholes.fwd_value(px, k_res[k_res_cols[colIdx]].values, \
         tau, k_data[k_cols[colIdx+2]], opt_type=opt_type)
    
#################################################################################
# Local DV01 calculations
#################################################################################
dv01_res_cols = ['atm_dv01','10dp_dv01','25dp_dv01','40dp_dv01',
                 '40dc_dv01', '25dc_dv01', '10dc_dv01']
dv01_res = pd.DataFrame(index=k_data.index, columns=dv01_res_cols, dtype=np.float)
for rowIdx in k_data.index:
    dv01_res.loc[rowIdx][dv01_res_cols] = qfb.bonds.dv01(y_res.loc[rowIdx].values / 100.0, 
                                          100.0 / all_data['FUT_CNVS_FACTOR'].loc[rowIdx], 
                                          all_data['CPN'].loc[rowIdx]/100.0, 
                                          all_data['CPN_FREQ'].loc[rowIdx],
                                          np.datetime64(all_data['FUT_CTD_MTY'].loc[rowIdx]),
                                          np.datetime64(all_data['FUT_DLV_DT_LAST'].loc[rowIdx]),
                                          dcc="30/360 US", eom=False, x_is_yield=True, step=0.5)



#################################################################################
# Yield price calculations
#################################################################################
yield_p_res_cols = ['atms_syP','10dp_cyP','25dp_cyP','40dp_cyP', 
                    '40dc_pyP', '25dc_pyP', '10dc_pyP']
# I scale the price of TY options based on the ATM yield levels, not the local 
# yield levels. 
# Scaling prices with the local dv01 levels at the strike
# yield_p_res = pd.DataFrame(-p_res.values / dv01_res[dv01_res_cols].values / 100,
#                            index=k_data.index, columns=yield_p_res_cols, dtype=np.float)
# Scaling prices with the atm fwd dv01 levels of the day, which makes all prices consistent
# and it is better for comparison
yield_p_res = pd.DataFrame(-p_res.values / 100 / 
                           np.repeat(dv01_res['atm_dv01'].values[:,np.newaxis], 
                                     len(yield_p_res_cols), axis=1),
                           index=k_data.index, columns=yield_p_res_cols, dtype=np.float)

#################################################################################
# Normal vols calculations based on yield prices
#################################################################################
normvol_res_cols = ['atms_sy_iNvol','10dp_cy_iNvol','25dp_cy_iNvol', '40dp_cy_iNvol', 
                    '40dc_py_iNvol', '25dc_py_iNvol', '10dc_py_iNvol']
normvol_res = pd.DataFrame(index=k_data.index, columns=normvol_res_cols, dtype=np.float)
for colName in normvol_res_cols:
    colIdx = normvol_res_cols.index(colName)
    opt_type = np.array([normvol_res_cols[colIdx][5] for i in range(normvol_res.shape[0])])
    sigma_guess = np.ones_like(opt_type, dtype=np.float)*0.75
    normvol_res[colName] = qfo.black_scholes.impl_vol(yield_p_res[yield_p_res_cols[colIdx]].values,
               y_res[y_res_cols[0]].values, y_res[y_res_cols[colIdx]].values, tau,
               opt_type, sigma_guess, model='n')


# vols_data = pd.concat([k_res, y_res, dv01_res, p_res, yield_p_res, normvol_res], axis=1)


#################################################################################
# SABR PARAMS calculations in price terms and in yield terms
#################################################################################

#################################################################################
# SABR PARAMS PRICE
# find SABR parameters for given combinations of strikes fwd and vols
#################################################################################
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

#################################################################################
# SABR PARAMS YIELDS
#################################################################################
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


#################################################################################
# Extended price calculation grid
#################################################################################

strk = np.arange(-k_minmax, k_minmax + k_step, k_step)
tp_lbl = ['CALL','PUT']
tp_lbl_name = 'opt_type'
strk_lbl = ['{:+.2f}'.format(n, 1) for n in strk]
strk_lbl_name = 'opt_rel_strike'
feat_lbl = ['period', 'expiry', 'tau', 
            'atm_k', 'atm_y', 'atm_dv01',
            'k', 'k_y', 'k_dv01',
            'p', 'yp',
            'yp_dv01', 'yp_dv01pc', 'yp_theta', 
            'yp_Nvega_0.01', 'yp_rho_0.1', 'yp_nu_0.1']
feat_lbl_name = 'features'
m_index = pd.MultiIndex.from_product([tp_lbl, strk_lbl, feat_lbl], 
                                     names=[tp_lbl_name, strk_lbl_name, feat_lbl_name])
idx = pd.IndexSlice

price_detail = pd.DataFrame(index=k_data.index,  columns=m_index)



for opt_type in tp_lbl:
    for strike in strk_lbl:
        price_detail.loc[:, idx[opt_type, strike, 'period']] = ty.loc[k_data.index, 'period']
        price_detail.loc[:, idx[opt_type, strike, 'expiry']] = ty.loc[k_data.index, 'OPT_EXP_DT']
        price_detail.loc[:, idx[opt_type, strike, 'tau']] = tau
        price_detail.loc[:, idx[opt_type, strike, 'atm_k']] = k_res.loc[:,'atm_k']
        price_detail.loc[:, idx[opt_type, strike, 'atm_y']] = y_res.loc[:,'atm_y']
        price_detail.loc[:, idx[opt_type, strike, 'atm_dv01']] = dv01_res.loc[:,'atm_dv01']

base_k = np.round(px*2)/2
base_y = y_res.loc[:,'atm_y'].values
atm_normvol = normvol_res['atms_sy_iNvol'].values
atm_normvol_up = atm_normvol + 0.005
atm_normvol_dn = atm_normvol - 0.005

alpha_normvol_up = qfo.sabr.alpha(base_y, tau, atm_normvol_up,
                                  sabr_res_y['beta_yield'].values, 
                                  sabr_res_y['rho_yield'].values,
                                  sabr_res_y['nu_yield'].values,
                                  alpha_guess=sabr_res_y['alpha_yield'].values,
                                  model='n')
alpha_normvol_dn = qfo.sabr.alpha(base_y, tau, atm_normvol_dn,
                                  sabr_res_y['beta_yield'].values, 
                                  sabr_res_y['rho_yield'].values,
                                  sabr_res_y['nu_yield'].values,
                                  alpha_guess=sabr_res_y['alpha_yield'].values,
                                  model='n')


for row in k_data.index:
    
    print(row)
    
    rowIdx = k_data.index.get_loc(row)
    CPN = all_data['CPN'].loc[row]/100.0
    CPN_FREQ = all_data['CPN_FREQ'].loc[row]
    FUT_CTD_MTY = np.datetime64(all_data['FUT_CTD_MTY'].loc[row])
    FUT_DLV_DT_LAST = np.datetime64(all_data['FUT_DLV_DT_LAST'].loc[row])
    FUT_CNVS_FACTOR = all_data['FUT_CNVS_FACTOR'].loc[row]
    
    # calculate strike and assign it
    k = base_k[rowIdx] + strk
    price_detail.loc[row, idx[:, :,  'k']] = np.concatenate((k,k))
    
    # calculatae yield at strike and assign it
    yyy = qfb.bonds.ytm(k * all_data['FUT_CNVS_FACTOR'].loc[row],
                        100.0, CPN, CPN_FREQ, FUT_CTD_MTY, FUT_DLV_DT_LAST,
                        dcc="30/360 US", eom=False)*100
    price_detail.loc[row, idx[:, :, 'k_y']] = np.concatenate((yyy,yyy))

    # calculate the strike DV01 and assign it
    ddvv = qfb.bonds.dv01(yyy/100.0, 100.0/FUT_CNVS_FACTOR, 
                          CPN, CPN_FREQ, FUT_CTD_MTY, FUT_DLV_DT_LAST,
                          dcc="30/360 US", eom=False, x_is_yield=True, step=0.5)
    price_detail.loc[row, idx[:, :, 'k_dv01']]  = np.concatenate((ddvv,ddvv))
    
    # calculate price
    sabr_vol = qfo.sabr.volatility(px[rowIdx], k, tau[rowIdx], 
                                   sabr_res['alpha_price'].iloc[rowIdx],
                                   sabr_res['beta_price'].iloc[rowIdx],
                                   sabr_res['rho_price'].iloc[rowIdx],
                                   sabr_res['nu_price'].iloc[rowIdx],
                                   model='l')
    p_CALL = qfo.black_scholes.fwd_value(px[rowIdx], k, tau[rowIdx], sabr_vol, opt_type='c')
    p_PUT = qfo.black_scholes.fwd_value(px[rowIdx], k, tau[rowIdx], sabr_vol, opt_type='p')
    p = np.concatenate((p_CALL, p_PUT))
    price_detail.loc[row, idx[:, :, 'p']] = np.round(p * FACTOR) / FACTOR

    # calculate yield price
    px_y = y_res.loc[row,'atm_y']
    sabr_vol = qfo.sabr.volatility(px_y, yyy, tau[rowIdx], 
                                   sabr_res_y['alpha_yield'].iloc[rowIdx],
                                   sabr_res_y['beta_yield'].iloc[rowIdx],
                                   sabr_res_y['rho_yield'].iloc[rowIdx],
                                   sabr_res_y['nu_yield'].iloc[rowIdx],
                                   model='n')
    p_PUT_y = qfo.black_scholes.fwd_value(px_y, yyy, tau[rowIdx], sabr_vol, opt_type='p')
    p_CALL_y = qfo.black_scholes.fwd_value(px_y, yyy, tau[rowIdx], sabr_vol, opt_type='c')
    price_detail.loc[row, idx[:, :, 'yp']] = np.concatenate((p_PUT_y, p_CALL_y))
    
    # calculate yield price sensitivities to fwd move
    sabr_vol_up = qfo.sabr.volatility(px_y + 0.005, yyy, tau[rowIdx], 
                                   sabr_res_y['alpha_yield'].iloc[rowIdx],
                                   sabr_res_y['beta_yield'].iloc[rowIdx],
                                   sabr_res_y['rho_yield'].iloc[rowIdx],
                                   sabr_res_y['nu_yield'].iloc[rowIdx],
                                   model='n')
    p_PUT_y_up = qfo.black_scholes.fwd_value(px_y + 0.005, yyy, tau[rowIdx], sabr_vol_up, opt_type='p')
    p_CALL_y_up = qfo.black_scholes.fwd_value(px_y + 0.005, yyy, tau[rowIdx], sabr_vol_up, opt_type='c')
    sabr_vol_dn = qfo.sabr.volatility(px_y - 0.005, yyy, tau[rowIdx], 
                                   sabr_res_y['alpha_yield'].iloc[rowIdx],
                                   sabr_res_y['beta_yield'].iloc[rowIdx],
                                   sabr_res_y['rho_yield'].iloc[rowIdx],
                                   sabr_res_y['nu_yield'].iloc[rowIdx],
                                   model='n')
    p_PUT_y_dn = qfo.black_scholes.fwd_value(px_y - 0.005, yyy, tau[rowIdx], sabr_vol_dn, opt_type='p')
    p_CALL_y_dn = qfo.black_scholes.fwd_value(px_y - 0.005, yyy, tau[rowIdx], sabr_vol_dn, opt_type='c')
    yp_dv01 = np.concatenate((p_PUT_y_up - p_PUT_y_dn, p_CALL_y_up - p_CALL_y_dn))
    price_detail.loc[row, idx[:, :, 'yp_dv01']] = yp_dv01
    yp_dv01pc = yp_dv01/0.01
    yp_dv01pc[np.abs(yp_dv01pc) > 1.0] = np.sign(yp_dv01pc[np.abs(yp_dv01pc) > 1.0])
    price_detail.loc[row, idx[:, :, 'yp_dv01pc']] = yp_dv01pc
    

    # calculate yield price sensitivities to time (1/365) which is opposite of time to expiry
    ttau = max(tau[rowIdx] - 1.0/365.0, 0.0)
    sabr_vol_tau = qfo.sabr.volatility(px_y, yyy, ttau, 
                                   sabr_res_y['alpha_yield'].iloc[rowIdx],
                                   sabr_res_y['beta_yield'].iloc[rowIdx],
                                   sabr_res_y['rho_yield'].iloc[rowIdx],
                                   sabr_res_y['nu_yield'].iloc[rowIdx],
                                   model='n')
    p_PUT_y_tau = qfo.black_scholes.fwd_value(px_y, yyy, ttau, sabr_vol_tau, opt_type='p')
    p_CALL_y_tau = qfo.black_scholes.fwd_value(px_y, yyy, ttau, sabr_vol_tau, opt_type='c')
    price_detail.loc[row, idx[:, :, 'yp_theta']] = \
                        np.concatenate((p_PUT_y_tau - p_PUT_y, p_CALL_y_tau - p_CALL_y))

    # calculate yield price sensitivities to normal vol
    sabr_vol_vol_up = qfo.sabr.volatility(px_y, yyy, tau[rowIdx], 
                                          alpha_normvol_up[rowIdx],
                                          sabr_res_y['beta_yield'].iloc[rowIdx],
                                          sabr_res_y['rho_yield'].iloc[rowIdx],
                                          sabr_res_y['nu_yield'].iloc[rowIdx],
                                          model='n')
    sabr_vol_vol_dn = qfo.sabr.volatility(px_y, yyy, tau[rowIdx], 
                                          alpha_normvol_dn[rowIdx],
                                          sabr_res_y['beta_yield'].iloc[rowIdx],
                                          sabr_res_y['rho_yield'].iloc[rowIdx],
                                          sabr_res_y['nu_yield'].iloc[rowIdx],
                                          model='n')
    p_PUT_y_vol_up = qfo.black_scholes.fwd_value(px_y, yyy, tau[rowIdx], 
                                                 sabr_vol_vol_up, opt_type='p')
    p_PUT_y_vol_dn = qfo.black_scholes.fwd_value(px_y, yyy, tau[rowIdx], 
                                                 sabr_vol_vol_dn, opt_type='p')
    p_CALL_y_vol_up = qfo.black_scholes.fwd_value(px_y, yyy, tau[rowIdx], 
                                                  sabr_vol_vol_up, opt_type='c')
    p_CALL_y_vol_dn = qfo.black_scholes.fwd_value(px_y, yyy, tau[rowIdx], 
                                                  sabr_vol_vol_dn, opt_type='c')
    price_detail.loc[row, idx[:, :, 'yp_Nvega_0.01']] = \
        np.concatenate((p_PUT_y_vol_up - p_PUT_y_vol_dn, p_CALL_y_vol_up - p_CALL_y_vol_dn))


    # calculate yield price sensitivities to rho
    rho_up = min(sabr_res_y['rho_yield'].iloc[rowIdx] + 0.05, 0.999)
    rho_dn = max(sabr_res_y['rho_yield'].iloc[rowIdx] - 0.05, -0.999)
    rho_delta = rho_up - rho_dn
    sabr_vol_rho_up = qfo.sabr.volatility(px_y, yyy, tau[rowIdx], 
                                          sabr_res_y['alpha_yield'].iloc[rowIdx],
                                          sabr_res_y['beta_yield'].iloc[rowIdx],
                                          rho_up,
                                          sabr_res_y['nu_yield'].iloc[rowIdx],
                                          model='n')
    sabr_vol_rho_dn = qfo.sabr.volatility(px_y, yyy, tau[rowIdx], 
                                          sabr_res_y['alpha_yield'].iloc[rowIdx],
                                          sabr_res_y['beta_yield'].iloc[rowIdx],
                                          rho_dn,
                                          sabr_res_y['nu_yield'].iloc[rowIdx],
                                          model='n')
    p_PUT_y_rho_up = qfo.black_scholes.fwd_value(px_y, yyy, tau[rowIdx], 
                                                 sabr_vol_rho_up, opt_type='p')
    p_PUT_y_rho_dn = qfo.black_scholes.fwd_value(px_y, yyy, tau[rowIdx], 
                                                 sabr_vol_rho_dn, opt_type='p')
    p_CALL_y_rho_up = qfo.black_scholes.fwd_value(px_y, yyy, tau[rowIdx], 
                                                  sabr_vol_rho_up, opt_type='c')
    p_CALL_y_rho_dn = qfo.black_scholes.fwd_value(px_y, yyy, tau[rowIdx], 
                                                  sabr_vol_rho_dn, opt_type='c')
    price_detail.loc[row, idx[:, :, 'yp_rho_0.1']] = \
        np.concatenate(((p_PUT_y_rho_up - p_PUT_y_rho_dn) / rho_delta, 
                        (p_CALL_y_rho_up - p_CALL_y_rho_dn) / rho_delta))
    
    # calculate yield price sensitivities to nu
    nu_up = sabr_res_y['nu_yield'].iloc[rowIdx] + 0.05
    nu_dn = max(sabr_res_y['nu_yield'].iloc[rowIdx] - 0.05, 0.0)
    nu_delta = nu_up - nu_dn
    sabr_vol_nu_up = qfo.sabr.volatility(px_y, yyy, tau[rowIdx], 
                                         sabr_res_y['alpha_yield'].iloc[rowIdx],
                                         sabr_res_y['beta_yield'].iloc[rowIdx],
                                         sabr_res_y['rho_yield'].iloc[rowIdx],
                                         nu_up,
                                         model='n')
    sabr_vol_nu_dn = qfo.sabr.volatility(px_y, yyy, tau[rowIdx], 
                                         sabr_res_y['alpha_yield'].iloc[rowIdx],
                                         sabr_res_y['beta_yield'].iloc[rowIdx],
                                         sabr_res_y['rho_yield'].iloc[rowIdx],
                                         nu_dn,
                                         model='n')
    p_PUT_y_nu_up = qfo.black_scholes.fwd_value(px_y, yyy, tau[rowIdx], 
                                                 sabr_vol_nu_up, opt_type='p')
    p_PUT_y_nu_dn = qfo.black_scholes.fwd_value(px_y, yyy, tau[rowIdx], 
                                                 sabr_vol_nu_dn, opt_type='p')
    p_CALL_y_nu_up = qfo.black_scholes.fwd_value(px_y, yyy, tau[rowIdx], 
                                                  sabr_vol_nu_up, opt_type='c')
    p_CALL_y_nu_dn = qfo.black_scholes.fwd_value(px_y, yyy, tau[rowIdx], 
                                                  sabr_vol_nu_dn, opt_type='c')
    price_detail.loc[row, idx[:, :, 'yp_nu_0.1']] = \
        np.concatenate(((p_PUT_y_nu_up - p_PUT_y_nu_dn) / nu_delta, 
                        (p_CALL_y_nu_up - p_CALL_y_nu_dn) / nu_delta))
    
    
#################################################################################
#
# File creation
#
#################################################################################
    
# now create the output in full with all information, drop the double time column
# output = pd.concat([ty, vols_data, sabr_res, sabr_res_y, price_detail],axis=1)

writer = pd.ExcelWriter(SHEET + '_analysis.xlsx', 
                        date_format = 'dd/mmm/yyyy',
                        datetime_format = 'dd/mmm/yyyy')

ty.to_excel(writer, sheet_name=SHEET, na_rep='na')


#################################################################################
# Create mutli-index dataframes that contain the calculated data on the original 
# information on the options. 
#################################################################################
ix1 = ['atms','10dp','25dp','40dp','40dc','25dc','10dc']
ix2 = ['tau','px','px_y','k','k_y','under_dv01','P','yP','ivol','iNvol']
m_index = pd.MultiIndex.from_product([ix1, ix2], names=['delta%','features'])
orig_data = pd.DataFrame(index=ty.index, columns=m_index)
   
orig_data.loc[k_data.index, (slice(None),'tau')] = \
            np.repeat(tau[:,np.newaxis], len(ix1), axis=1)
orig_data.loc[k_data.index, (slice(None),'px')] = \
            np.repeat(px[:,np.newaxis], len(ix1), axis=1)
orig_data.loc[k_data.index, (slice(None),'px_y')] = \
            np.repeat(y_res['atm_y'].values[:,np.newaxis], len(ix1), axis=1)
orig_data.loc[k_data.index, (slice(None), 'k')] = k_res.values
orig_data.loc[k_data.index, (slice(None), 'k_y')] = y_res.values
orig_data.loc[k_data.index, (slice(None), 'under_dv01')] = dv01_res.values
orig_data.loc[k_data.index, (slice(None), 'P')] = p_res.values
orig_data.loc[k_data.index, (slice(None), 'yP')] = yield_p_res.values
orig_data.loc[k_data.index, (slice(None), 'ivol')] = k_data.loc[:,ivols_res_cols].values
orig_data.loc[k_data.index, (slice(None), 'iNvol')] = normvol_res.values

orig_data.to_excel(writer, sheet_name=SHEET + '_orig_data', na_rep='na')

#################################################################################
# Create mutli-index dataframes that contain the sabr parameters for both  
# price and yield prices.
#################################################################################
ix1 = ['sabr_price', 'sabr_yield']
ix2 = ['alpha', 'beta', 'rho', 'nu']
m_index = pd.MultiIndex.from_product([ix1, ix2], names=['sabr_type','features'])
sabr_data = pd.DataFrame(index=ty.index, columns=m_index)
sabr_data.loc[k_data.index, ('sabr_price', slice(None))] = sabr_res.values
sabr_data.loc[k_data.index, ('sabr_yield', slice(None))] = sabr_res_y.values

sabr_data.to_excel(writer, sheet_name=SHEET + '_sabr_data', na_rep='na')

#################################################################################
# Store detail price data.
#################################################################################
price_detail.loc[:,idx['CALL',:,:]].to_excel(writer, 
                sheet_name=SHEET + '_price_detail_CALL', na_rep='na')
price_detail.loc[:,idx['PUT',:,:]].to_excel(writer,
                sheet_name=SHEET + '_price_detail_PUT', na_rep='na')



#################################################################################
# close output file
#################################################################################
writer.close()