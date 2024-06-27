#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""



import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
import seaborn as sns
import matplotlib.pyplot as plt
from epiweeks import Week


plotfolder = 'itsa_results_2024-06-23/figures/'


colors = ['#264653',# dark green
          '#2a9d8f', # medium green
          '#e9c46a', # yellow
          '#f4a261', #light orange
          '#e76f51', # darker orange
          '#843b62', # light magenta
          '#621940', # dark magenta
          ]


state_order = [#'Roraima', 'Amapá', 
               'Amazonas',  'Pará', 'Acre', 'Rondônia', 'Tocantins', # north
               'Mato Grosso', 'Goiás', 'Distrito Federal', 'Mato Grosso do Sul', # center-west
               'Paraná', 'Santa Catarina', 'Rio Grande do Sul', # south
               'Maranhão', 'Ceará', 'Piauí', 'Rio Grande do Norte', 'Paraíba', 'Pernambuco', 'Alagoas', 'Sergipe', 'Bahia', # northeast
               'Minas Gerais', 'Espírito Santo', 'Rio de Janeiro', 'São Paulo', # southeast
               ]

# -----------------------------------------------------------------------------
# functions
# -----------------------------------------------------------------------------

def scale(data_train, data_test):
    """normalize data to range [-1, 1]
    """
    val_min = data_train.min(axis=0)
    val_max = data_train.max(axis=0) 
    train_std = ((data_train - val_min) / (val_max - val_min))*2 - 1
    test_std = ((data_test - val_min) / (val_max - val_min))*2 - 1

    return train_std, test_std, val_min, val_max


def reverse_scale(pred_data, val_min, val_max):
    """reverse transform from [-1,1] range to raw case numbers
    """
    pred_rev = ( ((pred_data+1)/2) * (val_max - val_min)) + val_min
    
    return pred_rev


def dataprep_multiout(data, max_lags=12, length_out=6, min_date=None):
    """prepare data for multi-output NN model with lagged input 
    	and output features
    """
    
    # remove rows with NA date
    data.dropna(axis=0, subset=['date'], inplace=True)
    # sort by date
    data.sort_values(['date'], inplace=True)
    
    # set identifiers
    data.set_index(['date','mun_code'], inplace=True)
    
    # prep output with unlagged data:
    data_lags = data.copy()
    
    # first add the output lags (only y variable)
    for l in np.arange(1,length_out):
        data_lags['out_lag'+str(l)] = data.iloc[:,0].copy().groupby('mun_code').shift(l)
    # rename
    data_lags.rename({data_lags.columns[0]:'out_lag0'}, axis=1, inplace=True)
    
    # next the input lags
    for l in np.arange(length_out,max_lags+length_out):
        data_lags[data.columns+'_lag'+str(l)] = data.copy().groupby('mun_code').shift(l)

    # reset index    
    data_lags.reset_index(inplace=True)
    data_lags.sort_values(['mun_code', 'date'], inplace=True)

    # remove lag0 vars
    data_lags.drop(data.columns[1:], axis=1, inplace=True)


    # remove rows with missing values
    #data_lags.dropna(inplace=True)
    
    # drop data before min_date
    if min_date!=None:
        data_lags = data_lags[data_lags.date>=min_date]

    return data_lags


# -----------------------------------------------------------------------------
# load results from BSTS model
# -----------------------------------------------------------------------------


# cross validation
bsts_cv = pd.read_csv('itsa_results/bsts_cv.csv')


# 2020 predictions
bsts_preds_2020 = pd.read_csv('itsa_results/predictions-UF_bsts.csv', parse_dates=['date'])

# subset interruption time
bsts_preds_2020 = bsts_preds_2020[bsts_preds_2020.interruption_time=='2020-02-29']


# -----------------------------------------------------------------------------
# load results from multi-step NN model
# -----------------------------------------------------------------------------

nn_multiout = pd.read_csv('itsa_results/CV_predictions-UF-only-dengue_nn-multi.csv')

# reshape
nn_multiout_long = nn_multiout.melt(id_vars=['date', 'state','cv_fold'], 
                                    value_vars=['out_lag0', 'out_lag1', 'out_lag2', 'out_lag3',
                                           'out_lag4', 'out_lag5', 'out_lag6', 'out_lag7', 'out_lag8', 'out_lag9',
                                           'out_lag10', 'out_lag11', 'out_lag12', 'out_lag13', 'out_lag14',],
                                    var_name = 'variable',
                                    value_name='y_true_scaled')
nn_multiout_long['forecast_horizon'] = [int(x[7:]) for x in nn_multiout_long.variable]
# shifted dates
# y pred scaled
temp1 = nn_multiout.melt(id_vars=['date', 'state','cv_fold'], 
                                    value_vars=['pred_lag0', 'pred_lag1', 'pred_lag2', 'pred_lag3', 'pred_lag4',
                                    'pred_lag5', 'pred_lag6', 'pred_lag7', 'pred_lag8', 'pred_lag9',
                                    'pred_lag10', 'pred_lag11', 'pred_lag12', 'pred_lag13', 'pred_lag14',],
                                    var_name = 'variable',
                                    value_name='y_pred_scaled')
temp1['forecast_horizon'] = [int(x[8:]) for x in temp1.variable]
# y true real
temp2 = nn_multiout.melt(id_vars=['date', 'state','cv_fold'], 
                                    value_vars=['out_lag0_real', 'out_lag1_real', 'out_lag2_real', 'out_lag3_real',
                                    'out_lag4_real', 'out_lag5_real', 'out_lag6_real', 'out_lag7_real',
                                    'out_lag8_real', 'out_lag9_real', 'out_lag10_real', 'out_lag11_real',
                                    'out_lag12_real', 'out_lag13_real', 'out_lag14_real'],
                                    var_name = 'variable',
                                    value_name='y_true')
temp2['forecast_horizon'] = [int(x[7:-5]) for x in temp2.variable]
# y pred real
temp3 = nn_multiout.melt(id_vars=['date', 'state','cv_fold'], 
                                    value_vars=['pred_lag0_real',
                                    'pred_lag1_real', 'pred_lag2_real', 'pred_lag3_real', 'pred_lag4_real',
                                    'pred_lag5_real', 'pred_lag6_real', 'pred_lag7_real', 'pred_lag8_real',
                                    'pred_lag9_real', 'pred_lag10_real', 'pred_lag11_real',
                                    'pred_lag12_real', 'pred_lag13_real', 'pred_lag14_real'],
                                    var_name = 'variable',
                                    value_name='y_pred')
temp3['forecast_horizon'] = [int(x[8:-5]) for x in temp3.variable]

# combine
nn_multiout_long = pd.merge(nn_multiout_long, temp1.drop(['variable'], axis=1), on=['date', 'state','forecast_horizon', 'cv_fold'],
                                                         how='outer')
nn_multiout_long = pd.merge(nn_multiout_long, temp2.drop(['variable'], axis=1), on=['date', 'state','forecast_horizon', 'cv_fold'],
                                                         how='outer')
nn_multiout_long = pd.merge(nn_multiout_long, temp3.drop(['variable'], axis=1), on=['date', 'state','forecast_horizon', 'cv_fold'],
                                                         how='outer')

# add shifted dates
nn_multiout_long.reset_index(inplace=True, drop=True)
nn_multiout_long.rename({'date':'date_old'}, axis=1, inplace=True)
nn_multiout_long.date_old = pd.to_datetime(nn_multiout_long.date_old)
# adjust dates
nn_multiout_long['date'] = [nn_multiout_long.date_old[x] - timedelta(days=7.*nn_multiout_long.forecast_horizon[x]) for x in np.arange(nn_multiout_long.shape[0]) ]


# rename state to mun_code (to match rest of forecasts)
nn_multiout_long.rename({'state':'mun_code'}, axis=1, inplace=True)


# add ml method
nn_multiout_long['ml_method'] = 'nn_multi_out'
nn_multiout_long['forecast_horizon_rev'] = 9-nn_multiout_long.forecast_horizon


# --- 2020 predictions

nn_multi_2020 = pd.read_csv('itsa_results/predictions-UF_nn-multi_interruption-time-2020-02-29.csv')

# subset only test dataset
nn_multi_2020 = nn_multi_2020[nn_multi_2020.train_or_test=='test']

# reshape
# y true scaled
nn_multi_2020_long = nn_multi_2020.melt(id_vars=['date', 'mun_code'], 
                                    value_vars=['out_lag0', 'out_lag1', 'out_lag2', 'out_lag3',
                                           'out_lag4', 'out_lag5', 'out_lag6', 'out_lag7', 'out_lag8', 'out_lag9',
                                           'out_lag10', 'out_lag11', 'out_lag12', 'out_lag13', 'out_lag14',],
                                    var_name = 'variable',
                                    value_name='y_true_scaled')
nn_multi_2020_long['forecast_horizon'] = [int(x[7:]) for x in nn_multi_2020_long.variable]
# y pred scaled
temp1 = nn_multi_2020.melt(id_vars=['date', 'mun_code'], 
                                    value_vars=['pred_lag0', 'pred_lag1', 'pred_lag2', 'pred_lag3', 'pred_lag4',
                                    'pred_lag5', 'pred_lag6', 'pred_lag7', 'pred_lag8', 'pred_lag9',
                                    'pred_lag10', 'pred_lag11', 'pred_lag12', 'pred_lag13', 'pred_lag14',],
                                    var_name = 'variable',
                                    value_name='y_pred_scaled')
temp1['forecast_horizon'] = [int(x[8:]) for x in temp1.variable]
# y true real
temp2 = nn_multi_2020.melt(id_vars=['date', 'mun_code'], 
                                    value_vars=['out_lag0_real', 'out_lag1_real', 'out_lag2_real', 'out_lag3_real',
                                    'out_lag4_real', 'out_lag5_real', 'out_lag6_real', 'out_lag7_real',
                                    'out_lag8_real', 'out_lag9_real', 'out_lag10_real', 'out_lag11_real',
                                    'out_lag12_real', 'out_lag13_real', 'out_lag14_real'],
                                    var_name = 'variable',
                                    value_name='y_true')
temp2['forecast_horizon'] = [int(x[7:-5]) for x in temp2.variable]
# y pred real
temp3 = nn_multi_2020.melt(id_vars=['date', 'mun_code'], 
                                    value_vars=['pred_lag0_real',
                                    'pred_lag1_real', 'pred_lag2_real', 'pred_lag3_real', 'pred_lag4_real',
                                    'pred_lag5_real', 'pred_lag6_real', 'pred_lag7_real', 'pred_lag8_real',
                                    'pred_lag9_real', 'pred_lag10_real', 'pred_lag11_real',
                                    'pred_lag12_real', 'pred_lag13_real', 'pred_lag14_real'],
                                    var_name = 'variable',
                                    value_name='y_pred')
temp3['forecast_horizon'] = [int(x[8:-5]) for x in temp3.variable]

# combine
nn_multi_2020_long = pd.merge(nn_multi_2020_long, temp1.drop(['variable'], axis=1), on=['date', 'mun_code','forecast_horizon'],
                                                         how='outer')
nn_multi_2020_long = pd.merge(nn_multi_2020_long, temp2.drop(['variable'], axis=1), on=['date', 'mun_code','forecast_horizon'],
                                                         how='outer')
nn_multi_2020_long = pd.merge(nn_multi_2020_long, temp3.drop(['variable'], axis=1), on=['date', 'mun_code','forecast_horizon'],
                                                         how='outer')

# add shifted dates
nn_multi_2020_long.reset_index(inplace=True, drop=True)
nn_multi_2020_long.rename({'date':'date_old'}, axis=1, inplace=True)
nn_multi_2020_long.date_old = pd.to_datetime(nn_multi_2020_long.date_old)
# adjust dates
nn_multi_2020_long['date'] = [nn_multi_2020_long.date_old[x] - timedelta(days=7.*nn_multi_2020_long.forecast_horizon[x]) for x in np.arange(nn_multi_2020_long.shape[0]) ]



# add vars
nn_multi_2020_long['ml_method'] = 'nn_multi_out'
nn_multi_2020_long['forecast_horizon_rev'] = 9-nn_multi_2020_long.forecast_horizon



# -----------------------------------------------------------------------------
# read data
# -----------------------------------------------------------------------------

# read dengue data
data = pd.read_csv('sinan_uf_merged.csv', parse_dates=['date'])


# read covariates
dengue_climate = pd.read_csv('data for ps matching_all Brazil_2021-09-10.csv', parse_dates=['date'])



# read conversions
conv_state = pd.read_csv('RELATORIO_DTB_BRASIL_MUNICIPIO.csv')
conv_region = pd.read_csv('regions-to-states-conversion.csv')
# merge
conv_state['mun_code'] = [int(str(x)[:-1]) for x in conv_state.loc[:,'Código Município Completo']]


# -----------------------------------------------------------------------------
# merge and preprocess data
# -----------------------------------------------------------------------------

### aggregate climate to state level

dengue_climate = pd.merge(dengue_climate, conv_state.loc[:,['mun_code','Nome_UF']],
                          on = 'mun_code', how='left')
dengue_climate = pd.merge(dengue_climate, conv_region.loc[:,['region','state']],
                          left_on = 'Nome_UF', right_on='state',
                          how='left')

# agg
climate = dengue_climate.groupby(['Nome_UF','year', 'epi_week', 'date']).agg({'precipitacao':'mean','tempmaxima':'mean', 'temp comp media':'mean', 
                                'tempminima':'mean','umidade relativa media':'mean', 'velocidade do vento media':'mean'}).reset_index()

climate.rename({'Nome_UF':'state'}, axis=1, inplace=True)

# combine
use_data = pd.merge(data, climate, 
                    on=['state', 'year', 'epi_week', 'date'],
                    how='left')


# rename 'state' to 'mun_code' for code
use_data.rename({'state':'mun_code'}, axis=1, inplace=True)




###############################################################################
### step 1) MODEL SELECTION - Time Series Cross-validation
###############################################################################



n_features = 1
n_timesteps_in = 15
n_timesteps_out = 10
    
    
    

# -----------------------------------------------------------------------------
# data prep for NN model - only dengue
# -----------------------------------------------------------------------------

# data prep
data_0 = dataprep_multiout(use_data.loc[:,['date','mun_code','cases']],
                            max_lags = n_timesteps_in,
                            length_out = n_timesteps_out
                            )

# reshuffle randomly
data_0 = data_0.sample(frac=1, random_state=12).reset_index(drop=True)

# drop rows with missing values
data_0.dropna(inplace=True)


# full train set up to 2020
data_train_full = data_0[data_0.date<'2020-01-01'].copy()


# -----------------------------------------------------------------------------
# data prep for NN model - with climate
# -----------------------------------------------------------------------------

# data prep
data_1 = dataprep_multiout(use_data.loc[:,['date','mun_code','cases', 
                                           'precipitacao', 'tempmaxima', 'temp comp media', 
                                           'tempminima','umidade relativa media', 
                                           'velocidade do vento media']],
                            max_lags = n_timesteps_in,
                            length_out = n_timesteps_out
                            )

# reshuffle randomly
data_1 = data_1.sample(frac=1, random_state=12).reset_index(drop=True)

# drop rows with missing values
data_1.dropna(inplace=True)


# full train set up to 2020
data_train_full_1 = data_1[data_1.date<'2020-01-01'].copy()



# shifted dates
shifted_dates = pd.DataFrame()
for fh in np.arange(n_timesteps_out): 
    temp = data_train_full_1.date - timedelta(days= 7.*fh)
    shifted_dates['date_lag'+str(fh)] = temp

shifted_dates0 = pd.DataFrame()
for fh in np.arange(n_timesteps_out): 
    temp = data_train_full.date - timedelta(days= 7.*fh)
    shifted_dates0['date_lag'+str(fh)] = temp




# -----------------------------------------------------------------------------
# CV - multiple models
# -----------------------------------------------------------------------------

# define input features (dengue + climate)
input_feature_names = data_train_full_1.columns[(n_timesteps_out+2):]


# add epi week
data_train_full_1['epi_week'] = [int(str(x)[-2:]) for x in data_train_full_1.date.apply(Week.fromdate)]

# prep output
forecasts_all = pd.DataFrame()

for cv_fold, y in enumerate(np.arange(2016,2020)):
    
    # 1 year of test set, all prior data for training    
    data_train_cv = data_train_full_1[data_train_full_1.date.dt.year<y]
    
    data_test_cv = data_train_full_1[data_train_full_1.date.dt.year==y]
    
    # drop the overlapping observations from the training set (i.e. last 15 weeks of the year before test set)
    data_train_cv = data_train_cv[data_1.date <= ( data_test_cv.date.min()  - timedelta(days= 7.0 * n_timesteps_out ) ) ]

    
    shifted_dates_testcv = shifted_dates.loc[list(data_test_cv.index.values),:]
    
    # rescale 0-1
    train_scaled, test_scaled, val_min, val_max = scale(np.array(data_train_cv.iloc[:,2:]), 
                                                        np.array(data_test_cv.iloc[:,2:]))
    train_scaled_df = pd.DataFrame(train_scaled, columns=data_train_cv.columns[2:])
    test_scaled_df = pd.DataFrame(test_scaled, columns=data_test_cv.columns[2:])

    # input features remain the same over forecast horizons
    train_x = train_scaled_df.loc[:,input_feature_names]
    test_x = test_scaled_df.loc[:,input_feature_names]
    
    

    # ---------------------------------------------------------------------
    # add BSTS (from prior analysis)
    # ---------------------------------------------------------------------
            
    bsts_wclimate = bsts_cv[bsts_cv.year==y]
    
    # date variable
    bsts_wclimate.date = pd.to_datetime(bsts_wclimate.date)

    # ignore in-sample predictions
    bsts_wclimate = bsts_wclimate[bsts_wclimate.date>='01-01-'+str(y)]
    
    # add fh var
    bsts_wclimate['forecast_horizon_rev'] = 0
    
    # take only the first interruption time each year
    inttime = bsts_wclimate.interruption_time.min()
    # subset relevant interruption time
    temp = bsts_wclimate[bsts_wclimate.interruption_time==inttime]
    # subset dates after interruption time
    temp = temp[temp.date>inttime]
    temp.sort_values('date', ascending=True, inplace=True)
    # subset only 10 fhs
    temp = temp[temp.date<temp.date.unique()[10]]
    
    temp['y_pred_scaled'] = temp.post_preds.values
    # add fh var
    for fh, predtime in enumerate(temp.date.unique()):
        temp['forecast_horizon_rev'] = np.where(temp.date==predtime, fh, temp.forecast_horizon_rev)
        # scaled pred
        # rescale preds
        temp['y_pred_scaled'] = np.where(temp.date==predtime , ((temp.post_preds - val_min[fh]) / (val_max[fh] - val_min[fh]))*2 - 1, temp.y_pred_scaled)
    
    # subset variables
    temp = temp.loc[:,['date','post_preds', 'state',  'forecast_horizon_rev', 'y_pred_scaled']]
    # rename
    temp.rename({'post_preds':'y_pred', 'state':'mun_code'}, axis=1, inplace=True)
    
    # add y true (merge)
    temptrue = data_test_cv.loc[:,['date','mun_code', 'out_lag0']]
    temptrue.reset_index(inplace=True, drop=True)
    temptrue['y_true_scaled'] = test_scaled_df.loc[:,'out_lag0'].values
    temp = pd.merge(temp, temptrue, on=['date','mun_code'], how='left')
    temp.rename({'out_lag0':'y_true'}, axis=1, inplace=True)
    
    
    # add ml method
    temp['ml_method'] = 'bsts'
    # add the "reverse" forecast horizon to match the other models
    temp['forecast_horizon'] = 9-temp.forecast_horizon_rev
    # append to output
    forecasts_all = pd.concat((forecasts_all, temp))
        
    


    # loop over forecast horizons:
    for fh in np.arange(n_timesteps_out):

        # target feature varies by forecast horizon
        train_y = train_scaled_df.loc[:,'out_lag'+str(fh)]
        test_y = test_scaled_df.loc[:,'out_lag'+str(fh)]
        
        
        # ---------------------------------------------------------------------
        # model 1: Gradient boosting (with CI)
        # ---------------------------------------------------------------------
        all_models = {}
        
        for alpha in [0.05, 0.5, 0.95]:
            gbr = GradientBoostingRegressor(loss='quantile', alpha=alpha, 
                                            n_estimators=150, learning_rate=0.05,
                                            min_samples_split=3, max_depth=4)
            all_models["q %1.2f" % alpha] = gbr.fit(train_x, train_y)

        y_lower = all_models['q 0.05'].predict(test_x)
        y_upper = all_models['q 0.95'].predict(test_x)
        y_med = all_models['q 0.50'].predict(test_x)

        pred_result_gbr = pd.DataFrame({'mun_code':data_test_cv.mun_code,
                                    'date':shifted_dates_testcv['date_lag'+str(fh)],
                                    'y_true_scaled':test_y.values,
                                    'y_true':data_test_cv.loc[:,'out_lag'+str(fh)],
                                    'y_lower_scaled':y_lower,
                                    'y_upper_scaled':y_upper,
                                    'y_pred_scaled':y_med, # median
                                    'ml_method':'GBR',
                                   })
        
        pred_result_gbr['y_lower'] = reverse_scale(y_lower, val_min[fh], val_max[fh])
        pred_result_gbr['y_upper'] = reverse_scale(y_upper, val_min[fh], val_max[fh])
        
        # ---------------------------------------------------------------------
        # model 2: xgboost
        # ---------------------------------------------------------------------
        
        # build model
        bst = xgb.XGBRegressor(max_depth=4, min_child_weight = 3, learning_rate=0.05)
        # fit model
        bst.fit(train_x, train_y)
        # make predictions
        y_pred_xgb = bst.predict(test_x)
        # prep output
        pred_result_xgb = pd.DataFrame({'mun_code':data_test_cv.mun_code,
                                    'date':shifted_dates_testcv['date_lag'+str(fh)],
                                    'y_true_scaled':test_y.values,
                                    'y_true':data_test_cv.loc[:,'out_lag'+str(fh)],
                                    'y_pred_scaled':y_pred_xgb,
                                    'ml_method':'XGB',
                                   })
        
        
        
        # ---------------------------------------------------------------------
        # model 3: RF
        # ---------------------------------------------------------------------
        
        # build
        rfm = RandomForestRegressor(n_estimators=150, max_features='auto', 
                                    max_depth=4, min_samples_split=3)
        # fit
        rfm.fit(train_x, train_y)
        # predict
        y_pred_rf = rfm.predict(test_x)
        # prep output
        pred_result_rf = pd.DataFrame({'mun_code':data_test_cv.mun_code,
                                    'date':shifted_dates_testcv['date_lag'+str(fh)],
                                    'y_true_scaled':test_y.values,
                                    'y_true':data_test_cv.loc[:,'out_lag'+str(fh)],
                                    'y_pred_scaled':y_pred_rf,
                                    'ml_method':'RF',
                                   })
        
        
        
        # ---------------------------------------------------------------------
        # seasonal average model (naive) - same for all fh
        # ---------------------------------------------------------------------
        temp = train_scaled_df.copy()
        temp['mun_code'] = data_train_cv.mun_code.values
        temp['date'] = data_train_cv.date.values
        
        # add epi week
        temp['epi_week'] = [int(str(x)[-2:]) for x in temp.date.apply(Week.fromdate)]
        
        # calculate the average (by epi week)
        snaive = temp.groupby(['mun_code','epi_week']).agg({'out_lag'+str(fh):np.nanmean}).reset_index()
        snaive.rename({'out_lag'+str(fh):'y_pred_scaled'}, axis=1, inplace=True)
        
        # prep output
        pred_result_snaive = pd.DataFrame({'mun_code':data_test_cv.mun_code,
                                    'date':shifted_dates_testcv['date_lag'+str(fh)],
                                    'y_true_scaled':test_y.values,
                                    'y_true':data_test_cv.loc[:,'out_lag'+str(fh)],
                                    'ml_method':'seasonal_average',
                                   })
        # merge the observed with the seasonal average data
        pred_result_snaive['epi_week'] = [int(str(x)[-2:]) for x in pred_result_snaive.date.apply(Week.fromdate)]
        pred_result_snaive = pred_result_snaive.merge(snaive, on=['mun_code','epi_week'], how='left')
        
        
                
        # ---------------------------------------------------------------------
        # SVR
        # ---------------------------------------------------------------------
        
        
        svrmodel = SVR(kernel="rbf", C=10, gamma='scale', epsilon=0.1)
        
        # fit
        svrmodel.fit(train_x, train_y)
        # predict
        y_pred_svr = svrmodel.predict(test_x)
        # prep output
        pred_result_svr = pd.DataFrame({'mun_code':data_test_cv.mun_code,
                                    'date':shifted_dates_testcv['date_lag'+str(fh)],
                                    'y_true_scaled':test_y.values,
                                    'y_true':data_test_cv.loc[:,'out_lag'+str(fh)],
                                    'y_pred_scaled':y_pred_svr,
                                    'ml_method':'SVR',
                                   })
        
        

        
        # ---------------------------------------------------------------------
        # NN - single step
        # ---------------------------------------------------------------------
        
        
        nnmodel = MLPRegressor(hidden_layer_sizes=(128, 128, 128),
                               activation = 'relu',
                               solver = 'adam',
                               learning_rate = 'adaptive')
        
        # fit
        nnmodel.fit(train_x, train_y)
        # predict
        y_pred_nn = nnmodel.predict(test_x)
        # prep output
        pred_result_nn = pd.DataFrame({'mun_code':data_test_cv.mun_code,
                                    'date':shifted_dates_testcv['date_lag'+str(fh)],
                                    'y_true_scaled':test_y.values,
                                    'y_true':data_test_cv.loc[:,'out_lag'+str(fh)],
                                    'y_pred_scaled':y_pred_nn,
                                    'ml_method':'NN',
                                   })
        
        

        
        # ---------------------------------------------------------------------
        # combine output and rescale 
        # ---------------------------------------------------------------------
        
        out_temp = pd.concat((pred_result_gbr, pred_result_xgb, pred_result_rf, 
                              pred_result_snaive, pred_result_svr, pred_result_nn))
        
        # rescale
        out_temp['y_pred'] = reverse_scale(out_temp.y_pred_scaled, val_min[fh], val_max[fh])

        # add fh
        out_temp['forecast_horizon'] = fh
        out_temp['cv_fold'] = cv_fold
        
        
        # append
        forecasts_all = pd.concat((forecasts_all, out_temp))

forecasts_all['forecast_horizon_rev'] = 9-forecasts_all.forecast_horizon


# add nn multi
forecasts_all = pd.concat((forecasts_all, nn_multiout_long))



# save
forecasts_all.to_csv('pred_out_all-models_2024-06-13.csv', index=False)        

#forecasts_all = pd.read_csv('pred_out_all-models_2024-06-13.csv')        
        



bsts_dates = forecasts_all[forecasts_all.ml_method=='bsts'].date.unique()


# -----------------------------------------------------------------------------
# compute errors 
# -----------------------------------------------------------------------------

# prep
forecasts_all.reset_index(inplace=True, drop=True)

forecasts_all.date = pd.to_datetime(forecasts_all.date)


# compute
errors_out_subsetdates = pd.DataFrame()

# subset by state
for s, state in enumerate(forecasts_all.mun_code.unique()):
    sub_state = forecasts_all[forecasts_all.mun_code==state]
    
    # for forecast horizon
    for fh in np.arange(n_timesteps_out):
        subsub = sub_state[sub_state.forecast_horizon==fh]
        # for model type
        for mm in forecasts_all.ml_method.unique():
    
            subsubsub = subsub[subsub.ml_method==mm]
            
            # drop na rows
            subsubsub.dropna(axis=0, inplace=True, subset=['y_true', 'y_pred'])
            print(state, fh, mm)
            print(subsubsub.shape)
            
            # subset date range
            subsubsubsub = subsubsub[subsubsub.date.isin(bsts_dates)]
            # check if combo of fh and ml method exist
            if subsubsubsub.shape[0]>0:
                y_true = subsubsubsub.y_true
                y_pred = subsubsubsub.y_pred
                
                mae = mean_absolute_error(y_true, y_pred)
                mse = mean_squared_error(y_true, y_pred)
                rmse = np.sqrt(mse)
                
                errors_out_subsetdates = pd.concat((errors_out_subsetdates, pd.DataFrame({'state':state,
                                                           'mae':mae, 'rmse':rmse, 
                                                           'forecast_horizon':fh, 
                                                           'ml_method':mm,
                                                           }, index=[0])))
    print(state)
      

# add alternative horizon value
errors_out_subsetdates['forecast_horizon_rev'] = 9-errors_out_subsetdates.forecast_horizon

# save
errors_out_subsetdates.to_csv('errors_out_subset-dates_2024-06-13.csv', index=False)
# read
errors_out_subsetdates = pd.read_csv('errors_out_subset-dates_2024-06-13.csv')




# average error by method and horizon
errors_out_subsetdates.groupby(['forecast_horizon', 'ml_method', ]).agg({'mae':'mean', 'rmse':'mean'})
# average error by method
errors_out_subsetdates.groupby(['ml_method', ]).agg({'mae':'mean', 'rmse':'mean'})



# -----------------------------------------------------------------------------
# compute errors - expanded time period
# -----------------------------------------------------------------------------



errors_out = pd.DataFrame()

# subset by state
for s, state in enumerate(forecasts_all.mun_code.unique()):
    sub_state = forecasts_all[forecasts_all.mun_code==state]
    
    # for forecast horizon
    for fh in np.arange(n_timesteps_out):
        subsub = sub_state[sub_state.forecast_horizon==fh]
        # for model type
        for mm in forecasts_all.ml_method.unique():
    
            subsubsub = subsub[subsub.ml_method==mm]
            
            # drop na rows
            subsubsub.dropna(axis=0, inplace=True, subset=['y_true', 'y_pred'])
            print(state, fh, mm)
            print(subsubsub.shape)
            
            # check if combo of fh and ml method exist
            if subsubsub.shape[0]>0:
                y_true = subsubsub.y_true
                y_pred = subsubsub.y_pred
                
                mae = mean_absolute_error(y_true, y_pred)
                mse = mean_squared_error(y_true, y_pred)
                rmse = np.sqrt(mse)
                
                errors_out = pd.concat((errors_out, pd.DataFrame({'state':state,
                                                           'mae':mae, 'rmse':rmse, 
                                                           'forecast_horizon':fh, 
                                                           'ml_method':mm,
                                                           }, index=[0])))
    print(state)
      

# add alternative horizon value
errors_out['forecast_horizon_rev'] = 9-errors_out.forecast_horizon

# save
errors_out.to_csv('errors_out_all_2024-06-13.csv', index=False)
# read
errors_out = pd.read_csv('errors_out_all_2024-06-13.csv')





# average error by method and horizon
errors_out.groupby(['forecast_horizon', 'ml_method', ]).agg({'mae':'mean', 'rmse':'mean'})
# average error by method
errors_out.groupby(['ml_method', ]).agg({'mae':'mean', 'rmse':'mean'})






# -----------------------------------------------------------------------------
# make ensembles ( validation data )
# -----------------------------------------------------------------------------

# -- ensemble 1: simple average
ensemble_mean = forecasts_all.groupby(['mun_code', 'date', 'forecast_horizon', 'cv_fold', 
                                       'forecast_horizon_rev']).agg({'y_true_scaled':'mean', 
                                                                     'y_true':'mean', 
                                                                     'y_pred_scaled':'mean',
                                                                     'y_pred':'mean'}).reset_index()
ensemble_mean['ml_method'] = 'ensemble_average'



# -- ensemble top 3 
ensemble_top3 = forecasts_all[forecasts_all.ml_method.isin(['XGB','GBR', 'RF'])].groupby(['mun_code', 'date', 'forecast_horizon', 'cv_fold', 'forecast_horizon_rev']).agg({'y_true_scaled':'mean', 'y_true':'mean', 'y_pred_scaled':'mean','y_pred':'mean'}).reset_index()
ensemble_top3['ml_method'] = 'ensemble_top3'


# -- ensemble 3: RF 
# reshape data for ensemble model (long to wide)
data_ensemble_rf = forecasts_all.pivot(index = ['mun_code', 'date', 'y_true_scaled', 'y_true',],
                                       columns = ['ml_method', 'forecast_horizon_rev'],
                                       values = ['y_pred_scaled']).reset_index()

data_ensemble_rf.date = pd.to_datetime(data_ensemble_rf.date)


rf_ensemble_out = pd.DataFrame()

for fh in np.arange(1,9):
    data_ensemble_rf1 = forecasts_all[forecasts_all.forecast_horizon_rev==fh].pivot(index = ['mun_code', 'date', 'y_true_scaled', 'y_true',],
                                           columns = ['ml_method'],
                                           values = ['y_pred_scaled']).reset_index()

    # drop rows with missing values
    data_ensemble_rf1_nona = data_ensemble_rf1.dropna(axis=0)
    
    for cv_fold, y in enumerate(np.arange(2016,2020)):
        
        # --- start: get the max and min values for rescaling ---
        data_train_cv = data_train_full_1[data_train_full_1.date.dt.year<y]
        data_test_cv = data_train_full_1[data_train_full_1.date.dt.year==y]

        data_train_cv = data_train_cv[data_0.date <= ( data_test_cv.date.min()  - timedelta(days= 7.0 * n_timesteps_out ) ) ]
        bla, blb, val_min, val_max = scale(np.array(data_train_cv.iloc[:,2:]), 
                                                            np.array(data_test_cv.iloc[:,2:]))
        # --- end ---
        
        # split into train and validation set
        data_ensemble_rf_train = data_ensemble_rf1_nona[data_ensemble_rf1_nona.date<'01-01-'+str(y)]
        data_ensemble_rf_val = data_ensemble_rf1_nona[(data_ensemble_rf1_nona.date>='01-01-'+str(y))&
                                                (data_ensemble_rf1_nona.date<'01-01-'+str(y+1))]
        
        # drop missing
        data_ensemble_rf_train.dropna(axis=0, inplace=True)
        data_ensemble_rf_val.dropna(axis=0, inplace=True)

        if (data_ensemble_rf_train.shape[0]>0) & (data_ensemble_rf_val.shape[0]>0):
            # split into x and y
            # train set
            data_ensemble_rf_train_X = data_ensemble_rf_train.drop([(     'mun_code',    ''),
                                                                    (         'date',    ''),
                                                                    ('y_true_scaled',    ''),
                                                                    (       'y_true',    ''),], axis=1)
            data_ensemble_rf_train_Y = data_ensemble_rf_train.loc[:,[('y_true_scaled',       '')]]
            # same for val set
            data_ensemble_rf_val_X = data_ensemble_rf_val.drop([(     'mun_code',    ''),
                                                                (         'date',    ''),
                                                                ('y_true_scaled',    ''),
                                                                (       'y_true',    ''),], axis=1)
            data_ensemble_rf_val_Y = data_ensemble_rf_val.loc[:,[('y_true_scaled',       '')]]
        
            
            # build rf ensemble model
            rf_ensemble_model = RandomForestRegressor(n_estimators=20, max_features='auto', 
                                        max_depth=4, min_samples_split=3)
            # fit
            rf_ensemble_model.fit(data_ensemble_rf_train_X.values, data_ensemble_rf_train_Y.values)
            # predict
            rf_ensemble_pred = rf_ensemble_model.predict(data_ensemble_rf_val_X.values)
            # prep output
            rf_ensemble_out_temp = data_ensemble_rf_val.loc[:,[(     'mun_code',    ''),
                                                                    (         'date',    ''),
                                                                    ('y_true_scaled',    ''),
                                                                    (       'y_true',    ''),]]
            
            rf_ensemble_out_temp.columns = rf_ensemble_out_temp.columns.droplevel(1)
            
            rf_ensemble_out_temp['y_pred_scaled'] = rf_ensemble_pred
            rf_ensemble_out_temp['cv_fold'] = cv_fold
            rf_ensemble_out_temp['forecast_horizon_rev'] = fh
            
            
            # rescale
            rf_ensemble_out_temp['y_pred'] = reverse_scale(rf_ensemble_out_temp.y_pred_scaled, val_min[fh], val_max[fh])
    
            
            # append
            rf_ensemble_out = pd.concat((rf_ensemble_out, rf_ensemble_out_temp))
            

rf_ensemble_out['ml_method'] = 'ensemble_rf'

rf_ensemble_out['forecast_horizon'] = 9-rf_ensemble_out.forecast_horizon_rev



rf_ensemble_out.reset_index(inplace=True, drop=True)
ensemble_mean.reset_index(inplace=True, drop=True)
ensemble_top3.reset_index(inplace=True, drop=True)


# combine ensembles
ensembles_combined = pd.concat((rf_ensemble_out, 
                                ensemble_mean.loc[:,rf_ensemble_out.columns], 
                                ensemble_top3.loc[:,rf_ensemble_out.columns]))





# -----------------------------------------------------------------------------
# combine forecasts with ensembles
# -----------------------------------------------------------------------------
ensembles_combined.reset_index(inplace=True, drop=True)
forecasts_all.reset_index(inplace=True, drop=True)

# add identifier for individual model vs ensemble
forecasts_all['is_ensemble'] = False
ensembles_combined['is_ensemble'] = True

# concat
forecasts_with_ensembles = pd.concat((forecasts_all, ensembles_combined))

# update date
forecasts_with_ensembles.date = pd.to_datetime(forecasts_with_ensembles.date)


# -----------------------------------------------------------------------------
# calculate confidence intervals
# -----------------------------------------------------------------------------


# are errors normally distributed?

# calculate residual
forecasts_with_ensembles['residual'] = forecasts_with_ensembles.y_pred - forecasts_with_ensembles.y_true
forecasts_with_ensembles['residual_scaled'] = forecasts_with_ensembles.y_pred_scaled - forecasts_with_ensembles.y_true_scaled


# -- generate confidence intervals using bootstrapping

n_bootstrap = 200
bootstrap_out = pd.DataFrame()
for fh in np.arange(n_timesteps_out):
    # subset horizons
    forecasts_fh = forecasts_with_ensembles[forecasts_with_ensembles.forecast_horizon_rev==fh]
    # seperate interval for each state
    for s, state in enumerate(forecasts_with_ensembles.mun_code.unique()):
        # subset out-of-sample predictions from state
        forecasts_state = forecasts_fh[forecasts_fh.mun_code==state]
        for mm in forecasts_with_ensembles.ml_method.unique():
            forecasts_mm = forecasts_state[forecasts_state.ml_method==mm]
            for ii in np.arange(n_bootstrap):
                # sample with replacement
                bootstrap_sample = forecasts_mm.sample(n=forecasts_mm.shape[0], replace=True)
                # calculate std deviation
                bootstrap_std = np.std(bootstrap_sample.residual)
                # append
                bootstrap_out = pd.concat((bootstrap_out, 
                                           pd.DataFrame({'forecast_horizon_rev':fh,
                                                         'state': state,
                                                         'ml_method':mm,
                                                         'bootstrap_iteration':ii,
                                                         'bootstrap_std_residual':bootstrap_std}, index=[0])))
    print(fh)

# median std dev of residuals
bootstrap_measures = bootstrap_out.groupby(['state','ml_method','forecast_horizon_rev']).agg({'bootstrap_std_residual':'median'}).reset_index()
# prediction band
bootstrap_measures['bs_band_size'] = 1.96*bootstrap_measures.bootstrap_std_residual 

# save
bootstrap_measures.to_csv('bootstrap_measures_individual-models-and-ensembles_2024-06-13.csv', index=False)





# -----------------------------------------------------------------------------
# calculate all errors -  ensembles and individual models -> to select best models
# -----------------------------------------------------------------------------



# compute
errors_ensembles_subsetdates = pd.DataFrame()

# subset by state
for s, state in enumerate(forecasts_with_ensembles.mun_code.unique()):
    sub_state = forecasts_with_ensembles[forecasts_with_ensembles.mun_code==state]
    
    # for forecast horizon
    for fh in np.arange(n_timesteps_out):
        subsub = sub_state[sub_state.forecast_horizon==fh]
        # for model type
        for mm in forecasts_with_ensembles.ml_method.unique():
    
            subsubsub = subsub[subsub.ml_method==mm]
            
            # drop na rows
            subsubsub.dropna(axis=0, inplace=True, subset=['y_true', 'y_pred'])
            print(state, fh, mm)
            print(subsubsub.shape)
            
            # subset date range
            subsubsubsub = subsubsub[subsubsub.date.isin(bsts_dates)]
            # check if combo of fh and ml method exist
            if subsubsubsub.shape[0]>0:
                y_true = subsubsubsub.y_true
                y_pred = subsubsubsub.y_pred
                
                mae = mean_absolute_error(y_true, y_pred)
                mse = mean_squared_error(y_true, y_pred)
                rmse = np.sqrt(mse)
                
                # average y_true for comparison
                ytrue_avg = np.mean(y_true)
                
                errors_ensembles_subsetdates = pd.concat((errors_ensembles_subsetdates, 
                                                          pd.DataFrame({'state':state,
                                                           'mae':mae, 'rmse':rmse, 
                                                           'y_true_avg':ytrue_avg,
                                                           'forecast_horizon':fh, 
                                                           'ml_method':mm,
                                                           'is_ensemble':subsubsubsub.is_ensemble.unique()[0],
                                                           }, index=[0])))
    print(state)
      

# add alternative horizon value
errors_ensembles_subsetdates['forecast_horizon_rev'] = 9-errors_ensembles_subsetdates.forecast_horizon

# save
errors_ensembles_subsetdates.to_csv('errors_with-ensembles_subset-dates_2024-06-13.csv', index=False)
# read
errors_ensembles_subsetdates = pd.read_csv('errors_with-ensembles_subset-dates_2024-06-13.csv')



inspect_errors = errors_ensembles_subsetdates.groupby(['ml_method']).agg({'mean'}).reset_index()
inspect_errors2 = errors_ensembles_subsetdates.groupby(['ml_method', 'forecast_horizon']).agg({'mean'}).reset_index()

inspect_errors.to_csv('itsa_results_2024-06-23/errors-by-method_subset-dates.csv', index=False)
inspect_errors2.to_csv('itsa_results_2024-06-23/errors-by-method-and-fh_subset-dates.csv', index=False)


# -- full time period

errors_ensembles = pd.DataFrame()

# subset by state
for s, state in enumerate(forecasts_with_ensembles.mun_code.unique()):
    sub_state = forecasts_with_ensembles[(forecasts_with_ensembles.mun_code==state)&
                                   (forecasts_with_ensembles.date>'2015-12-31')]
    
    # for forecast horizon
    for fh in np.arange(1,9):
        subsub = sub_state[sub_state.forecast_horizon_rev==fh]
        
        for mm in forecasts_with_ensembles.ml_method.unique():
            subsubsub = subsub[subsub.ml_method==mm]
            # drop na
            subsubsub.dropna(axis=0, inplace=True, subset=['y_pred'])
            # check if combo of fh and ml method exist
            if subsubsub.shape[0]>0:
                #y_true = subsubsub.y_true_scaled
                #y_pred = subsubsub.y_pred_scaled
                y_true = subsubsub.y_true
                y_pred = subsubsub.y_pred
                
                mae = mean_absolute_error(y_true, y_pred)
                mse = mean_squared_error(y_true, y_pred)
                rmse = np.sqrt(mse)
                
                # average y_true for comparison
                ytrue_avg = np.mean(y_true)
                
                errors_ensembles = pd.concat((errors_ensembles, pd.DataFrame({'state':state,
                                                           'mae':mae, 'rmse':rmse, 
                                                           'y_true_avg':ytrue_avg,
                                                           'forecast_horizon':fh, 
                                                           'ml_method':mm,
                                                           'is_ensemble':subsubsub.is_ensemble.unique()[0],
                                                           }, index=[0])))
    print(state)
      

# save
errors_ensembles.to_csv('errors_out_ensembles_2024-06-13.csv', index=False)



inspect_errors3 = errors_ensembles.groupby(['ml_method']).agg({'mean'}).reset_index()
inspect_errors4 = errors_ensembles.groupby(['ml_method', 'forecast_horizon']).agg({'mean'}).reset_index()

inspect_errors3.to_csv('itsa_results_2024-06-23/errors-by-method_all-time.csv', index=False)
inspect_errors4.to_csv('itsa_results_2024-06-23/errors-by-method-and-fh_all-time.csv', index=False)





###############################################################################
### step 2) 2020 FORECASTS - for top3 ensemble
###############################################################################




interruption_time = '2020-02-29'    



# train < interruption time
data_train = data_1[data_1.date < interruption_time]

# test > interruption time minus any overlap
data_test = data_1[data_1.date >= ( pd.to_datetime(interruption_time) + timedelta(days= 7.0 * n_timesteps_out ) ) ].copy()

    
input_feature_names = data_1.columns[(n_timesteps_out+2):]
    
# rescale 0-1
train_scaled, test_scaled, val_min, val_max = scale(np.array(data_train.iloc[:,2:]), 
                                                    np.array(data_test.iloc[:,2:]))
train_scaled_df = pd.DataFrame(train_scaled, columns=data_train.columns[2:])
test_scaled_df = pd.DataFrame(test_scaled, columns=data_test.columns[2:])

# input features remain the same over forecast horizons
train_x = train_scaled_df.loc[:,input_feature_names]
test_x = test_scaled_df.loc[:,input_feature_names]


# shifted dates
shifted_dates_2020 = pd.DataFrame()
for fh in np.arange(n_timesteps_out): 
    temp = data_test.date - timedelta(days= 7.*fh)
    shifted_dates_2020['date_lag'+str(fh)] = temp


# prep output
forecasts_2020 = pd.DataFrame()

# for each forecast horizon:
for fh in np.arange(n_timesteps_out):

    # target feature varies by forecast horizon
    train_y = train_scaled_df.loc[:,'out_lag'+str(fh)]
    test_y = test_scaled_df.loc[:,'out_lag'+str(fh)]
    
    
    # ---------------------------------------------------------------------
    # model 1: Gradient boosting regression
    # ---------------------------------------------------------------------
    all_models = {}
    alpha=0.5
    gbr = GradientBoostingRegressor(loss='quantile', alpha=alpha, 
                                    n_estimators=150, learning_rate=0.05,
                                    min_samples_split=3, max_depth=4)
    all_models["q %1.2f" % alpha] = gbr.fit(train_x, train_y)

    y_med = all_models['q 0.50'].predict(test_x)

    pred_result_gbr = pd.DataFrame({'mun_code':data_test.mun_code,
                                'date':shifted_dates_2020['date_lag'+str(fh)],
                                'y_true_scaled':test_y.values,
                                'y_true':data_test.loc[:,'out_lag'+str(fh)],
                                'y_pred_scaled':y_med, # median
                                'ml_method':'GBR',
                               })
    
    
    # ---------------------------------------------------------------------
    # model 2: xgboost
    # ---------------------------------------------------------------------
    
    # build model
    bst = xgb.XGBRegressor(max_depth=4, min_child_weight = 3, learning_rate=0.05)
    # fit model
    bst.fit(train_x, train_y)
    # make predictions
    y_pred_xgb = bst.predict(test_x)
    # prep output
    pred_result_xgb = pd.DataFrame({'mun_code':data_test.mun_code,
                                'date':shifted_dates_2020['date_lag'+str(fh)],
                                'y_true_scaled':test_y.values,
                                'y_true':data_test.loc[:,'out_lag'+str(fh)],
                                'y_pred_scaled':y_pred_xgb,
                                'ml_method':'XGB',
                               })
    
    
    
    # ---------------------------------------------------------------------
    # model 3: RF
    # ---------------------------------------------------------------------
    
    # build
    rfm = RandomForestRegressor(n_estimators=150, max_features='auto', 
                                max_depth=4, min_samples_split=3)
    # fit
    rfm.fit(train_x, train_y)
    # predict
    y_pred_rf = rfm.predict(test_x)
    # prep output
    pred_result_rf = pd.DataFrame({'mun_code':data_test.mun_code,
                                'date':shifted_dates_2020['date_lag'+str(fh)],
                                'y_true_scaled':test_y.values,
                                'y_true':data_test.loc[:,'out_lag'+str(fh)],
                                'y_pred_scaled':y_pred_rf,
                                'ml_method':'RF',
                               })
    
    
    

    
    # ---------------------------------------------------------------------
    # combine output and rescale 
    # ---------------------------------------------------------------------
    
    out_temp = pd.concat((pred_result_gbr, pred_result_xgb, pred_result_rf))
    
    # rescale
    out_temp['y_pred'] = reverse_scale(out_temp.y_pred_scaled, val_min[fh], val_max[fh])

    # add fh
    out_temp['forecast_horizon'] = fh

    
    # append
    forecasts_2020 = pd.concat((forecasts_2020, out_temp))

forecasts_2020['forecast_horizon_rev'] = 9-forecasts_2020.forecast_horizon
  


# save
forecasts_2020.to_csv('itsa_results/predictions_2020_all_2024-06-22.csv')



# --- MAKE ENSEMBLE 

# ensemble top 3 
ensemble_top3_2020 = forecasts_2020[forecasts_2020.ml_method.isin(['XGB','GBR', 'RF'])].groupby(['mun_code', 'date', 'forecast_horizon', 'forecast_horizon_rev']).agg({'y_true_scaled':'mean', 'y_true':'mean', 'y_pred_scaled':'mean','y_pred':'mean'}).reset_index()
ensemble_top3_2020['ml_method'] = 'ensemble_top3'



# -----------------------------------------------------------------------------
# combine forecasts with ensembles
# -----------------------------------------------------------------------------

ensembles_combined_2020 = ensemble_top3_2020.loc[:,['mun_code', 'date', 'y_true_scaled', 
                                                    'y_true', 'y_pred_scaled','forecast_horizon_rev', 
                                                    'y_pred', 'ml_method', 'forecast_horizon']]


ensembles_combined_2020.reset_index(inplace=True, drop=True)
forecasts_2020.reset_index(inplace=True, drop=True)

# add identifier for individual model vs ensemble
forecasts_2020['is_ensemble'] = False
ensembles_combined_2020['is_ensemble'] = True


forecasts_with_ensembles_2020 = pd.concat((forecasts_2020, ensembles_combined_2020))

forecasts_with_ensembles_2020.date = pd.to_datetime(forecasts_with_ensembles_2020.date)

# save
forecasts_with_ensembles_2020.to_csv('itsa_results_predictions-with-ensembles_2020_2024-06-22.csv', index=False)

# -----------------------------------------------------------------------------
# plots
# -----------------------------------------------------------------------------



usedata = forecasts_with_ensembles_2020[forecasts_with_ensembles_2020.ml_method=='ensemble_top3']

# ---  forecasts in sequence

date_sequence = ['2020-03-07T00:00:00.000000000',
                 '2020-03-14T00:00:00.000000000', '2020-03-21T00:00:00.000000000',
       '2020-03-28T00:00:00.000000000', '2020-04-04T00:00:00.000000000',
       '2020-04-11T00:00:00.000000000', '2020-04-18T00:00:00.000000000',
       '2020-04-25T00:00:00.000000000', '2020-05-02T00:00:00.000000000',
       '2020-05-09T00:00:00.000000000']


pred_sequence = pd.DataFrame()

for s, state in enumerate(usedata.mun_code.unique()):
        
    temp1 = usedata[(usedata.mun_code==state)]

    for fh in np.arange(n_timesteps_out):
        # select the gap
        temp2 = temp1[(temp1.forecast_horizon_rev==fh)&
                       (usedata.date==date_sequence[fh])]
        # append
        pred_sequence = pd.concat((pred_sequence,temp2))
        

# plot
for s, state in enumerate(usedata.mun_code.unique()):
    fig, axs = plt.subplots(1,1, figsize=[10,5])
    # true
    sns.lineplot(x='date', y='out_lag0',color='black',
                 label='true',
                 data=data_1[(data_1.mun_code==state)&
                             (data_1.date>'2019-01-01')&
                             (data_1.date<'2020-05-10')],
                 ax=axs)
    sns.lineplot(x='date', y='y_pred', color=colors[1],
                 label='predicted', marker='o',
                data = pred_sequence[(pred_sequence.mun_code==state)], 
                ax=axs)
    plt.savefig(plotfolder+'top3_predictions-in-sequence_'+state+'.pdf', dpi=300)
    plt.close()





# --- lineplots with CI - individual

for s, state in enumerate(usedata.mun_code.unique()):
    bs_band_size = []
    for fh in np.arange(n_timesteps_out):
        # pick the band size corresponding to the method and state and fh
        temp = usedata[(usedata.mun_code==state)&
                       (usedata.forecast_horizon_rev==fh)]
        mm = temp.ml_method.unique()[0]
        # get band size
        bs_band_size_temp = bootstrap_measures[(bootstrap_measures.state==state)&
                                          (bootstrap_measures.ml_method==mm)&
                                          (bootstrap_measures.forecast_horizon_rev==fh)].bs_band_size.values[0]
        bs_band_size.append(bs_band_size_temp) 
    
    pred_sequence_sub = pred_sequence[(pred_sequence.mun_code==state)].sort_values('date', ascending=True)
    
    fig, axs = plt.subplots(1,1, figsize=(7,3))
    # true
    sns.lineplot(x='date', y='out_lag0',color='black',
                 label='true',
                 data=data_1[(data_1.mun_code==state)&
                             (data_1.date>'2019-01-01')&
                             (data_1.date<'2020-05-10')],
                 ax=axs)
    sns.lineplot(x='date', y='y_pred', color=colors[1],
                 label='predicted', marker='o',
                data = pred_sequence[(pred_sequence.mun_code==state)], 
                ax=axs)
    ci_lower = [np.max((x,0)) for x in pred_sequence_sub['y_pred']-bs_band_size]
    ci_upper = [np.max((x,0)) for x in pred_sequence_sub['y_pred']+bs_band_size]
    axs.fill_between(pred_sequence_sub.date, 
                     ci_lower, 
                     ci_upper, 
                     color=colors[1], alpha=.1)
    
    axs.set_ylabel('Cases')
    axs.set_xlabel('')
    axs.spines[['right', 'top']].set_visible(False)
    plt.savefig(plotfolder+'top3_lineplot_predictions-with-bootstrap95CI_state-'+state+'.pdf',
                dpi=300)
    plt.close()




# --- all states

state_order = ['Amazonas',  'Pará', 'Acre', 'Rondônia', 'Tocantins', # north (5)
               'Mato Grosso', 'Goiás', 'Distrito Federal', 'Mato Grosso do Sul', # center-west (4)
               'Minas Gerais', 'Espírito Santo', 'Rio de Janeiro', 'São Paulo', # southeast (4)
               'Maranhão', 'Ceará', 'Piauí', 'Rio Grande do Norte', 'Paraíba', 'Pernambuco', 'Alagoas', 'Sergipe', 'Bahia', # northeast (9)
                #'Roraima', 'Amapá', 
                'Paraná', 'Santa Catarina', 'Rio Grande do Sul', # south (3)
               ]

fig, axs = plt.subplots(nrows=13, ncols=2, figsize = (17,30), sharex=False)

for s, state in enumerate(state_order):
    if s<13:
        c = 0
        r = s
    else:
        c=1
        r = s-13
    
    bs_band_size = []
    for fh in np.arange(n_timesteps_out):
        # pick the band size corresponding to the method and state and fh
        temp = usedata[(usedata.mun_code==state)&
                       (usedata.forecast_horizon_rev==fh)]
        mm = temp.ml_method.unique()[0]
        # get band size
        bs_band_size_temp = bootstrap_measures[(bootstrap_measures.state==state)&
                                          (bootstrap_measures.ml_method==mm)&
                                          (bootstrap_measures.forecast_horizon_rev==fh)].bs_band_size.values[0]
        bs_band_size.append(bs_band_size_temp) 

    pred_sequence_sub = pred_sequence[(pred_sequence.mun_code==state)].sort_values('date', ascending=True)
    
    # true
    sns.lineplot(x='date', y='out_lag0',color='black',
                 label='true',
                 data=data_1[(data_1.mun_code==state)&
                             (data_1.date>'2019-01-01')&
                             (data_1.date<'2020-05-10')],
                 ax=axs[r,c])
    sns.lineplot(x='date', y='y_pred', color=colors[1],
                 label='predicted', marker='o',
                data = pred_sequence[(pred_sequence.mun_code==state)], 
                ax=axs[r,c])
    ci_lower = [np.max((x,0)) for x in pred_sequence_sub['y_pred']-bs_band_size]
    ci_upper = [np.max((x,0)) for x in pred_sequence_sub['y_pred']+bs_band_size]
    axs[r,c].fill_between(pred_sequence_sub.date, 
                     ci_lower, 
                     ci_upper, 
                     color=colors[1], alpha=.1)
    
    # plot interruption time
    axs[r,c].axvline(x=pd.to_datetime(interruption_time), 
       ymin=0, ymax=pred_sequence_sub.y_true.max(), 
       color='lightgray')
    
    axs[r,c].set_ylabel('Cases')
    axs[r,c].set_xlabel('')
    axs[r,c].spines[['right', 'top']].set_visible(False)
    axs[r,c].set_title(state)
    axs[r,c].legend(loc='upper left')
fig.delaxes(axs[12,1])
plt.tight_layout()

plt.savefig(plotfolder+'top3_lineplot_predictions-with-bootstrap95CI_all-states-2columns.pdf',
            dpi=300)
plt.close()
        


# --- 5 sample states

sample_states = ['Acre','Paraíba', 'São Paulo', 'Santa Catarina', 'Mato Grosso do Sul' ]

fig, axs = plt.subplots(nrows=5, ncols=1, figsize = (12,12))

for s, state in enumerate(sample_states):
    
    bs_band_size = []
    for fh in np.arange(n_timesteps_out):
        # pick the band size corresponding to the method and state and fh
        temp = usedata[(usedata.mun_code==state)&
                       (usedata.forecast_horizon_rev==fh)]
        mm = temp.ml_method.unique()[0]
        # get band size
        bs_band_size_temp = bootstrap_measures[(bootstrap_measures.state==state)&
                                          (bootstrap_measures.ml_method==mm)&
                                          (bootstrap_measures.forecast_horizon_rev==fh)].bs_band_size.values[0]
        bs_band_size.append(bs_band_size_temp) 

    pred_sequence_sub = pred_sequence[(pred_sequence.mun_code==state)].sort_values('date', ascending=True)
    
    # true
    sns.lineplot(x='date', y='out_lag0',color='black',
                 label='true',
                 data=data_1[(data_1.mun_code==state)&
                             (data_1.date>'2019-01-01')&
                             (data_1.date<'2020-05-10')],
                 ax=axs[s])
    sns.lineplot(x='date', y='y_pred', color=colors[1],
                 label='predicted', marker='o',
                data = pred_sequence[(pred_sequence.mun_code==state)], 
                ax=axs[s])
    ci_lower = [np.max((x,0)) for x in pred_sequence_sub['y_pred']-bs_band_size]
    ci_upper = [np.max((x,0)) for x in pred_sequence_sub['y_pred']+bs_band_size]
    axs[s].fill_between(pred_sequence_sub.date, 
                     ci_lower, 
                     ci_upper, 
                     color=colors[1], alpha=.1)
    # start of pandemic
    axs[s].axvline(x=pd.to_datetime(interruption_time), 
       ymin=0, ymax=pred_sequence_sub.y_true.max(), 
       color='lightgray')
    axs[s].set_ylabel('Cases')
    axs[s].set_xlabel('')
    axs[s].spines[['right', 'top']].set_visible(False)
    axs[s].set_title(state)
    axs[s].legend(loc='upper left')
plt.tight_layout()
plt.savefig(plotfolder+'top3_lineplot_predictions-with-bootstrap95CI_5-sample-states.pdf',
            dpi=300)
plt.close()





###############################################################################
### step 3) PREP FOR UNDERREPORTING ADJUSTMENT
###############################################################################

# -----------------------------------------------------------------------------
# aggregate total gap
# -----------------------------------------------------------------------------
itsa_agg = forecasts_with_ensembles_2020[(forecasts_with_ensembles_2020.ml_method=='ensemble_top3')&
                                         (forecasts_with_ensembles_2020.date<'2020-05-01')
                                         ].groupby(['mun_code'])



usedata = forecasts_with_ensembles_2020[forecasts_with_ensembles_2020.ml_method=='ensemble_top3']

# ---  forecasts in sequence

date_sequence = ['2020-03-07T00:00:00.000000000',
                 '2020-03-14T00:00:00.000000000', '2020-03-21T00:00:00.000000000',
       '2020-03-28T00:00:00.000000000', '2020-04-04T00:00:00.000000000',
       '2020-04-11T00:00:00.000000000', '2020-04-18T00:00:00.000000000',
       '2020-04-25T00:00:00.000000000']


pred_sequence = pd.DataFrame()

for s, state in enumerate(usedata.mun_code.unique()):
        
    temp1 = usedata[(usedata.mun_code==state)]

    for fh in np.arange(len(date_sequence)):
        # select the gap
        temp2 = temp1[(temp1.forecast_horizon_rev==fh)&
                       (usedata.date==date_sequence[fh])]
        
        # add CI
        
        bs_band_size = bootstrap_measures[(bootstrap_measures.state==state)&
                                          (bootstrap_measures.ml_method=='ensemble_top3')&
                                          (bootstrap_measures.forecast_horizon_rev==fh)].bs_band_size.values[0]
        temp2['bs_band_size'] = bs_band_size
        # append
        pred_sequence = pd.concat((pred_sequence,temp2))

pred_sequence.reset_index(inplace=True, drop=True)


pred_sequence['y_pred_lowerCI'] = pred_sequence.y_pred - pred_sequence.bs_band_size
pred_sequence['y_pred_upperCI'] = pred_sequence.y_pred + pred_sequence.bs_band_size

pred_sequence['y_pred_lowerCI'] = [np.max((x,0)) for x in pred_sequence['y_pred_lowerCI']]
pred_sequence['y_pred_upperCI'] = [np.max((x,0)) for x in pred_sequence['y_pred_upperCI']]


# agg
itsa_agg = pred_sequence.groupby(['mun_code']).agg('sum').reset_index()


# rename mun_code to state
itsa_agg.rename({'mun_code':'state'}, axis=1, inplace=True)

# save
itsa_agg.to_csv('itsa_results_2024-06-23/itsa_agg_top3.csv', index=False)
pred_sequence.to_csv('itsa_results_2024-06-23/pred_sequence_top3.csv', index=False)




