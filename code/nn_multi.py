#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""


import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Flatten
from datetime import timedelta



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
    pred_rev = ( (pred_data+1)/2 * (val_max - val_min)) + val_min
    
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

    
    # drop data before min_date
    if min_date!=None:
        data_lags = data_lags[data_lags.date>=min_date]

    return data_lags





# -----------------------------------------------------------------------------
# read data
# -----------------------------------------------------------------------------



# read dengue data
data = pd.read_csv('sinan_uf_merged.csv', parse_dates=['date'])

# read covariates
dengue_climate = pd.read_csv('data for ps matching_all Brazil_2021-09-10.csv', parse_dates=['date'])


# read conversions
conv_state = pd.read_csv('/Users/kirstinroster/Documents/PhD/mobility and covid-19/RELATORIO_DTB_BRASIL_MUNICIPIO.csv')
conv_region = pd.read_csv('/Users/kirstinroster/Documents/PhD/mobility and covid-19/regions-to-states-conversion.csv')
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

# -----------------------------------------------------------------------------
# loop over the interruption times
# -----------------------------------------------------------------------------

n_features = 7
n_timesteps_in = 12
n_timesteps_out = 10



interruption_time = '2020-02-29'



# -----------------------------------------------------------------------------
# data prep for NN model
# -----------------------------------------------------------------------------

# data prep
data_0 = dataprep_multiout(use_data.loc[:,['date','mun_code','cases', 
                                                 'precipitacao',
                                                 'tempmaxima', 'temp comp media', 'tempminima',
                                                 'umidade relativa media', 'velocidade do vento media']],
                            max_lags = 12,
                            length_out = n_timesteps_out
                            )

# reshuffle randomly
data_0 = data_0.sample(frac=1, random_state=12).reset_index(drop=True)

# drop rows with missing values
data_0.dropna(inplace=True)


# split into train and test sets
data_train = data_0[data_0.date<interruption_time].copy()
# because multi-output, need to add max output lags, 
#    so that earliest prediction is after interruption time
data_test = data_0[data_0.date >= ( pd.to_datetime(interruption_time) + timedelta(days= 7.0 * n_timesteps_out ) ) ].copy()


# rescale 0-1
train_scaled, test_scaled, val_min, val_max = scale(np.array(data_train.iloc[:,2:]), 
                                                    np.array(data_test.iloc[:,2:]))
    

## reshape data for NN

# split into x and y
train_x_temp = train_scaled[:, n_timesteps_out:]
train_y_temp = train_scaled[:, :n_timesteps_out]
test_x_temp = test_scaled[:, n_timesteps_out:]
test_y_temp = test_scaled[:, :n_timesteps_out]


# reshape input into 3d cube. dimensions: samples, time steps, features
train_x = np.array(train_x_temp).reshape((train_x_temp.shape[0], n_timesteps_in, n_features))
test_x = np.array(test_x_temp).reshape((test_x_temp.shape[0], n_timesteps_in, n_features))

# reshape output to 2d. dimensions: samples, time steps
train_y = np.array(train_y_temp).reshape((train_y_temp.shape[0], n_timesteps_out))
test_y = np.array(test_y_temp).reshape((test_y_temp.shape[0], n_timesteps_out))



# -----------------------------------------------------------------------------
# training (up to interruption point)
# -----------------------------------------------------------------------------


# run the NN model

verbose = 1
epochs = 300
batch_size = 16


dense = Sequential()
dense.add(Flatten())
dense.add(Dense(128, activation='relu', input_shape=(n_timesteps_in, n_features)))
dense.add(Dense(128, activation='relu'))
dense.add(Dense(n_timesteps_out))

dense.compile(loss='mse', optimizer='adam')
# fit network
dense.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)



# -----------------------------------------------------------------------------
# predictions (after interruption point)
# -----------------------------------------------------------------------------

# predictions on test set
pred_nn_multi = dense.predict(test_x, verbose=verbose)

# predictions on train set
val_nn_multi = dense.predict(train_x, verbose=verbose)


# -----------------------------------------------------------------------------
# set up output dataframe
# -----------------------------------------------------------------------------

# ----- test data -----

out_col_names = ['out_lag'+str(x) for x in np.arange(n_timesteps_out)]
pred_col_names = ['pred_lag'+str(x) for x in np.arange(n_timesteps_out)]

# mun_code, date
out_nn_multi = data_test.iloc[:,:2]
out_nn_multi.reset_index(inplace=True, drop=True)

# predicted and true values
out_nn_multi = out_nn_multi.join(pd.DataFrame(test_y, columns=out_col_names))
out_nn_multi = out_nn_multi.join(pd.DataFrame(pred_nn_multi, columns=pred_col_names))

# real y values
temp = data_test.iloc[:,2:2+n_timesteps_out].reset_index(drop=True)
temp.columns = [x+'_real' for x in out_col_names]
out_nn_multi = out_nn_multi.join(temp)

# rescale to append real yhat values
temp2 = reverse_scale(pred_nn_multi, val_min[:n_timesteps_out], val_max[:n_timesteps_out])
out_nn_multi = out_nn_multi.join(pd.DataFrame(temp2[:,-n_timesteps_out:], 
                                                columns = [x+'_real' for x in pred_col_names]))


# ----- training data -----

# mun_code, date
out_nn_multi_val = data_train.iloc[:,:2]
out_nn_multi_val.reset_index(inplace=True, drop=True)

out_nn_multi_val = out_nn_multi_val.join(pd.DataFrame(train_y, columns=out_col_names))
out_nn_multi_val = out_nn_multi_val.join(pd.DataFrame(val_nn_multi, columns=pred_col_names))

# append un-scaled real y values
temp = data_train.iloc[:,2:2+n_timesteps_out].reset_index(drop=True)
temp.columns = [x+'_real' for x in out_col_names]
out_nn_multi_val = out_nn_multi_val.join(temp)

# rescale
temp2 = reverse_scale(val_nn_multi, val_min[:n_timesteps_out], val_max[:n_timesteps_out])
out_nn_multi_val = out_nn_multi_val.join(pd.DataFrame(temp2[:,-n_timesteps_out:], 
                                                        columns = [x+'_real' for x in pred_col_names]))


# ----- combine train and test output dataframes -----

out_nn_multi_val['train_or_test'] = 'train'
out_nn_multi['train_or_test'] = 'test'
out = out_nn_multi_val.append(out_nn_multi)


# -----------------------------------------------------------------------------
# save output
# -----------------------------------------------------------------------------

out.to_csv('itsa_results/predictions-UF_nn-multi_interruption-time-'+interruption_time+'.csv', index=False)


