#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import pandas as pd

from scipy import stats
from sklearn.preprocessing import MinMaxScaler

# viz
import matplotlib.pyplot as plt
import seaborn as sns

# causal
from causalimpact import CausalImpact

# ML
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten
#from keras.layers import Dense

# maps
import geopandas as gpd
from mpl_toolkits.axes_grid1 import make_axes_locatable

# evaluation
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

# seasonality
import pywt

from datetime import timedelta
from epiweeks import Week


colors = ['#264653',# dark green
          '#2a9d8f', # medium green
          '#e9c46a', # yellow
          '#f4a261', #light orange
          '#e76f51', # darker orange
          '843b62', # light magenta
          '621940', # dark magenta
          ]

state_order = ['Roraima', 'Amapá', 'Amazonas',  'Pará', 'Acre', 'Rondônia', 'Tocantins', # north
               'Mato Grosso', 'Goiás', 'Distrito Federal', 'Mato Grosso do Sul', # center-west
               'Paraná', 'Santa Catarina', 'Rio Grande do Sul', # south
               'Maranhão', 'Ceará', 'Piauí', 'Rio Grande do Norte', 'Paraíba', 'Pernambuco', 'Alagoas', 'Sergipe', 'Bahia', # northeast
               'Minas Gerais', 'Espírito Santo', 'Rio de Janeiro', 'São Paulo', # southeast
               ]

# -----------------------------------------------------------------------------
# read data
# -----------------------------------------------------------------------------


# read conversions
conv_state = pd.read_csv('/Users/kirstinroster/Documents/PhD/mobility and covid-19/RELATORIO_DTB_BRASIL_MUNICIPIO.csv')
conv_region = pd.read_csv('/Users/kirstinroster/Documents/PhD/mobility and covid-19/regions-to-states-conversion.csv')
# merge
conv_state['mun_code'] = [int(str(x)[:-1]) for x in conv_state.loc[:,'Código Município Completo']]



# state level data with climate variables
dengue_uf_climate = pd.read_csv('dengue-climate-uf.csv')
# remove rows with missing values
dengue_uf_climate_nona = dengue_uf_climate.dropna(axis=0)





# -----------------------------------------------------------------------------
# run models - cross-validation
# -----------------------------------------------------------------------------



# 2019 predictions, same interruption points
# only dengue
interruption_times = ['2016-01-16', '2017-01-14', '2018-01-13', '2019-01-12']


bsts_cv = pd.DataFrame()


for i, y in enumerate([2016,2017,2018,2019]):
    
    interruption_time = interruption_times[i]
    
    for s,state in enumerate(state_order):

        sub = dengue_uf_climate_nona[dengue_uf_climate_nona.state==state]
        sub.sort_values('date', inplace=True)

        pre_period  = [ pd.Timestamp(sub.date.min()) , 
                       pd.Timestamp( pd.to_datetime(interruption_time) - timedelta(days= 7.0) ) ]
        post_period = [ pd.Timestamp(interruption_time) , pd.Timestamp(str(y)+'-06-29') ]

        sub.set_index('date', inplace=True)



        ci = CausalImpact(sub.loc[:,"cases"], 
                          pre_period, post_period, 
                          nseasons=[{'period': 52}],
                          prior_level_sd=None)


        # combine
        temp = ci.inferences
        temp['state'] = state
        temp['interruption_time'] = interruption_time
        temp['year'] = y
        bsts_cv = bsts_cv.append(temp)

        print(state+', interruption time: '+interruption_time)

        ci.plot()



bsts_cv.to_csv('itsa_results/bsts_cv.csv', index=True)


# -----------------------------------------------------------------------------
# run models - 2020
# -----------------------------------------------------------------------------


# interruption point: Jan 11th
interruption_times = ['2020-01-11', '2020-01-18', '2020-01-25',
                      '2020-02-01', '2020-02-08', '2020-02-15', '2020-02-22', '2020-02-29',
                      '2020-03-07', '2020-03-14']


bsts_2020 = pd.DataFrame()

for i, interruption_time in enumerate(interruption_times):
    for s,state in enumerate(state_order):


        sub = dengue_uf_climate[dengue_uf_climate.state==state]
        sub.sort_values('date', inplace=True)

        pre_period  = [ pd.Timestamp(sub.date.min()) , 
                       pd.Timestamp( pd.to_datetime(interruption_time) - timedelta(days= 7.0) ) ]
        post_period = [ pd.Timestamp(interruption_time) , pd.Timestamp('2020-06-27') ]

        sub.set_index('date', inplace=True)



        ci = CausalImpact(sub.loc[:,"cases"], 
                          pre_period, post_period, 
                          nseasons=[{'period': 52}],
                          prior_level_sd=None)


        # combine
        temp = ci.inferences
        temp['state'] = state
        temp['interruption_time'] = interruption_time
        bsts_2020 = bsts_2020.append(temp)

        print(state+', interruption time: '+interruption_time)

        ci.plot()


# save
bsts_2020.to_csv('itsa_results/predictions-UF_bsts.csv', index=True)


