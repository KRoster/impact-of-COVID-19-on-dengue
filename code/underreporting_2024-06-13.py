#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from parse_pt_month import parse_pt_month
from pandas.tseries.offsets import DateOffset


colors = ['#264653',# dark green - 0
          '#2a9d8f', # medium green - 1
          '#e9c46a', # yellow - 2
          '#f4a261', #light orange - 3
          '#e76f51', # darker orange - 4
          '#843b62', # light magenta - 5
          '#621940', # dark magenta - 6
          ]


state_order = ['Amazonas',  'Pará', 'Acre', 'Rondônia', 'Tocantins', # north (5)
               'Mato Grosso', 'Goiás', 'Distrito Federal', 'Mato Grosso do Sul', # center-west (4)
               'Minas Gerais', 'Espírito Santo', 'Rio de Janeiro', 'São Paulo', # southeast (4)
               'Maranhão', 'Ceará', 'Piauí', 'Rio Grande do Norte', 'Paraíba', 'Pernambuco', 'Alagoas', 'Sergipe', 'Bahia', # northeast (9)
                'Paraná', 'Santa Catarina', 'Rio Grande do Sul', # south (3)
               ]


# -----------------------------------------------------------------------------
# read data
# -----------------------------------------------------------------------------


# 0. conversions
conv_state = pd.read_csv('RELATORIO_DTB_BRASIL_MUNICIPIO.csv')
conv_state['mun_code'] = [int(str(x)[:-1]) for x in conv_state.loc[:,'Código Município Completo']]

conv_region = pd.read_csv('regions-to-states-conversion.csv')



# 1. time series - for lineplot
# itsa predictions
pred_sequence = pd.read_csv('itsa_results_2024-06-23/pred_sequence_top3.csv', parse_dates=['date'])
pred_sequence.rename({'mun_code':'state'}, axis=1, inplace=True)
# agg to monthly
pred_sequence['month'] = pred_sequence.date.dt.month
pred_sequence_agg = pred_sequence.groupby(['state','month']).agg({'y_true':'sum', 'y_pred':'sum'}).reset_index()

# underreporting adjustments
itsa_hiv = pd.read_csv('underreporting_results/itsa_hiv.csv', parse_dates=['date'])
itsa_elhosp = pd.read_csv('underreporting_results/itsa_elhosp.csv',
                         parse_dates=['date']).drop('Unnamed: 0', axis=1)


# 2. aggregated  - for bar plot

# itsa march+april
itsa_agg = pd.read_csv('itsa_results_2024-06-23/itsa_agg_top3.csv')

# underreporting march+april
underreporting_agg = pd.read_csv('underreporting_results/combined_data_underreporting+itsa_agg-march-to-april.csv')

# combine
itsa_under_agg = pd.merge(underreporting_agg.loc[:,['state', 'hiv_adjusted_cases', 'elhosp_adjusted_cases']],
                      itsa_agg.loc[:,['state', 'y_true', 'y_pred']],
                      on=['state'])


# 3. raw hiv and elhosp data 

hiv_treatments = pd.read_csv('data_underreporting/HIV-related-hospital-procedures.csv', encoding='latin1')
hiv_treatments.drop(['Total'], axis=1, inplace=True)
# reshape wide to long
hiv_treatments = hiv_treatments.melt(id_vars='Unidade da Federação', var_name='year_month', value_name='treatments')
# separate month and year
hiv_treatments['year'] = hiv_treatments.year_month.str.split('/', n=1, expand=True)[0].astype(int)
hiv_treatments['month'] = hiv_treatments.year_month.str.split('/', n=1, expand=True)[1]
month_converter = {'Jan':1, 'Fev':2, 'Mar':3, 'Abr':4, 'Mai':5, 'Jun':6, 'Jul':7, 'Ago':8, 'Set':9,
                   'Out':10, 'Nov':11, 'Dez':12}
hiv_treatments['month'].replace(month_converter, inplace=True)
# separate UF code and name
hiv_treatments['uf_code'] = hiv_treatments['Unidade da Federação'].str.split(' ', n=1, expand=True)[0]
hiv_treatments['state'] = hiv_treatments['Unidade da Federação'].str.split(' ', n=1, expand=True)[1]
# date variable
hiv_treatments['day'] = 1
hiv_treatments['date'] = pd.to_datetime(hiv_treatments.loc[:,['year','month','day']])
hiv_treatments.drop(['day'], axis=1, inplace=True)
# replace - with 0
hiv_treatments.treatments.replace({'-':0}, inplace=True)
hiv_treatments.treatments = hiv_treatments.treatments.astype(int)


# -

# read in the new data source (elective surgeries)
hospitals = pd.read_csv("datasus_elective internations.csv")

# drop "totals" column
hospitals.drop('Total', axis=1, inplace=True)

# reshape
hospitals_long = hospitals.melt(id_vars = 'Município', var_name = 'year_month', value_name = 'hospitalizations')

# replace "-" with 0
hospitals_long["hospitalizations"].replace({"-": "0"}, inplace=True)
# convert to numeric
hospitals_long['hospitalizations'] = hospitals_long.hospitalizations.str.replace('.', '').astype(float)
# replace NaN with 0
hospitals_long['hospitalizations'] = hospitals_long['hospitalizations'].fillna(value=0)


# split year and month into separate variables
hospitals_long[['year','month']] = hospitals_long['year_month'].str.split('/', expand=True)
# add a datetime variable
hospitals_long['date'] = parse_pt_month(hospitals_long.year_month, sep='/', year_position=0, month_position=1, day_position=None, is_long=False)
# replace month string with numeric
hospitals_long['month'] = hospitals_long.date.dt.month
# convert year to numeric
hospitals_long['year'] = hospitals_long.year.astype(float)

# split municipalities into name and code
hospitals_long[['mun_code','mun_name']] = hospitals_long['Município'].str.split(pat=' ', n=1, expand=True)
# convert mun_code to numeric
hospitals_long['mun_code'] = hospitals_long.mun_code.astype(float)

# add state & region conversions
hospitals_long = pd.merge(hospitals_long, conv_state.loc[:,['mun_code', 'Nome_UF', 'Nome_Mesorregião']],
                         on='mun_code', how='left')

# agg to state level
hospitals_state = hospitals_long.groupby(['Nome_UF', 'date']).agg({'hospitalizations':'sum'}).reset_index()
hospitals_state.rename({'Nome_UF':'state'}, axis=1, inplace=True)



# 4. raw dengue data
dengue_uf = pd.read_csv('sinan_uf_merged.csv', parse_dates=['date'])

# monthly dengue cases (to merge with monthly underreporting data)
dengue_monthly = dengue_uf.copy()
dengue_monthly['month'] = dengue_uf.date.dt.month
dengue_monthly = dengue_monthly.groupby(['state', 'month', 'year']).cases.agg('sum').reset_index()
# make new date variable
dengue_monthly['day'] = 1
dengue_monthly['date'] = pd.to_datetime(dengue_monthly.loc[:,['year','month','day']])
dengue_monthly.drop('day', axis=1, inplace=True)


# add date to pred sequence
pred_sequence_agg['year'] = 2020
pred_sequence_agg['day'] = 1
pred_sequence_agg['date'] = pd.to_datetime(pred_sequence_agg.loc[:,['day', 'month','year']])



# -----------------------------------------------------------------------------
# data prep
# -----------------------------------------------------------------------------

### HIV-adjusted (demand)

# merge the real values 
itsa_hiv_dengue = pd.merge(itsa_hiv, hiv_treatments.loc[:,['state', 'date', 'treatments']],
                   on=['state', 'date'], how='left')
# calculate point-wise percentage effect of itsa (percent of true treatments)
itsa_hiv_dengue['pointwise_percent'] = itsa_hiv_dengue.point_effects / itsa_hiv_dengue.treatments


# merge dengue and itsa results
hiv_adjusted_dengue = pd.merge(dengue_monthly, itsa_hiv_dengue.loc[:,['state','date', 'preds','treatments',
                                                               'pointwise_percent']], 
                               on=['state','date'], how='right')

# compute extra column with percent point-adjusted dengue cases
hiv_adjusted_dengue['hiv_adjusted_cases'] = hiv_adjusted_dengue.cases - (hiv_adjusted_dengue.pointwise_percent * hiv_adjusted_dengue.cases)


### Elective hospitalizations-adjusted (supply)


# merge the real values 
itsa_elhosp_dengue = pd.merge(itsa_elhosp, hospitals_state.loc[:,['state', 'date', 'hospitalizations']],
                   on=['state', 'date'], how='left')
# calculate point-wise percentage effect of itsa (percent of true treatments)
itsa_elhosp_dengue['pointwise_percent'] = itsa_elhosp_dengue.point_effects / itsa_elhosp_dengue.hospitalizations

# merge dengue and itsa results
elhosp_adjusted_dengue = pd.merge(dengue_monthly, itsa_elhosp_dengue.loc[:,['state','date', 'preds','hospitalizations',
                                                               'pointwise_percent']], 
                               on=['state','date'], how='right')

# compute extra column with percent point-adjusted dengue cases
elhosp_adjusted_dengue['elhosp_adjusted_cases'] = elhosp_adjusted_dengue.cases - (elhosp_adjusted_dengue.pointwise_percent * elhosp_adjusted_dengue.cases)


# EXCLUDE STATES WITH OBSERVED > EXPECTED

exclude_states = itsa_agg[itsa_agg.y_true>itsa_agg.y_pred].state.values


state_order_sub = [x for x in state_order if x not in exclude_states]




# adjust dates so that plotting end of month
elhosp_adjusted_dengue['date_offset'] = elhosp_adjusted_dengue.date + DateOffset(days=27)
hiv_adjusted_dengue['date_offset'] = hiv_adjusted_dengue.date + DateOffset(days=27)
dengue_monthly['date_offset'] = dengue_monthly.date + DateOffset(days=27)
pred_sequence_agg['date_offset'] = pred_sequence_agg.date + DateOffset(days=27)




# -----------------------------------------------------------------------------
# plots
# -----------------------------------------------------------------------------

## all states, but in two columns

# excluding Roraima and Amapa (no itsa) and states with observed > expected



fig, axs = plt.subplots(nrows=10, ncols=2, figsize = (17,22), sharex=False)

for s, state in enumerate(state_order_sub):
    if s<10:
        c = 0
    else:
        c=1
        s = s-10
        
        
    # subset states and date range
    sub1 = elhosp_adjusted_dengue[(elhosp_adjusted_dengue.state==state)&
                                  (elhosp_adjusted_dengue.date>pd.to_datetime('2019-01-01'))&
                                  (elhosp_adjusted_dengue.date<pd.to_datetime('2020-04-02'))
                                 ]
    sub2 = hiv_adjusted_dengue[(hiv_adjusted_dengue.state==state)&
                                (hiv_adjusted_dengue.date>pd.to_datetime('2019-01-01'))&
                                (hiv_adjusted_dengue.date<pd.to_datetime('2020-04-02'))
                              ]
    sub3 = pred_sequence_agg[(pred_sequence_agg.state==state)
                         ]
    sub4 = dengue_monthly[(dengue_monthly.state==state)&
                            (dengue_monthly.date>pd.to_datetime('2019-01-01'))&
                            (dengue_monthly.date<pd.to_datetime('2020-04-02'))
                         ]
    
    
    # observed
    sns.lineplot(x='date_offset', y='cases',
                 data = sub4, 
                 ax=axs[s,c],
                 color='gray',
                 label='observed dengue',
                 linewidth=4
                )
    

    # itsa predictions
    sns.lineplot(x='date_offset', y='y_pred',
                data = sub3, 
                 color=colors[1],
                 label='predicted', marker='o',
                 linewidth=4,
                ax=axs[s,c])
    
    # el. hosp. adjusted
    sns.lineplot(x='date_offset', y='elhosp_adjusted_cases',
                 data = sub1, 
                 ax=axs[s,c],
                 color=colors[4],
                 label='El. Hosp. adjusted dengue' ,
                 linewidth=2,
                 marker = 'o', 
                )

    # hiv adjusted
    sns.lineplot(x='date_offset', y='hiv_adjusted_cases',
                 data = sub2, 
                 ax=axs[s,c],
                 color=colors[5],
                 label='HIV-adjusted dengue',
                 linewidth=2,
                 marker = 'o', 
                )

    axs[s,c].axvline(pd.to_datetime('2020-03-01'), color='black')
    axs[s,c].set_xlabel('')
    axs[s,c].set_title(state)
    axs[s,c].set_xlim(left=pd.to_datetime('2019-01-01'), right=pd.to_datetime('2020-05-02'))
    
    axs[s,c].spines.top.set_visible(False)
    axs[s,c].spines.right.set_visible(False)
    
    axs[s,c].legend(bbox_to_anchor=(1.1, 1.05))
    
    # keep legend only in a single subplot
    if ((s!=0) or (c!=1)):
        axs[s,c].legend().remove()

fig.delaxes(axs[9,1])
    
plt.tight_layout()

plt.savefig('underreporting_results_2024-06-23/lineplots_reporting-adjusted_all-states_2-columns.pdf', dpi=300)




## 5 example states --> 4 states (exclude state with observed > expected)
fig, axs = plt.subplots(4,1,figsize=(12,9))

sample_states = ['Acre','Paraíba', 'São Paulo', #'Santa Catarina',
                 'Mato Grosso do Sul' ]



for s,state in enumerate(sample_states):
    # subset states and date range
    sub1 = elhosp_adjusted_dengue[(elhosp_adjusted_dengue.state==state)&
                                  (elhosp_adjusted_dengue.date>pd.to_datetime('2019-01-01'))&
                                  (elhosp_adjusted_dengue.date<pd.to_datetime('2020-04-02'))
                                 ]
    sub2 = hiv_adjusted_dengue[(hiv_adjusted_dengue.state==state)&
                                (hiv_adjusted_dengue.date>pd.to_datetime('2019-01-01'))&
                                (hiv_adjusted_dengue.date<pd.to_datetime('2020-04-02'))
                              ]
    sub3 = pred_sequence_agg[(pred_sequence_agg.state==state)
                         ]
    sub4 = dengue_monthly[(dengue_monthly.state==state)&
                            (dengue_monthly.date>pd.to_datetime('2019-01-01'))&
                            (dengue_monthly.date<pd.to_datetime('2020-04-02'))
                         ]
    
    
    # observed
    sns.lineplot(x='date_offset', y='cases',
                 data = sub4, 
                 ax=axs[s],
                 color='gray',
                 label='observed dengue',
                 linewidth=4
                )
    
    
    # itsa predictions
    sns.lineplot(x='date_offset', y='y_pred',
                data = sub3, 
                 color=colors[1],
                 label='predicted', marker='o',
                 linewidth=4,
                ax=axs[s])
    
    # el. hosp. adjusted
    sns.lineplot(x='date_offset', y='elhosp_adjusted_cases',
                 data = sub1, 
                 ax=axs[s],
                 color=colors[4],
                 label='El. Hosp. adjusted dengue' ,
                 linewidth=2,
                 marker = 'o', 
                )

    # hiv adjusted
    sns.lineplot(x='date_offset', y='hiv_adjusted_cases',
                 data = sub2, 
                 ax=axs[s],
                 color=colors[5],
                 label='HIV-adjusted dengue',
                 linewidth=2,
                 marker = 'o', 
                )
    
    axs[s].axvline(pd.to_datetime('2020-03-01'), color='black')
    
    axs[s].set_title(state)
    axs[s].set_xlim(left=pd.to_datetime('2019-01-01'), right=pd.to_datetime('2020-05-02'))
    axs[s].set_xlabel('')
    axs[s].spines.top.set_visible(False)
    axs[s].spines.right.set_visible(False)
    
    axs[s].legend(bbox_to_anchor=(1.1, 1.05))
    if s>0:
        axs[s].legend().remove()
    
plt.tight_layout()

plt.savefig('underreporting_results_2024-06-23/lineplots_reporting-adjusted_5-sample-states.pdf', dpi=300)



### -- data prep for bar plot

interruption_time = '2020-02-29'
cumulative_all = elhosp_adjusted_dengue[(elhosp_adjusted_dengue.date>=interruption_time)&
                                        (elhosp_adjusted_dengue.date<'2020-05-01')].loc[:,['state','cases','date','elhosp_adjusted_cases']]
cumulative_all = pd.merge(cumulative_all, hiv_adjusted_dengue[(hiv_adjusted_dengue.date>=interruption_time)&
                                                              (hiv_adjusted_dengue.date<'2020-05-01')].loc[:,['state','date','hiv_adjusted_cases']],
                         on=['state','date'],
                         how='outer')
cumulative_all = pd.merge(cumulative_all, pred_sequence_agg.loc[:,['state','date',
                                                                'y_pred']],
                          on=['state','date'],
                          how='outer')
                          
cumulative_all_agg = cumulative_all.groupby(['state']).agg('sum')

# only march & april
cumulative_all_agg_marchapril = cumulative_all[cumulative_all.date<'2020-05-01'].groupby(['state']).agg('sum')

# compute the cumulative gap
cumulative_all_agg_marchapril['observed_predicted_gap'] = cumulative_all_agg_marchapril.cases - cumulative_all_agg_marchapril.y_pred

# gaps between adjusted and predicted
cumulative_all_agg_marchapril['observed_hiv_adjusted_gap'] = cumulative_all_agg_marchapril.hiv_adjusted_cases - cumulative_all_agg_marchapril.y_pred
cumulative_all_agg_marchapril['observed_elhosp_adjusted_gap'] = cumulative_all_agg_marchapril.elhosp_adjusted_cases - cumulative_all_agg_marchapril.y_pred

# compute remainder, i.e. after percent explained is removed
cumulative_all_agg_marchapril['remainder_hiv'] = cumulative_all_agg_marchapril.observed_predicted_gap - cumulative_all_agg_marchapril.observed_hiv_adjusted_gap
cumulative_all_agg_marchapril['remainder_elhosp'] = cumulative_all_agg_marchapril.observed_predicted_gap - cumulative_all_agg_marchapril.observed_elhosp_adjusted_gap

# sort by states
cumulative_all_agg_marchapril = cumulative_all_agg_marchapril.loc[state_order,:]


# percentages
# where observed-predicted gap is 100% (neg if observed)
# note: absolute value to keep the same sign
cumulative_all_agg_marchapril['observed_predicted_gap_percent'] = cumulative_all_agg_marchapril.observed_predicted_gap / np.abs(cumulative_all_agg_marchapril.observed_predicted_gap)
cumulative_all_agg_marchapril['observed_hiv_adjusted_gap_percent'] = cumulative_all_agg_marchapril.observed_hiv_adjusted_gap / np.abs(cumulative_all_agg_marchapril.observed_predicted_gap)
cumulative_all_agg_marchapril['observed_elhosp_adjusted_gap_percent'] = cumulative_all_agg_marchapril.observed_elhosp_adjusted_gap / np.abs(cumulative_all_agg_marchapril.observed_predicted_gap)


# melt for paired barplot
temp = cumulative_all_agg_marchapril.reset_index().melt(id_vars='state', 
                                             value_vars=['observed_hiv_adjusted_gap', 
                                                         'observed_elhosp_adjusted_gap'], 
                                            var_name='variable')

# melt for paired barplot
temp2 = cumulative_all_agg_marchapril.reset_index().melt(id_vars='state', 
                                             value_vars=['observed_hiv_adjusted_gap_percent', 
                                                         'observed_elhosp_adjusted_gap_percent'], 
                                            var_name='variable')



# march & april
cumulative_all_agg_marchapril['cases_percent'] = cumulative_all_agg_marchapril.cases / cumulative_all_agg_marchapril.cases
cumulative_all_agg_marchapril['elhosp_adjusted_cases_percent'] = cumulative_all_agg_marchapril.elhosp_adjusted_cases / cumulative_all_agg_marchapril.cases
cumulative_all_agg_marchapril['hiv_adjusted_cases_percent'] = cumulative_all_agg_marchapril.hiv_adjusted_cases / cumulative_all_agg_marchapril.cases
cumulative_all_agg_marchapril['ensemble_average_percent'] = cumulative_all_agg_marchapril.y_pred / cumulative_all_agg_marchapril.cases




# melt for paired barplot
temp2 = cumulative_all_agg_marchapril.reset_index().melt(id_vars='state', 
                                             value_vars=['observed_hiv_adjusted_gap_percent', 
                                                         'observed_elhosp_adjusted_gap_percent'], 
                                            var_name='variable')

# don't want to show the positive l
temp3 = temp2.copy()
t = temp3.value.copy()
t[t>0] = 0
temp3.value = t
# rename variables
temp3.replace({'observed_hiv_adjusted_gap_percent':'HIV-related treatments',
               'observed_elhosp_adjusted_gap_percent':'Elective internations'}, inplace=True)
# merge with state data


# want to show ONLY the positive
temp4 = temp2.copy()
t = temp4.value.copy()
t[t<=0] = 0
temp4.value = t
# rename variables
temp4.replace({'observed_hiv_adjusted_gap_percent':'HIV-related treatments',
               'observed_elhosp_adjusted_gap_percent':'Elective internations'}, inplace=True)



# make a single dataframe

groupdf = cumulative_all_agg_marchapril.copy().reset_index()

# first, drop the 2 incomplete states Roraima and Amapa
groupdf = groupdf[(groupdf.state.isin(state_order_sub))==True]
# first consider group 4), where adjustment lowers observed cases (e.g. HIV adjustment in MGdS)
# conditions: adjustment < observed
groupdf['group_4'] = (  ((groupdf.cases > groupdf.hiv_adjusted_cases ) | 
                           (groupdf.cases > groupdf.elhosp_adjusted_cases)) ).astype(int)
# then exclude these from the other groups

t = groupdf.observed_hiv_adjusted_gap_percent.copy()
t[(groupdf.cases > groupdf.hiv_adjusted_cases)] = 0

groupdf['observed_hiv_adjusted_gap_percent_excl_group_4'] = t

t = groupdf.observed_elhosp_adjusted_gap_percent.copy()
t[(groupdf.cases > groupdf.elhosp_adjusted_cases)] = 0

groupdf['observed_elhosp_adjusted_gap_percent_excl_group_4'] = t


# group 1 indicator
# conditions: observed < predicted AND one of the two adjusted between observed and predicted (i.e. neg percent)
# observed < adjusted < expected
'''
groupdf['group_1'] = ( (groupdf.observed_predicted_gap_percent<=0) &
                         ((groupdf.observed_hiv_adjusted_gap_percent<=0) | 
                           (groupdf.observed_elhosp_adjusted_gap_percent<=0)) ).astype(int)
'''
# alt
groupdf['group_1'] = ( (groupdf.cases <= groupdf.y_pred) &
                         ((groupdf.hiv_adjusted_cases > groupdf.cases) | 
                           (groupdf.elhosp_adjusted_cases > groupdf.cases)) ).astype(int)




# group 2 indicator
# conditions: observed < predicted AND one of the two adjusted above predicted (i.e. neg percent)
# observed < expected < adjusted
'''
groupdf['group_2'] = ( (groupdf.observed_predicted_gap_percent<=0) & 
                         ((groupdf.observed_hiv_adjusted_gap_percent>0) | 
                          (groupdf.observed_elhosp_adjusted_gap_percent>0)) ).astype(int)
'''

groupdf['group_2'] = ( (groupdf.cases <= groupdf.y_pred) &
                         ((groupdf.hiv_adjusted_cases > groupdf.y_pred) | 
                           (groupdf.elhosp_adjusted_cases > groupdf.y_pred)) ).astype(int)


# group 3 indicator
# conditions: observed > predicted 
# expected < observed < adjusted
groupdf['group_3'] = ( (groupdf.cases > groupdf.y_pred) &
                         ((groupdf.hiv_adjusted_cases > groupdf.cases) | 
                           (groupdf.elhosp_adjusted_cases > groupdf.cases)) ).astype(int)


# add region
groupdf = groupdf.merge(conv_region, on='state', how='left')


# group 4 indicator 
# % adjustment < observed
groupdf['group_4_hiv'] = ( groupdf.hiv_adjusted_cases < groupdf.cases ).astype(int)
groupdf['group_4_elhosp'] = ( groupdf.elhosp_adjusted_cases < groupdf.cases ).astype(int)



# compute the required value for each group
# group 1: percent of gap explained (0<value<1)
groupdf['value_group1_hiv'] = (groupdf.hiv_adjusted_cases - groupdf.cases) / (groupdf.y_pred - groupdf.cases)
groupdf['value_group1_elhosp'] = (groupdf.elhosp_adjusted_cases - groupdf.cases) / (groupdf.y_pred - groupdf.cases)
# group 2: same as group 1, but value > 1
groupdf['value_group2_hiv'] = groupdf.value_group1_hiv.copy()
groupdf['value_group2_elhosp'] = groupdf.value_group1_elhosp.copy()
# group 3: % increase in gap (relative to gap)
groupdf['value_group3_hiv'] = (groupdf.hiv_adjusted_cases - groupdf.y_pred) / (groupdf.cases - groupdf.y_pred)
groupdf['value_group3_elhosp'] = (groupdf.elhosp_adjusted_cases - groupdf.y_pred) / (groupdf.cases - groupdf.y_pred)



# multiply value by group (zero if not in group)
groupdf['value_group1_hiv'] = groupdf.value_group1_hiv * groupdf.group_1
groupdf['value_group1_elhosp'] = groupdf.value_group1_elhosp * groupdf.group_1
groupdf['value_group2_hiv'] = groupdf.value_group2_hiv * groupdf.group_2
groupdf['value_group2_elhosp'] = groupdf.value_group2_elhosp * groupdf.group_2
groupdf['value_group3_hiv'] = groupdf.value_group3_hiv * groupdf.group_3
groupdf['value_group3_elhosp'] = groupdf.value_group3_elhosp * groupdf.group_3



# remove group 4 entries  (set to zero)
groupdf['value_group1_hiv'] = groupdf.value_group1_hiv * (1-groupdf.group_4_hiv)
groupdf['value_group1_elhosp'] = groupdf.value_group1_elhosp * (1-groupdf.group_4_elhosp)
groupdf['value_group2_hiv'] = groupdf.value_group2_hiv * (1-groupdf.group_4_hiv)
groupdf['value_group2_elhosp'] = groupdf.value_group2_elhosp * (1-groupdf.group_4_elhosp)
groupdf['value_group3_hiv'] = groupdf.value_group3_hiv * (1-groupdf.group_4_hiv)
groupdf['value_group3_elhosp'] = groupdf.value_group3_elhosp * (1-groupdf.group_4_elhosp)



# melt 

### group 1
groupdf_melted_g1 = groupdf.reset_index().melt(id_vars='state', 
                                             value_vars=['value_group1_hiv', 
                                                         'value_group1_elhosp'], 
                                            var_name='variable')
groupdf_melted_g1.replace({'value_group1_hiv':'HIV-related treatments',
                        'value_group1_elhosp':'Elective internations'}, inplace=True)

# replace the values greater than 1 with exactly 1
groupdf_melted_g1.loc[(groupdf_melted_g1.value > 1),'value'] = 1


### group 2
groupdf_melted_g2 = groupdf.reset_index().melt(id_vars='state', 
                                             value_vars=['value_group2_hiv', 
                                                         'value_group2_elhosp'], 
                                            var_name='variable')
groupdf_melted_g2.replace({'value_group2_hiv':'HIV-related treatments',
                        'value_group2_elhosp':'Elective internations'}, inplace=True)

# remove values <1 (i.e. group 1)
groupdf_melted_g2.loc[(groupdf_melted_g2.value < 1), 'value'] = 0


### group 3
groupdf_melted_g3 = groupdf.reset_index().melt(id_vars='state', 
                                             value_vars=['value_group3_hiv', 
                                                         'value_group3_elhosp'], 
                                            var_name='variable')
groupdf_melted_g3.replace({'value_group3_hiv':'HIV-related treatments',
                           'value_group3_elhosp':'Elective internations'}, inplace=True)




# PLOT

subplot_titles = ['Region', 
                  'Observed < Expected: \n Adjustment partially \n explains gap',
                  'Observed < Expected: \n Adjustment over-corrects \n gap',
                  #'Observed > Expected: \n Adjustment increases \n gap'
                 ]

ncols = 3
fig, axs = plt.subplots(nrows=1, ncols=ncols, 
                        figsize=(12,10), 
                        sharey=True,
                       gridspec_kw={'width_ratios': [1, 4,3]})

# add a color block for the region
sns.scatterplot(x=[1]*groupdf.shape[0], y=groupdf.state, 
                hue=groupdf.region, palette=[colors[0]] + colors[2:6],
                marker='s', s=100,
                ax=axs[0]
               )
# group 1: partial explanation of gap
sns.barplot(x='group_1', y='state', 
            data=groupdf,
            ax=axs[1], color='white', edgecolor='black',
            linewidth=1.5
           )
sns.barplot(x='value', y='state', 
            hue='variable',
            data=groupdf_melted_g1,
            ax=axs[1], 
            palette=[colors[4], colors[5]], edgecolor=None, alpha=1,
           )

# group 2: overcompensation
sns.barplot(x='value', y='state', 
            hue='variable',
            data=groupdf_melted_g2,
            ax=axs[2], 
            palette=[colors[4], colors[5]], edgecolor=None, alpha=1,
           )

"""
# group 3: predicted < observed
sns.barplot(x='value', y='state', 
            hue='variable',
            data=groupdf_melted_g3,
            ax=axs[3], 
            palette=[colors[1], colors[0]], edgecolor=None, alpha=0.7,
           )
"""
# in all subplots
for i in np.arange(ncols):
    axs[i].spines.top.set_visible(False)
    axs[i].spines.right.set_visible(False)
    #axs[i].spines.left.set_visible(False)
    
    axs[i].set_ylabel('')
    axs[i].set_xlabel('Share of gap', fontsize=12)

    axs[i].tick_params(axis='y', which='both', labelsize=12, 
                width=0, length=0)
    axs[i].spines.left.set_linewidth(2)
    axs[i].spines.bottom.set_linewidth(2)
    axs[i].tick_params(axis='x',width=2, length=4)
    axs[i].tick_params(axis='y',width=0, length=0)
    
    # title
    axs[i].set_title(subplot_titles[i], fontsize=12)
    
# in some subplots
  
# legends
axs[2].legend(bbox_to_anchor=(1.1, 0.8), #labels=['HIV-related treatments','Elective internations'],
               title='Adjustment', fontsize=12, title_fontsize=12,
              )
#axs[2].legend().remove()
axs[1].legend().remove()
axs[0].legend().remove()

# axis
axs[0].spines.left.set_visible(False)
axs[0].spines.bottom.set_visible(False)
axs[1].spines.left.set_visible(False)
axs[1].spines.bottom.set_visible(False)

# ticks and labels
axs[0].tick_params(width=0, length=0, labelsize=15)
axs[0].set_xticklabels('')
axs[1].set_xlim(-0.15,1.15)

# x labels
axs[0].set_xlabel('')


plt.tight_layout()
plt.savefig('underreporting_results_2024-06-23/combined_barplots_groups1-3.pdf', dpi=300)



# save the data
groupdf_melted_g1['group_type'] = 'group 1'
groupdf_melted_g2['group_type'] = 'group 2'
groupdf_melted_g3['group_type'] = 'group 3'

groupdf_melted_combo = pd.concat((groupdf_melted_g1, groupdf_melted_g2, groupdf_melted_g3))

# save
groupdf_melted_combo.to_csv('underreporting_results_2024-06-23/combined_data_underreporting_groups.csv', index=False)



