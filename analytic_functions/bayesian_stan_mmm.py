import warnings
warnings.filterwarnings("ignore")
import pystan
from mlxtend.feature_selection import SequentialFeatureSelector
from mlxtend.feature_selection import ExhaustiveFeatureSelector
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae
import scipy.stats.kde as kde
import numpy as np
import pandas as pd
import math
import random
import json
import pickle
from hpd import hpd_grid
import sys
import time
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import arviz as az
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sns.color_palette("husl")
sns.set_style('darkgrid')


# import plotly.io as pio
# pio.renderers.default = "browser"


def extract_mmm_model(fit_result, dep_var, media_vars=[], comp_media_vars=[], positive_vars=[], neutral_vars=[], negative_vars=[], extract_param_list=True):

    mmm_model_output = {
        'dep_var': dep_var,
        'media_vars': media_vars,
        'comp_media_vars': comp_media_vars,
        'positive_vars': positive_vars,
        'neutral_vars': neutral_vars,
        'negative_vars': negative_vars,
        'tau_mean': fit_result['tau'].mean(),
        'beta_mean': fit_result['beta'].mean(axis=0).tolist() if len(media_vars) > 0 else [],
        'beta1_mean': fit_result['beta1'].mean(axis=0).tolist() if len(positive_vars) > 0 else [],
        'beta2_mean': fit_result['beta2'].mean(axis=0).tolist() if len(neutral_vars) > 0 else [],
        'beta3_mean': fit_result['beta3'].mean(axis=0).tolist() if len(negative_vars) > 0 else [],
        'decay_mean': fit_result['decay'].mean(axis=0).tolist() if len(media_vars) > 0 else [],
        'alpha_mean': fit_result['alpha'].mean(axis=0).tolist() if len(media_vars) > 0 else [],
        'beta_comp_mean': fit_result['beta_comp'].mean(axis=0).tolist() if len(comp_media_vars) > 0 else [],
        'decay_comp_mean': fit_result['decay_comp'].mean(axis=0).tolist() if len(comp_media_vars) > 0 else [],
        'alpha_comp_mean': fit_result['alpha_comp'].mean(axis=0).tolist() if len(comp_media_vars) > 0 else []
    }
    if extract_param_list:
        mmm_model_output['tau'] = fit_result['tau'].tolist()
        mmm_model_output['beta1'] = fit_result['beta1'].tolist() if len(positive_vars) > 0 else []
        mmm_model_output['beta2'] = fit_result['beta2'].tolist() if len(neutral_vars) > 0 else []
        mmm_model_output['beta3'] = fit_result['beta3'].tolist() if len(negative_vars) > 0 else []
        mmm_model_output['beta'] = fit_result['beta'].tolist() if len(media_vars) > 0 else []
        mmm_model_output['decay'] = fit_result['decay'].tolist() if len(media_vars) > 0 else []
        mmm_model_output['alpha'] = fit_result['alpha'].tolist() if len(media_vars) > 0 else []
        mmm_model_output['beta_comp'] = fit_result['beta_comp'].tolist() if len(comp_media_vars) > 0 else []
        mmm_model_output['decay_comp'] = fit_result['decay_comp'].tolist() if len(comp_media_vars) > 0 else []
        mmm_model_output['alpha_comp'] = fit_result['alpha_comp'].tolist() if len(comp_media_vars) > 0 else []
        mmm_model_output['predicted_y'] = fit_result['predicted_y'].tolist() if 'predicted_y' in fit_result.keys() else []
    media_names = [shorten_name(col) for col in media_vars]
    mmm_model_output['transformation_params'] = {shorten_name(var): {'decay_rate': mmm_model_output['decay_mean'][i], 'diminishing_rate': mmm_model_output['alpha_mean'][i]} for i, var in enumerate(media_names)}
    if len(comp_media_vars) > 0:
        for i, var in enumerate(comp_media_vars):
            mmm_model_output['transformation_params'][shorten_name(var)] = {'decay_rate': mmm_model_output['decay_comp_mean'][i], 'diminishing_rate': mmm_model_output['alpha_comp_mean'][i]}
    if 'X_media_transformed' in list(fit_result.keys()):
        mmm_model_output['X_media_transformed'] = fit_result['X_media_transformed']
    if 'X_comp_media_transformed' in list(fit_result.keys()):
        mmm_model_output['X_comp_media_transformed'] = fit_result['X_comp_media_transformed']

    return mmm_model_output


def get_model_output_parameters(mmm_model_output, selected_model_index=None):

    if selected_model_index is None:
        tau, beta, beta_comp, beta1, beta2, beta3 = mmm_model_output['tau_mean'], np.array(mmm_model_output['beta_mean']), np.array(mmm_model_output['beta_comp_mean']), np.array(mmm_model_output['beta1_mean']), np.array(mmm_model_output['beta2_mean']), np.array(mmm_model_output['beta3_mean'])
        transformation_params = mmm_model_output['transformation_params']
    else:
        media_vars = mmm_model_output['media_vars']
        comp_media_vars = mmm_model_output['comp_media_vars']
        tau = mmm_model_output['tau'][selected_model_index] if len(mmm_model_output['tau']) > 0 else []
        beta = np.array(mmm_model_output['beta'][selected_model_index]) if len(mmm_model_output['beta']) > 0 else []
        beta_comp = np.array(mmm_model_output['beta_comp'][selected_model_index]) if len(mmm_model_output['beta_comp']) > 0 else []
        beta1 = np.array(mmm_model_output['beta1'][selected_model_index]) if len(mmm_model_output['beta1']) > 0 else []
        beta2 = np.array(mmm_model_output['beta2'][selected_model_index]) if len(mmm_model_output['beta2']) > 0 else []
        beta3 = np.array(mmm_model_output['beta3'][selected_model_index]) if len(mmm_model_output['beta3']) > 0 else []
        transformation_params = {shorten_name(var): {'decay_rate': mmm_model_output['decay'][selected_model_index][i], 'diminishing_rate': mmm_model_output['alpha'][selected_model_index][i]} for i, var in enumerate(media_vars)}
        if len(comp_media_vars) > 0:
            for i, var in enumerate(comp_media_vars):
                transformation_params[shorten_name(var)] = {'decay_rate': mmm_model_output['decay_comp'][selected_model_index][i], 'diminishing_rate': mmm_model_output['alpha_comp'][selected_model_index][i]}

    return tau, beta, beta_comp, beta1, beta2, beta3, transformation_params


def predict_mmm_model(mmm_model_output, df_model, model_form='linear'):

    media_vars, comp_media_vars, positive_vars, neutral_vars, negative_vars = mmm_model_output['media_vars'], mmm_model_output['comp_media_vars'], mmm_model_output['positive_vars'], mmm_model_output['neutral_vars'], mmm_model_output['negative_vars']
    x_media, x_comp_media, x_control_positive, x_control_neutral, x_control_negative = subset_df(df_model, media_vars), subset_df(df_model, comp_media_vars), subset_df(df_model, positive_vars), subset_df(df_model, neutral_vars), subset_df(df_model, negative_vars)
    tau, beta, beta_comp, beta1, beta2, beta3, transformation_params = get_model_output_parameters(mmm_model_output)

    predicted_y = tau
    if len(x_media) > 0:
        x_media_dim = apply_diminishing(x_media, media_vars, transformation_params)
        x_media_adstocked = apply_carry_over(x_media_dim, media_vars, transformation_params)
        x_media_transformed, x_media_avg = apply_mean_log(x_media_adstocked, media_vars) if model_form.lower() == 'log-log' else apply_mean_center(x_media_adstocked, media_vars)
        predicted_y += np.dot(x_media_transformed, beta)

    if len(x_comp_media) > 0:
        x_comp_media_dim = apply_diminishing(x_comp_media, comp_media_vars, transformation_params)
        x_comp_media_adstocked = apply_carry_over(x_comp_media_dim, comp_media_vars, transformation_params)
        x_comp_media_transformed, x_comp_media_avg = apply_mean_log(x_comp_media_adstocked, comp_media_vars) if model_form.lower() == 'log-log' else apply_mean_center(x_comp_media_adstocked, comp_media_vars)
        predicted_y += np.dot(x_comp_media_transformed, beta_comp)

    for item in [{'coeff': beta1, 'data': x_control_positive}, {'coeff': beta2, 'data': x_control_neutral}, {'coeff': beta3, 'data': x_control_negative}]:
        if len(item['data']) > 0:
            predicted_y += np.dot(item['data'], item['coeff'])

    return predicted_y


def count_obs(df):
    if len(df.index[0]) == 1:
        return {'ALL': {'nobs': len(df), 'start': 0, 'end': len(df)}}
    else:
        temp_df = df.copy()
        temp_df['CrossSection'] = [i[0] for i in temp_df.index]
        temp_df['RowIndex'] = [i for i in range(len(temp_df))]
        s = temp_df.groupby('CrossSection').apply(lambda x: {'nobs': len(x), 'start': min(x['RowIndex']), 'end': max(x['RowIndex'])})
        return {k: v for k, v in s.items()}


def decompose_model(mmm_model_output, df_model, df_media_spend=None, selected_model_index=None, model_form='linear', min_max_adjustment=False):

    # linear model: y = X[0]*beta[0] + ... + X[10]*beta[10] + tau
    # log-log model: log1(y) = log1(X[0])*beta[0] + ... + log1(X[10])*beta[10] + tau
    # -------------- y+1 = X[0]^beta[0] * ... * X[10]^beta[10] * e^tau
    # semi-log model: log1(y) = X[0]*beta[0] + ... + X[10]*beta[10] + tau
    # -------------- y+1 = e^(X[0]*beta[0] + ... + X[10]*beta[10] + tau)

    dep_var, media_vars, comp_media_vars, positive_vars, neutral_vars, negative_vars = mmm_model_output['dep_var'], mmm_model_output['media_vars'], mmm_model_output['comp_media_vars'], mmm_model_output['positive_vars'], mmm_model_output['neutral_vars'], mmm_model_output['negative_vars']
    x_media, x_comp_media, x_control_positive, x_control_neutral, x_control_negative = subset_df(df_model, media_vars), subset_df(df_model, comp_media_vars), subset_df(df_model, positive_vars), subset_df(df_model, neutral_vars), subset_df(df_model, negative_vars)
    dep_var_raw_col = dep_var + '_raw'
    y_raw = df_model[[dep_var_raw_col]].copy()
    y_norm, y_mean = apply_mean_center(y_raw, [dep_var_raw_col])
    y_mean = y_mean[dep_var_raw_col]
    y_norm = y_norm + 1
    cross_sections_summary = count_obs(df_model)
    cross_sections = list(cross_sections_summary.keys())
    cross_sections_total = cross_sections + ['[TOTAL]']

    has_spend_data = True if df_media_spend is not None and len(df_media_spend) > 0 else False

    tau, beta, beta_comp, beta1, beta2, beta3, transformation_params = get_model_output_parameters(mmm_model_output, selected_model_index)
    if len(x_media) > 0:
        x_media_dim = apply_diminishing(x_media, media_vars, transformation_params)
        x_media_adstocked = apply_carry_over(x_media_dim, media_vars, transformation_params)
        x_media_transformed, x_media_avg = apply_mean_log(x_media_adstocked, media_vars) if model_form.lower() == 'log-log' else apply_mean_center(x_media_adstocked, media_vars)
    else:
        x_media_transformed = pd.DataFrame()
    if len(x_comp_media) > 0:
        x_comp_media_dim = apply_diminishing(x_comp_media, comp_media_vars, transformation_params)
        x_comp_media_adstocked = apply_carry_over(x_comp_media_dim, comp_media_vars, transformation_params)
        x_comp_media_transformed, x_comp_media_avg = apply_mean_log(x_comp_media_adstocked, comp_media_vars) if model_form.lower() == 'log-log' else apply_mean_center(x_comp_media_adstocked, comp_media_vars)
    else:
        x_comp_media_transformed = pd.DataFrame()
    model_contributions = {
        'model_contributions': {'media_vars': {}, 'baseline': {}},
        'lifetime_contributions': {'media_vars': {}},
        'spend': {'media_vars': {}}
    }

    # Spend
    model_contributions['spend']['media_vars'] = {cross_section: {} for cross_section in cross_sections + ['[TOTAL]']}
    for cross_section in cross_sections:
        model_contributions['spend']['media_vars'][cross_section] = {}
        nobs = cross_sections_summary[cross_section]['nobs']
        start = cross_sections_summary[cross_section]['start']
        end = cross_sections_summary[cross_section]['end']
        model_contributions['spend']['media_vars'][cross_section] = {shorten_name(var): df_media_spend[shorten_name(var)].values[start:end+1] if has_spend_data and shorten_name(var) in df_media_spend.columns else [0] * nobs for var in media_vars}
    total_obs = sum([cross_sections_summary[cross_section]['nobs'] for cross_section in cross_sections])
    model_contributions['spend']['media_vars']['[TOTAL]'] = {shorten_name(var): df_media_spend[shorten_name(var)].values if has_spend_data and shorten_name(var) in df_media_spend.columns else [0] * total_obs for var in media_vars}

    # Decomposition
    model_contributions['model_contributions']['baseline'] = {cross_section: {} for cross_section in cross_sections + ['[TOTAL]']}
    model_contributions['model_contributions']['media_vars'] = {cross_section: {} for cross_section in cross_sections + ['[TOTAL]']}
    model_contributions['lifetime_contributions']['media_vars'] = {cross_section: {} for cross_section in cross_sections + ['[TOTAL]']}
    if model_form.lower() == 'log-log':
        var_factors = {}
        unadjusted_control_contributions = {}
        unadjusted_media_contributions = {}
        intercept_con = np.exp([tau] * total_obs)

        # Baseline
        baseline = intercept_con.copy()
        model_contributions['model_contributions']['baseline']['[TOTAL]']['intercept'] = intercept_con.copy()
        for item in [{'vars': comp_media_vars, 'coeff': beta_comp, 'data': x_comp_media_transformed}, {'vars': positive_vars, 'coeff': beta1, 'data': x_control_positive}, {'vars': neutral_vars, 'coeff': beta2, 'data': x_control_neutral}, {'vars': negative_vars, 'coeff': beta3, 'data': x_control_negative}]:
            if len(item['vars']) > 0:
                for v, var in enumerate(item['vars']):
                    baseline *= np.exp(item['data'][var] * item['coeff'][v])
        for item in [{'vars': comp_media_vars, 'coeff': beta_comp, 'data': x_comp_media_transformed}, {'vars': positive_vars, 'coeff': beta1, 'data': x_control_positive}, {'vars': neutral_vars, 'coeff': beta2, 'data': x_control_neutral}, {'vars': negative_vars, 'coeff': beta3, 'data': x_control_negative}]:
            if len(item['vars']) > 0:
                for v, var in enumerate(item['vars']):
                    unadjusted_control_contributions[var] = baseline - baseline / np.exp(item['data'][var] * item['coeff'][v])

        actual_control_contributions = baseline - intercept_con
        predicted_control_contributions = np.array([0] * total_obs)
        for var in unadjusted_control_contributions.keys():
            predicted_control_contributions = predicted_control_contributions + unadjusted_control_contributions[var]

        # Intercept
        model_contributions['model_contributions']['baseline']['[TOTAL]']['intercept'] = np.array([0] * total_obs)
        for cross_section in cross_sections:
            start = cross_sections_summary[cross_section]['start']
            end = cross_sections_summary[cross_section]['end']
            model_contributions['model_contributions']['baseline'][cross_section]['intercept'] = (intercept_con[start:end + 1] - 1) * y_mean[cross_section]
            model_contributions['model_contributions']['baseline']['[TOTAL]']['intercept'][start:end + 1] = (intercept_con[start:end + 1] - 1) * y_mean[cross_section]

        # Control / Competitive
        for var in unadjusted_control_contributions.keys():
            model_contributions['model_contributions']['baseline']['[TOTAL]'][var] = np.array([0] * total_obs)
            for cross_section in cross_sections:
                start = cross_sections_summary[cross_section]['start']
                end = cross_sections_summary[cross_section]['end']
                adjusted_var_con = unadjusted_control_contributions[var][start:end+1] * actual_control_contributions[start:end+1] / predicted_control_contributions[start:end+1] * y_mean[cross_section]
                if min_max_adjustment and var not in comp_media_vars:
                    if sum(adjusted_var_con) > 0:  # Min Adjustment
                        adjustment_amount = min(adjusted_var_con)
                        adjusted_var_con = adjusted_var_con - adjustment_amount
                    else:  # Max Adjustment
                        adjustment_amount = max(adjusted_var_con)
                        adjusted_var_con = adjusted_var_con - adjustment_amount
                    model_contributions['model_contributions']['baseline'][cross_section][var] = adjusted_var_con
                    model_contributions['model_contributions']['baseline']['[TOTAL]'][var][start:end + 1] = adjusted_var_con
                    model_contributions['model_contributions']['baseline'][cross_section]['intercept'] = model_contributions['model_contributions']['baseline'][cross_section]['intercept'] + adjustment_amount
                    model_contributions['model_contributions']['baseline']['[TOTAL]']['intercept'][start:end + 1] = model_contributions['model_contributions']['baseline']['[TOTAL]']['intercept'][start:end + 1] + adjustment_amount
                else:
                    model_contributions['model_contributions']['baseline'][cross_section][var] = adjusted_var_con
                    model_contributions['model_contributions']['baseline']['[TOTAL]'][var][start:end + 1] = adjusted_var_con

        # Media
        predicted_y_norm = baseline.copy()
        for v, var in enumerate(media_vars):
            var_factor = np.exp(x_media_transformed[var] * beta[v])
            var_factors[var] = [1 if f == 0 else f for f in var_factor]
            predicted_y_norm *= var_factors[var]
        # Calc unadjusted contributions by removing media variable one by one from predicted y norm. Adjusted contributions takes into account the scale diff between true vs. predicted
        true_media_contributions = y_norm[dep_var_raw_col].values - baseline.values
        predicted_media_contributions = np.array([0] * total_obs)
        for v, var in enumerate(media_vars):
            unadjusted_media_contributions[var] = predicted_y_norm.values - predicted_y_norm.values / var_factors[var]
            predicted_media_contributions = predicted_media_contributions + unadjusted_media_contributions[var]
        for v, var in enumerate(media_vars):
            model_contributions['model_contributions']['media_vars']['[TOTAL]'][var] = np.array([0] * total_obs)
            model_contributions['lifetime_contributions']['media_vars']['[TOTAL]'][var] = np.array([0] * total_obs)
            for cross_section in cross_sections:
                start = cross_sections_summary[cross_section]['start']
                end = cross_sections_summary[cross_section]['end']
                model_adjusted_var_con = unadjusted_media_contributions[var][start:end+1] * true_media_contributions[start:end+1] / predicted_media_contributions[start:end+1] * y_mean[cross_section]
                adjusted_beta = sum(model_adjusted_var_con) / sum(x_media_adstocked[var].values[start:end+1])
                life_time_adjusted_var_con = x_media_dim[var].values[start:end+1] * adjusted_beta / transformation_params[shorten_name(var)]['decay_rate']  # Short term contribution divided by decay rate for geometric decay
                # life_time_adjusted_var_con = model_adjusted_var_con  # Need to change the logic here
                model_contributions['model_contributions']['media_vars'][cross_section][var] = model_adjusted_var_con
                model_contributions['lifetime_contributions']['media_vars'][cross_section][var] = life_time_adjusted_var_con
                model_contributions['model_contributions']['media_vars']['[TOTAL]'][var][start:end+1] = model_adjusted_var_con
                model_contributions['lifetime_contributions']['media_vars']['[TOTAL]'][var][start:end+1] = life_time_adjusted_var_con

    elif model_form.lower() == 'linear':
        # Baseline
        model_contributions['model_contributions']['baseline']['[TOTAL]']['intercept'] = []
        for cross_section in cross_sections:
            model_contributions['model_contributions']['baseline'][cross_section]['intercept'] = np.array([tau * y_mean[cross_section]] * cross_sections_summary[cross_section]['nobs'])
            model_contributions['model_contributions']['baseline']['[TOTAL]']['intercept'] += list(model_contributions['model_contributions']['baseline'][cross_section]['intercept'])
        model_contributions['model_contributions']['baseline']['[TOTAL]']['intercept'] = np.array(model_contributions['model_contributions']['baseline']['[TOTAL]']['intercept'])

        for item in [{'vars': comp_media_vars, 'coeff': beta_comp, 'data': x_comp_media_transformed}, {'vars': positive_vars, 'coeff': beta1, 'data': x_control_positive}, {'vars': neutral_vars, 'coeff': beta2, 'data': x_control_neutral}, {'vars': negative_vars, 'coeff': beta3, 'data': x_control_negative}]:
            if len(item['vars']) > 0:
                for v, var in enumerate(item['vars']):
                    model_contributions['model_contributions']['baseline']['[TOTAL]'][var] = []
                    var_con = item['data'][var].values * item['coeff'][v]
                    for cross_section in cross_sections:
                        start = cross_sections_summary[cross_section]['start']
                        end = cross_sections_summary[cross_section]['end']
                        adjusted_var_con = var_con[start:end+1] * y_mean[cross_section]
                        if min_max_adjustment and var not in comp_media_vars:
                            if sum(adjusted_var_con) > 0:  # Min Adjustment
                                adjustment_amount = min(adjusted_var_con)
                                adjusted_var_con = adjusted_var_con - adjustment_amount
                            else:  # Max Adjustment
                                adjustment_amount = max(adjusted_var_con)
                                adjusted_var_con = adjusted_var_con - adjustment_amount
                            model_contributions['model_contributions']['baseline'][cross_section][var] = adjusted_var_con
                            model_contributions['model_contributions']['baseline']['[TOTAL]'][var] += list(model_contributions['model_contributions']['baseline'][cross_section][var])
                            model_contributions['model_contributions']['baseline'][cross_section]['intercept'] = model_contributions['model_contributions']['baseline'][cross_section]['intercept'] + adjustment_amount
                            model_contributions['model_contributions']['baseline']['[TOTAL]']['intercept'][start:end + 1] = model_contributions['model_contributions']['baseline']['[TOTAL]']['intercept'][start:end + 1] + adjustment_amount
                        else:
                            model_contributions['model_contributions']['baseline'][cross_section][var] = adjusted_var_con
                            model_contributions['model_contributions']['baseline']['[TOTAL]'][var] += list(model_contributions['model_contributions']['baseline'][cross_section][var])
                    model_contributions['model_contributions']['baseline']['[TOTAL]'][var] = np.array(model_contributions['model_contributions']['baseline']['[TOTAL]'][var])

        # Media
        for v, var in enumerate(media_vars):
            model_contributions['model_contributions']['media_vars']['[TOTAL]'][var] = []
            model_contributions['lifetime_contributions']['media_vars']['[TOTAL]'][var] = []
            var_con = x_media_transformed[var].values * beta[v]
            life_time_con = x_media_dim[var].values * beta[v] / transformation_params[shorten_name(var)]['decay_rate']
            for cross_section in cross_sections:
                start = cross_sections_summary[cross_section]['start']
                end = cross_sections_summary[cross_section]['end']
                model_var_con = var_con[start:end+1] * y_mean[cross_section]
                life_time_var_con = life_time_con[start:end+1] / x_media_avg[var][cross_section] * y_mean[cross_section]
                model_contributions['model_contributions']['media_vars'][cross_section][var] = model_var_con
                model_contributions['lifetime_contributions']['media_vars'][cross_section][var] = life_time_var_con
                model_contributions['model_contributions']['media_vars']['[TOTAL]'][var] += list(model_contributions['model_contributions']['media_vars'][cross_section][var])
                model_contributions['lifetime_contributions']['media_vars']['[TOTAL]'][var] += list(model_contributions['lifetime_contributions']['media_vars'][cross_section][var])
            model_contributions['model_contributions']['media_vars']['[TOTAL]'][var] = np.array(model_contributions['model_contributions']['media_vars']['[TOTAL]'][var])
            model_contributions['lifetime_contributions']['media_vars']['[TOTAL]'][var] = np.array(model_contributions['lifetime_contributions']['media_vars']['[TOTAL]'][var])

    return model_contributions


def simulate_mmm_model_prediction(mmm_model_output, df_model, df_media_spend=None, model_form='linear', num_sample=None, min_max_adjustment=False):

    dep_var, media_vars, comp_media_vars, positive_vars, neutral_vars, negative_vars = mmm_model_output['dep_var'], mmm_model_output['media_vars'], mmm_model_output['comp_media_vars'], mmm_model_output['positive_vars'], mmm_model_output['neutral_vars'], mmm_model_output['negative_vars']
    x_media, x_comp_media, x_control_positive, x_control_neutral, x_control_negative = subset_df(df_model, media_vars), subset_df(df_model, comp_media_vars), subset_df(df_model, positive_vars), subset_df(df_model, neutral_vars), subset_df(df_model, negative_vars)
    dep_var_raw_col = dep_var + '_raw'
    y_raw = df_model[[dep_var_raw_col]].copy()
    y_norm, y_mean = apply_mean_center(y_raw, [dep_var_raw_col])
    y_mean = y_mean[dep_var_raw_col]
    cross_sections_summary = count_obs(df_model)
    cross_sections = list(cross_sections_summary.keys())
    cross_sections_total = cross_sections + ['[TOTAL]']
    has_spend_data = True if df_media_spend is not None and len(df_media_spend) > 0 else False

    simulated_results = {
        'ModelIndex': [],
        'ModelStats': {
            cross_section: {
                'Predicted Y': [],
                'MAPE': [],
                'R2': [],
                'SOSSOC': [],
                'MAPE_SOSSOC': []
            } for cross_section in cross_sections_total
        },
        'ModelContributions': {cross_section: {shorten_name(var): [] for var in media_vars} for cross_section in cross_sections_total},
        'LifeTimeContributions': {cross_section: {shorten_name(var): [] for var in media_vars} for cross_section in cross_sections_total},
        'ROAS': {cross_section: {shorten_name(var): [] for var in media_vars} for cross_section in cross_sections_total},
        'LifeTimeROAS': {cross_section: {shorten_name(var): [] for var in media_vars} for cross_section in cross_sections_total},
        'ResponseCurves': {cross_section: {shorten_name(var): [] for var in media_vars} for cross_section in cross_sections_total}
    }
    if num_sample is None:
        sample_iterations = [i for i in range(len(mmm_model_output['tau']))]
    else:
        sample_iterations = sorted(random.sample([i for i in range(len(mmm_model_output['tau']))], num_sample)) if num_sample <= len(mmm_model_output['tau']) else [i for i in range(len(mmm_model_output['tau']))]
    count = 0
    for i in sample_iterations:
        simulated_results['ModelIndex'].append(i)
        tau, beta, beta_comp, beta1, beta2, beta3, transformation_params = get_model_output_parameters(mmm_model_output, selected_model_index=i)
        predicted_y = tau
        if len(x_media) > 0:
            x_media_dim = apply_diminishing(x_media, media_vars, transformation_params)
            x_media_adstocked = apply_carry_over(x_media_dim, media_vars, transformation_params)
            x_media_transformed, x_media_avg = apply_mean_log(x_media_adstocked, media_vars) if model_form.lower() == 'log-log' else apply_mean_center(x_media_adstocked, media_vars)
            x_media_contribution = np.dot(x_media_transformed, beta)
            predicted_y = predicted_y + x_media_contribution

        if len(x_comp_media) > 0:
            x_comp_media_dim = apply_diminishing(x_comp_media, comp_media_vars, transformation_params)
            x_comp_media_adstocked = apply_carry_over(x_comp_media_dim, comp_media_vars, transformation_params)
            x_comp_media_transformed, x_comp_media_avg = apply_mean_log(x_comp_media_adstocked, comp_media_vars) if model_form.lower() == 'log-log' else apply_mean_center(x_comp_media_adstocked, comp_media_vars)
            x_comp_media_contribution = np.dot(x_comp_media_transformed, beta_comp)
            predicted_y = predicted_y + x_comp_media_contribution

        for item in [{'coeff': beta1, 'data': x_control_positive}, {'coeff': beta2, 'data': x_control_neutral}, {'coeff': beta3, 'data': x_control_negative}]:
            if len(item['data']) > 0:
                predicted_y = predicted_y + np.dot(item['data'], item['coeff'])

        # Model Stats
        true_y = df_model[dep_var].values
        for cross_section in cross_sections:
            start = cross_sections_summary[cross_section]['start']
            end = cross_sections_summary[cross_section]['end']
            simulated_results['ModelStats'][cross_section]['MAPE'].append(calc_mape(true_y[start:end+1], predicted_y[start:end+1]))
            simulated_results['ModelStats'][cross_section]['R2'].append(calc_r2(true_y[start:end+1], predicted_y[start:end+1]))
            cross_section_predicted_y = sum(predicted_y[start:end+1])  # * y_mean[cross_section]
            simulated_results['ModelStats'][cross_section]['Predicted Y'].append(cross_section_predicted_y)
        simulated_results['ModelStats']['[TOTAL]']['MAPE'].append(calc_mape(true_y, predicted_y))
        simulated_results['ModelStats']['[TOTAL]']['R2'].append(calc_r2(true_y, predicted_y))
        simulated_results['ModelStats']['[TOTAL]']['Predicted Y'].append(sum(predicted_y))

        # Contributions
        model_contributions = decompose_model(mmm_model_output, df_model, df_media_spend, selected_model_index=i, model_form=model_form, min_max_adjustment=min_max_adjustment)
        summary_spend = {cross_section: {} for cross_section in cross_sections_total}
        share_of_spend = {cross_section: {} for cross_section in cross_sections_total}
        summary_contributions = {cross_section: {} for cross_section in cross_sections_total}
        share_of_contributions = {cross_section: {} for cross_section in cross_sections_total}
        for cross_section in cross_sections_total:
            for v, var in enumerate(media_vars):
                total_spend = sum(model_contributions['spend']['media_vars'][cross_section][shorten_name(var)])
                summary_spend[cross_section][shorten_name(var)] = total_spend
                total_model_var_con = sum(model_contributions['model_contributions']['media_vars'][cross_section][var])
                total_roas = total_model_var_con / total_spend if total_spend > 0 else 0
                total_life_time_var_con = sum(model_contributions['lifetime_contributions']['media_vars'][cross_section][var])
                total_life_time_roas = total_life_time_var_con / total_spend if total_spend > 0 else 0
                summary_contributions[cross_section][shorten_name(var)] = total_model_var_con
                simulated_results['ModelContributions'][cross_section][shorten_name(var)].append(total_model_var_con)
                simulated_results['LifeTimeContributions'][cross_section][shorten_name(var)].append(total_life_time_var_con)
                simulated_results['ROAS'][cross_section][shorten_name(var)].append(total_roas)
                simulated_results['LifeTimeROAS'][cross_section][shorten_name(var)].append(total_life_time_roas)
                # simulated_results['Coeff'][cross_section][shorten_name(var)].append()
            total_spend = sum([v for v in summary_spend[cross_section].values()])
            share_of_spend[cross_section] = {shorten_name(var): summary_spend[cross_section][shorten_name(var)] / total_spend if total_spend > 0 else 0 for var in media_vars}
            total_contribution = sum([v for v in summary_contributions[cross_section].values()])
            share_of_contributions[cross_section] = {shorten_name(var): summary_contributions[cross_section][shorten_name(var)] / total_contribution for var in media_vars}
            sos_soc = math.sqrt(sum([(share_of_contributions[cross_section][shorten_name(var)] - share_of_spend[cross_section][shorten_name(var)]) ** 2 for var in media_vars]) / len(media_vars) * 100)
            simulated_results['ModelStats'][cross_section]['SOSSOC'].append(sos_soc)
            simulated_results['ModelStats'][cross_section]['MAPE_SOSSOC'].append(math.sqrt(sos_soc ** 2 + calc_mape(true_y, predicted_y) ** 2))

        # Response Curves
        if has_spend_data:
            for cross_section in cross_sections:
                start = cross_sections_summary[cross_section]['start']
                end = cross_sections_summary[cross_section]['end']
                for v, var in enumerate(media_vars):
                    cpm = summary_spend[cross_section][shorten_name(var)] / sum(x_media[var][start:end+1]) * 1000
                    weekly_spend = [summary_spend[cross_section][shorten_name(var)] / cross_sections_summary[cross_section]['nobs'] * i / 10 for i in range(30)]  # Average Weekly Spend
                    diminishing_rate = mmm_model_output['alpha'][i][v]
                    decay_rate = mmm_model_output['decay'][i][v]
                    cross_section_coeff = sum(model_contributions['model_contributions']['media_vars'][cross_section][var]) / sum(x_media_adstocked[var].values[start:end+1])
                    life_time_con = [cross_section_coeff * calc_diminishing(s / cpm * 1000, diminishing_rate) / decay_rate for s in weekly_spend]
                    simulated_results['ResponseCurves'][cross_section][shorten_name(var)].append({'x': weekly_spend, 'y': life_time_con})

        count += 1
        if (count % 100 == 0) or (count == len(sample_iterations)):
            print(' > Completed {} / {} sample iterations'.format(count, len(sample_iterations)))

    return simulated_results


def select_model(simulated_results, metric='R2', alpha=0.05, cross_section='[TOTAL]'):
    if metric.lower() == 'r2':
        metric = 'R2'
    elif metric.lower() == 'mape':
        metric = 'MAPE'
    elif metric.lower() == 'sossoc':
        metric = 'SOSSOC'
    elif metric.lower() == 'mape_sossoc':
        metric = 'MAPE_SOSSOC'
    else:
        metric = 'MAPE'
    samples = simulated_results['ModelStats'][cross_section][metric]
    hpd_mu, x_mu, y_mu, modes_mu = hpd_grid(samples, alpha, roundto=10)  # High Density Region
    if metric == 'R2':
        best_score = max([v for v in samples if v <= hpd_mu[0][1]])
    else:
        best_score = min([v for v in samples if v >= hpd_mu[0][0]])
    selected_sample_counter = next(i for i, v in enumerate(samples) if v == best_score)
    selected_model_index = simulated_results['ModelIndex'][selected_sample_counter]
    print('Selected Model Index: {} - {}: {}'.format(selected_model_index, metric, best_score))
    return selected_model_index, selected_sample_counter


# -----------------------------------------------------------------------------------------
# VARIABLE TRANSFORMATIONS
# -----------------------------------------------------------------------------------------
def calc_diminishing(x, alpha):
    return x ** alpha


def apply_diminishing(df, media_vars, transformation_params):
    df_out = pd.DataFrame(index=df.index)
    if len(df.index[0]) == 1:
        for media_var in media_vars:
            media = shorten_name(media_var)
            diminishing_rate = transformation_params[media]['diminishing_rate']
            df_out[media_var] = [calc_diminishing(v, diminishing_rate) for v in df[media_var].values]
    else:  # Cross Section
        df['CrossSection'] = [i[0] for i in df.index]
        g = df.groupby('CrossSection')
        for media_var in media_vars:
            media = shorten_name(media_var)
            diminishing_rate = transformation_params[media]['diminishing_rate']
            temp = g.apply(lambda x: calc_diminishing(x[media_var].values, diminishing_rate))
            df_out[media_var] = [x for k, v in temp.items() for x in v]
    return df_out


def calc_carry_over(x, carry_over_rate=0.9):
    carry_over_array = [x[0]] + [0] * (len(x) - 1)
    for i, v in enumerate(x):
        if i > 0:
            carry_over_array[i] = carry_over_array[i - 1] * carry_over_rate + v
    return carry_over_array


def apply_carry_over(df, media_vars, transformation_params):
    df_out = pd.DataFrame(index=df.index)
    if len(df.index[0]) == 1:
        for media_var in media_vars:
            media = shorten_name(media_var)
            carry_over_rate = 1 - transformation_params[media]['decay_rate']
            df_out[media_var] = calc_carry_over(df[media_var].values, carry_over_rate)
    else:
        df['CrossSection'] = [i[0] for i in df.index]
        g = df.groupby('CrossSection')
        for media_var in media_vars:
            media = shorten_name(media_var)
            carry_over_rate = 1 - transformation_params[media]['decay_rate']
            temp = g.apply(lambda x: calc_carry_over(x[media_var].values, carry_over_rate))
            df_out[media_var] = [x for k, v in temp.items() for x in v]
    return df_out


def calc_mean_center(x):
    avg = np.mean(x)
    x_mean = x / avg
    return x_mean, avg


def apply_mean_center(df, cols):
    df_out = pd.DataFrame(index=df.index)
    scale = {}
    if len(df.index[0]) == 1:
        for col in cols:
            df_out[col], avg = calc_mean_center(df[col].values)
            scale[col] = {'ALL': avg}
    else:
        df['CrossSection'] = [i[0] for i in df.index]
        g = df.groupby('CrossSection')
        for col in cols:
            temp = g.apply(lambda x: calc_mean_center(x[col].values))
            df_out[col] = [x for k, v in temp.items() for x in v[0]]
            scale[col] = {k: v[1] for k, v in temp.items()}
    return df_out, scale


def apply_mean_log(df, cols):
    df_out = pd.DataFrame(index=df.index)
    scale = {}
    if len(df.index[0]) == 1:
        for col in cols:
            x = df[col].values
            x_mean, avg = calc_mean_center(x)
            df_out[col] = np.log1p(x_mean)
            scale[col] = {'ALL': avg}
    else:
        df['CrossSection'] = [i[0] for i in df.index]
        g = df.groupby('CrossSection')
        for col in cols:
            temp = g.apply(lambda x: calc_mean_center(x[col].values))
            df_out[col] = [np.log1p(x) for k, v in temp.items() for x in v[0]]
            scale[col] = {k: v[1] for k, v in temp.items()}
    return df_out, scale


# -----------------------------------------------------------------------------------------
# MODEL OUTPUTS
# -----------------------------------------------------------------------------------------
def calc_model_contributions(df, df_model, mmm_model_output, df_media_spend=None, selected_model_index=None, model_form='linear', min_max_adjustment=True):

    dep_var, media_vars, comp_media_vars, positive_vars, neutral_vars, negative_vars = mmm_model_output['dep_var'], mmm_model_output['media_vars'], mmm_model_output['comp_media_vars'], mmm_model_output['positive_vars'], mmm_model_output['neutral_vars'], mmm_model_output['negative_vars']
    x_media, x_comp_media, x_control_positive, x_control_neutral, x_control_negative = subset_df(df_model, media_vars), subset_df(df_model, comp_media_vars), subset_df(df_model, positive_vars), subset_df(df_model, neutral_vars), subset_df(df_model, negative_vars)
    tau, beta, beta_comp, beta1, beta2, beta3, transformation_params = get_model_output_parameters(mmm_model_output, selected_model_index)
    cross_sections_summary = count_obs(df_model)
    cross_sections = list(cross_sections_summary.keys())
    cross_sections_total = cross_sections + ['[TOTAL]']

    # Dep Var Raw
    y_raw = df[dep_var].copy()
    y_raw.index = df_model.index

    # Media Variables
    if len(x_media) > 0:
        x_media_dim = apply_diminishing(x_media, media_vars, transformation_params)
        x_media_adstocked = apply_carry_over(x_media_dim, media_vars, transformation_params)
        x_media_transformed, x_media_avg = apply_mean_log(x_media_adstocked, media_vars) if model_form.lower() == 'log-log' else apply_mean_center(x_media_adstocked, media_vars)
        x_media_transformed.index = df_model.index

    if len(x_comp_media) > 0:
        x_comp_media_dim = apply_diminishing(x_comp_media, comp_media_vars, transformation_params)
        x_comp_media_adstocked = apply_carry_over(x_comp_media_dim, comp_media_vars, transformation_params)
        x_comp_media_transformed, x_comp_media_avg = apply_mean_log(x_comp_media_adstocked, comp_media_vars) if model_form.lower() == 'log-log' else apply_mean_center(x_comp_media_adstocked, comp_media_vars)
        x_comp_media_transformed.index = df_model.index
    else:
        x_comp_media_transformed = pd.DataFrame()

    # Dep Var
    y_norm = df_model[dep_var]

    # Compute each media / control variables
    model_contributions = decompose_model(mmm_model_output, df_model, df_media_spend, selected_model_index=selected_model_index, model_form=model_form, min_max_adjustment=min_max_adjustment)
    df_contributions = pd.DataFrame(columns=media_vars + comp_media_vars + positive_vars + neutral_vars + negative_vars + ['intercept'], index=df_model.index)
    df_lifetime_contributions = pd.DataFrame(columns=media_vars + positive_vars + neutral_vars + negative_vars + ['intercept'], index=df_model.index)
    for var in media_vars:
        df_contributions[var] = model_contributions['model_contributions']['media_vars']['[TOTAL]'][var]
        df_lifetime_contributions[var] = model_contributions['lifetime_contributions']['media_vars']['[TOTAL]'][var]
    for item in [{'vars': comp_media_vars, 'coeff': beta_comp, 'data': x_comp_media_transformed}, {'vars': positive_vars, 'coeff': beta1, 'data': x_control_positive}, {'vars': neutral_vars, 'coeff': beta2, 'data': x_control_neutral}, {'vars': negative_vars, 'coeff': beta3, 'data': x_control_negative}]:
        if len(item['vars']) > 0:
            for i, var in enumerate(item['vars']):
                df_contributions[var] = model_contributions['model_contributions']['baseline']['[TOTAL]'][var]
                df_lifetime_contributions[var] = model_contributions['model_contributions']['baseline']['[TOTAL]'][var]
    df_contributions['intercept'] = model_contributions['model_contributions']['baseline']['[TOTAL]']['intercept']
    df_lifetime_contributions['intercept'] = model_contributions['model_contributions']['baseline']['[TOTAL]']['intercept']

    # Calculate predicted y
    df_contributions['predicted_' + dep_var] = df_contributions.apply(np.sum, axis=1)
    df_contributions[dep_var] = y_raw
    if len(df_contributions.index[0]) == 1:
        df_contributions['Date'] = df_contributions.index
        df_lifetime_contributions['Date'] = df_lifetime_contributions.index
    else:
        df_contributions['Date'] = [i[1] for i in df_contributions.index]
        df_contributions['CrossSection'] = [i[0] for i in df_contributions.index]
        df_lifetime_contributions['Date'] = [i[1] for i in df_lifetime_contributions.index]
        df_lifetime_contributions['CrossSection'] = [i[0] for i in df_lifetime_contributions.index]
    df_contributions = df_contributions[[col for col in ['CrossSection', 'Date'] + [c for c in df_contributions.columns if c not in ['CrossSection', 'Date']]]]
    df_lifetime_contributions = df_lifetime_contributions[[col for col in ['CrossSection', 'Date'] + [c for c in df_lifetime_contributions.columns if c not in ['CrossSection', 'Date']]]]

    # Model Specification
    records = []
    for cross_section in cross_sections:
        start = cross_sections_summary[cross_section]['start']
        end = cross_sections_summary[cross_section]['end']
        for item in [{'vars': media_vars, 'coeff': beta}, {'vars': comp_media_vars, 'coeff': beta_comp}, {'vars': positive_vars, 'coeff': beta1}, {'vars': neutral_vars, 'coeff': beta2}, {'vars': negative_vars, 'coeff': beta3}, {'vars': ['intercept'], 'coeff': tau}]:
            for i, var in enumerate(item['vars']):
                if var in media_vars:
                    coeff = sum(model_contributions['model_contributions']['media_vars'][cross_section][var]) / sum(x_media_adstocked[start:end+1][var])
                elif var in comp_media_vars:
                    coeff = sum(model_contributions['model_contributions']['baseline'][cross_section][var]) / sum(x_comp_media_adstocked[start:end+1][var])
                elif var == 'intercept':
                    coeff = sum(model_contributions['model_contributions']['baseline'][cross_section][var]) / len(model_contributions['model_contributions']['baseline'][cross_section][var])
                else:
                    coeff = sum(model_contributions['model_contributions']['baseline'][cross_section][var]) / sum(df[var][start:end+1])
                records.append({
                    'CrossSection': cross_section,
                    'Variable': var,
                    'DiminishingFunction': 'power' if var in media_vars else 'linear',
                    'DiminishingRate': transformation_params[shorten_name(var)]['diminishing_rate'] if var in media_vars else 1.0,
                    'DecayFunction': 'geometric',
                    'DecayRate': transformation_params[shorten_name(var)]['decay_rate'] if var in media_vars else 1.0,
                    'Coefficient': coeff
                })
    df_model_specifications = pd.DataFrame(records)

    # Model Stats
    mape = calc_mape(df_contributions[dep_var], df_contributions['predicted_' + dep_var])
    r2 = calc_r2(df_contributions[dep_var], df_contributions['predicted_' + dep_var])
    print('MAPE: {:.1f}% - R2: {:.1f}%'.format(mape, r2))

    return df_contributions, df_lifetime_contributions, df_model_specifications


def calc_model_decomposition(df, df_model, mmm_model_output, df_media_spend=None, selected_model_index=None, model_form='linear', min_max_adjustment=True, num_weeks=52):

    dep_var, media_vars, positive_vars, neutral_vars, negative_vars = mmm_model_output['dep_var'], mmm_model_output['media_vars'], mmm_model_output['positive_vars'], mmm_model_output['neutral_vars'], mmm_model_output['negative_vars']
    model_contributions = decompose_model(mmm_model_output, df_model, df_media_spend, selected_model_index=selected_model_index, model_form=model_form, min_max_adjustment=min_max_adjustment)
    cross_sections_summary = count_obs(df_model)
    cross_sections = list(cross_sections_summary.keys())
    cross_sections_total = cross_sections + ['[TOTAL]']

    # Share of spend
    if num_weeks is None:
        summary_spend = {cross_section: {shorten_name(var): sum(model_contributions['spend']['media_vars'][cross_section][shorten_name(var)]) if shorten_name(var) in list(model_contributions['spend']['media_vars'][cross_section].keys()) else 0 for var in media_vars} for cross_section in cross_sections_total}
        total_spend = {cross_section: sum([v for v in summary_spend[cross_section].values()]) for cross_section in cross_sections_total}
        share_of_spend = {cross_section: {shorten_name(var): summary_spend[cross_section][shorten_name(var)] / total_spend[cross_section] if total_spend[cross_section] > 0 else 0 for var in media_vars} for cross_section in cross_sections_total}
    else:
        summary_spend = {cross_section: {shorten_name(var): sum(model_contributions['spend']['media_vars'][cross_section][shorten_name(var)][-num_weeks:]) if shorten_name(var) in list(model_contributions['spend']['media_vars'][cross_section].keys()) else 0 for var in media_vars} for cross_section in cross_sections_total}
        total_spend = {cross_section: sum([v for v in summary_spend[cross_section].values()]) for cross_section in cross_sections_total}
        share_of_spend = {cross_section: {shorten_name(var): summary_spend[cross_section][shorten_name(var)] / total_spend[cross_section] if total_spend[cross_section] > 0 else 0 for var in media_vars} for cross_section in cross_sections_total}

    # Share of contributions
    summary_contributions = {cross_section: {} for cross_section in cross_sections_total}
    total_contributions = {cross_section: {} for cross_section in cross_sections_total}
    share_of_contributions = {cross_section: {} for cross_section in cross_sections_total}
    for cross_section in cross_sections_total:
        for var, var_con in model_contributions['model_contributions']['media_vars'][cross_section].items():
            summary_contributions[cross_section][shorten_name(var)] = sum(var_con) if num_weeks is None else sum(var_con[-num_weeks:])
        for var, var_con in model_contributions['model_contributions']['baseline'][cross_section].items():
            summary_contributions[cross_section][shorten_name(var)] = sum(var_con) if num_weeks is None else sum(var_con[-num_weeks:])
        total_contributions[cross_section] = sum(var_con for var, var_con in summary_contributions[cross_section].items()) if len(summary_contributions[cross_section].keys()) > 0 else 0
        share_of_contributions[cross_section] = {var: var_con / total_contributions[cross_section] if total_contributions[cross_section] > 0 else 0 for var, var_con in summary_contributions[cross_section].items()}

    summary_media_contributions = {cross_section: {var: var_con for var, var_con in summary_contributions[cross_section].items() if var in [shorten_name(v) for v in media_vars]} for cross_section in cross_sections_total}
    total_media_contributions = {cross_section: sum(var_con for var, var_con in summary_media_contributions[cross_section].items()) if len(summary_media_contributions[cross_section].keys()) > 0 else 0 for cross_section in cross_sections_total}
    share_of_media_contributions = {cross_section: {var: var_con / total_media_contributions[cross_section] if total_media_contributions[cross_section] > 0 else 0 for var, var_con in summary_media_contributions[cross_section].items()} for cross_section in cross_sections_total}

    return {
        'SpendSummary': summary_spend,
        'SpendShare': share_of_spend,
        'MediaContributionsSummary': summary_media_contributions,
        'MediaContributionsShare': share_of_media_contributions,
        'ContributionsSummary': summary_contributions,
        'ContributionsShare': share_of_contributions
    }


# -----------------------------------------------------------------------------------------
# CHARTING
# -----------------------------------------------------------------------------------------
def plot_transformation():
    fig, ax = plt.subplots(figsize=(15, 8))
    param_range = [0.1 * i for i in range(10) if i > 0]
    x_media = np.arange(0, 2, 0.05)
    for i in range(len(param_range)):
        r = param_range[i]
        sns.lineplot(x=x_media, y=calc_diminishing(x_media, r), ax=ax, label='Power=%.2f' % (r))
        ax.lines[i].set_linestyle("--")
    ax.set_title('Diminishing Return Function', fontsize=16)


def plot_adstock():
    fig, ax = plt.subplots(figsize=(15, 8))
    carry_over_range = [0.1 * i for i in range(10)]
    x_media = [100] + [0] * 51
    sns.lineplot(x=range(52), y=x_media[-52:], ax=ax, label='original')
    for i in range(len(carry_over_range)):
        r = carry_over_range[i]
        x_media_adstocked = calc_carry_over(x_media, r)
        sns.lineplot(x=range(52), y=x_media_adstocked[-52:], ax=ax, label='carry_over_rate=%.1f' % (r))
        ax.lines[i+1].set_linestyle("--")
    ax.set_title('Adstock Parameter: Decay Rate', fontsize=16)


def plot_model_fit(model_fit_result, true_y, x, percentile_list=[10, 50, 90], x_index=None):
    samples = np.percentile(model_fit_result['predicted_y'], q=percentile_list, axis=0)
    if x_index is None:
        fig, ax = plt.subplots(figsize=(15, 8))
        sns.lineplot(x=x, y=true_y, ax=ax, label='true_y')
        for i in range(len(samples)):
            sns.lineplot(x=x, y=samples[i], ax=ax, label='{0:.0f}% percentile - MAPE: {1:.1f}%'.format(percentile_list[i], calc_mape(true_y, samples[i])))
            ax.lines[i+1].set_linestyle("--")
        ax.set_title('True Y vs. Fitted Y', fontsize=16)
    else:
        f = plt.figure(figsize=(35, 30))
        cross_sections = x_index['CrossSection'].drop_duplicates().to_list()
        for ci, c in enumerate(cross_sections):
            cross_section_index = x_index.loc[x_index['CrossSection'] == c, 'RowIndex'].to_list()
            ax = f.add_subplot(4, 2, ci + 1)
            ax = sns.lineplot(x=x[cross_section_index], y=true_y[cross_section_index], ax=ax, label='true_y')
            for i in range(len(samples)):
                sns.lineplot(x=x[cross_section_index], y=samples[i][cross_section_index], ax=ax, label='{0:.0f}% percentile - MAPE: {1:.1f}%'.format(percentile_list[i], calc_mape(true_y[cross_section_index], samples[i][cross_section_index])))
                ax.lines[i + 1].set_linestyle("--")
            ax.set_title('{} - True Y vs. Fitted Y'.format(c), fontsize=16)


def plot_model_stats(simulated_results, x='MAPE', y='SOSSOC', metric='MAPE_SOSSOC', selected_model_index=None, alpha=0.05, cross_section='[TOTAL]'):

    selected_model_index = -1 if selected_model_index is None else selected_model_index
    colors = ['#1f77b4' if i != selected_model_index else '#ff7f0e' for i in simulated_results['ModelIndex']]
    hpd_mu, x_mu, y_mu, modes_mu = hpd_grid(simulated_results['ModelStats'][cross_section][metric], alpha, roundto=10)
    for i in range(len(colors)):
        for (x0, x1) in hpd_mu:
            if x0 <= simulated_results['ModelStats'][cross_section][metric][i] <= x1 and simulated_results['ModelIndex'][i] != selected_model_index:
                colors[i] = 'blue'
    data = simulated_results['ModelStats'][cross_section]
    my_fig = go.Figure(
                go.Scatter(
                    x=data[x],
                    y=data[y],
                    mode='markers',
                    marker=dict(
                        # line=dict(color='blue'),
                        symbol=['circle' if i != selected_model_index else 'star' for i in simulated_results['ModelIndex']],
                        size=[8 if i != selected_model_index else 16 for i in simulated_results['ModelIndex']],
                        color=colors
                    ),
                    hovertext=['<b> Cross Section: {} </b><br><b> Model Index: {} </b><br> R2: {:,.1f} <br> MAPE: {:,.1f} <br> SOSSOC: {:,.1f}'.format(cross_section.upper(), i, data['R2'][c], data['MAPE'][c], data['SOSSOC'][c]) for c, i in enumerate(simulated_results['ModelIndex'])],
                )
            )
    my_fig.update_layout(get_layout(chart_title='<b>{} vs. {}</b>'.format(y, x)))
    my_fig.update_xaxes(title='<b>{}</b>'.format(x), showline=True, linewidth=1, linecolor='grey')
    my_fig.update_yaxes(title='<b>{}</b>'.format(y), showline=True, linewidth=1, linecolor='grey')
    my_fig.show()

    return my_fig


def plot_posterior_distribution(variables, params, para_name='beta', selected_model_index=None, alpha=0.05):
    # red line: mean, blue line: median, orange line: selected
    para_values = {var: np.array([params[para_name][j][i] for j in range(len(params[para_name]))]) for i, var in enumerate(variables)}
    if para_name == 'beta':
        para_values['tau'] = np.array(params['tau'])
    f = plt.figure(figsize=(35, 30))
    for i, p in enumerate(list(para_values.keys())):
        ax = f.add_subplot(7, 5, i + 1)
        x = para_values[p]
        mean_x = x.mean()
        median_x = np.median(x)
        ax = sns.distplot(x)
        ax.axvline(mean_x, color='r', linestyle='-')
        ax.axvline(median_x, color='b', linestyle='-')
        if selected_model_index is not None:
            best_x = x[selected_model_index]
            ax.axvline(best_x, color='orange', linestyle='--')

        # Compute high density regions
        hpd_mu, x_mu, y_mu, modes_mu = hpd_grid(para_values[p], alpha, roundto=10)
        for (x0, x1) in hpd_mu:
            ax.hlines(y=0, xmin=x0, xmax=x1, linewidth=5)
            ax.axvline(x=x0, color='grey', linestyle='--', linewidth=1)
            ax.axvline(x=x1, color='grey', linestyle='--', linewidth=1)
        ax.set_title(p)


def plot_model_output_distribution(params, selected_model_index=None, alpha=0.05, cross_section='[TOTAL]'):
    # red line: mean, blue line: median, orange line: selected
    f = plt.figure(figsize=(35, 30))
    data = params[cross_section]
    for i, para_name in enumerate(list(data.keys())):
        ax = f.add_subplot(7, 5, i + 1)
        x = np.array(data[para_name])
        mean_x = x.mean()
        median_x = np.median(x)
        ax = sns.distplot(x)
        ax.axvline(mean_x, color='r', linestyle='-')
        ax.axvline(median_x, color='b', linestyle='-')
        if selected_model_index is not None:
            selected_x = x[selected_model_index]
            ax.axvline(selected_x, color='orange', linestyle='--')

        # Compute high density regions
        hpd_mu, x_mu, y_mu, modes_mu = hpd_grid(data[para_name], alpha, roundto=10)
        for (x0, x1) in hpd_mu:
            ax.hlines(y=0, xmin=x0, xmax=x1, linewidth=5)
            ax.axvline(x=x0, color='grey', linestyle='--', linewidth=1)
            ax.axvline(x=x1, color='grey', linestyle='--', linewidth=1)
        ax.set_title(para_name)


def plot_response_curves(params, selected_model_index=None, cross_section='[TOTAL]'):
    from random import randrange
    f = plt.figure(figsize=(35, 30))
    random_indices = [randrange(len(params[list(params.keys())[0]])) for j in range(1000)]
    selected_model_index = -1 if selected_model_index is None else selected_model_index
    if selected_model_index >= 0:
        random_indices.append(selected_model_index)
    data = params[cross_section]
    for i, para_name in enumerate(list(data.keys())):
        ax = f.add_subplot(7, 5, i + 1)
        for ci, curve in enumerate(data[para_name]):
            if ci in random_indices or ci == selected_model_index:
                x = np.array(curve['x'])
                y = np.array(curve['y'])
                if selected_model_index is not None and ci == selected_model_index:
                    sns.scatterplot(x=x, y=y, ax=ax, label='Selected Model', color='orange', s=10)
                else:
                    sns.scatterplot(x=x, y=y, ax=ax, color='blue', s=1)
        ax.set_title(para_name)


def plot_model_actual_vs_predicted(df_model_contributions, mmm_model_output):

    y_var = mmm_model_output['dep_var']
    df_model_contributions['residual'] = df_model_contributions[y_var] - df_model_contributions['predicted_' + y_var]
    columns_to_plot = [y_var, 'predicted_' + y_var, 'residual']
    colors = ['blue', 'orange', 'green']
    layout = get_layout(chart_title='<b>Actual vs. Predicted</b>')
    if 'CrossSection' not in df_model_contributions.columns:
        my_fig = go.Figure(layout=layout)
        for index, col in enumerate(columns_to_plot):
            my_fig.add_trace(
                go.Scatter(
                    x=df_model_contributions['Date'],
                    y=df_model_contributions[col],
                    name=col,
                    line=dict(color=colors[index])
                ))
        my_fig.update_xaxes(tickformat='%Y-%m-%d', tickangle=-90)
        my_fig.show()
    else:
        cross_sections = df_model_contributions['CrossSection'].drop_duplicates().to_list()
        columns_count = 2
        rows_count = int(round_up(len(cross_sections) / columns_count))
        for row_index in range(rows_count):
            my_fig = make_subplots(
                rows=1, cols=columns_count,
                subplot_titles=['<b>{}</b>'.format(cross_section.upper()) for cross_section in cross_sections[(row_index * columns_count):((row_index + 1) * columns_count)]]
            )
            my_fig.update_layout(layout)
            for col_index in range(columns_count):
                cross_section_index = row_index * columns_count + col_index
                if cross_section_index == len(cross_sections):
                    break
                cross_section = cross_sections[cross_section_index]
                df_cross_section = df_model_contributions[df_model_contributions['CrossSection'] == cross_section]
                for index, col in enumerate(columns_to_plot):
                    my_fig.add_trace(
                        go.Scatter(
                            x=df_cross_section['Date'],
                            y=df_cross_section[col],
                            name=col,
                            line=dict(color=colors[index]),
                            legendgroup=col,
                            showlegend=True if row_index == col_index == 0 else False
                        ),
                        row=1, col=col_index + 1
                    )
                my_fig.update_xaxes(tickformat='%Y-%m-%d', tickangle=-90)
                # my_fig.layout.annotations[col_index].update(text='MAPE: {} - R2: {}'.format(calc_mape(df_cross_section[y_var], df_cross_section['predicted_' + y_var]), calc_r2(df_cross_section[y_var], df_cross_section['predicted_' + y_var])))
            my_fig.show()

    return my_fig


def plot_model_contributions(df_model_contributions, mmm_model_output):

    y_var, media_vars, comp_media_vars, positive_vars, neutral_vars, negative_vars = mmm_model_output['dep_var'], mmm_model_output['media_vars'], mmm_model_output['comp_media_vars'], mmm_model_output['positive_vars'], mmm_model_output['neutral_vars'], mmm_model_output['negative_vars']

    columns_to_plot = comp_media_vars + ['intercept'] + media_vars + positive_vars + neutral_vars + negative_vars
    colors = ['blue', 'orange']
    contribution_colors = get_default_colors() + [get_random_color() for x in columns_to_plot]
    layout = get_layout(chart_title='<b>Model Contributions</b>')
    if 'CrossSection' not in df_model_contributions.columns:
        my_fig = go.Figure(layout=layout)
        for index, col in enumerate([y_var, 'predicted_' + y_var]):
            my_fig.add_trace(
                go.Scatter(
                    x=df_model_contributions['Date'],
                    y=df_model_contributions[col],
                    name=col,
                    line=dict(color=colors[index])
                ))
        for index, col in enumerate(columns_to_plot):
            my_fig.add_trace(
                go.Bar(
                    x=df_model_contributions['Date'],
                    y=df_model_contributions[col],
                    name=shorten_name('|'.join(col.split('|')[0:4])),
                    marker_color=contribution_colors[index],
                    opacity=1.0
                ))
        my_fig.update_layout(barmode='stack')
        my_fig.show()
    else:
        cross_sections = df_model_contributions['CrossSection'].drop_duplicates().to_list()
        columns_count = 2
        rows_count = int(round_up(len(cross_sections) / columns_count))
        for row_index in range(rows_count):
            my_fig = make_subplots(
                rows=1, cols=columns_count,
                subplot_titles=['<b>{}</b>'.format(cross_section.upper()) for cross_section in cross_sections[(row_index * columns_count):((row_index + 1) * columns_count)]]
            )
            my_fig.update_layout(layout)
            for col_index in range(columns_count):
                cross_section_index = row_index * columns_count + col_index
                if cross_section_index == len(cross_sections):
                    break
                cross_section = cross_sections[cross_section_index]
                df = df_model_contributions[df_model_contributions['CrossSection'] == cross_section]
                for index, col in enumerate([y_var, 'predicted_' + y_var]):
                    my_fig.add_trace(
                        go.Scatter(
                            x=df['Date'],
                            y=df[col],
                            name=col,
                            line=dict(color=colors[index]),
                            legendgroup=col,
                            showlegend=True if row_index == col_index == 0 else False
                        ),
                        row=1, col=col_index + 1
                    )
                for index, col in enumerate(columns_to_plot):
                    my_fig.add_trace(
                        go.Bar(
                            x=df['Date'],
                            y=df[col],
                            name=shorten_name('|'.join(col.split('|')[0:4])),
                            marker_color=contribution_colors[index],
                            opacity=1.0,
                            legendgroup=col,
                            showlegend=True if row_index == col_index == 0 else False
                        ),
                        row=1, col=col_index + 1
                    )
                my_fig.update_xaxes(tickformat='%Y-%m-%d', tickangle=-90)
            my_fig.update_layout(barmode='stack')
            my_fig.show()

    return my_fig


def plot_model_decomposition(df_model_contributions, mmm_model_output, media_only=True):

    y_var, media_vars, comp_media_vars, positive_vars, neutral_vars, negative_vars = mmm_model_output['dep_var'], mmm_model_output['media_vars'], mmm_model_output['comp_media_vars'], mmm_model_output['positive_vars'], mmm_model_output['neutral_vars'], mmm_model_output['negative_vars']
    columns_to_plot = comp_media_vars + ['intercept'] + media_vars + positive_vars + neutral_vars + negative_vars if not media_only else media_vars
    contribution_colors = get_default_colors() + [get_random_color() for x in columns_to_plot]
    layout = get_layout(chart_title='<b>Model Decomposition - {}</b>'.format(y_var.upper()))
    predicted_y_var = 'predicted_' + y_var
    if 'CrossSection' not in df_model_contributions.columns:
        filtered_records = df_model_contributions[predicted_y_var].notnull()
        model_decomposition = {}
        for col in columns_to_plot:
            model_decomposition[col] = {
                'a': df_model_contributions[filtered_records][col].sum(),
                'p': df_model_contributions[filtered_records][col].sum() / df_model_contributions[filtered_records][predicted_y_var].sum()
            }
        my_fig = go.Figure(
            go.Bar(
                y=[values['a'] for col, values in model_decomposition.items()],
                x=['<b>' + shorten_name('|'.join(col.split('|')[0:4])) + '</b>' for col in model_decomposition],
                text=['{:,.0f}'.format(values['a']) + ' (' + '{:.1%}'.format(values['p']) + ')' for col, values in model_decomposition.items()],
                textposition='auto',
                hovertext=['<b>' + shorten_name(col) + '</b>: <br>' + '{:,.0f}'.format(values['a']) + ' (' + '{:.1%}'.format(values['p']) + ')' for col, values in model_decomposition.items()],
                orientation='v',
                marker_color=contribution_colors[0:len(model_decomposition.keys())],
                # marker_color='blue',
                opacity=1.0,
            ),
            layout=layout)
        my_fig.show()
    else:
        cross_sections = df_model_contributions['CrossSection'].drop_duplicates().to_list()
        columns_count = 2
        rows_count = int(round_up(len(cross_sections) / columns_count))
        for row_index in range(rows_count):
            my_fig = make_subplots(
                rows=1, cols=columns_count,
                subplot_titles=['<b>{}</b>'.format(cross_section.upper()) for cross_section in cross_sections[(row_index*columns_count):((row_index+1)*columns_count)]]
            )
            my_fig.update_layout(layout)
            for col_index in range(columns_count):
                cross_section_index = row_index * columns_count + col_index
                if cross_section_index == len(cross_sections):
                    break
                cross_section = cross_sections[cross_section_index]
                model_decomposition = {}
                filtered_records = (df_model_contributions[predicted_y_var].notnull()) & (df_model_contributions['CrossSection'] == cross_section)
                for col in columns_to_plot:
                    model_decomposition[col] = {
                        'a': df_model_contributions[filtered_records][col].sum(),
                        'p': df_model_contributions[filtered_records][col].sum() / df_model_contributions[filtered_records][predicted_y_var].sum()
                    }
                my_fig.add_trace(
                    go.Bar(
                        y=[values['a'] for col, values in model_decomposition.items()],
                        x=['<b>' + shorten_name('|'.join(col.split('|')[0:4])) + '</b>' for col in model_decomposition],
                        text=['{:,.0f}'.format(values['a']) + ' (' + '{:.1%}'.format(values['p']) + ')' for col, values in model_decomposition.items()],
                        textposition='auto',
                        hovertext=['<b>' + shorten_name(col) + '</b>: <br>' + '{:,.0f}'.format(values['a']) + ' (' + '{:.1%}'.format(values['p']) + ')' for col, values in model_decomposition.items()],
                        orientation='v',
                        marker_color='blue',
                        opacity=1.0
                    ),
                    row=1, col=col_index + 1
                )
                my_fig.update_xaxes(tickangle=0)
                my_fig.update_yaxes(tickformat=',.0f')
            my_fig.show()

    return my_fig


def plot_waterfall(dict_decomposition, cross_sections=None):

    if cross_sections is None or len(cross_sections) == 0:
        cross_sections = [c for c in list(dict_decomposition['SpendShare'].keys())]
    else:
        cross_sections = [c for c in list(dict_decomposition['SpendShare'].keys()) if c in cross_sections]
        cross_sections = ['[TOTAL]'] if len(cross_sections) == 0 else cross_sections
    columns_count = 2 if len(cross_sections) > 1 else 1
    rows_count = int(round_up(len(cross_sections) / columns_count))
    layout = get_layout(chart_title='<b>Model Decomposition</b>')
    for row_index in range(rows_count):
        my_fig = make_subplots(
            rows=1, cols=columns_count,
            subplot_titles=['<b>{}</b>'.format(cross_section.upper()) for cross_section in cross_sections[(row_index * columns_count):((row_index + 1) * columns_count)]]
        )
        my_fig.update_layout(layout)
        for col_index in range(columns_count):
            cross_section_index = row_index * columns_count + col_index
            if cross_section_index == len(cross_sections):
                break
            cross_section = cross_sections[cross_section_index]
            media_contributions = dict_decomposition['MediaContributionsSummary'][cross_section]
            summary_contributions = dict_decomposition['ContributionsSummary'][cross_section]
            baseline_contribution = sum([v for k, v in summary_contributions.items() if k not in list(media_contributions.keys())])
            total_contributions = sum([v for k, v in summary_contributions.items()])
            share_of_contributions = dict_decomposition['MediaContributionsShare'][cross_section]
            baseline_share_of_contributions = baseline_contribution / total_contributions if total_contributions > 0 else 0

            channels = ['<b>Baseline</b>'] + ['<b>{}</b>'.format(k) for k in media_contributions.keys()]
            measures = ['relative'] * len(channels) + ['total']
            channels += ['<b>TOTAL</b>']

            colors = get_default_colors() + [get_random_color()] * len(channels)
            my_fig.add_trace(
                go.Waterfall(
                    name='Media Contributions',
                    orientation='v',
                    measure=measures,
                    x=channels,
                    y=[baseline_contribution] + [v for k, v in media_contributions.items()] + [total_contributions],
                    textposition='auto',
                    text=['<b>{:,.0f}<br>({:,.1%})</b>'.format(baseline_contribution, baseline_share_of_contributions)] + ['{:,.0f}<br>({:,.1%})'.format(v, share_of_contributions[k]) for k, v in media_contributions.items()] + [
                        '<b>{:,.0f}<br>({:,.1%})</b>'.format(total_contributions, 1)],
                    hovertext=['<b>Baseline</b>'] + ['<b>{}</b> <br>Media Contributions: {:,.0f} ({:,.1%})'.format(k, v, share_of_contributions[k]) for k, v in media_contributions.items()] + ['<b>TOTAL</b>'],
                    # marker_color=colors,
                    connector={"line": {"color": "rgb(63, 63, 63)"}}
                ),
                row=1, col=col_index + 1
            )
            my_fig.update_xaxes(tickangle=0)
            my_fig.update_yaxes(tickformat=',.0f')
        my_fig.show()

    return my_fig


def plot_share_of_spend_vs_contributions(dict_decomposition, cross_sections=None):

    if cross_sections is None or len(cross_sections) == 0:
        cross_sections = [c for c in list(dict_decomposition['SpendShare'].keys())]
    else:
        cross_sections = [c for c in list(dict_decomposition['SpendShare'].keys()) if c in cross_sections]
        cross_sections = ['[TOTAL]'] if len(cross_sections) == 0 else cross_sections
    columns_count = 2 if len(cross_sections) > 1 else 1
    rows_count = int(round_up(len(cross_sections) / columns_count))
    layout = get_layout(chart_title='<b>Share of Media Spend vs. Share of Media Contributions</b>')
    for row_index in range(rows_count):
        my_fig = make_subplots(
            rows=1, cols=columns_count,
            subplot_titles=['<b>{}</b>'.format(cross_section.upper()) for cross_section in cross_sections[(row_index * columns_count):((row_index + 1) * columns_count)]]
        )
        my_fig.update_layout(layout)
        for col_index in range(columns_count):
            cross_section_index = row_index * columns_count + col_index
            if cross_section_index == len(cross_sections):
                break
            cross_section = cross_sections[cross_section_index]
            channels = ['<b>{}</b>'.format(k) for k in dict_decomposition['SpendShare'][cross_section].keys()]
            share_of_spend = dict_decomposition['SpendShare'][cross_section]
            share_of_contributions = dict_decomposition['MediaContributionsShare'][cross_section]

            my_fig.add_trace(
                go.Bar(
                    x=channels,
                    y=[v for k, v in share_of_spend.items()],
                    name='Share of Media Spend',
                    text=['{:,.1%}'.format(v) for k, v in share_of_spend.items()],
                    textposition='auto',
                    hovertext=['<b>{}</b> <br>Share of Media Spend: {:,.1%}'.format(k, v) for k, v in share_of_spend.items()],
                    marker_color='#1f77b4'
                ),
                row=1, col=col_index + 1
            )
            my_fig.add_trace(
                go.Bar(
                    x=channels,
                    y=[v for k, v in share_of_contributions.items()],
                    name='Share of Media Contributions',
                    text=['{:,.1%}'.format(v) for k, v in share_of_contributions.items()],
                    textposition='auto',
                    hovertext=['<b>{}</b> <br>Share of Media Contributions: {:,.1%}'.format(k, v) for k, v in share_of_contributions.items()],
                    marker_color='#ff7f0e'
                ),
                row=1, col=col_index + 1
            )
            my_fig.update_xaxes(tickangle=0)
            my_fig.update_yaxes(tickformat=',.0%')
        my_fig.show()

    return my_fig


def plot_contributions_vs_roas(dict_decomposition, cross_sections=None):

    if cross_sections is None or len(cross_sections) == 0:
        cross_sections = [c for c in list(dict_decomposition['SpendShare'].keys())]
    else:
        cross_sections = [c for c in list(dict_decomposition['SpendShare'].keys()) if c in cross_sections]
        cross_sections = ['[TOTAL]'] if len(cross_sections) == 0 else cross_sections
    columns_count = 2 if len(cross_sections) > 1 else 1
    rows_count = int(round_up(len(cross_sections) / columns_count))
    layout = get_layout(chart_title='<b>Media Contributions vs. Media ROAS</b>')
    for row_index in range(rows_count):
        my_fig = make_subplots(
            rows=1, cols=columns_count,
            subplot_titles=['<b>{}</b>'.format(cross_section.upper()) for cross_section in cross_sections[(row_index * columns_count):((row_index + 1) * columns_count)]],
            specs=[[{"secondary_y": True}] * columns_count]
        )
        my_fig.update_layout(layout)
        for col_index in range(columns_count):
            cross_section_index = row_index * columns_count + col_index
            if cross_section_index == len(cross_sections):
                break
            cross_section = cross_sections[cross_section_index]
            channels = ['<b>{}</b>'.format(k) for k in dict_decomposition['SpendSummary'][cross_section].keys()]
            media_spend = dict_decomposition['SpendSummary'][cross_section]
            media_contributions = dict_decomposition['MediaContributionsSummary'][cross_section]
            share_of_spend = dict_decomposition['SpendShare'][cross_section]
            share_of_contributions = dict_decomposition['MediaContributionsShare'][cross_section]

            colors = get_default_colors() + [get_random_color()] * len(channels)
            my_fig.add_trace(
                go.Bar(
                    x=channels,
                    y=[v for k, v in media_contributions.items()],
                    name='Media Contributions',
                    text=['{:,.0f}<br>({:,.1%})'.format(v, share_of_contributions[k]) for k, v in media_contributions.items()],
                    textposition='auto',
                    hovertext=['<b>{}</b> <br>Media Contributions: {:,.0f} ({:,.1%})'.format(k, v, share_of_contributions[k]) for k, v in media_contributions.items()],
                    marker_color=colors
                ),
                secondary_y=False,
                row=1, col=col_index + 1
            )
            my_fig.add_trace(
                go.Scatter(
                    x=channels,
                    y=[v / media_spend[k] for k, v in media_contributions.items()],
                    name='ROAS',
                    # text=['{:,.0f}'.format(v / media_spend[k]) for k, v in media_contributions.items()],
                    # textposition='top center',
                    hovertext=['<b>{}</b> <br>ROAS: {:,.1f}'.format(k, v / media_spend[k]) for k, v in media_contributions.items()],
                    marker_color='#ff7f0e'
                ),
                secondary_y=True,
                row=1, col=col_index + 1
            )
            my_fig.update_xaxes(tickangle=0)
            my_fig.update_yaxes(tickformat=',.0f')
        my_fig.show()

    return my_fig


def plot_spend_vs_roas(dict_decomposition, cross_sections=None):

    if cross_sections is None or len(cross_sections) == 0:
        cross_sections = [c for c in list(dict_decomposition['SpendShare'].keys())]
    else:
        cross_sections = [c for c in list(dict_decomposition['SpendShare'].keys()) if c in cross_sections]
        cross_sections = ['[TOTAL]'] if len(cross_sections) == 0 else cross_sections
    columns_count = 2 if len(cross_sections) > 1 else 1
    rows_count = int(round_up(len(cross_sections) / columns_count))
    layout = get_layout(chart_title='<b>Media Spend vs. Media ROAS</b>')
    for row_index in range(rows_count):
        my_fig = make_subplots(
            rows=1, cols=columns_count,
            subplot_titles=['<b>{}</b>'.format(cross_section.upper()) for cross_section in cross_sections[(row_index * columns_count):((row_index + 1) * columns_count)]],
            specs=[[{"secondary_y": True}] * columns_count]
        )
        my_fig.update_layout(layout)
        for col_index in range(columns_count):
            cross_section_index = row_index * columns_count + col_index
            if cross_section_index == len(cross_sections):
                break
            cross_section = cross_sections[cross_section_index]
            channels = ['<b>{}</b>'.format(k) for k in dict_decomposition['SpendSummary'][cross_section].keys()]
            media_spend = dict_decomposition['SpendSummary'][cross_section]
            media_contributions = dict_decomposition['MediaContributionsSummary'][cross_section]
            share_of_spend = dict_decomposition['SpendShare'][cross_section]
            share_of_contributions = dict_decomposition['MediaContributionsShare'][cross_section]

            colors = get_default_colors() + [get_random_color()] * len(channels)
            my_fig.add_trace(
                go.Bar(
                    x=channels,
                    y=[v for k, v in media_spend.items()],
                    name='Media Spend',
                    text=['{:,.0f}<br>({:,.1%})'.format(v, share_of_spend[k]) for k, v in media_spend.items()],
                    textposition='auto',
                    hovertext=['<b>{}</b> <br>Media Spend: {:,.0f} ({:,.1%})'.format(k, v, share_of_spend[k]) for k, v in media_spend.items()],
                    marker_color=colors
                ),
                secondary_y=False,
                row=1, col=col_index + 1
            )
            my_fig.add_trace(
                go.Scatter(
                    x=channels,
                    y=[v / media_spend[k] for k, v in media_contributions.items()],
                    name='ROAS',
                    # text=['{:,.0f}'.format(v / media_spend[k]) for k, v in media_contributions.items()],
                    # textposition='auto',
                    hovertext=['<b>{}</b> <br>ROAS: {:,.1f}'.format(k, v / media_spend[k]) for k, v in media_contributions.items()],
                    marker_color='#ff7f0e'
                ),
                secondary_y=True,
                row=1, col=col_index + 1
            )
            my_fig.update_xaxes(tickangle=0)
            my_fig.update_yaxes(tickformat=',.0f')
        my_fig.show()

    return my_fig


def get_layout(chart_title):
    return go.Layout(
        title=chart_title,
        autosize=True,
        barmode='group',
        plot_bgcolor='#ffffff',
        showlegend=False,
        uniformtext_minsize=8,
        uniformtext_mode='hide'
        # legend=dict(yanchor="top", y=1.02, xanchor="left", x=0.01)
    )


# -----------------------------------------------------------------------------------------
# MATH
# -----------------------------------------------------------------------------------------
def calc_mape(true_y, predicted_y):
    true_y, predicted_y = np.array(true_y), np.array(predicted_y)
    return np.mean(np.abs((true_y - predicted_y) / true_y)) * 100


def calc_r2(true_y, predicted_y):
    y_mean = true_y.mean()
    sst = sum([math.pow(v - y_mean, 2) for v in true_y])
    ssr = sum([math.pow(v, 2) for v in true_y - predicted_y])
    return (1 - ssr / sst) * 100


# -----------------------------------------------------------------------------------------
# EXPORT
# -----------------------------------------------------------------------------------------
def export_model_output(mmm_model_output, df, df_model, model_params, selected_model_index=None):

    model_name = model_params['ModelName']
    model_form = model_params['ModelForm']

    dep_var = model_params['DependentVariable']
    media_vars = model_params['MediaVariables']['MediaQuantity']
    media_spend_columns = model_params['MediaVariables']['MediaSpend']
    comp_media_vars = model_params['CompetitiveMediaVariables']['MediaQuantity']
    positive_vars = model_params['ControlVariables']['Positive']
    neutral_vars = model_params['ControlVariables']['Neutral']
    negative_vars = model_params['ControlVariables']['Negative']

    cross_section_col = model_params['CrossSection'] if 'CrossSection' in model_params.keys() else None
    date_col = model_params['Date']
    df_date = df[date_col]

    df_media_spend = df[media_spend_columns].copy()
    df_media_spend.index = df[date_col]
    df_media_spend = df_media_spend.rename(columns={col: shorten_name(col) for col in df_media_spend.columns})

    # Data
    id_columns = [date_col] if cross_section_col is None else [cross_section_col, date_col]
    variables_to_melt = media_vars + media_spend_columns + comp_media_vars + positive_vars + neutral_vars + negative_vars
    df_melted = pd.melt(df, id_vars=id_columns, value_vars=variables_to_melt)
    df_melted.columns = id_columns + ['Variable', 'Quantity']
    df_melted['Channel'] = df_melted['Variable'].apply(lambda r: shorten_name(r))
    df_melted['VariableCategory'] = df_melted['Variable'].apply(lambda r: 'spend' if 'spend' in r.lower().split('_') else 'quantity')
    df_melted['Spend'] = 0
    df_melted.loc[df_melted['VariableCategory'] == 'spend', 'Spend'] = df_melted.loc[df_melted['VariableCategory'] == 'spend', 'Quantity']
    df_melted.loc[df_melted['VariableCategory'] == 'spend', 'Quantity'] = 0

    # Model Contributions
    df_contributions, df_lifetime_contributions, df_model_specifications = calc_model_contributions(df, df_model, mmm_model_output, df_media_spend, selected_model_index, model_form)

    # Model Contributions
    id_columns = ['Date'] if 'CrossSection' not in df_contributions.columns else ['CrossSection', 'Date']
    variables_to_melt = media_vars + comp_media_vars + positive_vars + neutral_vars + negative_vars
    df_con_melted = pd.melt(df_contributions, id_vars=id_columns, value_vars=variables_to_melt)
    df_con_melted.columns = id_columns + ['Variable', 'ModelContribution']
    df_con_melted['Channel'] = df_con_melted['Variable'].apply(lambda r: shorten_name(r))

    # LifeTime Contributions
    if df_lifetime_contributions is not None and len(df_lifetime_contributions) > 0:
        variables_to_melt = media_vars + comp_media_vars + positive_vars + neutral_vars + negative_vars
        df_lifetime_con_melted = pd.melt(df_lifetime_contributions, id_vars=id_columns, value_vars=variables_to_melt)
        df_lifetime_con_melted.columns = id_columns + ['Variable', 'LifeTimeContribution']
        df_lifetime_con_melted['Channel'] = df_lifetime_con_melted['Variable'].apply(lambda r: shorten_name(r))
        df_con_melted = pd.merge(df_con_melted, df_lifetime_con_melted[id_columns + ['Channel', 'LifeTimeContribution']], on=id_columns + ['Channel'], how='left')
    else:
        df_con_melted['LifeTimeContribution'] = np.nan

    # Media Quantity frame
    df_media_quantity = df[[col for col in df.columns if col in media_vars]].copy()
    if df_media_quantity is not None and len(df_media_quantity) > 0:
        if len(df_media_quantity.index[0]) == 1:
            df_media_quantity['Date'] = df_date
        else:
            df_media_quantity['CrossSection'] = [i[0] for i in df_media_quantity.index]
            df_media_quantity['Date'] = [i[1] for i in df_media_quantity.index]
        variables_to_melt = [col for col in df_media_quantity.columns if col in media_vars]
        df_quantity_melted = pd.melt(df_media_quantity, id_vars=id_columns, value_vars=variables_to_melt)
        df_quantity_melted.columns = id_columns + ['Variable', 'Impressions']
        df_quantity_melted['Channel'] = df_quantity_melted['Variable'].apply(lambda r: shorten_name(r))
        df_con_melted = pd.merge(df_con_melted, df_quantity_melted[id_columns + ['Channel', 'Impressions']], on=id_columns + ['Channel'], how='left')
    else:
        df_con_melted['Impressions'] = np.nan

    # Spend data frame
    df_media_spend = df[[col for col in df.columns if col in media_spend_columns]].copy()
    if df_media_spend is not None and len(df_media_spend) > 0:
        if len(df_media_spend.index[0]) == 1:
            df_media_spend['Date'] = df_date
        else:
            df_media_spend['CrossSection'] = [i[0] for i in df_media_spend.index]
            df_media_spend['Date'] = [i[1] for i in df_media_spend.index]
        variables_to_melt = [col for col in df_media_spend.columns if col in media_spend_columns]
        df_spend_melted = pd.melt(df_media_spend, id_vars=id_columns, value_vars=variables_to_melt)
        df_spend_melted.columns = id_columns + ['Variable', 'Spend']
        df_spend_melted['Channel'] = df_spend_melted['Variable'].apply(lambda r: shorten_name(r))
        df_con_melted = pd.merge(df_con_melted, df_spend_melted[id_columns + ['Channel', 'Spend']], on=id_columns + ['Channel'], how='left')
    else:
        df_con_melted['Spend'] = np.nan

    df_model_contributions = df_con_melted[id_columns + ['Channel', 'Variable', 'Impressions', 'Spend', 'ModelContribution', 'LifeTimeContribution']]

    return df_model_contributions, df_model_specifications, df_melted


# -----------------------------------------------------------------------------------------
# UTILITIES
# -----------------------------------------------------------------------------------------
def subset_df(df, cols):
    if len(cols) > 0:
        df_out = df[cols]
        df_out.index = df.index
        return df_out
    else:
        return pd.DataFrame()


def save_json(data, file_name):
    with open(file_name, 'w') as fp:
        json.dump(data, fp)


def load_json(file_name):
    with open(file_name, 'r') as fp:
        data = json.load(fp)
    return data


def save_pickle(data, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(file_name):
    with open(file_name, 'rb') as fp:
        data = pickle.load(fp)
    return data


def shorten_name(col):
    for v in ['MM|Impressions|', 'MM|Clicks|', 'MM|GRP|', 'MM|Spend|', 'impr_', 'spend_', '_impressions', '_impression', '_spend', '_clicks']:
        col = col.replace(v, '')
    return col


def write_worksheet(df, sheet_name, writer, freeze_panes=(1, 10), zoom_level=90):
    if df is not None:
        print(' - Writing {} sheet data. {} Rows & {} columns'.format(sheet_name, len(df), len(df.columns)))
        df.to_excel(writer, sheet_name=sheet_name, header=True, index=False, startrow=0, freeze_panes=freeze_panes)  # note that we start from Row 1
        worksheet = writer.sheets[sheet_name]
        worksheet.set_zoom(zoom_level)

        try:
            header_format = writer.book.add_format({
                'bold': True,
                'text_wrap': True
            })
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)
        except:
            pass
        return worksheet
    else:
        print(' - Skipped {} sheet data.'.format(sheet_name))
        return None


def get_column_index(df, col_name):
    try:
        col_num = df.columns.tolist().index(col_name)
    except:
        col_num = -1
    return col_num


def write_excel(data, file_name):
    # Write to Excel
    writer = pd.ExcelWriter(file_name, engine='xlsxwriter')
    workbook = writer.book
    zoom_level = 90
    header_format = workbook.add_format({'bold': True, 'fg_color': '#d4ebf2', 'text_wrap': True})

    for k, df_data in data.items():
        output_sheet = write_worksheet(df_data, k, writer, freeze_panes=(1, 0), zoom_level=zoom_level)

        for col_name in df_data.columns:
            col_num = get_column_index(df_data, col_name)
            output_sheet.write(0, col_num, col_name, header_format)

    writer.save()


def round_up(n, decimals=0):
    round_multiplier = 10 ** decimals
    return math.ceil(n*round_multiplier) / round_multiplier


def get_random_color():
    return 'rgb(' + ', '.join([str(k) for k in list(np.random.choice(range(256), size=3))]) + ')'


def get_default_colors():
    return ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


def beauty_display(df, num_rows=5):
    from IPython.core.display import display, HTML
    display(df[0:num_rows].style.set_properties(**{'text-align': 'right', 'white-space': 'nowrap'}))


