# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 16:08:59 2026

@author: internet
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import label, sum as ndi_sum, generate_binary_structure, maximum_position
from skimage import measure
import os
import pydicom
from tkinter import Tk, filedialog
from skimage.transform import resize
import nrrd
from datetime import datetime, date
import tkinter as tk
from tkinter import ttk
from tkcalendar import Calendar
from radiomics import featureextractor, getTestCase
from tkinter import simpledialog
from sklearn.linear_model import ElasticNetCV, LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from collections import deque
import ast
import seaborn as sns
from scipy.optimize import curve_fit
import itertools
from scipy.stats import ttest_ind, chi2_contingency
from sklearn.feature_selection import VarianceThreshold

#הפונ' עושה רגרסיה לינארית על 2 עמודות
def plot_linear_regression(df, x_col, y_col, test_size=0.2, upper_threshold=300):

    # Drop NA values
    df_clean = df[[x_col, y_col]].dropna()
    df_clean = df_clean[df_clean[x_col] <upper_threshold]
    # Split into training and testing
    train_df, test_df = train_test_split(df_clean, test_size=test_size)

    # Prepare data
    X_train = train_df[[x_col]].values
    y_train = train_df[y_col].values
    X_test = test_df[[x_col]].values
    y_test = test_df[y_col].values

    # Fit model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on train and test
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    mse_train=np.abs(y_pred_train-y_train)
    mse_test=np.abs(y_pred_test-y_test)
    # Evaluate
    slope = model.coef_[0]
    intercept = model.intercept_
    r2 = r2_score(y_train, y_pred_train)
    
    low_mask = test_df[x_col] < upper_threshold  # לדוגמה
    mse_low = mean_squared_error(y_test[low_mask], y_pred_test[low_mask])
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.scatter(X_train, y_train, color='blue', label='Training data')
    ax1.scatter(X_test, y_test, color='orange', label='Test data')
    ax1.set_xlabel(x_col)
    ax1.set_ylabel(y_col)

    sorted_idx_train = np.argsort(X_train.ravel())
    sorted_idx_test = np.argsort(X_test.ravel())

    ax2.plot(X_train.ravel()[sorted_idx_train],
    (mse_train / X_train.ravel())[sorted_idx_train],
    color='blue', marker='x', label='Training MSE per X')

    ax2.plot(X_test.ravel()[sorted_idx_test],
    (mse_test / X_test.ravel())[sorted_idx_test],
    color='orange', marker='x', label='Test MSE per X')
    ax2.set_ylabel('MSE (%)')

    # Plot regression line over full x range
    x_all = np.linspace(df_clean[x_col].min(), df_clean[x_col].max(), 100).reshape(-1, 1)
    y_line = model.predict(x_all)
    ax1.plot(x_all, y_line, 'm--', linewidth=2, label='Regression line')

    # Labels and legend
    ax1.legend()

    # Add slope and R^2 as annotation
    ax1.text(0.75, 0.55, f"Slope: {slope:.17f}\nIntercept: {intercept:.10f}\nR²: {r2:.3f}\nmse (<{upper_threshold}): {mse_low:.3f}",
             transform=ax1.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax1.set_title("Linear Regression")
    ax1.grid(True)
    ax2.grid(True)
    #plt.tight_layout()
    plt.show()
    weights = pd.Series(model.coef_, index=y_col)
    intercept_unscaled = model.intercept_ 
    weights['__intercept__'] = intercept_unscaled
    return weights
#הפונ' עושה רנדום פורסט על צפיפות אקטיביות

def random_forest_activity_density(X_train, X_test, y_train, y_test, n=100):
    model = RandomForestRegressor(n_estimators=n)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    importances = model.feature_importances_
    features = X_train.columns
    iif=pd.DataFrame({'importances':importances,'features':features}).sort_values(by='importances', ascending=False)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.plot(X_train['Mean_Intensity_counts_NM_40'],y_train,'.b', label='train')
    ax1.plot(X_test['Mean_Intensity_counts_NM_40'],y_test,'.r', label='test')
    ax1.set_xlabel('Counts')
    ax1.set_ylabel('Activity density (\u03BCCi/ml)')
    ax1.legend(loc='best')
    max_t=max(y_test)
    max_p=max(y_pred)
    ax2.plot(y_test, y_pred,'.r')
    ax2.set_xlabel('True Activity density (\u03BCCi/ml)')
    ax2.set_ylabel('Predicted Activity density (\u03BCCi/ml)')
    ax2.set_xlim(0, max_t*1.1)
    ax2.set_ylim(0, max_p*1.1)
    ax2.set_title(f'estinators={n}')
    txt = f"RMSE: {rmse:.2f} \nR²:   {r2:.2f}"
    ax2.annotate(txt, xy=(0.25, 0.9),xycoords='axes fraction',ha='right', va='bottom')
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.2f}")
    return iif


#הפונ' מחשבת רגרסיה לינארית, על אחד הפיצ'רים, ואז רנדום פורסט על השאר כדי לחשב צפיפות אקטיביות
def linearRegression_fo_randomForest_so_activity_density(X_train, X_test, y_train, y_test, main_feature='Mean_Intensity_counts_NM_40', n=100):
    # === שלב 1: רגרסיה ליניארית על הפיצ'ר הראשי ===
    model_lin = LinearRegression()
    model_lin.fit(X_train[[main_feature]], y_train)
    y_train_lin = model_lin.predict(X_train[[main_feature]])
    y_test_lin = model_lin.predict(X_test[[main_feature]])
    
    # === שלב 2: חישוב השאריות ===
    residuals_train = y_train - y_train_lin
    
    # === שלב 3: אימון Random Forest על השאריות (ללא הפיצ'ר הראשי אם רוצים) ===
    X_train_rf = X_train.drop(columns=[main_feature])
    X_test_rf = X_test.drop(columns=[main_feature])
    
    model_rf = RandomForestRegressor(n_estimators=n)
    model_rf.fit(X_train_rf, residuals_train)
    residuals_pred = model_rf.predict(X_test_rf)
    
    # === שלב 4: חיבור התחזיות ===
    y_pred_final = y_test_lin + residuals_pred
    
    # === חשיבות הפיצ'רים של הרנדום פורסט ===
    importances = model_rf.feature_importances_
    features = X_train_rf.columns
    iif = pd.DataFrame({'importances': importances, 'features': features}) \
            .sort_values(by='importances', ascending=False)
    
    # === חישוב מדדי ביצועים ===
    mse = mean_squared_error(y_test, y_pred_final)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred_final)
    r2_lin = r2_score(y_test, y_test_lin)
    
    # === גרפים ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # גרף 1: נתוני אימון ובדיקה
    ax1.plot(X_train[main_feature], y_train, '.b', label='Train')
    ax1.plot(X_test[main_feature], y_test, '.r', label='Test')
    ax1.plot(X_test[main_feature], y_test_lin, 'k-', label='Linear fit', alpha=0.7)
    ax1.set_xlabel('Counts')
    ax1.set_ylabel('Activity density (\u03BCCi/ml)')
    ax1.legend(loc='best')
    
    # גרף 2: תחזית מול אמת
    max_t = max(y_test)
    max_p = max(y_pred_final)
    ax2.plot(y_test, y_pred_final, '.r', label="Final Prediction")
    ax2.plot(y_test, y_test_lin, '.g', alpha=0.6, label="Linear Only")
    ax2.set_xlabel('True Activity density (\u03BCCi/ml)')
    ax2.set_ylabel('Predicted Activity density (\u03BCCi/ml)')
    ax2.set_xlim(0, max_t*1.1)
    ax2.set_ylim(0, max_p*1.1)
    ax2.legend(loc="best")
    txt = f"RMSE: {rmse:.2f} \nR²:   {r2:.2f}"
    ax2.annotate(txt, xy=(0.35, 0.9), xycoords='axes fraction', ha='left', va='bottom')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.2f}")
    print(f"R² linear:   {r2_lin:.2f}")
    
    return iif

#הפונ' מחשבת רנדום פורסט על הנפח
def random_forest_Volume(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    importances = model.feature_importances_
    features = X_train.columns
    iif=pd.DataFrame([importances,features]).T
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.plot(X_train['Volume_[ml]_NM_40'],y_train,'.b', label='train')
    ax1.plot(X_test['Volume_[ml]_NM_40'],y_test,'.r', label='test')
    ax1.set_xlabel('Volume_NM_40 [ml]')
    ax1.set_ylabel('Volume_CT [ml]')
    ax1.legend(loc='best')
    max_t=max(y_test)
    max_p=max(y_pred)
    ax2.plot(y_test, y_pred,'.r')
    ax2.plot([0, max_p], [0, max_p],'--r')
    ax2.set_xlabel('True Volume_CT [ml]')
    ax2.set_ylabel('Predicted Volume_CT [ml]')
    ax2.set_xlim(0, max_t*1.1)
    ax2.set_ylim(0, max_p*1.1)
    txt = f"RMSE: {rmse:.2f} \nR²:   {r2:.2f}"
    ax2.annotate(txt, xy=(0.25, 0.9),xycoords='axes fraction',ha='right', va='bottom')
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.2f}")

#הפונ' מחשבת אלסטיק נט על הנפח
def elasticNet_Volume(X_train, X_test, y_train, y_test, alpha_values=None, l1_ratio_vals=None, random_state= None):
    
    if alpha_values is None:
        alpha_values = np.logspace(-4, 0, 50)
    if l1_ratio_vals is None:
        l1_ratio_vals = np.linspace(0, 1,50)[0:]
    
    scaler = StandardScaler()
#    X_numeric=X_train.select_dtypes(include=[np.number])
#    X_encoded = X_numeric.loc[:, ~X_numeric.T.duplicated()]
#    feature_names = X_encoded.columns.tolist()
#    X_scaled = scaler.transform(X_encoded)
#    X_numeric1=X_test.select_dtypes(include=[np.number])
#    X_encoded1 = X_numeric1.loc[:, ~X_numeric1.T.duplicated()]
#    X_scaled_test = scaler.fit_transform(X_encoded1)

    scaler = StandardScaler()
    scaler.fit(X_train)  # מחשב ממוצע וסטיית תקן רק על train

    X_scaled_train = scaler.transform(X_train)
    X_scaled_test = scaler.transform(X_test)  # 
    X_numeric=X_train.select_dtypes(include=[np.number])
    X_encoded = X_numeric.loc[:, ~X_numeric.T.duplicated()]#משתמש באותם פרמטרים
    feature_names = X_encoded.columns.tolist()
    model = ElasticNetCV(alphas=alpha_values, l1_ratio=l1_ratio_vals, cv=5, random_state=random_state)
    model.fit(X_scaled_train, y_train)
    y_pred = model.predict(X_scaled_test)
    weights = pd.Series(model.coef_/ scaler.scale_, index=feature_names)
    intercept_unscaled = model.intercept_ - np.sum((model.coef_ * scaler.mean_) / scaler.scale_)
    weights['__intercept__'] = intercept_unscaled
    #weights.name = f"Fold_{i+1}"
    #importances = model.feature_importances_
    #features = X_train.columns
    #iif=pd.DataFrame([importances,features]).T
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.plot(X_train['Volume_[ml]_NM_40'],y_train,'.b', label='train')
    ax1.plot(X_test['Volume_[ml]_NM_40'],y_test,'.r', label='test')
    ax1.set_xlabel('Volume_NM_40 [ml]')
    ax1.set_ylabel('True Volume [ml]')
    ax1.legend(loc='best')
    max_t=max(y_test)
    max_p=max(y_pred)
    ax2.plot(y_test, y_pred,'.r')
    ax2.plot([0, max_p], [0, max_p],'--m')
    ax2.set_xlabel('True Volume (ml)')
    ax2.set_ylabel('Predicted Volume (ml)')
    ax2.set_xlim(0, max_t*1.1)
    ax2.set_ylim(0, max_p*1.1)
    txt = f"RMSE: {rmse:.2f} \nR²:   {r2:.2f}"
    ax2.annotate(txt, xy=(0.25, 0.9),xycoords='axes fraction',ha='right', va='bottom')
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.2f}")
    print("Best alpha:", model.alpha_)
    print("Best l1_ratio:", model.l1_ratio_)

#הפונ' מחשבת אלסטיק נט על צפיפות אקטיביות
def elasticNet_activityDensity(X_train, X_test, y_train, y_test, alpha_values=None, l1_ratio_vals=None, random_state= None):
    
    if alpha_values is None:
        alpha_values = np.logspace(-4, 0, 50)
    if l1_ratio_vals is None:
        l1_ratio_vals = np.linspace(0, 1,50)[0:]
    
    scaler = StandardScaler()
    scaler.fit(X_train)  # מחשב ממוצע וסטיית תקן רק על train

    X_scaled_train = scaler.transform(X_train)
    X_scaled_test = scaler.transform(X_test)  # 
    X_numeric=X_train.select_dtypes(include=[np.number])
    X_encoded = X_numeric.loc[:, ~X_numeric.T.duplicated()]#משתמש באותם פרמטרים
    feature_names = X_encoded.columns.tolist()
    model = ElasticNetCV(alphas=alpha_values, l1_ratio=l1_ratio_vals, cv=5, random_state=random_state)
    model.fit(X_scaled_train, y_train)
    y_pred = model.predict(X_scaled_test)
    weights = pd.Series(model.coef_/ scaler.scale_, index=feature_names)
    intercept_unscaled = model.intercept_ - np.sum((model.coef_ * scaler.mean_) / scaler.scale_)
    weights['__intercept__'] = intercept_unscaled
    weights=weights.sort_values(key=abs, ascending=False)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.plot(X_train['Mean_Intensity_counts_NM_40'],y_train,'.b', label='train')
    ax1.plot(X_test['Mean_Intensity_counts_NM_40'],y_test,'.r', label='test')
    ax1.set_xlabel('Mean_Intensity_counts_NM_40')
    ax1.set_ylabel('activity density')
    ax1.legend(loc='best')
    max_t=max(y_test)
    max_p=max(y_pred)
    ax2.plot(y_test, y_pred,'.r')
    ax2.plot([0, max_p], [0, max_p],'--m')
    ax2.set_xlabel('True activity density')
    ax2.set_ylabel('Predicted activity density')
    ax2.set_xlim(0, max_t*1.1)
    ax2.set_ylim(0, max_p*1.1)
    txt = f"RMSE: {rmse:.2f} \nR²:   {r2:.2f}"
    ax2.annotate(txt, xy=(0.25, 0.9),xycoords='axes fraction',ha='right', va='bottom')
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.2f}")
    print("Best alpha:", model.alpha_)
    print("Best l1_ratio:", model.l1_ratio_)
    return weights

