import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline

import Preprocessing
import Test
from Preprocessing import (ColumnSelector, DropColumnsTransformer, DateTimeTransformer,
                           WinsorizationImpute, DataFrameImputer, StandardScaleTransform)



class FullPipeline1:
    def __init__(self):
        
        self.x_cols = ['Date/Time', 'Wind Speed (m/s)','Theoretical_Power_Curve (KWh)', 'Wind Direction (°)']
        self.y_cols = ['LV ActivePower (kW)']
        self.median_cols = ['Wind Speed (m/s)','Theoretical_Power_Curve (KWh)', 'Wind Direction (°)']
        self.freq_cols = ['Date/Time']
        self.scaling_cols = ['Wind Speed (m/s)','Theoretical_Power_Curve (KWh)', 'Wind Direction (°)', 'day', 'month']

        self.x_pipeline = Pipeline([
            ('selector', ColumnSelector(columns=self.x_cols)),
            ('imputer', DataFrameImputer(median_cols=self.median_cols, freq_cols=self.freq_cols)),
            ('DateTime',DateTimeTransformer(column='Date/Time',date_format='%d %m %Y %H:%M',drop_original=True)),
            ('drop_cols', DropColumnsTransformer(columns=['year'])),
            ('winsorization', WinsorizationImpute(columns=['Wind Speed (m/s)'],outlier_handling='random_in_distribution',
                                       p=0,q=0.95)),
            ('scale', StandardScaleTransform(self.scaling_cols))
        ])
    
        self.y_pipeline = Pipeline([
            ('selector', ColumnSelector(columns=self.y_cols))
        ])
    
    def fit_transform(self, X_train, y_train):
        X_train = self.x_pipeline.fit_transform(X_train)
        y_train = self.y_pipeline.fit_transform(y_train)
        return X_train, y_train
    
    def transform(self, X_test, y_test):
        X_test = self.x_pipeline.transform(X_test)
        y_test = self.y_pipeline.transform(y_test)
        return X_test, y_test
    
    
class FullPipeline2:
    def __init__(self):
        
        self.x_cols = ['Date/Time', 'Wind Speed (m/s)','Theoretical_Power_Curve (KWh)', 'Wind Direction (°)']
        self.y_cols = ['LV ActivePower (kW)']
        self.median_cols = ['Wind Speed (m/s)','Theoretical_Power_Curve (KWh)', 'Wind Direction (°)']
        self.freq_cols = ['Date/Time']
        self.scaling_cols = ['Wind Speed (m/s)','Theoretical_Power_Curve (KWh)', 'Wind Direction (°)', 'day', 'month']

        self.x_pipeline = Pipeline([
            ('selector', ColumnSelector(columns=self.x_cols)),
            ('imputer', DataFrameImputer(median_cols=self.median_cols, freq_cols=self.freq_cols)),
            ('DateTime',DateTimeTransformer(column='Date/Time',date_format='%d %m %Y %H:%M',drop_original=True)),
            ('drop_cols', DropColumnsTransformer(columns=['year'])),
            ('scale', StandardScaleTransform(self.scaling_cols))
        ])
    
        self.y_pipeline = Pipeline([
            ('selector', ColumnSelector(columns=self.y_cols))
        ])
    
    def fit_transform(self, X_train, y_train):
        X_train = self.x_pipeline.fit_transform(X_train)
        y_train = self.y_pipeline.fit_transform(y_train)
        return X_train, y_train
    
    def transform(self, X_test, y_test):
        X_test = self.x_pipeline.transform(X_test)
        y_test = self.y_pipeline.transform(y_test)
        return X_test, y_test