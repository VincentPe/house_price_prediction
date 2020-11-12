import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression


def standard_preprocessing_function(data, NA_means_not_there_cols, scale_cont_cols, train=True):
    """
    Prepare the dataset across train and testset.
    No data leakage issues at this stage
    """
    if train:
        # Replace salesprice with a log scaled version of it
        data['LogSalePrice'] = np.log(data['SalePrice'])
        # Use np.exp on predictions to scale back to actual sales price
    
    # Impute missing values where they are not at random
    data[NA_means_not_there_cols] = data[NA_means_not_there_cols].fillna('Not_present') 
    data['LotFrontage'] = data['LotFrontage'].fillna(0)
    
    # Transform existing variables
    data = replace_ordinal_values(data)
    
    # Create combined features
    data['BsmtScore'] = data['BsmtFinSF1'] * data['BsmtFinType1'] + data['BsmtFinSF2'] * data['BsmtFinType2']
    data['AllBathsSum'] = np.sum(data[['BsmtHalfBath', 'HalfBath', 'BsmtFullBath', 'FullBath']], axis=1)
    data['TotalSFInclBsmnt'] = np.sum(data[['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']], axis=1)
    data['YardArea'] = data['LotArea'] - data['1stFlrSF'] - data['GarageArea']
    
    # Make a more even distribution for continuous features as well
    for col in scale_cont_cols:
        data['Log' + col] = np.log1p(data[col])
    
    if train:
        # Remove outliers
        data = data[data['LogTotalSFInclBsmnt'] < 8.9].reset_index(drop=True)
    
    return data


class leakage_preventive_preprocessing_function():
    
    def __init__(self, target, cont_impute_cols, cat_impute_cols, mean_enc_cols, square_features, keep_features):
        self.target = target
        self.cont_impute_cols = cont_impute_cols
        self.cat_impute_cols = cat_impute_cols
        self.mean_enc_cols = mean_enc_cols
        self.keep_features = keep_features
        self.square_features = square_features
        self.mean_enc_dict = {}
    
    def fit(self, X, y):
        # Fit regression to impute NAs for GarageYrBlt
        self.reg = LinearRegression().fit(X.loc[X['GarageYrBlt'].notna(), ['GarageYrBlt']], 
                                          y[X['GarageYrBlt'].notna()])
        self.avg_houseprice_nogarage = np.mean(y[X['GarageYrBlt'].isna()])
        self.garage_yearbuilt_impute = (self.avg_houseprice_nogarage - self.reg.intercept_) / self.reg.coef_[0]
        
        # Fit imputer to impute missing values
        self.num_imputer = SimpleImputer(missing_values=np.nan, strategy='median').fit(X[self.cont_impute_cols])
        self.cat_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent').fit(X[self.cat_impute_cols])
        
        # Mean encode based on the current split
        X[self.target] = y
        for col in self.mean_enc_cols:
            self.mean_enc_dict[col] = X.groupby(col)[self.target].mean()
        X = X.drop([self.target], axis=1)
        
        # Save mean of target when ME's could not be created due to specific split
        self.target_mean = np.mean(y) # use median?
        
        # Transform mean encodings for standard scaler 
        for col in self.mean_enc_cols:
            X['ME_' + col] = X[col].map(self.mean_enc_dict[col])
            
            # Median impute NA's for the encoding
            X.loc[X['ME_' + col].isnull(), 'ME_' + col] = self.target_mean
        
        self.scaler = preprocessing.RobustScaler().fit(X[self.keep_features])
    
    def transform(self, X):
        # Impute missing values based on specific strategy
        X.loc[X['GarageYrBlt'].isna(), 'GarageYrBlt'] = self.garage_yearbuilt_impute
        
        # Use imputer to impute missing values
        X[self.cont_impute_cols] = self.num_imputer.transform(X[self.cont_impute_cols])
        X[self.cat_impute_cols] = self.cat_imputer.transform(X[self.cat_impute_cols])
        
        # Transform mean encodings
        for col in self.mean_enc_cols:
            X['ME_' + col] = X[col].map(self.mean_enc_dict[col])
            
            # Median impute NA's for the encoding
            X.loc[X['ME_' + col].isnull(), 'ME_' + col] = self.target_mean

        X[self.keep_features] = self.scaler.transform(X[self.keep_features])
        
        # After all transformations add squared features
        X = X[self.keep_features]
        for col in self.square_features:
            X['sq_' + col] = X[col]*X[col]
        
        return X #[self.keep_features]
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        X = self.transform(X)
        return X
    
def replace_ordinal_values(X):
    """
    Check whether any numerical variables are actually categorical and vice versa
    """
    clean_up_dict = {
        # Categorical to numerical
                    "LotShape": {'IR3': 0, 'IR2': 1, 'IR1': 2, 'Reg': 3},
                    "LandSlope": {"Gtl": 2, "Mod": 1, "Sev": 0},
                    "ExterQual": {"Ex": 4, "Gd": 3, "TA": 2, "Fa": 1, "Po": 0},
                    "ExterCond": {"Ex": 4, "Gd": 3, "TA": 2, "Fa": 1, "Po": 0},
                    "BsmtQual": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "Not_present": 0},
                    "BsmtCond": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "Not_present": 0},
                    "BsmtExposure": {"Gd": 4, "Av": 3, "Mn": 2, "No": 1, "Not_present": 0},
                    "BsmtFinType1": {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "Not_present": 0},
                    "BsmtFinType2": {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "Not_present": 0},
                    "HeatingQC": {"Ex": 4, "Gd": 3, "TA": 2, "Fa": 1, "Po": 0},
                    "CentralAir": {"Y": 1, "N": 0},
                    "KitchenQual": {"Ex": 4, "Gd": 3, "TA": 2, "Fa": 1, "Po": 0},
                    "Functional": {"Typ": 7, "Min1": 6, "Min2": 5, "Mod": 4, "Maj1": 3, "Maj2": 2, "Sev": 1, "Sal": 0},
                    "FireplaceQu": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "Not_present": 0},
                    "GarageFinish": {"Fin": 3, "RFn": 2, "Unf": 1, "Not_present": 0},
                    "GarageQual": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "Not_present": 0},
                    "GarageCond": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "Not_present": 0},
                    "PavedDrive": {"Y": 2, "P": 1, "N": 0},
                    "PoolQC": {"Ex": 4, "Gd": 3, "TA": 2, "Fa": 1, "Not_present": 0},
                    "Fence": {"GdPrv": 4, "MnPrv": 3, "GdWo": 2, "MnWw": 1, "Not_present": 0},
        # Numerical to categorical
                    "MSSubClass": {20: "A", 30: "B", 40: "C", 45: "D", 50: "E", 60: "F", 70: "G", 75: "H",
                                   80: "I", 85: "J", 90: "K", 120: "L", 150: "M", 160: "N", 180: "O", 190: "P"}
    }
    X.replace(clean_up_dict, inplace=True)
    
    return X