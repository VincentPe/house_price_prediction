# modelling test settings
dev_seed = 0
testset_size = 0.25
n_jobs = -1

# Path to shapefiles
shapefile_path = 'Shapefile'

# For the following list, the data desciption already indicates that missing means not present in the house
NA_means_not_there_cols = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 
                           'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']

# Median impute columns
cont_impute_cols = ['BsmtHalfBath', 'TotalBsmtSF', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFullBath', 
                    'GarageCars', 'BsmtScore', 'LogYardArea']

# Mode impute columns
cat_impute_cols = ['KitchenQual', 'MSZoning']

# Scale continuous columns
scale_cont_cols = ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'TotalSFInclBsmnt', 'YardArea']

# Categorical columns to mean encode
mean_enc_cols = ['Neighborhood', 'KitchenQual', 'MSSubClass', 'MSZoning']

# Features to square to find non-linear relations
square_features = ['YearBuilt', 'YearRemodAdd', 'LogTotalSFInclBsmnt', 'FireplaceQu']

# Variables and features
target = 'LogSalePrice'
keep_features = [
        'OverallQual',
        'LogTotalSFInclBsmnt',
        'GarageCars',
        'ME_Neighborhood',
        'LogYardArea',
        'AllBathsSum',
        'YearBuilt',
        'YearRemodAdd',
        'ME_KitchenQual',
        'ME_MSSubClass',
        'HeatingQC',
        'Fireplaces',
        'FireplaceQu',
        'ME_MSZoning',
        'BsmtScore'        
]