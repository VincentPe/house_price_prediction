# House price prediction
Kaggle competition to predict housing prices based on about 80 features. 
The size of the dataset in nr of observations is small (~1500 in trainset) making it vulnerable 
for overfitting. I stopped at the point at which I estimated this feature and model combination
would be generalizing good to new unseen data (outside of the Kaggle testset).

# Some things I included this project
- Collect shapefiles from neighborhoods in Ames to create geographical features.
- Try to create time series based features based on fluctuations in housing market.
- Use polynomial features to smoothen non linear relations in the data.
- Log/boxcox transforming skewed distributions (e.g. surface features)
- Create fit/transform class for feature generation to prevent data leakage in cross validation
- Use forward selection by adding features one by one to test it's importance
- Try PCA to reduce dimentionality (and multicolliniarity) of dataset while keeping its variance.
- Create prediction and hyper parameter tuning pipelines to iterate through experiments fast.
- Ensamble models to create more robust and accurate predictions.
- Use regression to impute missing values when yearbuilt was missing for the garage.
- Mean encoding some categories that add more value this way then when using dummies.
- Make use for macro's to make jupyter notebooks more readible and easier to use.