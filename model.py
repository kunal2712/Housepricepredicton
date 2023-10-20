import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error , mean_squared_error


prices_data = pd.read_csv('melb_data.csv')

data = prices_data.drop(["Suburb" , "Address" , "SellerG" , "Date" , "CouncilArea" , "Regionname", "BuildingArea" , 'Method' , 'Propertycount', 'Type'] , axis=1)

from sklearn.impute import SimpleImputer
imputer_mode = SimpleImputer(strategy='most_frequent')

data['YearBuilt'] = imputer_mode.fit_transform(data[['YearBuilt']])
data['Car'] = imputer_mode.fit_transform(data[['Car']])




# categorial_cols = ['Type'  ]

# cleaned_data = pd.get_dummies(data,columns=categorial_cols)

#splitting the data
y = data['Price']
X = data.drop('Price' , axis=1)

model = RandomForestRegressor(n_estimators=50)
model.fit(X, y)


# try:
#   
# except UnicodeDecodeError:
#   print('UnicodeDecodeError: XGBoost does not support Unicode strings in Python 2. Please use Python 3 or set the `XGB_FORCE_JSON_VALUE` environment variable to `1`.')

# import xgboost as xgb
# model = xgb.XGBRegressor()
# model.fit(X, y)

pickle.dump(model,open('Forest_model.pkl','wb'))
#pickle.dump(model,open('XG_model.pkl','wb'))