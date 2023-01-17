from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

import pandas as pd

melbourne_data = pd.read_csv('melb_data.csv')

filtered_melbourne_data = melbourne_data.dropna(axis=0)
y = filtered_melbourne_data.Price

melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
                        'YearBuilt', 'Lattitude', 'Longtitude','Type', 'Regionname','Method']

X = filtered_melbourne_data[melbourne_features]

from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# Get list of categorical variables
s = (train_X.dtypes == 'object')
object_cols = list(s[s].index)

# Get list of numerical variables
s = (train_X.dtypes == 'float')
numerical_cols = list(s[s].index)

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant') # strategy can be mean , median too

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, object_cols)
    ]) 


model = RandomForestRegressor(n_estimators=100, random_state=0)

 # Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                               ])
                
from sklearn.model_selection import cross_val_score
# Multiply by -1 since sklearn calculates *negative* MAE
scores = -1 * cross_val_score(my_pipeline, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')

print("MAE scores:\n", scores)  
print("Average MAE score (across experiments):")
print(scores.mean())

# Preprocessing of training data, fit model 
my_pipeline.fit(X, y)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(val_X)

# Evaluate the model
score = mean_absolute_error(val_y, preds)
print('MAE:', score)



