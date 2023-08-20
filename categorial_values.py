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
def get_mae(max_leaf_nodes , train_X , val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes , random_state=0)
    model.fit(train_X,train_y)
    preds_value = model.predict(val_X)
    mae = mean_absolute_error(val_y,preds_value)
    return (mae)

# Get list of categorical variables
s = (train_X.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables:")
print(object_cols)


'''
Bad columns

# Columns that can be safely ordinal encoded
good_label_cols = [col for col in object_cols if 
                   set(X_valid[col]).issubset(set(X_train[col]))]
        
# Problematic columns that will be dropped from the dataset
bad_label_cols = list(set(object_cols)-set(good_label_cols))
        
print('Categorical columns that will be ordinal encoded:', good_label_cols)
print('\nCategorical columns that will be dropped from the dataset:', bad_label_cols)

'''

# Ordinal encoding assigns each unique value to a different integer.
# an ordering of the categories: "Never" (0) < "Rarely" (1) < "Most days" (2) < "Every day" (3).
# ordinal variables

from sklearn.preprocessing import OrdinalEncoder

# Make copy to avoid changing original data 
label_X_train = X.copy()
label_X_valid = val_X.copy()

# Apply ordinal encoder to each column with categorical data
ordinal_encoder = OrdinalEncoder()
label_X_train[object_cols] = ordinal_encoder.fit_transform(X[object_cols])
label_X_valid[object_cols] = ordinal_encoder.transform(val_X[object_cols])

print("MAE from Approach 1 (Ordinal Encoding):") 
print(get_mae(500, label_X_train, label_X_valid, y, val_y))

'''
    
# one hot encoding 

from sklearn.preprocessing import OneHotEncoder

# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(train_X[object_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(val_X[object_cols]))

# One-hot encoding removed index; put it back
OH_cols_train.index = train_X.index
OH_cols_valid.index = val_X.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_train = train_X.drop(object_cols, axis=1)
num_X_valid = val_X.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

print("MAE from Approach 2 (One-Hot Encoding):") 
print(get_mae(500 ,OH_X_train, OH_X_valid, train_y, val_y))



'''

