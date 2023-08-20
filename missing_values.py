import pandas as pd

melbourne_data = pd.read_csv('melb_data.csv')

filtered_melbourne_data = melbourne_data.dropna(axis=0)
y = filtered_melbourne_data.Price

melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
                        'YearBuilt', 'Lattitude', 'Longtitude']

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

best_model = DecisionTreeRegressor(max_leaf_nodes = 500)
best_model.fit(X,y) # since we know the best size , we can work upon the whole dataset
val_preds = best_model.predict(val_X)
print(mean_absolute_error(val_y,val_preds)) #minimum absolute error

# Get names of columns with missing values
cols_with_missing = [col for col in train_X.columns
                     if X[col].isnull().any()]  

from sklearn.impute import SimpleImputer

#imputation => replacing with mean value

my_imputer = SimpleImputer()

imputed_train_X = pd.DataFrame(my_imputer.fit_transform(X))
imputed_val_X = pd.DataFrame(my_imputer.transform(val_X))

imputed_train_X.columns = X.columns
imputed_val_X.columns = val_X.columns


print("MAE from  (Imputation) approach :")
print(get_mae(500 , imputed_train_X, imputed_val_X, y, val_y))



'''
# Extension to imputation => impute the missing values, while also 
# keeping track of which values were imputed ie creating separate boolean columns

# Make copy to avoid changing original data (when imputing)
train_X_plus =train_X.copy()
val_X_plus = val_X.copy()

# Make new columns indicating what will be imputed
for col in cols_with_missing:
   train_X_plus[col + '_was_missing'] =train_X_plus[col].isnull()
   val_X_plus[col + '_was_missing'] = val_X_plus[col].isnull()

# Imputation
my_imputer = SimpleImputer()
imputed_train_X_plus = pd.DataFrame(my_imputer.fit_transform(train_X_plus))
imputed_val_X_plus = pd.DataFrame(my_imputer.transform(val_X_plus))

# Imputation removed column names; put them back
imputed_train_X_plus.columns =train_X_plus.columns
imputed_val_X_plus.columns = val_X_plus.columns

print("MAE from Approach 3 (An Extension to Imputation):")
print(get_mae(500, imputed_train_X_plus, imputed_val_X_plus, train_y, val_y))

'''