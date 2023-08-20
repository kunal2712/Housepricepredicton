import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
data = pd.read_csv('melb_data.csv')

# Select subset of predictors
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]

# Select target
y = data.Price

# Separate data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y)


from xgboost import XGBRegressor

'''
parameters -

n_estimators  : specifies how many times to go through the modeling
cycle described above. It is equal to the number of models that we include in the ensemble.

early_stopping_rounds :  causes the model to stop iterating when the validation score stops 
improving, even if we aren't at the hard stop for n_estimators. It's smart to set a high value
for n_estimators and then use early_stopping_rounds to find the optimal time to stop iterating.

learning_rate :Instead of getting predictions by simply adding up the predictions from each component model, we can multiply the predictions from each model by a small number (known as the learning rate) before adding them in.
This means each tree we add to the ensemble helps us less. So, we can set a higher value for n_estimators without overfitting. If we use early stopping, the appropriate number of trees will be determined automatically.
In general, a small learning rate and large number of estimators will yield more accurate XGBoost models, though it will also take the model longer to train since it does more iterations through the cycle. As default, XGBoost sets learning_rate=0.1. 

n_jobs : common to set the parameter n_jobs equal to the number of cores on your machine

'''

my_model = XGBRegressor(n_estimators = 500,learning_rate=0.05,n_jobs=4)
my_model.fit(X_train,y_train,
early_stopping_rounds =5,
eval_set = [(X_valid,y_valid)],
verbose=False)

from sklearn.metrics import mean_absolute_error

predictions = my_model.predict(X_valid)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))