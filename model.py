import pandas as pd

melbourne_data = pd.read_csv('melb_data.csv')

filtered_melbourne_data = melbourne_data.dropna(axis=0)

y = filtered_melbourne_data.Price

melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
                        'YearBuilt', 'Lattitude', 'Longtitude']

X = filtered_melbourne_data[melbourne_features]

from sklearn.tree import DecisionTreeRegressor

#melbourne_model = DecisionTreeRegressor()
#melbourne_model.fit(X,y)

from sklearn.metrics import mean_absolute_error
#predicted_home_prices = melbourne_model.predict(X)

from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

'''
melbourne_model.fit(train_X,train_y)
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y,val_predictions))
'''

def get_mae(max_leaf_nodes , train_X , val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes , random_state=0)
    model.fit(train_X,train_y)
    preds_value = model.predict(val_X)
    mae = mean_absolute_error(val_y,preds_value)
    return (mae)


candidate_nodes = [5, 25, 50, 100, 250, 500]
scores = {leaf_size : get_mae(leaf_size,train_X , val_X, train_y, val_y) for leaf_size in candidate_nodes}
best_tree_size = min(scores,key=scores.get)
print(best_tree_size)

best_model = DecisionTreeRegressor(max_leaf_nodes = best_tree_size)
best_model.fit(X,y) # since we know the best size , we can work upon the whole dataset
val_preds = best_model.predict(val_X)
print(mean_absolute_error(val_y,val_preds)) #minimum absolute error


'''
#random forest

from sklearn.ensemble import RandomForestRegressor
forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))

'''

