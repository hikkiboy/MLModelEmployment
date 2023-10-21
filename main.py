import pandas as pd
from sklearn.model_selection import  train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


artist_path = "./data/employment_to_population.csv"

data = pd.read_csv(artist_path)

y = data.bachelors_degree

featureList = [ 'less_than_hs','high_school', 'some_college']

X = data[featureList]

X_train, X_val, y_train, y_val = train_test_split(X,y, random_state=0)


artist_model = RandomForestRegressor(random_state=0)

artist_model.fit(X_train, y_train)


artist_model_predict = artist_model.predict(X_val)

MAE_artist_model = mean_absolute_error(y_val,artist_model_predict)

print('esse são os dados: ')
print(X.describe())
print('Essa são as previsões')
print(artist_model.predict(X_val))
print('Esses são so valores reais: ')
print(y_val)

print("Validation MAE sem numero maximo de nodes: {:,.0f}".format(MAE_artist_model))
print("Validation MAE para o modelo de floresta: {}".format(MAE_artist_model))


