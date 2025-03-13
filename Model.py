import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from xgboost import XGBClassifier #Model used for prediction
from sklearn.ensemble import RandomForestClassifier #Random forest for feature selection
from colorama import Fore #Used to print colored console output for readability
from sklearn.model_selection import GridSearchCV #Model used to optimize parameters of XGBoost model
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("prepped_data.csv")

#Code in this file is based on code provided in the workshop at the following link: https://github.com/wiscosac/wiscosac.github.io/blob/master/files/ML_Mania_Workshop.ipynb

#***** RECORD = 0.19773 -> seedDiff, offRankDiff, 3PG, FTPG, PDiffPG ******
#***** SECOND = 0.2012 ->  seedDiff, offRankDiff, H/A SPLITS(3PG, FTPG, PDiffPG) ******

#Features the model will be using
features = ['Seed_Diff', 'offRankDiff', 'T1_HThreepg', 'T1_AThreepg', 'T2_HThreepg', 'T2_AThreepg', 'T1_HFTPG', 'T1_AFTPG', 'T2_HFTPG', 'T2_AFTPG', 'T1_HPDiffPG', 'T1_APDiffPG', 'T2_HPDiffPG', 'T2_APDiffPG']

#Normalize the data
scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])

#Train with tournament games 2003-2019, test with tournament games 2021-2024
train = data[data['Season'] < 2020]
test = data[data['Season'] > 2020]

#Assign test and train values
Xtrain = train[features]
ytrain = (train['T1_Score'] > train['T2_Score']).astype(int) #We are predicting if T1 will beat T2 for each game in the data
Xtest = test[features]
ytest = (test['T1_Score'] > test['T2_Score']).astype(int)

#Random Forest for Feature Selection
rf = RandomForestClassifier(n_estimators=100, random_state=32)
rf.fit(Xtrain, ytrain)

feature_importance = pd.Series(rf.feature_importances_, index = Xtrain.columns)
top_features = feature_importance.nlargest(11).index #Selects the n top features

#Filter for only selected features
Xtrain_selected = Xtrain[top_features]
Xtest_selected = Xtest[top_features]

#Train the XGBoost model
m1 = XGBClassifier()
m1.fit(Xtrain_selected, ytrain)
predictions = m1.predict_proba(Xtest_selected)

#Get the error score for this model
output = pd.DataFrame(predictions[:,1], columns = ['Predictions'])
output['Actual'] = ytest.astype(int).reset_index(drop=True)
output["Score"] = (output["Actual"] - output["Predictions"])**2

# Score = 0.25 indicates each team is given an equal chance, no pattern association
print(Fore.BLUE + "********** PREDITCTION SCORE: " + Fore.YELLOW + str(output["Score"].mean()) + Fore.BLUE + " ***************" + Fore.WHITE)

#Optimized parameters to avoid overfitting and improve the model
param_grid = {
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'learning_rate': [0.1, 0.01, 0.001],
    'subsample': [0.6, 0.75, 0.9]
}

#Fit a gridSearch model to find the best parameters
grid_search = GridSearchCV(estimator=m1, param_grid=param_grid, scoring='accuracy', cv=5)
grid_search.fit(Xtrain_selected, ytrain)

best_params = grid_search.best_params_

#Train another XGBoost model, but this time with tuned parameters
m2 = XGBClassifier(reg_alpha = 0.15, reg_lambda = 0.01, **best_params)

m2.fit(Xtrain_selected, ytrain)
predictions2 = m2.predict_proba(Xtest_selected)

#Get our new error score
output = pd.DataFrame(predictions2[:,1], columns = ['Predictions'])
output['Actual'] = ytest.astype(int).reset_index(drop=True)
output["Score"] = (output["Actual"] - output["Predictions"])**2
print(Fore.BLUE + "********** OPTIMIZED PREDITCTION SCORE: " + Fore.GREEN + str(output["Score"].mean()) + Fore.BLUE + " ***************" + Fore.WHITE)

#Updated Predictions
test = test.copy()
test['Pred'] = predictions2[:,1]
test['rounded_preds'] = np.round(predictions2[:,1])
test['Actual'] = (test['T1_Score'] > test['T2_Score']).astype(int)
test['Correct'] = test['rounded_preds'] == test['Actual']
summary = test[['Season', 'DayNum', 'T1_TeamID', 'T1_Score', 'T2_TeamID', 'T2_Score', 'Pred', 'rounded_preds', 'Actual', 'Correct']]

#Save our Predictions for viewing
summary.to_csv('Predictions.csv', index=False)
