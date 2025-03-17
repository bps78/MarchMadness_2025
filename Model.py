import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from colorama import Fore
from sklearn.model_selection import GridSearchCV

data = pd.read_csv("prepped_data.csv")

# RECORD = 0.18954 -> Random Tree (seedDiff, offRankDiff, defRankDiff, fgEff, ftRate, 3PG, FTPG, PDiffPG)

# ********TODO****************
# Add Barttovrik Rankings Data?? !**! (THE DATA SET IS IMPORTED AND RENAMED)
# Add resume feature??

features = ['Seed_Diff', 'offRankDiff', 'defRankDiff', 'T1_fgEff', 'T2_fgEff', 'T1_ftRate', 'T2_ftRate', 'T1_Threepg', 'T2_Threepg', 'T1_FTPG', 'T2_FTPG', 'T1_PDiffPG', 'T2_PDiffPG']

train = data[data['Season'] < 2020]
test = data[data['Season'] > 2020]

Xtrain = train[features]
ytrain = (train['T1_Score'] > train['T2_Score']).astype(int)
Xtest = test[features]
ytest = (test['T1_Score'] > test['T2_Score']).astype(int)

#Random Forest for Feature Selection
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(Xtrain, ytrain)

feature_importance = pd.Series(rf.feature_importances_, index = Xtrain.columns)
top_features = feature_importance.nlargest(10).index #Selects the 10 top features

#Filter for only selected features
Xtrain_selected = Xtrain[top_features]
Xtest_selected = Xtest[top_features]

#Train the XGBoost model
m1 = XGBClassifier()
m1.fit(Xtrain_selected, ytrain)
predictions = m1.predict_proba(Xtest_selected)

output = pd.DataFrame(predictions[:,1], columns = ['Predictions'])
output['Actual'] = ytest.astype(int).reset_index(drop=True)
output["Score"] = (output["Actual"] - output["Predictions"])**2

# Score = 0.25 indicates each team is given an equal chance, no pattern association
print(Fore.BLUE + "********** PREDITCTION SCORE: " + Fore.YELLOW + str(output["Score"].mean()) + Fore.BLUE + " ***************" + Fore.WHITE)

#Optimized parameters to avoid overfitting and improve the model
param_grid = {
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'learning_rate': [0.1, 0.01],
    'subsample': [0.6, 0.75, 0.9]
}

grid_search = GridSearchCV(estimator=m1, param_grid=param_grid, scoring='accuracy', cv=5)
grid_search.fit(Xtrain_selected, ytrain)

best_params = grid_search.best_params_

m2 = XGBClassifier(reg_alpha = 0.15, reg_lambda = 0.01, **best_params)

m2.fit(Xtrain_selected, ytrain)
predictions2 = m2.predict_proba(Xtest_selected)

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
