import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from xgboost import XGBClassifier
from colorama import Fore
from sklearn.model_selection import GridSearchCV

data = pd.read_csv("prepped_data.csv")

# RECORD = 0.203 - seedDiff & offRankDiff

# ********TODO****************
# Add more features

features = ['Seed_Diff', 'offRankDiff']

train = data[data['Season'] < 2020]
test = data[data['Season'] > 2020]

Xtrain = train[features]
ytrain = (train['T1_Score'] > train['T2_Score']).astype(int)
Xtest = test[features]
ytest = (test['T1_Score'] > test['T2_Score']).astype(int)


m1 = XGBClassifier()
m1.fit(Xtrain, ytrain)
predictions = m1.predict_proba(Xtest)

output = pd.DataFrame(predictions[:,1], columns = ['Predictions'])
output['Actual'] = ytest.astype(int).reset_index(drop=True)
output["Score"] = (output["Actual"] - output["Predictions"])**2

# Score = 0.25 indicates each team is given an equal chance, no pattern association
print(Fore.BLUE + "********** PREDITCTION SCORE: " + Fore.GREEN + str(output["Score"].mean()) + Fore.BLUE + " ***************" + Fore.WHITE)

test['Pred'] = predictions[:,1]
test['rounded_preds'] = np.round(predictions[:,1])
test['Actual'] = (test['T1_Score'] > test['T2_Score']).astype(int)
summary = test[['Season', 'DayNum', 'T1_TeamID', 'T1_Score', 'T2_TeamID', 'T2_Score', 'Pred', 'rounded_preds', 'Actual']]

summary.to_csv('Predictions.csv', index=False)

#Optimized parameters to avoid overfitting and improve the model
param_grid = {
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'learning_rate': [0.1, 0.01, 0.001]
}

grid_search = GridSearchCV(estimator=m1, param_grid=param_grid, scoring='accuracy', cv=5)
grid_search.fit(Xtrain, ytrain)

best_params = grid_search.best_params_

m2 = XGBClassifier(reg_alpha = 0.15, reg_lambda = 0.01, **best_params)

m2.fit(Xtrain, ytrain)
predictions2 = m2.predict_proba(Xtest)

output = pd.DataFrame(predictions2[:,1], columns = ['Predictions'])
output['Actual'] = ytest.astype(int).reset_index(drop=True)
output["Score"] = (output["Actual"] - output["Predictions"])**2
print(Fore.BLUE + "********** OPTIMIZED PREDITCTION SCORE: " + Fore.GREEN + str(output["Score"].mean()) + Fore.BLUE + " ***************" + Fore.WHITE)


