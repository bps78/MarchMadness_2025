import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from xgboost import XGBClassifier, plot_importance

data = pd.read_csv("prepped_data.csv")

features = ['T1_OffRating', 'T1_Tempo', 'T2_OffRating', 'T2_Tempo']

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
print("********** PREDITCTION SCORE: ", output["Score"].mean(), " ***************")

test['Pred'] = predictions[:,1]
test['rounded_preds'] = np.round(predictions[:,1])
test['Actual'] = (test['T1_Score'] > test['T2_Score']).astype(int)
summary = test[['Season', 'DayNum', 'T1_TeamID', 'T1_Score', 'T2_TeamID', 'T2_Score', 'Pred', 'rounded_preds', 'Actual']]

summary.to_csv('Predictions.csv', index=False)