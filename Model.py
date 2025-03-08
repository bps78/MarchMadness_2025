import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from xgboost import XGBClassifier
from colorama import Fore

data = pd.read_csv("prepped_data.csv")

# ********TODO****************
# Model with just seeding should be getting better score?
# Try adding differential features (ex. **tempo diff, offRating diff)
# Add more features

features = ['T1_OffRating', 'T1_Tempo', 'T2_OffRating', 'T2_Tempo', 'T1_seed', 'T2_seed', 'Seed_Diff']

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