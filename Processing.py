import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

kenpom = pd.read_csv("Renamed_KenPom.csv") #Chat-GPT renamed teams to match spelling in MTeams.csv
teams = pd.read_csv("MTeams.csv")
season = pd.read_csv("MRegularSeasonDetailedResults.csv")
tourney = pd.read_csv("MNCAATourneyCompactResults.csv")
seeds = pd.read_csv("MNCAATourneySeeds.csv")

my_data = pd.DataFrame()
year = 2003

for y in range(2003, 2025):
    year = y
    if(year != 2020):
        for x, z in zip(teams['TeamName'], teams['TeamID']):
            if(kenpom.loc[(kenpom['Team'] == x) & (kenpom['Season'] == year)]["Adjusted Tempo"].any()):
                offRank = kenpom.loc[(kenpom['Team'] == x) & (kenpom['Season'] == year)]["Adjusted Offensive Efficiency Rank"].values[0]
                temp = kenpom.loc[(kenpom['Team'] == x) & (kenpom['Season'] == year)]["Adjusted Tempo"].values[0]
                off = kenpom.loc[(kenpom['Team'] == x) & (kenpom['Season'] == year)]["Adjusted Offensive Efficiency"].values[0]
                new_data = pd.DataFrame({'Year': y, 'ID': z, 'Team': x, 'offRank': offRank, 'offRating': off, 'tempo': temp}, index = [x])

                if(not new_data.empty):
                    my_data = pd.concat([my_data, new_data])

print(my_data.head())

tourney = tourney.loc[tourney['Season'] > 2002]

#Get each game from the tournament since 2002
for index, row in tourney.iterrows():
    wID = row['WTeamID']
    lID = row['LTeamID']
    year = row['Season']

    if(my_data.loc[(my_data['Year'] == year) & (my_data['ID'] == wID)]['offRating'].any() and my_data.loc[(my_data['Year'] == year) & (my_data['ID'] == lID)]['offRating'].any()):
        tourney.loc[index,'WOffRating'] = my_data.loc[(my_data['Year'] == year) & (my_data['ID'] == wID)]['offRating'].values[0]
        tourney.loc[index,'LOffRating'] = my_data.loc[(my_data['Year'] == year) & (my_data['ID'] == lID)]['offRating'].values[0]
        
        tourney.loc[index,'WTempo'] = my_data.loc[(my_data['Year'] == year) & (my_data['ID'] == wID)]['tempo'].values[0]
        tourney.loc[index,'LTempo'] = my_data.loc[(my_data['Year'] == year) & (my_data['ID'] == lID)]['tempo'].values[0]

        tourney.loc[index,'WOffRank'] = my_data.loc[(my_data['Year'] == year) & (my_data['ID'] == wID)]['offRank'].values[0]
        tourney.loc[index,'LOffRank'] = my_data.loc[(my_data['Year'] == year) & (my_data['ID'] == lID)]['offRank'].values[0]

print(tourney.head())

prepped_data = tourney.dropna()
print(prepped_data.shape)


def prepare_data(df):
    dfswap = df[['Season', 'DayNum', 'LTeamID', 'LScore', 'WTeamID', 'WScore', 'WLoc', 'NumOT', 'WOffRating', 'LOffRating', 'WOffRank', 'LOffRank', 'WTempo', 'LTempo']]
  
    df.columns = [x.replace('W','T1_').replace('L','T2_') for x in list(df.columns)]
    dfswap.columns = [x.replace('L','T1_').replace('W','T2_') for x in list(dfswap.columns)]

    output = pd.concat([df, dfswap]).reset_index(drop=True)    
    output['PointDiff'] = output['T1_Score'] - output['T2_Score']
    
    return output


prepped_data = prepare_data(prepped_data)

#Create Differential Stats
prepped_data['offRatingDiff'] = prepped_data['T1_OffRating'] - prepped_data['T2_OffRating']
prepped_data['tempoDiff'] = prepped_data['T1_Tempo'] - prepped_data['T2_Tempo']
prepped_data['offRankDiff'] = prepped_data['T1_OffRank'] - prepped_data['T2_OffRank']


#Apply seeds
seeds['seed'] = seeds['Seed'].apply(lambda x: int(x[1:3]))
seeds_T1 = seeds[['Season','TeamID','seed']].copy()
seeds_T2 = seeds[['Season','TeamID','seed']].copy()
seeds_T1.columns = ['Season','T1_TeamID','T1_seed']
seeds_T2.columns = ['Season','T2_TeamID','T2_seed']
prepped_data = pd.merge(prepped_data, seeds_T1, on = ['Season', 'T1_TeamID'], how = 'left')
prepped_data = pd.merge(prepped_data, seeds_T2, on = ['Season', 'T2_TeamID'], how = 'left')

prepped_data['Seed_Diff'] = prepped_data['T1_seed'] - prepped_data['T2_seed']

print(prepped_data.head())

prepped_data.to_csv('prepped_data.csv', index=False)


