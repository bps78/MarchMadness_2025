import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#Code in this file is based on code provided in the workshop at the following link: https://github.com/wiscosac/wiscosac.github.io/blob/master/files/ML_Mania_Workshop.ipynb

kenpom = pd.read_csv("Renamed_KenPom.csv") #Chat-GPT renamed teams to match spelling in MTeams.csv, original data from March Madness Historical Data Set 2002-2025 on Kaggle
#The following data is all from the March Learning Mania 2025 competition on Kaggle
teams = pd.read_csv("MTeams.csv")
season = pd.read_csv("MRegularSeasonDetailedResults.csv")
tourney = pd.read_csv("MNCAATourneyCompactResults.csv")
seeds = pd.read_csv("MNCAATourneySeeds.csv")

my_data = pd.DataFrame()
year = 2003

#Gets season averages for the given teamID and year
def getAggregateStats(my_team, year):
    #my_team = 1104 #Alabama
    #year = 2021
    
    w_games = season.loc[((season['WTeamID'] == my_team)) & (season['Season'] == year)]
    l_games = season.loc[((season['LTeamID'] == my_team)) & (season['Season'] == year)]
    
   # points_made = (w_games['WFGM3'].sum() * 3) + (l_games['LFGM3'].sum() * 3) + ((w_games['WFGM'].sum() - w_games['WFGM3'].sum()) * 2) + ((l_games['LFGM'].sum() - l_games['LFGM3'].sum()) * 2)
   #posessions = w_games['WFGA'].sum() + w_games['WTO'].sum() + l_games['LFGA'].sum() + l_games['LTO'].sum()
    gameCount = (len(w_games) + len(l_games))

    threesPerG = (w_games['WFGM3'].sum() + l_games['LFGM3'].sum()) / gameCount  #Three Pointers Made / Game
    ftpg = (w_games['WFTM'].sum() + l_games['LFTM'].sum()) / gameCount #Free Throws Made / Game
    pdiffpg = ((w_games['WScore'].sum() - w_games['LScore'].sum()) + (l_games['LScore'].sum() - l_games['WScore'].sum())) / gameCount #Mean Point differential per game
   
    return (threesPerG, ftpg, pdiffpg)

#Get the regular season kenpom data we need for each team and each year 2003 - 2024, excluding 2020
for y in range(2003, 2025):
    year = y
    if(year != 2020): #There was no tournament this year
        for x, z in zip(teams['TeamName'], teams['TeamID']):
            if(kenpom.loc[(kenpom['Team'] == x) & (kenpom['Season'] == year)]["Adjusted Tempo"].any()):
                offRank = kenpom.loc[(kenpom['Team'] == x) & (kenpom['Season'] == year)]["Adjusted Offensive Efficiency Rank"].values[0]
                temp = kenpom.loc[(kenpom['Team'] == x) & (kenpom['Season'] == year)]["Adjusted Tempo"].values[0]
                off = kenpom.loc[(kenpom['Team'] == x) & (kenpom['Season'] == year)]["Adjusted Offensive Efficiency"].values[0]
                threesPG, ftpg, pdiffpg = getAggregateStats(z, year) #Add in the aggregate regular season stats

                new_data = pd.DataFrame({'Year': y, 'ID': z, 'Team': x, 'offRank': offRank, 'offRating': off, 'tempo': temp, 'threepg': threesPG, 'ftpg': ftpg, 'pDiffpg': pdiffpg}, index = [x])

                if(not new_data.empty):
                    my_data = pd.concat([my_data, new_data])

print(my_data.head())

tourney = tourney.loc[tourney['Season'] > 2002]

#Get each game from the tournament since 2002
for index, row in tourney.iterrows():
    wID = row['WTeamID']
    lID = row['LTeamID']
    year = row['Season']

    #Set up our data to have a column for each stat for both the winner and loser of each tournament game since 2003
    if(my_data.loc[(my_data['Year'] == year) & (my_data['ID'] == wID)]['offRating'].any() and my_data.loc[(my_data['Year'] == year) & (my_data['ID'] == lID)]['offRating'].any()):
        tourney.loc[index,'WOffRating'] = my_data.loc[(my_data['Year'] == year) & (my_data['ID'] == wID)]['offRating'].values[0]
        tourney.loc[index,'LOffRating'] = my_data.loc[(my_data['Year'] == year) & (my_data['ID'] == lID)]['offRating'].values[0]
        
        tourney.loc[index,'WTempo'] = my_data.loc[(my_data['Year'] == year) & (my_data['ID'] == wID)]['tempo'].values[0]
        tourney.loc[index,'LTempo'] = my_data.loc[(my_data['Year'] == year) & (my_data['ID'] == lID)]['tempo'].values[0]

        tourney.loc[index,'WThreepg'] = my_data.loc[(my_data['Year'] == year) & (my_data['ID'] == wID)]['threepg'].values[0]
        tourney.loc[index,'LThreepg'] = my_data.loc[(my_data['Year'] == year) & (my_data['ID'] == lID)]['threepg'].values[0]

        tourney.loc[index,'WFTPG'] = my_data.loc[(my_data['Year'] == year) & (my_data['ID'] == wID)]['ftpg'].values[0]
        tourney.loc[index,'LFTPG'] = my_data.loc[(my_data['Year'] == year) & (my_data['ID'] == lID)]['ftpg'].values[0]

        tourney.loc[index,'WPDiffPG'] = my_data.loc[(my_data['Year'] == year) & (my_data['ID'] == wID)]['pDiffpg'].values[0]
        tourney.loc[index,'LPDiffPG'] = my_data.loc[(my_data['Year'] == year) & (my_data['ID'] == lID)]['pDiffpg'].values[0]

        tourney.loc[index,'WOffRank'] = my_data.loc[(my_data['Year'] == year) & (my_data['ID'] == wID)]['offRank'].values[0]
        tourney.loc[index,'LOffRank'] = my_data.loc[(my_data['Year'] == year) & (my_data['ID'] == lID)]['offRank'].values[0]

print(tourney.head())

prepped_data = tourney.dropna()
print(prepped_data.shape)

#Replace the W and L with T_1 and T_2 and make 2 copies of each game so that the model predicts off features and not just the position/names of the variables
def prepare_data(df):
    dfswap = df[['Season', 'DayNum', 'LTeamID', 'LScore', 'WTeamID', 'WScore', 'WLoc', 'NumOT', 'WOffRating', 'LOffRating', 'WOffRank', 'LOffRank', 'WTempo', 'LTempo', 'WThreepg', 'LThreepg', 'WFTPG', 'LFTPG', 'WPDiffPG', 'LPDiffPG']]
  
    df.columns = [x.replace('W','T1_').replace('L','T2_') for x in list(df.columns)]
    dfswap.columns = [x.replace('L','T1_').replace('W','T2_') for x in list(dfswap.columns)]

    output = pd.concat([df, dfswap]).reset_index(drop=True)    
    output['PointDiff'] = output['T1_Score'] - output['T2_Score']
    
    return output


prepped_data = prepare_data(prepped_data)

#Create Differential Stats

#prepped_data['offRatingDiff'] = prepped_data['T1_OffRating'] - prepped_data['T2_OffRating']
#prepped_data['tempoDiff'] = prepped_data['T1_Tempo'] - prepped_data['T2_Tempo']
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

#Save the processed data for use with the model
prepped_data.to_csv('prepped_data.csv', index=False)


