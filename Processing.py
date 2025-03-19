import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#See sources of below data in Pre_processing.py
kenpom = pd.read_csv("Kenpom_eff_renamed.csv")
kenpom_off = pd.read_csv("Kenpom_off_renamed.csv")

bart = pd.read_csv("Barttorvik_renamed.csv")

#Below data is from https://www.kaggle.com/competitions/march-machine-learning-mania-2025/data
teams = pd.read_csv("MTeams.csv")
season = pd.read_csv("MRegularSeasonDetailedResults.csv")
tourney = pd.read_csv("MNCAATourneyCompactResults.csv")
seeds = pd.read_csv("MNCAATourneySeeds.csv")


my_data = pd.DataFrame()

#Gets certain stats that we will be using that need to be summed up over the course of an entire season
def getAggregateStats(my_team, year):
    
    w_games = season.loc[((season['WTeamID'] == my_team)) & (season['Season'] == year)]
    l_games = season.loc[((season['LTeamID'] == my_team)) & (season['Season'] == year)]
    
    gameCount = (len(w_games) + len(l_games))

    threesPerG = (w_games['WFGM3'].sum() + l_games['LFGM3'].sum()) / gameCount  #Three Pointers Made / Game
    ftpg = (w_games['WFTM'].sum() + l_games['LFTM'].sum()) / gameCount #Free Throws Made / Game
    pdiffpg = ((w_games['WScore'].sum() - w_games['LScore'].sum()) + (l_games['LScore'].sum() - l_games['WScore'].sum())) / gameCount #Mean Point differential per game
   
    return (threesPerG, ftpg, pdiffpg)

#Get stats that have already been collected for us that represent an entire season for one team
#We are making a dataframe that has a different team and different year along with all their stats in each row
for y in range(2008, 2026):
    year = y
    if(year != 2020):
        for x, z in zip(teams['TeamName'], teams['TeamID']):
            if(kenpom.loc[(kenpom['Team'] == x) & (kenpom['Season'] == year)]["Adjusted Temo"].any() and kenpom_off.loc[(kenpom_off['TeamName'] == x) & (kenpom_off['Season'] == year)]["eFGPct"].any() and bart.loc[(bart['TEAM'] == x) & (bart['YEAR'] == year)]['WAB'].any()):
                offRank = kenpom.loc[(kenpom['Team'] == x) & (kenpom['Season'] == year)]["Adjusted Offensive Efficiency Rank"].values[0]
                defRank = kenpom.loc[(kenpom['Team'] == x) & (kenpom['Season'] == year)]["Adjusted Defensive Efficiency Rank"].values[0]
                temp = kenpom.loc[(kenpom['Team'] == x) & (kenpom['Season'] == year)]["Adjusted Temo"].values[0]
                off = kenpom.loc[(kenpom['Team'] == x) & (kenpom['Season'] == year)]["Adjusted Offensive Efficiency"].values[0]

                fgEff = kenpom_off.loc[(kenpom_off['TeamName'] == x) & (kenpom_off['Season'] == year)]["eFGPct"].values[0]
                ftRate = kenpom_off.loc[(kenpom_off['TeamName'] == x) & (kenpom_off['Season'] == year)]["FTRate"].values[0]

                wab = bart.loc[(bart['TEAM'] == x) & (bart['YEAR'] == year)]['WAB'].values[0]
                talent = bart.loc[(bart['TEAM'] == x) & (bart['YEAR'] == year)]['TALENT'].values[0]
                sos = bart.loc[(bart['TEAM'] == x) & (bart['YEAR'] == year)]['ELITE SOS'].values[0]

                threesPG, ftpg, pdiffpg = getAggregateStats(z, year)

                new_data = pd.DataFrame({'Year': y, 'ID': z, 'Team': x, 'offRank': offRank, 'defRank': defRank, 'offRating': off, 'tempo': temp, 'fgEff': fgEff, 'ftRate': ftRate, 'wab': wab, 'talent': talent, 'sos': sos, 'threepg': threesPG, 'ftpg': ftpg, 'pDiffpg': pdiffpg}, index = [x])

                if(not new_data.empty):
                    my_data = pd.concat([my_data, new_data])




tourney = tourney.loc[tourney['Season'] > 2007]
print("# of tourney games since 2008 ", len(tourney['Season']))
#Get each game from the tournament since 2008 to use for training and testing the model
for index, row in tourney.iterrows():
    wID = row['WTeamID']
    lID = row['LTeamID']
    year = row['Season']

    if(my_data.loc[(my_data['Year'] == year) & (my_data['ID'] == wID)]['offRating'].any() and my_data.loc[(my_data['Year'] == year) & (my_data['ID'] == lID)]['offRating'].any()):
        tourney.loc[index,'WTeam'] = my_data.loc[(my_data['Year'] == year) & (my_data['ID'] == wID)]['Team'].values[0]
        tourney.loc[index,'LTeam'] = my_data.loc[(my_data['Year'] == year) & (my_data['ID'] == lID)]['Team'].values[0]

        tourney.loc[index,'WOffRating'] = my_data.loc[(my_data['Year'] == year) & (my_data['ID'] == wID)]['offRating'].values[0]
        tourney.loc[index,'LOffRating'] = my_data.loc[(my_data['Year'] == year) & (my_data['ID'] == lID)]['offRating'].values[0]
        
        tourney.loc[index,'WTempo'] = my_data.loc[(my_data['Year'] == year) & (my_data['ID'] == wID)]['tempo'].values[0]
        tourney.loc[index,'LTempo'] = my_data.loc[(my_data['Year'] == year) & (my_data['ID'] == lID)]['tempo'].values[0]

        tourney.loc[index,'WfgEff'] = my_data.loc[(my_data['Year'] == year) & (my_data['ID'] == wID)]['fgEff'].values[0]
        tourney.loc[index,'LfgEff'] = my_data.loc[(my_data['Year'] == year) & (my_data['ID'] == lID)]['fgEff'].values[0]

        tourney.loc[index,'WftRate'] = my_data.loc[(my_data['Year'] == year) & (my_data['ID'] == wID)]['ftRate'].values[0]
        tourney.loc[index,'LftRate'] = my_data.loc[(my_data['Year'] == year) & (my_data['ID'] == lID)]['ftRate'].values[0]

        tourney.loc[index,'Wwab'] = my_data.loc[(my_data['Year'] == year) & (my_data['ID'] == wID)]['wab'].values[0]
        tourney.loc[index,'Lwab'] = my_data.loc[(my_data['Year'] == year) & (my_data['ID'] == lID)]['wab'].values[0]

        tourney.loc[index,'Wtalent'] = my_data.loc[(my_data['Year'] == year) & (my_data['ID'] == wID)]['talent'].values[0]
        tourney.loc[index,'Ltalent'] = my_data.loc[(my_data['Year'] == year) & (my_data['ID'] == lID)]['talent'].values[0]

        tourney.loc[index,'Wsos'] = my_data.loc[(my_data['Year'] == year) & (my_data['ID'] == wID)]['sos'].values[0]
        tourney.loc[index,'Lsos'] = my_data.loc[(my_data['Year'] == year) & (my_data['ID'] == lID)]['sos'].values[0]

        tourney.loc[index,'WThreepg'] = my_data.loc[(my_data['Year'] == year) & (my_data['ID'] == wID)]['threepg'].values[0]
        tourney.loc[index,'LThreepg'] = my_data.loc[(my_data['Year'] == year) & (my_data['ID'] == lID)]['threepg'].values[0]

        tourney.loc[index,'WFTPG'] = my_data.loc[(my_data['Year'] == year) & (my_data['ID'] == wID)]['ftpg'].values[0]
        tourney.loc[index,'LFTPG'] = my_data.loc[(my_data['Year'] == year) & (my_data['ID'] == lID)]['ftpg'].values[0]

        tourney.loc[index,'WPDiffPG'] = my_data.loc[(my_data['Year'] == year) & (my_data['ID'] == wID)]['pDiffpg'].values[0]
        tourney.loc[index,'LPDiffPG'] = my_data.loc[(my_data['Year'] == year) & (my_data['ID'] == lID)]['pDiffpg'].values[0]

        tourney.loc[index,'WOffRank'] = my_data.loc[(my_data['Year'] == year) & (my_data['ID'] == wID)]['offRank'].values[0]
        tourney.loc[index,'LOffRank'] = my_data.loc[(my_data['Year'] == year) & (my_data['ID'] == lID)]['offRank'].values[0]

        tourney.loc[index,'WDefRank'] = my_data.loc[(my_data['Year'] == year) & (my_data['ID'] == wID)]['defRank'].values[0]
        tourney.loc[index,'LDefRank'] = my_data.loc[(my_data['Year'] == year) & (my_data['ID'] == lID)]['defRank'].values[0]

#Drop NA values that did not get asigned properly to prevent errors
print('Pre NA drop ', tourney.shape)
prepped_data = tourney.dropna()
print("filter data")
print('Post NA drop ', prepped_data.shape)


#Mapping from the MTeams naming conventions to the submission file naming conventions
sub_mapping = {
    "St John's": "St. John's",
    'Michigan St': 'Michigan St.',
    'Iowa St': 'Iowa St.',
    "St Mary's CA": "Saint Mary's",
    'Mississippi St': 'Mississippi St.',
    'Utah St': 'Utah St.',
    'San Diego St': 'San Diego St.',
    'McNeese St': 'McNeese St.',
    'Colorado St': 'Colorado St.',
    'NE Omaha': 'Omaha',
    'Norfolk St': 'Norfolk St.',
    'American Univ': 'American',
    "Mt St Mary's": "Mount St. Mary's",
    'Alabama St': 'Alabama St.',
    'St Francis NY': 'Saint Francis',
    'UC San Diego': 'San Diego',
    'SIU Edwardsville': 'SIUE',
    'St Francis PA': 'Saint Francis'
}

def map_teams(team_name):
    return sub_mapping.get(team_name, team_name)


#Create a dataframe with data for submission matchups and the data needed to make predictions
submission = pd.read_csv("competition_submission.csv")
for index, row in submission.iterrows():
    t1 = row['higher_seed']
    t2 = row['lower_seed']
    year = 2025

    if(not my_data.loc[(my_data['Year'] == year) & (my_data['Team'].apply(map_teams) == t1)]['offRating'].any()):
        print(t1)
    if(not my_data.loc[(my_data['Year'] == year) & (my_data['Team'].apply(map_teams) == t2)]['offRating'].any()):
        print(t2)

    submission.loc[index,'T1_OffRating'] = my_data.loc[(my_data['Year'] == year) & (my_data['Team'].apply(map_teams) == t1)]['offRating'].values[0]
    submission.loc[index,'T2_OffRating'] = my_data.loc[(my_data['Year'] == year) & (my_data['Team'].apply(map_teams) == t2)]['offRating'].values[0]

    submission.loc[index,'T1_tempo'] = my_data.loc[(my_data['Year'] == year) & (my_data['Team'].apply(map_teams) == t1)]['tempo'].values[0]
    submission.loc[index,'T2_tempo'] = my_data.loc[(my_data['Year'] == year) & (my_data['Team'].apply(map_teams) == t2)]['tempo'].values[0]

    submission.loc[index,'T1_fgEff'] = my_data.loc[(my_data['Year'] == year) & (my_data['Team'].apply(map_teams) == t1)]['fgEff'].values[0]
    submission.loc[index,'T2_fgEff'] = my_data.loc[(my_data['Year'] == year) & (my_data['Team'].apply(map_teams) == t2)]['fgEff'].values[0]

    submission.loc[index,'T1_ftRate'] = my_data.loc[(my_data['Year'] == year) & (my_data['Team'].apply(map_teams) == t1)]['ftRate'].values[0]
    submission.loc[index,'T2_ftRate'] = my_data.loc[(my_data['Year'] == year) & (my_data['Team'].apply(map_teams) == t2)]['ftRate'].values[0]

    submission.loc[index,'T1_wab'] = my_data.loc[(my_data['Year'] == year) & (my_data['Team'].apply(map_teams) == t1)]['wab'].values[0]
    submission.loc[index,'T2_wab'] = my_data.loc[(my_data['Year'] == year) & (my_data['Team'].apply(map_teams) == t2)]['wab'].values[0]

    submission.loc[index,'T1_talent'] = my_data.loc[(my_data['Year'] == year) & (my_data['Team'].apply(map_teams) == t1)]['talent'].values[0]
    submission.loc[index,'T2_talent'] = my_data.loc[(my_data['Year'] == year) & (my_data['Team'].apply(map_teams) == t2)]['talent'].values[0]

    submission.loc[index,'T1_sos'] = my_data.loc[(my_data['Year'] == year) & (my_data['Team'].apply(map_teams) == t1)]['sos'].values[0]
    submission.loc[index,'T2_sos'] = my_data.loc[(my_data['Year'] == year) & (my_data['Team'].apply(map_teams) == t2)]['sos'].values[0]

    submission.loc[index,'T1_Threepg'] = my_data.loc[(my_data['Year'] == year) & (my_data['Team'].apply(map_teams) == t1)]['threepg'].values[0]
    submission.loc[index,'T2_Threepg'] = my_data.loc[(my_data['Year'] == year) & (my_data['Team'].apply(map_teams) == t2)]['threepg'].values[0]

    submission.loc[index,'T1_FTPG'] = my_data.loc[(my_data['Year'] == year) & (my_data['Team'].apply(map_teams) == t1)]['ftpg'].values[0]
    submission.loc[index,'T2_FTPG'] = my_data.loc[(my_data['Year'] == year) & (my_data['Team'].apply(map_teams) == t2)]['ftpg'].values[0]

    submission.loc[index,'T1_PDiffPG'] = my_data.loc[(my_data['Year'] == year) & (my_data['Team'].apply(map_teams) == t1)]['pDiffpg'].values[0]
    submission.loc[index,'T2_PDiffPG'] = my_data.loc[(my_data['Year'] == year) & (my_data['Team'].apply(map_teams) == t2)]['pDiffpg'].values[0]

    submission.loc[index,'T1_OffRank'] = my_data.loc[(my_data['Year'] == year) & (my_data['Team'].apply(map_teams) == t1)]['offRank'].values[0]
    submission.loc[index,'T2_OffRank'] = my_data.loc[(my_data['Year'] == year) & (my_data['Team'].apply(map_teams) == t2)]['offRank'].values[0]

    submission.loc[index,'T1_DefRank'] = my_data.loc[(my_data['Year'] == year) & (my_data['Team'].apply(map_teams) == t1)]['defRank'].values[0]
    submission.loc[index,'T2_DefRank'] = my_data.loc[(my_data['Year'] == year) & (my_data['Team'].apply(map_teams) == t2)]['defRank'].values[0]



#Replaces the W and L with T1 and T2 as well as creates 2 copies of every game so that the team listed first does not always win every time
def prepare_data(df):
    dfswap = df[['Season', 'DayNum', 'LTeam', 'LTeamID', 'LScore', 'WTeam', 'WTeamID', 'WScore', 'WLoc', 'NumOT', 'WOffRating', 'LOffRating', 'WOffRank', 'LOffRank', 'WDefRank', 'LDefRank', 'WTempo', 'LTempo', 'WfgEff', 'LfgEff', 'WftRate', 'LftRate', 'Wwab', 'Lwab', 'Wtalent', 'Ltalent', 'Wsos', 'Lsos', 'WThreepg', 'LThreepg', 'WFTPG', 'LFTPG', 'WPDiffPG', 'LPDiffPG']]
  
    df.columns = [x.replace('W','T1_').replace('L','T2_') for x in list(df.columns)]
    dfswap.columns = [x.replace('L','T1_').replace('W','T2_') for x in list(dfswap.columns)]

    output = pd.concat([df, dfswap]).reset_index(drop=True)    
    output['PointDiff'] = output['T1_Score'] - output['T2_Score']
    
    return output

#Apply the function
prepped_data = prepare_data(prepped_data)


#Create Differential Stats
prepped_data['offRankDiff'] = prepped_data['T1_OffRank'] - prepped_data['T2_OffRank']
prepped_data['defRankDiff'] = prepped_data['T1_DefRank'] - prepped_data['T2_DefRank']
prepped_data['talentDiff'] = prepped_data['T1_talent'] - prepped_data['T2_talent']
prepped_data['sosDiff'] = prepped_data['T1_sos'] - prepped_data['T2_sos']

submission['offRankDiff'] = submission['T1_OffRank'] - submission['T2_OffRank']
submission['defRankDiff'] = submission['T1_DefRank'] - submission['T2_DefRank']
submission['talentDiff'] = submission['T1_talent'] - submission['T2_talent']
submission['sosDiff'] = submission['T1_sos'] - submission['T2_sos']
submission['Seed_Diff'] = submission['higher_seed_num'] - submission['lower_seed_num']


#Apply seeds
seeds['seed'] = seeds['Seed'].apply(lambda x: int(x[1:3]))
seeds_T1 = seeds[['Season','TeamID','seed']].copy()
seeds_T2 = seeds[['Season','TeamID','seed']].copy()
seeds_T1.columns = ['Season','T1_TeamID','T1_seed']
seeds_T2.columns = ['Season','T2_TeamID','T2_seed']
prepped_data = pd.merge(prepped_data, seeds_T1, on = ['Season', 'T1_TeamID'], how = 'left')
prepped_data = pd.merge(prepped_data, seeds_T2, on = ['Season', 'T2_TeamID'], how = 'left')

prepped_data['Seed_Diff'] = prepped_data['T1_seed'] - prepped_data['T2_seed']

#Save our data to be used by the model
prepped_data.to_csv('prepped_data.csv', index=False)
submission.to_csv('2025_prepped_matchups.csv', index=False)


