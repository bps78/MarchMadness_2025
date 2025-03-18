import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

kenpom = pd.read_csv("Kenpom_eff_renamed.csv") #Chat-GPT renamed teams to match spelling in MTeams.csv
kenpom_off = pd.read_csv("Kenpom_off_renamed.csv")

bart = pd.read_csv("Barttorvik_renamed.csv")

teams = pd.read_csv("MTeams.csv")
season = pd.read_csv("MRegularSeasonDetailedResults.csv")
tourney = pd.read_csv("MNCAATourneyCompactResults.csv")
seeds = pd.read_csv("MNCAATourneySeeds.csv")

my_data = pd.DataFrame()
year = 2003

def getAggregateStats(my_team, year):
    
    w_games = season.loc[((season['WTeamID'] == my_team)) & (season['Season'] == year)]
    l_games = season.loc[((season['LTeamID'] == my_team)) & (season['Season'] == year)]
    
    gameCount = (len(w_games) + len(l_games))

    threesPerG = (w_games['WFGM3'].sum() + l_games['LFGM3'].sum()) / gameCount  #Three Pointers Made / Game
    ftpg = (w_games['WFTM'].sum() + l_games['LFTM'].sum()) / gameCount #Free Throws Made / Game
    pdiffpg = ((w_games['WScore'].sum() - w_games['LScore'].sum()) + (l_games['LScore'].sum() - l_games['WScore'].sum())) / gameCount #Mean Point differential per game
   
    return (threesPerG, ftpg, pdiffpg)

for y in range(2008, 2025):
    year = y
    if(year != 2020):
        for x, z in zip(teams['TeamName'], teams['TeamID']):
            if(kenpom.loc[(kenpom['Team'] == x) & (kenpom['Season'] == year)]["Adjusted Tempo"].any() and kenpom_off.loc[(kenpom_off['TeamName'] == x) & (kenpom_off['Season'] == year)]["eFGPct"].any() and bart.loc[(bart['TEAM'] == x) & (bart['YEAR'] == year)]['WAB'].any()):
                offRank = kenpom.loc[(kenpom['Team'] == x) & (kenpom['Season'] == year)]["Adjusted Offensive Efficiency Rank"].values[0]
                defRank = kenpom.loc[(kenpom['Team'] == x) & (kenpom['Season'] == year)]["Adjusted Defensive Efficiency Rank"].values[0]
                temp = kenpom.loc[(kenpom['Team'] == x) & (kenpom['Season'] == year)]["Adjusted Tempo"].values[0]
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

print("Num of Unique teams data was found for",len(my_data['Team'].unique()))
print(my_data['Team'].unique())
my_data.to_csv('myInitialData.csv', index=False)

tourney = tourney.loc[tourney['Season'] > 2007]
print("# of tourney games", len(tourney['Season']))
#Get each game from the tournament since 2008
for index, row in tourney.iterrows():
    wID = row['WTeamID']
    lID = row['LTeamID']
    year = row['Season']

    if(my_data.loc[(my_data['Year'] == year) & (my_data['ID'] == wID)]['offRating'].any() and my_data.loc[(my_data['Year'] == year) & (my_data['ID'] == lID)]['offRating'].any()):
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

    else:
        #Debugging
        if(not my_data.loc[(my_data['Year'] == year) & (my_data['ID'] == wID)]['offRating'].any()):
            print(teams.loc[teams['TeamID'] == wID, 'TeamName'].values[0], year)
        else:
            print(teams.loc[teams['TeamID'] == lID, 'TeamName'].values[0], year)

print(tourney.shape)
prepped_data = tourney.dropna()
print("filter data")
print(prepped_data.shape)


def prepare_data(df):
    dfswap = df[['Season', 'DayNum', 'LTeamID', 'LScore', 'WTeamID', 'WScore', 'WLoc', 'NumOT', 'WOffRating', 'LOffRating', 'WOffRank', 'LOffRank', 'WDefRank', 'LDefRank', 'WTempo', 'LTempo', 'WfgEff', 'LfgEff', 'WftRate', 'LftRate', 'Wwab', 'Lwab', 'Wtalent', 'Ltalent', 'Wsos', 'Lsos', 'WThreepg', 'LThreepg', 'WFTPG', 'LFTPG', 'WPDiffPG', 'LPDiffPG']]
  
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
prepped_data['defRankDiff'] = prepped_data['T1_DefRank'] - prepped_data['T2_DefRank']


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


