import pandas as pd


#These dictionaries were made for me with the help of Chat GPT

""" Prompt given:
Now I want you to manually pair up items from these lists that you belive are meant to be refering to the same thing 
(one item from each list per pair, item from kenpom on the left, item from Mteams on the right). You are not allowed 
to use any of the items from either list more than once and the items must be listed exactly as they are found in the CSV. 
Do not modify them at all, simply pair them together."""

#Mapping from the kenpom data set naming conventions to the MTeams file naming conventions
kenpomToMTeams = {
    'Oregon St.': 'Oregon St',
    'Maryland Eastern Shore': 'MD E Shore',
    'Loyola Marymount': 'Loy Marymount',
    'Florida St.': 'Florida St',
    'Eastern Illinois': 'E Illinois',
    'Wisconsin Milwaukee': 'WI Milwaukee',
    'Eastern Washington': 'E Washington',
    'Loyola Chicago': 'Loyola-Chicago',
    'Middle Tennessee St.': 'MTSU',
    'FIU': 'Florida Intl',
    'Southern Illinois': 'S Illinois',
    'The Citadel': 'Citadel',
    "Mount St. Mary's": "Mt St Mary's",
    'Arkansas Little Rock': 'Ark Little Rock',
    "Saint Peter's": "St Peter's",
    'Washington St.': 'Washington St',
    'Kent St.': 'Kent St',
    'UMKC': 'Missouri KC',
    'Cal St. Fullerton': 'CS Fullerton',
    'Mississippi Valley St.': 'MS Valley St',
    'College of Charleston': 'Col Charleston',
    'North Carolina Central': 'NC Central',
    'Northern Illinois': 'N Illinois',
    'Florida Atlantic': 'FL Atlantic',
    'Kennesaw St.': 'Kennesaw',
    'Texas A&M Commerce': 'TAM C. Christi',
    'North Dakota St.': 'N Dakota St',
    'Texas Pan American': 'TX Southern',
    'Youngstown St.': 'Youngstown St',
    'Monmouth': 'Monmouth NJ',
    'Little Rock': 'Little Rock',
    'Morgan St.': 'Morgan St',
    'East Tennessee St.': 'ETSU',
    'George Washington': 'G Washington',
    'Kansas St.': 'Kansas St',
    'Southern': 'Southern Univ',
    'McNeese St.': 'McNeese St',
    'McNeese': 'McNeese St',
    'Albany': 'SUNY Albany',
    'Saint Francis': 'St Francis PA',
    'Penn St.': 'Penn St',
    'Montana St.': 'Montana St',
    'Louisiana Lafayette': 'Louisiana Lafayette',
    'Cleveland St.': 'Cleveland St',
    'Arkansas St.': 'Arkansas St',
    'Middle Tennessee': 'MTSU',
    'Murray St.': 'Murray St',
    'Prairie View A&M': 'Prairie View',
    'UNC Charlotte': 'NC Charlotte',
    'Mississippi St.': 'Mississippi St',
    'Boise St.': 'Boise St',
    'Savannah St.': 'Savannah St',
    'UT Rio Grande Valley': 'UTRGV',
    "Saint Joseph's": "St Joseph's PA",
    'Western Kentucky': 'WKU',
    'Purdue Fort Wayne': 'PFW',
    'Southwest Texas St.': 'Texas St',
    'IPFW': 'IPFW',
    'St. Francis PA': 'St Francis PA',
    'St. Francis NY': 'St Francis NY',
    'Delaware St.': 'Delaware St',
    'Western Illinois': 'W Illinois',
    'Missouri St.': 'Missouri St',
    'Western Michigan': 'W Michigan',
    'Boston University': 'Boston Univ',
    'Ohio St.': 'Ohio St',
    'Utah St.': 'Utah St',
    'Arizona St.': 'Arizona St',
    'Long Beach St.': 'Long Beach St',
    'St. Thomas': 'St Thomas MN',
    'Weber St.': 'Weber St',
    'Iowa St.': 'Iowa St',
    'Southeast Missouri': 'SE Missouri St',
    'South Dakota St.': 'S Dakota St',
    'Jackson St.': 'Jackson St',
    'Southwest Missouri St.': 'Southwest Missouri St',
    'Green Bay': 'WI Green Bay',
    'Nicholls St.': 'Nicholls St',
    'Portland St.': 'Portland St',
    'Wright St.': 'Wright St',
    'Nicholls': 'Nicholls',
    'Texas Southern': 'Texas Southern',
    'Nebraska Omaha': 'NE Omaha',
    'San Jose St.': 'San Jose St',
    'Oklahoma St.': 'Oklahoma St',
    'Northwestern St.': 'Northwestern LA',
    'Western Carolina': 'W Carolina',
    'Eastern Kentucky': 'E Kentucky',
    'Georgia St.': 'Ga Southern',
    'New Mexico St.': 'New Mexico St',
    'Morehead St.': 'Morehead St',
    'Indiana St.': 'Indiana St',
    'Northern Colorado': 'N Colorado',
    'North Carolina A&T': 'NC A&T',
    'Abilene Christian': 'Abilene Chr',
    'Fairleigh Dickinson': 'F Dickinson',
    'Eastern Michigan': 'E Michigan',
    'Stephen F. Austin': 'SF Austin',
    'SIU Edwardsville': 'SIUE',
    'Charleston Southern': 'Charleston So',
    'Chicago St.': 'Chicago St',
    "Saint Mary's": "St Mary's CA",
    'Northeast Louisiana': 'N Louisiana',
    'Dixie St.': 'Dixie St',
    'Southeast Missouri St.': 'SE Missouri St',
    'Fresno St.': 'Fresno St',
    'Ball St.': 'Ball St',
    'American': 'American Univ',
    'Georgia Southern': 'Ga Southern',
    'Tarleton St.': 'Tarleton St',
    'Saint Louis': 'St Louis',
    'Wisconsin Green Bay': 'WI Green Bay',
    'Utah Valley St.': 'Utah Valley St',
    'Jacksonville St.': 'Jacksonville St',
    'Milwaukee': 'Milwaukee',
    'Colorado St.': 'Colorado St',
    'Houston Baptist': 'Houston Chr',
    'Birmingham Southern': 'Birmingham So',
    'Coastal Carolina': 'Coastal Car',
    'St. Bonaventure': 'St Bonaventure',
    'Houston Christian': 'Houston Christian',
    'CSUN': 'CSUN',
    'Charleston': 'Charleston',
    'UTSA': 'UT San Antonio',
    'Norfolk St.': 'Norfolk St',
    'Sam Houston St.': 'Sam Houston St',
    'Kansas City': 'Kansas City',
    'Idaho St.': 'Idaho St',
    'UMass Lowell': 'MA Lowell',
    'Central Michigan': 'C Michigan',
    'Illinois St.': 'Illinois St',
    'Winston Salem St.': 'W Salem St',
    'Michigan St.': 'Michigan St',
    'Texas St.': 'Texas St',
    'San Diego St.': 'San Diego St',
    'Tennessee St.': 'Tennessee St',
    'Sacramento St.': 'Sacramento St',
    "St. John's": "St John's",
    'Coppin St.': 'Coppin St',
    'Southeastern Louisiana': 'SE Louisiana',
    'Fort Wayne': 'Fort Wayne',
    'Cal St. Northridge': 'CS Northridge',
    'Florida Gulf Coast': 'FGCU',
    'South Carolina St.': 'S Carolina St',
    'Arkansas Pine Bluff': 'Ark Pine Bluff',
    'Northern Kentucky': 'N Kentucky',
    'Troy St.': 'Troy St',
    'LIU': 'LIU',
    'Wichita St.': 'Wichita St',
    'Texas Southern': 'TX Southern',
    'Georgia St.': 'Georgia St',
    'Grambling St.': 'Grambling',
    'Western Kentucky': 'WKU',
    'College of Charleston': 'Col Charleston',
    'Murray St.': 'Murray St',
    'Alabama St.': 'Alabama St'
}

def map_teams(team_name):
    return kenpomToMTeams.get(team_name, team_name)

#https://www.kaggle.com/datasets/jonathanpilafas/2024-march-madness-statistical-analysis?select=INT+_+KenPom+_+Offense.csv
kenpomEff = pd.read_csv("INT _ KenPom _ Efficiency.csv")
kenpomEff['Team'] = kenpomEff['Team'].apply(map_teams)
kenpomEff.to_csv('Kenpom_eff_renamed.csv', index=False)

#https://www.kaggle.com/datasets/jonathanpilafas/2024-march-madness-statistical-analysis?select=INT+_+KenPom+_+Offense.csv
kenpomOff = pd.read_csv("INT _ KenPom _ Offense.csv")
kenpomOff['TeamName'] = kenpomOff['TeamName'].apply(map_teams)
kenpomOff.to_csv('Kenpom_off_renamed.csv', index=False)

#Mapping from the barttorvik dataset naming conventions to the MTeams csv naming conventions
barttorvikToMTeams = team_pairings = {
    'Boise St.': 'Boise St',
    'College of Charleston': 'Col Charleston',
    'Colorado St.': 'Colorado St',
    'Florida Atlantic': 'FL Atlantic',
    'Grambling St.': 'Grambling',
    'Iowa St.': 'Iowa St',
    'Long Beach St.': 'Long Beach St',
    'McNeese St.': 'McNeese St',
    'Nebraska Omaha': 'NE Omaha',
    'Michigan St.': 'Michigan St',
    'Mississippi St.': 'Mississippi St',
    'Montana St.': 'Montana St',
    'Morehead St.': 'Morehead St',
    'North Carolina St.': 'NC State',
    "Saint Mary's": "St Mary's CA",
    "Saint Peter's": "St Peter's",
    'Saint Francis': 'St Francis PA',
    'San Diego St.': 'San Diego St',
    'South Dakota St.': 'S Dakota St',
    'SIU Edwardsville': 'SIUE',
    'Utah St.': 'Utah St',
    'Washington St.': 'Washington St',
    'Western Kentucky': 'WKU',
    'Arizona St.': 'Arizona St',
    'Fairleigh Dickinson': 'F Dickinson',
    'Kansas St.': 'Kansas St',
    'Kennesaw St.': 'Kennesaw',
    'Kent St.': 'Kent',
    'Louisiana Lafayette': 'Louisiana',
    'Northern Kentucky': 'N Kentucky',
    'Penn St.': 'Penn St',
    'Southeast Missouri St.': 'SE Missouri St',
    'Texas A&M Corpus Chris': 'TAM C. Christi',
    'Texas Southern': 'TX Southern',
    'Cal St. Fullerton': 'CS Fullerton',
    'Georgia St.': 'Georgia St',
    'Jacksonville St.': 'Jacksonville St',
    'Loyola Chicago': 'Loyola-Chicago',
    'Murray St.': 'Murray St',
    'New Mexico St.': 'New Mexico St',
    'Norfolk St.': 'Norfolk St',
    'Ohio St.': 'Ohio St',
    'Wright St.': 'Wright St',
    'Abilene Christian': 'Abilene Chr',
    'Appalachian St.': 'Appalachian St',
    'Cleveland St.': 'Cleveland St',
    'Eastern Washington': 'E Washington',
    'Florida St.': 'Florida St',
    "Mount St. Mary's": "Mt St Mary's",
    'Oklahoma St.': 'Oklahoma St',
    'Oregon St.': 'Oregon St',
    'St. Bonaventure': 'St Bonaventure',
    'Wichita St.': 'Wichita St',
    'North Carolina Central': 'NC Central',
    'North Dakota St.': 'N Dakota St',
    'Prairie View A&M': 'Prairie View',
    'Saint Louis': 'St Louis',
    "St. John's": "St John's",
    'Stephen F. Austin': 'SF Austin',
    'East Tennessee St.': 'ETSU',
    'Florida Gulf Coast': 'FGCU',
    'Middle Tennessee': 'MTSU',
    'Cal St. Bakersfield': 'CS Bakersfield',
    'Fresno St.': 'Fresno St',
    'Green Bay': 'WI Green Bay',
    'Little Rock': 'Ark Little Rock',
    "Saint Joseph's": "St Joseph's PA",
    'Southern': 'Southern Univ',
    'Weber St.': 'Weber St',
    'Albany': 'SUNY Albany',
    'Coastal Carolina': 'Coastal Car',
    'American': 'American Univ',
    'Eastern Kentucky': 'E Kentucky',
    'George Washington': 'G Washington',
    'Milwaukee': 'WI Milwaukee',
    'Western Michigan': 'W Michigan',
    'North Carolina A&T': 'NC A&T',
    'Northwestern St.': 'Northwestern LA',
    'Mississippi Valley St.': 'MS Valley St',
    'Alabama St.': 'Alabama St',
    'Boston University': 'Boston Univ',
    'Indiana St.': 'Indiana St',
    'Northern Colorado': 'N Colorado',
    'UTSA': 'UT San Antonio',
    'Arkansas Pine Bluff': 'Ark Pine Bluff',
    'Morgan St.': 'Morgan St',
    'Sam Houston St.': 'Sam Houston St',
    'Cal St. Northridge': 'CS Northridge',
    'Portland St.': 'Portland St',
    'Coppin St.': 'Coppin St'
}

def map_barttorvik_teams(team_name):
    return barttorvikToMTeams.get(team_name, team_name)

#https://www.kaggle.com/datasets/nishaanamin/march-madness-data
barttorvik = pd.read_csv("KenPom Barttorvik.csv")
barttorvik['TEAM'] = barttorvik['TEAM'].apply(map_barttorvik_teams)
barttorvik.to_csv('Barttorvik_renamed.csv', index=False)
