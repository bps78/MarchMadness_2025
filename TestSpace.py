import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

kenpom = pd.read_csv("Renamed_KenPom.csv") #Chat-GPT renamed teams to match spelling in MTeams.csv
teams = pd.read_csv("MTeams.csv")
season = pd.read_csv("MRegularSeasonDetailedResults.csv")
tourney = pd.read_csv("MNCAATourneyCompactResults.csv")

tourney = tourney.loc[tourney['Season'] > 2002]
print(tourney.shape)

    