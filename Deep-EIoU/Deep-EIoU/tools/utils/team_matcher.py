import pandas as pd
import numpy as np

class teamMatcher:
    def __init__(self):
        self.team_colors = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255)}
        self.player_team_matcher = pd.DataFrame({
            'id': list(range(23)),
            'team1': [0] * 23,
            'team2': [1] * 23,
            'current_team': [2] * 23})
        
    def update_matches(self, matches: np.array) -> pd.DataFrame:
        for i in range(len(matches)):
            if (matches[i] == 1) & ((self.player_team_matcher.loc[i, 'team2'] + \
                                    self.player_team_matcher.loc[i, 'team1']) < 102):
                
                self.player_team_matcher.loc[i, 'team1'] += 1
                if self.player_team_matcher.loc[i, 'team2'] !=  0:
                    self.player_team_matcher.loc[i, 'team2'] -= 1
            elif (matches[i]  ==  2) & ((self.player_team_matcher.loc[i, 'team2'] + \
                                    self.player_team_matcher.loc[i, 'team1']) <  102):
                
                self.player_team_matcher.loc[i, 'team2'] += 1
                if self.player_team_matcher.loc[i, 'team1'] !=  0:
                    self.player_team_matcher.loc[i, 'team1'] -= 1
                
        return self.player_team_matcher.apply(
            lambda row: 1 if row['team1'] > row['team2'] else 2, axis=1
        )
