import pandas as pd
import numpy as np


class teamMatcher:
    def __init__(self):
        self.player_team_matcher = pd.DataFrame(
            {
                "id": list(range(1, 24)),
                "team1": [0] * 23,
                "team2": [1] * 23,
                "current_team": [1] * 23,
            }
        )

    def update_matches(self, ids: np.array, matches: np.array) -> pd.DataFrame:
        res_list = []
        df = self.player_team_matcher
        for i, id in enumerate(ids):
            if id not in df["id"].tolist():
                df.loc[len(df)] = {"id": id, "team1": 0, "team2": 0, "current_team": 0}

            if df[df.loc[:, "id"] == id][["team1", "team2"]].max().max() < 50:

                if matches[i] == 0:  # & ((self.player_team_matcher.loc[i, 'team2'] + \
                    #          self.player_team_matcher.loc[i, 'team1']) < 102):
                    self._update_counters(df, id, "team1", "team2", res_list)
                elif (
                    matches[i] == 1
                ):  # & ((self.player_team_matcher.loc[i, 'team2'] + \
                    #      self.player_team_matcher.loc[i, 'team1']) <  102):
                    self._update_counters(df, id, "team2", "team1", res_list)
            else:
                res_list.append(df.loc[i, "current_team"])
        print(df)
        return res_list

    def _update_counters(self, df, id, arg1, arg2, res_list):
        idx = df[df.loc[:, "id"] == id].index[0]
        df.loc[idx, arg1] += 1
        if df.loc[idx, arg2] != 0:
            df.loc[idx, arg2] -= 1
        df.loc[idx, "current_team"] = (
            1 if (df.loc[idx, "team1"] > df.loc[idx, "team2"]) else 2
        )
        res_list.append(df.loc[idx, "current_team"])
