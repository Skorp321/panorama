import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

class TrackMacher:
    def __init__(self):
        self.tracks = []
        self.buffer_df = pd.DataFrame({"pred_track_id": [], "cur_track_id": []})        
        self.matched_tracs_id_dict = {}
        self.prev_df = pd.DataFrame(list(range(1,24)), columns=['id'])
        self.buffer_df = pd.DataFrame()
                
    def mach(self, tracks_id_dataFrame: pd.DataFrame, prev_dets: pd.DataFrame) -> pd.DataFrame:
        
        '''tracks_id_dataFrame['team1'] = 
        
        for i in range(len(tracks_id_dataFrame['pred_team'])):
            if (tracks_id_dataFrame['pred_team'][i]  ==  1) & ((results_df.loc[i, 'team2'] + results_df.loc[i, 'team1']) < 102):
                results_df.loc[i, 'team1'] += 1
                if results_df.loc[i, 'team2'] !=  0:
                    results_df.loc[i, 'team2'] -= 1
            elif (pred_team[i]  ==  2) & ((results_df.loc[i, 'team2'] + results_df.loc[i, 'team1']) < 102):
                results_df.loc[i, 'team2'] += 1
                if results_df.loc[i, 'team1'] !=  0:
                    results_df.loc[i, 'team1'] -= 1
                    
        results_df['team'] == results_df.apply(lambda row: 1 if row['x1'] > row['y1'] else 2, axis=1)

        teams_stats = results_df[['id', 'team1',  'team2',  'team']]'''
        
        tracks = self.prev_df['id'].values.astype(int).tolist()
        matched_tracks = []
        unmatched_tracks = []
        free_tracks = list(range(1,24))
        matched_tracs_id_dict = self.matched_tracs_id_dict
        prev_tracs = [int(x) for x in prev_dets['id']]
        copy_tracks_id_dataFrame = tracks_id_dataFrame.copy()
        #print(f"Dets trecs: {tracks_id_dataFrame['id'].tolist()}")
        for _, track in copy_tracks_id_dataFrame.iterrows():
            if int(track['id']) in tracks:
                matched_tracks.append(int(track['id']))
                #print(f"remuv: {int(track['id'])}")
                free_tracks.remove(int(track['id']))
            elif int(track['id']) in list(matched_tracs_id_dict.keys()) and (matched_tracs_id_dict[int(track['id'])] not in tracks_id_dataFrame['id'].tolist()):
                matched_tracks.append(int(track['id']))
                #print(f"remuv: {int(track['id'])}")
                free_tracks.remove(matched_tracs_id_dict[int(track['id'])])
                tracks_id_dataFrame['id']  = tracks_id_dataFrame['id'].replace(int(track['id']), matched_tracs_id_dict[int(track['id'])])                  
            else:
                unmatched_tracks.append(int(track['id']))
        #print(f'matched tracks: {matched_tracks}, unmatched tracks: {unmatched_tracks}, free tracks: {free_tracks}')
        #print(matched_tracs_id_dict)

        if not unmatched_tracks:
            pass
        elif (len(unmatched_tracks)  ==  1)  &  (len(free_tracks)  ==  1)  &  (unmatched_tracks[0]  >  23):
            matched_tracs_id_dict[unmatched_tracks[0]] = free_tracks[0]

        else:
            current_frame_id_unmatched = tracks_id_dataFrame[tracks_id_dataFrame['id'].isin(unmatched_tracks)]
            #print(current_frame_id_unmatched)
            prev_frame_id_unmatched = prev_dets[prev_dets['id'].isin(free_tracks)]
            #print(prev_frame_id_unmatched)

        '''
        
        if len(unmatched_tracks) == 0:
            self.matched_tracs_id_dict = matched_tracs_id_dict
            self.buffer_df  = tracks_id_dataFrame.copy()
            print(matched_tracs_id_dict)
            return tracks_id_dataFrame
        elif (len(unmatched_tracks) == 1) & (len(self.tracks)==1) & (unmatched_tracks[0] > 23):
            matched_tracs_id_dict[unmatched_tracks[0]] = self.tracks[0]
            tracks_id_dataFrame['id'] = tracks_id_dataFrame['id'].replace(unmatched_tracks[0], self.tracks[0])
            #tracks_id_dataFrame.loc[tracks_id_dataFrame['id'] == list(matched_tracs_id_dict.keys())[0], 'id'] = list(matched_tracs_id_dict.values())[0]
            self.matched_tracs_id_dict = matched_tracs_id_dict
            print(matched_tracs_id_dict)
            return tracks_id_dataFrame
        else:
            current_frame_id_unmatched = tracks_id_dataFrame[tracks_id_dataFrame['id'].isin(unmatched_tracks)]
            print(current_frame_id_unmatched['id'])
            prev_frame_id_unmatched = prev_dets[prev_dets['id'].isin(prev_tracs)]
            print(prev_frame_id_unmatched['id'])
            if len(current_frame_id_unmatched['id'])  ==  0:
                pass
            current_frame_id_unmatched['x_center'] = current_frame_id_unmatched['x1'] + current_frame_id_unmatched['w'] / 2.0
            current_frame_id_unmatched['y_center'] = current_frame_id_unmatched['y1'] + current_frame_id_unmatched['h'] / 2.0
            current_frame_id_unmatched = current_frame_id_unmatched.sort_values(['x_center', 'y_center'], ascending=[True, True]).reset_index(drop=True)
            prev_frame_id_unmatched['x_center'] = prev_frame_id_unmatched['x1'] + prev_frame_id_unmatched['w'] / 2.0
            prev_frame_id_unmatched['y_center'] = prev_frame_id_unmatched['y1'] + prev_frame_id_unmatched['h'] / 2.0
            prev_frame_id_unmatched = prev_frame_id_unmatched.sort_values(['x_center', 'y_center'], ascending=[True, True]).reset_index(drop=True)

            selected_columns = ['x_center', 'y_center']
            curent_coords = current_frame_id_unmatched[selected_columns].to_numpy()
            prev_coords = prev_frame_id_unmatched[selected_columns].to_numpy()

            # Вычисление матрицы евклидовых расстояний
            distance_matrix = cdist(curent_coords, prev_coords, metric='euclidean')

            # Решение задачи сопоставления с минимальной стоимостью (расстоянием)
            row_ind, col_ind = linear_sum_assignment(distance_matrix)

            for i, j in zip(row_ind, col_ind):
                tracks_id_dataFrame.loc[tracks_id_dataFrame['id'] == current_frame_id_unmatched.loc[i, 'id'], 'id'] = prev_frame_id_unmatched.loc[j, 'id']
                matched_tracs_id_dict[current_frame_id_unmatched.loc[i, 'id']] = prev_frame_id_unmatched.loc[j, 'id']
                print(matched_tracs_id_dict)
        
        self.matched_tracs_id_dict = matched_tracs_id_dict'''
        self.buffer_df =  tracks_id_dataFrame.copy()
        return tracks_id_dataFrame