import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

class TrackMacher:
    def __init__(self):
        self.tracks = []
        self.buffer_df = pd.DataFrame({"pred_track_id": [], "cur_track_id": []})        
        self.matched_tracs_id_dict = {}
                
    def mach(self, tracks_id_dataFrame: pd.DataFrame, prev_dets: pd.DataFrame) -> pd.DataFrame:
        # sourcery skip: simplify-len-comparison
        self.tracks = list(range(1,24))
        matched_tracks = []
        unmatched_tracks = []
        matched_tracs_id_dict = self.matched_tracs_id_dict
        prev_tracs = [int(x) for x in prev_dets['id']]
        print(tracks_id_dataFrame['id'])
        for _, track in tracks_id_dataFrame.iterrows():
            if int(track['id']) in self.tracks:
                self.tracks.remove(int(track['id']))
                if int(track['id']) in prev_tracs:
                    prev_tracs.remove(int(track['id']))
            elif int(track['id']) in list(matched_tracs_id_dict.values()):
                tracks_id_dataFrame['id'] = tracks_id_dataFrame['id'].replace(matched_tracs_id_dict)
                #tracks_id_dataFrame.loc[tracks_id_dataFrame['id'] == list(matched_tracs_id_dict.keys())[0], 'id'] = list(matched_tracs_id_dict.values())[0]
                self.tracks.remove(matched_tracs_id_dict[int(track['id'])])
                if matched_tracs_id_dict[int(track['id'])] in prev_tracs:
                    prev_tracs.remove(matched_tracs_id_dict[int(track['id'])])
            else:
                unmatched_tracks.append(int(track['id']))

        if len(unmatched_tracks) == 0:
            self.matched_tracs_id_dict = matched_tracs_id_dict
            return tracks_id_dataFrame
        elif (len(unmatched_tracks) == 1) & (len(self.tracks)==1):
            matched_tracs_id_dict[unmatched_tracks[0]] = self.tracks[0]
            tracks_id_dataFrame['id'] = tracks_id_dataFrame['id'].replace(unmatched_tracks[0], self.tracks[0])
            #tracks_id_dataFrame.loc[tracks_id_dataFrame['id'] == list(matched_tracs_id_dict.keys())[0], 'id'] = list(matched_tracs_id_dict.values())[0]
            self.matched_tracs_id_dict = matched_tracs_id_dict
            return tracks_id_dataFrame
        else:
            current_frame_id_unmatched = tracks_id_dataFrame[tracks_id_dataFrame['id'].isin(unmatched_tracks)]
            prev_frame_id_unmatched = prev_dets[prev_dets['id'].isin(prev_tracs)]
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
        self.matched_tracs_id_dict = matched_tracs_id_dict
        return tracks_id_dataFrame