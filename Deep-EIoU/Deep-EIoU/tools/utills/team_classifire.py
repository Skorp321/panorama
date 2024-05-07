import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
import torch

from torchreid.utils.feature_extractor import FeatureExtractor


class TeamClassifier:
    def __init__(self, weights_path=None, model_name='osnet_ain_x1_0', device='cuda:0'):
        '''
        model_name   - название ипсользуемой модели ('osnet_ain_x1_0'),
        wieghts_path - путь к весам модели
        '''
        if weights_path is not None:
            self.extractor = FeatureExtractor(model_name=model_name, model_path=weights_path, device=device)
        else:
            self.extractor = None
        self.device = device
        self.embs_for_train = pd.DataFrame()
        self.k_means = KMeans(init='k-means++', n_clusters=2, n_init=10, algorithm='elkan')
        self.db = DBSCAN(eps=0.25, min_samples=8)

    def extract_features(self, img_paths):
        assert self.extractor is not None

        # Extract features:
        embs = self.extractor(img_paths)

        return list(map(list, embs.cpu().numpy()))

    #@staticmethod
    def classify(self, reid_features, count):
        # Classification by kmeans:
        pd_embs = pd.DataFrame(reid_features)
        
        if count < 3:
            self.embs_for_train = pd.concat([self.embs_for_train, pd_embs], ignore_index=True)
            pd_shuffled = self.embs_for_train.sample(frac=1)
            #self.embs_for_train = pd_embs
            #print(self.embs_for_train)
            #db = self.db.fit(self.embs_for_train)
            self.k_means.fit(pd_shuffled)
                
        team_preds = self.k_means.predict(pd_embs)
        #team_preds = db.labels_


        return team_preds