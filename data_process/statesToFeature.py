import pickle
import pandas as pd
from FeatureExtractor.ExtractGameFeatures import featureExtractor
import numpy as np


def stateToFeature():
    data = pd.read_pickle("../Data/process/bi-20_weight_norm.pkl")

    for i in range(len(data)):
        if data["energizers"][i] == []:
            data["energizers"][i] = np.nan
    feature_extractor = featureExtractor(map_name="originalClassic")
    feature_extractor.state_num = 10
    feature = feature_extractor.extract_feature(data)
    columns = list(data.columns)
    feature[columns] = data[columns]

    feature.to_pickle("../Data/output/bi-20_feature.pkl")


if __name__ == '__main__':
    stateToFeature()
