import pickle
import pandas as pd
from FeatureExtractor.ExtractGameFeatures import featureExtractor
import numpy as np


def stateToFeature():
    data = pd.read_pickle("../Data/process/one_0.pkl")

    for i in range(len(data)):
        if data["energizers"][i] == []:
            data["energizers"][i] = np.nan
    feature_extractor = featureExtractor(map_name="originalClassic")
    feature_extractor.state_num = 10
    feature = feature_extractor.extract_feature(data)
    columns = list(data.columns)
    feature[columns] = data[columns]

    feature.to_pickle("../Data/output/one_0_feature.pkl")


if __name__ == '__main__':
    stateToFeature()
