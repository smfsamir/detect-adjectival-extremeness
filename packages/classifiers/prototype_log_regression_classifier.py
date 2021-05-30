import numpy as np
from ..utils.calculations import cos
from sklearn.linear_model import LogisticRegression

class PrototypeLogisticRegressionClassifier():

    def __init__(self):
        """
        """
        self.centroid = None
        self.lr_classifier = None

    def fit(self, X, y):
        """
        """
        # TODO: construct centroid for positive class
        extreme_embeds = X.loc[y==1]
        extreme_centroid = extreme_embeds.mean(axis=0).values
        self.centroid = extreme_centroid

        # TODO: fit logistic regression classifier
        sim_to_extreme_centroid = X.apply(lambda vec: cos(vec.values, self.centroid), axis=1)
        sim_to_extreme_centroid = np.expand_dims(sim_to_extreme_centroid.values, -1)
        self.lr_classifier = LogisticRegression().fit(sim_to_extreme_centroid, y)
        return self

    def predict_proba(self, X):
        """
        """
        sim_to_extreme_centroid = X.apply(lambda vec: cos(vec.values, self.centroid), axis=1)
        sim_to_extreme_centroid = np.expand_dims(sim_to_extreme_centroid.values, -1)
        return self.lr_classifier.predict_proba(sim_to_extreme_centroid)

    def predict(self, X):
        """
        """
        sim_to_extreme_centroid = X.apply(lambda vec: cos(vec.values, self.centroid), axis=1)
        sim_to_extreme_centroid = np.expand_dims(sim_to_extreme_centroid.values, -1)
        return self.lr_classifier.predict(sim_to_extreme_centroid)