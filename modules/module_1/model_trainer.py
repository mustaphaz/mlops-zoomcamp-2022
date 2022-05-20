import logging
from dataclasses import dataclass

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)



@dataclass
class ModelTrainer:
    """
    Trains linear regression model.
    """

    features: dict
    target: pd.Series

    def train_model(self) -> Pipeline:
        """Trains the model"""
        logger.debug("Training model")
        vectorizer = DictVectorizer()
        vectorizer.fit(self.features)
        transformed = vectorizer.transform(self.features)
        model = LinearRegression()
        model.fit(transformed, self.target)

        return Pipeline([("vect", vectorizer), ("lr", model)])
