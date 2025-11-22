from abc import ABC, abstractmethod


class PredictionModel(ABC):

    @abstractmethod
    def predict_next_time_step(self):
        raise NotImplementedError


# TODO: Implement models
