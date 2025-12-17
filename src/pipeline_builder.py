from abc import ABC, abstractmethod
from sklearn.pipeline import Pipeline

class PipelineStrategy(ABC):
    @abstractmethod
    def build(self, ctx: ExpContext) -> Pipeline:
        ...


class LogRegPipeline(PipelineBuilder):
    pass


class HGBCPipeline(PipelineBuilder):
    pass