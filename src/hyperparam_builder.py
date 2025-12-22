from abc import ABC, abstractmethod


class ParamGridBuilder(ABC):
    raise NotImplementedError


class LogRegParamGrid(ParamGridBuilder):
    pass


class HGBCParamGrid(ParamGridBuilder):
    pass
