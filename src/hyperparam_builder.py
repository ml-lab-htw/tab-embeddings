from abc import ABC, abstractmethod

from src.exp_context import ExpContext


def prefix_grid(grid: dict[str, list], prefix: str) -> dict[str, list]:
    return {f"{prefix}{k}": v for k, v in grid.items()}

LR_GRID = {
    "classifier__C": [2, 10],
}

GBDT_GRID = {
    "classifier__min_samples_leaf": [5, 10, 15, 20],
}

TEXT_GRID = {
    "embedding_aggregator__method": [
        "embedding_cls",
        "embedding_mean_with_cls_and_sep",
        "embedding_mean_without_cls_and_sep",
    ]
}

RTE_GRID = {
    "embedding__n_estimators": [10, 100],
    "embedding__max_depth": [2, 5],
}

class ParamGridStrategy(ABC):
    @abstractmethod
    def build(self, ctx):
        """Return a parameter grid dict."""
        pass

class BasicGrid(ParamGridStrategy):
    def build(self, ctx:ExpContext):
        if ctx.flags.is_lr:
            return LR_GRID
        elif ctx.flags.is_gbdt:
            return GBDT_GRID
        raise NotImplementedError

class RTEGrid(ParamGridStrategy):
    def build(self, ctx):
        base = LR_GRID if ctx.flags.is_lr else GBDT_GRID
        return {
            **base,
            **RTE_GRID,
        }

class TextGrid(ParamGridStrategy):
    def build(self, ctx):
        base = LR_GRID if ctx.flags.is_lr else GBDT_GRID
        return {
            **base,
            **TEXT_GRID,
        }

class ConcatTextGrid(ParamGridStrategy):
    def build(self, ctx):
        base = LR_GRID if ctx.flags.is_lr else GBDT_GRID
        return {
            **base,
            **prefix_grid(TEXT_GRID, "transformer__text__"),
        }


class ConcatRTEGrid(ParamGridStrategy):
    def build(self, ctx):
        base = LR_GRID if ctx.flags.is_lr else GBDT_GRID
        return {
            **base,
            **prefix_grid(RTE_GRID, "feature_combiner__embeddings__"),
        }
