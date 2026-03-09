from abc import ABC, abstractmethod

from config.config_manager import ConfigManager
from src.exp_context import ExpContext


def prefix_grid(grid: dict[str, list], prefix: str) -> dict[str, list]:
    return {f"{prefix}{k}": v for k, v in grid.items()}


class ParamGridStrategy(ABC):
    @abstractmethod
    def build(self, ctx):
        """Return a parameter grid dict."""
        pass

class BasicGrid(ParamGridStrategy):
    def build(self, ctx:ExpContext):
        if ctx.flags.is_lr:
            LR_GRID = ctx.cfg.hyperparams_cfg['lr']
            return LR_GRID
        elif ctx.flags.is_gbdt:
            GBDT_GRID = ctx.cfg.hyperparams_cfg['gbdt']
            return GBDT_GRID
        raise NotImplementedError

class RTEGrid(ParamGridStrategy):
    def build(self, ctx):
        base = ctx.cfg.hyperparams_cfg['lr'] if ctx.flags.is_lr else ctx.cfg.hyperparams_cfg['gbdt']
        RTE_GRID = ctx.cfg.hyperparams_cfg['rte']
        return {
            **base,
            **RTE_GRID,
        }

class TextGrid(ParamGridStrategy):
    def build(self, ctx):
        base = ctx.cfg.hyperparams_cfg['lr'] if ctx.flags.is_lr else ctx.cfg.hyperparams_cfg['gbdt']
        TEXT_GRID = ctx.cfg.hyperparams_cfg['text_emb']
        return {
            **base,
            **TEXT_GRID,
        }

class ConcatTextGrid(ParamGridStrategy):
    def build(self, ctx):
        base = ctx.cfg.hyperparams_cfg['lr'] if ctx.flags.is_lr else ctx.cfg.hyperparams_cfg['gbdt']
        TEXT_GRID = ctx.cfg.hyperparams_cfg['text_emb']
        return {
            **base,
            **prefix_grid(TEXT_GRID, "transformer__text__"),
        }


class ConcatRTEGrid(ParamGridStrategy):
    def build(self, ctx):
        base = ctx.cfg.hyperparams_cfg['lr'] if ctx.flags.is_lr else ctx.cfg.hyperparams_cfg['gbdt']
        RTE_GRID = ctx.cfg.hyperparams_cfg['rte']
        return {
            **base,
            **prefix_grid(RTE_GRID, "feature_combiner__embeddings__"),
        }
