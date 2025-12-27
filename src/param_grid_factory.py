from src.exp_context import ExpContext
from src.hyperparam_builder import TextGrid, RTEGrid, BasicGrid, ConcatTextGrid, ConcatRTEGrid, ParamGridStrategy


class ParamGridFactory:
    @staticmethod
    def get_strategy(ctx: ExpContext) -> ParamGridStrategy:
        if ctx.flags.has_text:
            if ctx.flags.is_concat:
                return ConcatTextGrid()
            return TextGrid()

        if ctx.flags.has_rte:
            if ctx.flags.is_concat:
                return ConcatRTEGrid()
            return RTEGrid()

        return BasicGrid()
