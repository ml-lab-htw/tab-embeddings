from src.exp_context import ExpContext
from src.pipeline_builder import ConcatTextPipeline, ConcatRTEPipeline, PipelineStrategy, TextPipeline, RTEPipeline, \
    BasicPipeline


class PipelineFactory:
    @staticmethod
    def get_strategy(ctx: ExpContext) -> PipelineStrategy:
        if ctx.flags.has_text:
            if ctx.flags.is_concat:
                return ConcatTextPipeline()
            return TextPipeline()

        elif ctx.flags.has_rte:
            if ctx.flags.is_concat:
                return ConcatRTEPipeline()
            return RTEPipeline()

        return BasicPipeline()
