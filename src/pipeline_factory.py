class PipelineFactory:
    @staticmethod
    def get_strategy(ctx: ExpContext) -> PipelineStrategy:

        if ctx.flags.is_concat:
            return ConcatPipeline()

        if ctx.flags.has_text and ctx.flags.has_rte:
            return TextRTEPipeline()

        if ctx.flags.has_text:
            return TextPipeline()

        if ctx.flags.has_rte:
            return RTEPipeline()

        return BasicPipeline()
