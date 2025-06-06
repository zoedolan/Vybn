class _Completions:
    def create(self, *args, **kwargs):
        raise NotImplementedError("openai stub: override in tests")

class chat:
    completions = _Completions()

api_key = None
