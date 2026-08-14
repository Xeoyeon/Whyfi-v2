from .chains import RAGAgent, get_agent  # noqa: F401


def __getattr__(name):
    if name == "ce_agent":
        return get_agent()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")