from importlib import import_module

__all__ = ["compute_metrics", "compute_full_eval", "meets_targets"]


def __getattr__(name: str):
    if name in __all__:
        metrics_module = import_module(".metrics", __name__)
        value = getattr(metrics_module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
