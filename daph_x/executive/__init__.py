"""DAPH-X cognitive executive package."""
from daph_x.executive.budget import BudgetEnvelope

# Registry imports operators.external.base which imports executive.budget,
# so we defer the registry import to avoid a circular dependency.
def __getattr__(name: str):
    if name in ("OperatorRegistry", "OperatorStatus", "RegistryEntry"):
        from daph_x.executive import registry as _reg
        if name == "OperatorRegistry":
            return _reg.OperatorRegistry
        if name == "OperatorStatus":
            return _reg.OperatorStatus
        if name == "RegistryEntry":
            return _reg.RegistryEntry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BudgetEnvelope",
    "OperatorRegistry",
    "OperatorStatus",
    "RegistryEntry",
]
