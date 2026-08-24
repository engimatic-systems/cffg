from cffg import structured_hybrid as structured_hybrid
from cffg import switching_stability as switching_stability
from cffg.params import set_params

__all__ = ["hello", "set_params", "structured_hybrid", "switching_stability"]


def hello() -> str:
    return "Hello from cffg!"
