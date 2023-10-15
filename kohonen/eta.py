from typing import Callable, Dict, Any

def _constant_eta(eta: float) -> float:
    return eta

def _linear_eta(epoch: int) -> float:
    return 1 / (epoch + 1)

EtaFunction = Callable[[int], float]

def get_eta_function(config: Dict[str, Any]) -> EtaFunction:
    eta_type = config["eta_type"]
    eta = config["eta"]

    if eta_type == "constant":
        return lambda _: _constant_eta(eta)
    elif eta_type == "linear":
        return _linear_eta

    raise ValueError(f"Unknown eta type: {eta_type}")


    
