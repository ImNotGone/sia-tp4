from typing import Callable, Dict, Any

def _constant_radius(radius: float) -> float:
    return radius

def _linear_radius(epoch: int, initial_radius: float) -> float:
    if epoch == 0:
        return initial_radius

    current_radius = initial_radius / (epoch + 1)

    if current_radius < 1:
        return 1
    
    return current_radius

RadiusFunction = Callable[[int], float]

def get_radius_function(config: Dict[str, Any]) -> RadiusFunction:
    radius_type = config["radius_type"]
    radius = config["radius"]

    if radius_type == "constant":
        return lambda _: _constant_radius(radius)
    elif radius_type == "linear":
        return lambda epoch: _linear_radius(epoch, radius)

    raise ValueError(f"Unknown radius type: {radius_type}")
