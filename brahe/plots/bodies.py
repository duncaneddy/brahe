"""Body visual registry for 3D plotting: radius, default texture, NAIF ID."""

import brahe as bh

# Radii: brahe packaged constants (IAU/WGCCRE 2015). Ceres has no packaged
# constant deliberately (no built-in body), so it uses a user-defined literal.
BODY_VISUALS = {
    "sun": {"radius": bh.R_SUN, "texture": "sun", "naif_id": 10},
    "mercury": {"radius": bh.R_MERCURY, "texture": "mercury", "naif_id": 199},
    "venus": {"radius": bh.R_VENUS, "texture": "venus", "naif_id": 299},
    "earth": {"radius": bh.R_EARTH, "texture": "blue_marble", "naif_id": 399},
    "moon": {"radius": bh.R_MOON, "texture": "moon", "naif_id": 301},
    "mars": {"radius": bh.R_MARS, "texture": "mars", "naif_id": 499},
    "jupiter": {"radius": bh.R_JUPITER, "texture": "jupiter", "naif_id": 599},
    "saturn": {"radius": bh.R_SATURN, "texture": "saturn", "naif_id": 699},
    "uranus": {"radius": bh.R_URANUS, "texture": "uranus", "naif_id": 799},
    "neptune": {"radius": bh.R_NEPTUNE, "texture": "neptune", "naif_id": 899},
    "ceres": {"radius": 469.7e3, "texture": "ceres", "naif_id": 2000001},
}

BODY_VISUALS_BY_NAIF_ID = {
    spec["naif_id"]: {"name": name.capitalize(), **spec}
    for name, spec in BODY_VISUALS.items()
}


def resolve_body(central_body) -> dict:
    """Resolve a central-body specification into rendering parameters.

    Args:
        central_body: Either a registry key from ``BODY_VISUALS``
            (e.g. ``'earth'``, ``'moon'``) or a dict with keys ``name`` (str),
            ``radius`` (float, m), and optionally ``texture``
            (str | Path | None).

    Returns:
        dict: ``{name, radius, texture, naif_id}`` (``naif_id`` is ``None``
        for custom dict specifications without one).

    Raises:
        ValueError: If a string key is not in the registry or a dict is
            missing ``name``/``radius``.
    """
    if isinstance(central_body, str):
        key = central_body.lower()
        if key not in BODY_VISUALS:
            raise ValueError(
                f"Unknown central body '{central_body}'. "
                f"Available: {', '.join(sorted(BODY_VISUALS))}; or pass a "
                "dict with 'name', 'radius', and optional 'texture'."
            )
        spec = BODY_VISUALS[key]
        return {"name": key.capitalize(), **spec}
    if isinstance(central_body, dict):
        if "name" not in central_body or "radius" not in central_body:
            raise ValueError(
                "Custom central_body dict requires 'name' and 'radius' keys"
            )
        return {
            "name": central_body["name"],
            "radius": float(central_body["radius"]),
            "texture": central_body.get("texture"),
            "naif_id": central_body.get("naif_id"),
        }
    raise ValueError(
        f"central_body must be a registry name or dict, got {type(central_body)}"
    )
