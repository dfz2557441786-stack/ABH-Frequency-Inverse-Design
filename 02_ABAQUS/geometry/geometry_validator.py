"""Deterministic, pre-Abaqus ABH geometry validity checks."""
from __future__ import division

from math import cos, sin

import config
from damping_mapping import calculate_envelope, map_damping_dimensions


REQUIRED_PARAMETERS = tuple(config.FIXED_BASELINE_PARAMETERS.keys())


def validate_geometry(parameters):
    """Return ``(is_valid, failure_reason, derived_geometry)`` without repairs."""
    missing = [name for name in REQUIRED_PARAMETERS if name not in parameters]
    if missing:
        return False, "Missing required parameters: {0}".format(", ".join(missing)), {}
    try:
        numeric = dict(parameters)
        positive = ("Rou_2", "Rou_3", "Rou_4", "equal_t_length", "equal_t_width",
                    "platform_height", "platform_length", "platform_width", "cutoff_thickness",
                    "power_law", "hole_radius", "damp_elastic", "damp_thickness",
                    "damp_length", "damp_width")
        invalid = [name for name in positive if numeric[name] <= 0.0]
        if invalid:
            return False, "Non-positive dimension/property: {0}".format(", ".join(invalid)), {}
        for name, bounds in config.PARAMETER_BOUNDS.items():
            if name in numeric and not bounds[0] <= numeric[name] <= bounds[1]:
                return False, "{0} is outside configured range [{1}, {2}].".format(
                    name, bounds[0], bounds[1]), {}
        if not 0.0 < numeric["theta"] < 0.5 * 3.141592653589793:
            return False, "theta must be in (0, pi/2).", {}
        right_radical = numeric["Rou_4"] ** 2 - (numeric["Rou_3"] * sin(numeric["theta"])) ** 2
        up_radical = numeric["Rou_2"] ** 2 - (numeric["Rou_3"] * cos(numeric["theta"])) ** 2
        if right_radical <= 0.0:
            return False, "Rou_4**2-(Rou_3*sin(theta))**2 must be > 0.", {}
        if up_radical <= 0.0:
            return False, "Rou_2**2-(Rou_3*cos(theta))**2 must be > 0.", {}
        if not numeric["platform_height"] > numeric["cutoff_thickness"] > 0.0:
            return False, "Require platform_height > cutoff_thickness > 0.", {}
        if numeric["power_law"] <= 0.0:
            return False, "power_law must be > 0.", {}
        if 2.0 * numeric["hole_radius"] >= min(numeric["platform_length"], numeric["platform_width"]):
            return False, "hole_radius exceeds the allowed center-platform range.", {}
        envelope = calculate_envelope(numeric)
        mapping = map_damping_dimensions(numeric, config.GEOMETRY_EQUALITY_TOLERANCE)
        if numeric["damp_length"] >= envelope["L_all"] or numeric["damp_width"] >= envelope["W_all"]:
            return False, "Damping opening exceeds the ABH envelope.", mapping
        if mapping["damp_inward_length_x"] < config.MIN_DAMP_RING_WIDTH or mapping["damp_inward_length_y"] < config.MIN_DAMP_RING_WIDTH:
            return False, "Damping ring width is below the configured manufacturing limit.", mapping
        return True, "", mapping
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        return False, "Geometry evaluation failed: {0}".format(error), {}
