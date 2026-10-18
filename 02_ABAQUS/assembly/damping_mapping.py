"""Explicit bridge between new damping dimensions and legacy ring geometry.

The verified source uses one ``damp_inward_length`` for both in-plane axes.
This project exposes the more transparent rectangular central-opening
dimensions ``damp_length`` and ``damp_width``.  No value is silently altered:
the two implied offsets are returned separately.  The unmodified legacy
function can be called only when they are equal.
"""
from __future__ import division

from math import cos, sin, sqrt
from numeric_audit import assert_real_finite


def calculate_envelope(parameters):
    right = sqrt(parameters["Rou_4"] ** 2 - (parameters["Rou_3"] * sin(parameters["theta"])) ** 2)
    up = sqrt(parameters["Rou_2"] ** 2 - (parameters["Rou_3"] * cos(parameters["theta"])) ** 2)
    left = parameters["Rou_3"] * cos(parameters["theta"])
    down = parameters["Rou_3"] * sin(parameters["theta"])
    length = right + left + parameters["platform_length"] + 2.0 * parameters["equal_t_length"]
    width = up + down + parameters["platform_width"] + 2.0 * parameters["equal_t_width"]
    for name,value in (("Rou_2",parameters["Rou_2"]),("Rou_3",parameters["Rou_3"]),("Rou_4",parameters["Rou_4"]),("theta",parameters["theta"]),("abh_length_right",right),("abh_length_up",up),("abh_length_left",left),("abh_length_down",down),("L_all",length),("W_all",width)):
        assert_real_finite(name,value,"calculate_envelope")
    return {"abh_length_right": right, "abh_length_up": up, "abh_length_left": left,
            "abh_length_down": down, "L_all": length, "W_all": width}


def shell_outer_dimensions(parameters):
    """Return the real V2 ABH-shell XY outer-envelope dimensions in metres.

    ``L_all`` and ``W_all`` are calculated from the same geometric quantities
    used by the verified Shell_create model.  They describe the aluminium ABH
    shell only: the damping and constrained layers are deliberately excluded.
    """
    envelope = calculate_envelope(parameters)
    return {
        "ABH_Shell_Overall_Length_X_m": envelope["L_all"],
        "ABH_Shell_Overall_Width_Y_m": envelope["W_all"],
    }


def map_damping_dimensions(parameters, equality_tolerance=1.0e-10):
    envelope = calculate_envelope(parameters)
    inset_x = (envelope["L_all"] - parameters["damp_length"]) / 2.0
    inset_y = (envelope["W_all"] - parameters["damp_width"]) / 2.0
    return {
        "damp_length_definition": "central rectangular opening dimension parallel to L_all",
        "damp_width_definition": "central rectangular opening dimension parallel to W_all",
        "damp_inward_length_x": inset_x,
        "damp_inward_length_y": inset_y,
        "legacy_damp_inward_length": inset_x if abs(inset_x - inset_y) <= equality_tolerance else None,
        "legacy_compatible": abs(inset_x - inset_y) <= equality_tolerance,
        "envelope": envelope,
    }
