"""Shared pre-submit damping-policy assertions for the controlled validation."""
from __future__ import print_function

import config


POLICY = config.CONTROLLED_VALIDATION_DAMPING_POLICY
TOLERANCE = 1.0e-12


def fixed_band_width_to_rx_ry(geometry_parameters, band_width_x_m,
                              band_width_y_m=None):
    """Return the geometry-specific ``rx``/``ry`` for a fixed physical band.

    The V2 damping frame uses dimensionless inward offsets::

        dx = rx * L_all / 2
        dy = ry * W_all / 2

    ``band_width_x_m`` is the physical edge-band width on the two ends
    parallel to ``L_all``.  ``band_width_y_m`` is the corresponding width on
    the two sides parallel to ``W_all``.  If ``band_width_y_m`` is omitted,
    a square, equal-width frame is requested for backward convenience.

    This helper is the sole conversion to use when a study specifies real
    edge-band widths (for example 15 mm / 15 mm) rather than a fixed D01--D04 ratio.
    It does *not* modify the supplied mapping or submit an Abaqus job.
    """
    try:
        width_x = float(band_width_x_m)
        width_y = float(band_width_x_m if band_width_y_m is None else band_width_y_m)
    except (TypeError, ValueError):
        raise ValueError("FIXED_DAMPING_BAND_WIDTH_M_MUST_BE_NUMERIC")
    if width_x <= 0.0 or width_y <= 0.0:
        raise ValueError("FIXED_DAMPING_BAND_WIDTH_M_MUST_BE_POSITIVE")

    # ``calculate_envelope`` requires both the variable geometry and the
    # project-wide fixed platform dimensions.
    from damping_mapping import calculate_envelope
    complete = dict(config.FRAME_FIXED_PARAMETERS)
    complete.update(dict(geometry_parameters))
    envelope = calculate_envelope(complete)
    length = float(envelope["L_all"])
    width_total = float(envelope["W_all"])
    if 2.0 * width_x >= length or 2.0 * width_y >= width_total:
        raise ValueError("FIXED_DAMPING_BAND_WIDTH_LEAVES_NO_INNER_OPENING")

    rx = 2.0 * width_x / length
    ry = 2.0 * width_y / width_total
    if not (0.0 < rx < 1.0 and 0.0 < ry < 1.0):
        raise ValueError("FIXED_DAMPING_BAND_WIDTH_PRODUCED_INVALID_RX_RY")
    return {
        "Damping_Configuration": "FIXED_BAND_X{0:g}_Y{1:g}MM".format(
            width_x * 1.0e3, width_y * 1.0e3),
        "Fixed_Damping_Band_Width_X_m": width_x,
        "Fixed_Damping_Band_Width_Y_m": width_y,
        "L_all": length,
        "W_all": width_total,
        "damp_inward_x": width_x,
        "damp_inward_y": width_y,
        "rx": rx,
        "ry": ry,
        "L_inner": length - 2.0 * width_x,
        "W_inner": width_total - 2.0 * width_y,
    }


def _structural_value(material):
    damping = getattr(material, "damping", None)
    if damping is None:
        raise RuntimeError("DAMPING_POLICY_MISSING_DAMPING")
    value = getattr(damping, "structural", None)
    if value is None:
        raise RuntimeError("DAMPING_POLICY_MISSING_STRUCTURAL_VALUE")
    return float(value)


def assert_model_damping(model, include_rabh):
    """Check the already-created materials without changing their definitions."""
    expected = {"Al-Plate": POLICY["HOST_STRUCTURAL_DAMPING"]}
    if include_rabh:
        expected.update({
            "Al": POLICY["RABH_AL_DAMPING"],
            "Constrained_Layer_Material": POLICY["CONSTRAINED_LAYER_DAMPING"],
            "Damp_Material": POLICY["DAMPING_LAYER_LOSS_FACTOR"],
        })
    actual = {}
    for name, target in expected.items():
        if name not in model.materials:
            raise RuntimeError("DAMPING_POLICY_MATERIAL_MISSING:%s" % name)
        value = _structural_value(model.materials[name])
        actual[name] = value
        if abs(value - target) > TOLERANCE:
            raise RuntimeError("DAMPING_POLICY_MISMATCH:%s:%s:%s" %
                               (name, value, target))
    return actual
