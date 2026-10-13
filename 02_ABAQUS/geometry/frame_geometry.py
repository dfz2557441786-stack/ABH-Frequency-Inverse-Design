from __future__ import division
from damping_mapping import calculate_envelope


def validate_frame(parameters):
    envelope = calculate_envelope(parameters)
    dx, dy = parameters["damp_inward_x"], parameters["damp_inward_y"]
    if parameters["damp_thickness"] != 3.0e-3:
        return False, "damp_thickness must remain 3 mm", {}
    if not (0.0 < 2.0 * dx < envelope["L_all"] and 0.0 < 2.0 * dy < envelope["W_all"]):
        return False, "Frame inner rectangle has invalid dimensions", {}
    inner_l, inner_w = envelope["L_all"] - 2.0 * dx, envelope["W_all"] - 2.0 * dy
    rx, ry = 2.0 * dx / envelope["L_all"], 2.0 * dy / envelope["W_all"]
    area_direct=envelope["L_all"] * envelope["W_all"] - inner_l * inner_w
    area_dimensionless=envelope["L_all"] * envelope["W_all"] * (rx + ry - rx * ry)
    if abs(area_direct-area_dimensionless)>1.0e-12:
        return False, "Dimensionless frame-area identity failed", {}
    return True, "", dict(envelope, Outer_Length=envelope["L_all"], Outer_Width=envelope["W_all"],
                            Inner_Length=inner_l, Inner_Width=inner_w,
                            rx=rx, ry=ry, A_damp=area_direct, A_damp_dimensionless=area_dimensionless,
                            damp_inward_x=dx, damp_inward_y=dy)
