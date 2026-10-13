"""Fresh, isolated Abaqus job runner based on the verified source functions."""
from __future__ import print_function

import imp
import json
import os
import uuid

import config
from damping_mapping import map_damping_dimensions
from frequency_extractor import extract_first_positive_frequencies
from geometry_validator import validate_geometry
from mass_extractor import extract_attachment_masses


def _load_verified_module(module_name, path):
    if not os.path.isfile(path):
        raise RuntimeError("Verified source module is unavailable: {0}".format(path))
    return imp.load_source(module_name, path)


def _two_axis_source_transform(source_text):
    """Apply the sole geometry extension needed for independent ring opening axes.

    The imported verified function remains otherwise byte-for-byte sourced from
    the audited project.  This avoids duplicating or reimplementing material,
    thickness-field, mesh, tie, and boundary-condition code.
    """
    signature = "damp_inward_length=40e-3, \n"
    replacement_signature = (
        "damp_inward_length=40e-3, damp_inward_length_x=None, "
        "damp_inward_length_y=None, \n"
    )
    geometry = (
        "s.rectangle(point1=(-L_all/2+damp_inward_length, -W_all/2+damp_inward_length), "
        "point2=(L_all/2-damp_inward_length, W_all/2-damp_inward_length))"
    )
    replacement_geometry = (
        "s.rectangle(point1=(-L_all/2+damp_inward_length_x, -W_all/2+damp_inward_length_y), "
        "point2=(L_all/2-damp_inward_length_x, W_all/2-damp_inward_length_y))"
    )
    insertion = "    # 固定材料参数 (Fixed Material Parameters)\n"
    replacement_insertion = (
        "    # Independent x/y opening sizes preserve the legacy equal-inset "
        "case when either new value is omitted.\n"
        "    if damp_inward_length_x is None:\n"
        "        damp_inward_length_x = damp_inward_length\n"
        "    if damp_inward_length_y is None:\n"
        "        damp_inward_length_y = damp_inward_length\n\n" + insertion
    )
    anchors = (signature, geometry, insertion)
    if any(source_text.count(anchor) != 1 for anchor in anchors):
        raise RuntimeError("Verified Shell_create.py no longer matches the constrained two-axis adapter.")
    transformed = source_text.replace(signature, replacement_signature).replace(
        geometry, replacement_geometry
    ).replace(insertion, replacement_insertion)
    audit_anchor = "    L_all=abh_length_right+abh_length_left+platform_length+equal_t_length*2\n    W_all=abh_length_up+abh_length_down+platform_width+equal_t_width*2\n"
    audit_text = audit_anchor + "    from numeric_audit import assert_real_finite\n    for __name, __value in (('Rou_2', Rou_2), ('Rou_3', Rou_3), ('Rou_4', Rou_4), ('theta', theta), ('power_law', power_law), ('abh_length_right', abh_length_right), ('abh_length_up', abh_length_up), ('L_all', L_all), ('W_all', W_all)):\n        assert_real_finite(__name, __value, 'transformed Shell_create')\n"
    if transformed.count(audit_anchor) != 1:
        raise RuntimeError("Verified Shell_create.py no longer matches the numerical-audit anchor.")
    return transformed.replace(audit_anchor, audit_text)


def _load_two_axis_verified_shell_module():
    """Load the verified builder with only the damping-opening x/y extension."""
    with open(config.VERIFIED_SHELL_CREATE, "r", encoding="utf-8") as source_file:
        source_text = source_file.read()
    namespace = {"__name__": "verified_shell_create_two_axis", "__file__": config.VERIFIED_SHELL_CREATE}
    exec(compile(_two_axis_source_transform(source_text), config.VERIFIED_SHELL_CREATE, "exec"), namespace)
    return namespace


def _continuous_power_safe_source_transform(source_text):
    """Protect global off-face ExpressionField evaluation without altering faces."""
    transformed = _two_axis_source_transform(source_text)
    anchor = "    model.ExpressionField(name='ABH-1', localCsys=None, description='', expression=str(expression_abh_1))\n"
    insertion = (
        "    # On the assigned face each B is in [0,1]. max(B,0) is used only\n"
        "    # for Abaqus global off-face field evaluation before writeInput.\n"
        "    __b1 = 'max((({0}-X)/{0}),0.0)'.format(abh_length_right)\n"
        "    __b2 = 'max((({0}+{1}-Y)/{0}),0.0)'.format(abh_length_up, platform_width)\n"
        "    __b3 = 'max((({0}+{1}+X)/{1}),0.0)'.format(platform_length, abh_length_left)\n"
        "    __b4 = 'max((({0}+Y)/{0}),0.0)'.format(abh_length_down)\n"
        "    expression_abh_1 = '({0})*(({1})**({2}))+({3})'.format(platform_height-cutoff_thickness, __b1, power_law, cutoff_thickness)\n"
        "    expression_abh_2 = '({0})*(({1})**({2}))+({3})'.format(platform_height-cutoff_thickness, __b2, power_law, cutoff_thickness)\n"
        "    expression_abh_3 = '({0})*(({1})**({2}))+({3})'.format(platform_height-cutoff_thickness, __b3, power_law, cutoff_thickness)\n"
        "    expression_abh_4 = '({0})*(({1})**({2}))+({3})'.format(platform_height-cutoff_thickness, __b4, power_law, cutoff_thickness)\n"
    )
    if transformed.count(anchor) != 1:
        raise RuntimeError("Verified Shell_create.py no longer matches continuous-power field anchor.")
    return transformed.replace(anchor, insertion + anchor)


def _load_continuous_power_safe_shell_module():
    with open(config.VERIFIED_SHELL_CREATE, "r", encoding="utf-8") as source_file:
        source_text = source_file.read()
    namespace = {"__name__": "verified_shell_create_continuous_power_safe", "__file__": config.VERIFIED_SHELL_CREATE}
    exec(compile(_continuous_power_safe_source_transform(source_text), config.VERIFIED_SHELL_CREATE, "exec"), namespace)
    return namespace


def load_variable_power_shell_module(implementation):
    """The sole permitted variable-power thickness implementation.

    Dataset labels deliberately do not enter this decision.
    """
    if implementation != config.THICKNESS_FIELD_IMPLEMENTATION:
        raise RuntimeError("UNKNOWN_THICKNESS_FIELD_IMPLEMENTATION: {0}".format(implementation))
    return _load_continuous_power_safe_shell_module()


def _rect_patch_source_transform(source_text):
    """Reuse the verified builder while replacing only the legacy ring sketch.

    The patch is centred at the same translated envelope centre as the legacy
    DAMP part.  Its two layers remain partitioned at ``damp_thickness``.
    """
    signature = "damp_inward_length=40e-3, \n"
    outer = "    s.rectangle(point1=(-L_all/2, -W_all/2), point2=(L_all/2, W_all/2))\n"
    inner = "    s.rectangle(point1=(-L_all/2+damp_inward_length, -W_all/2+damp_inward_length), point2=(L_all/2-damp_inward_length, W_all/2-damp_inward_length))\n"
    old_points = (
        "    p.DatumPointByCoordinate(coords=(-L_all/2, W_all/2, damp_thickness))\n"
        "    p.DatumPointByCoordinate(coords=(-L_all/2, -W_all/2, damp_thickness))\n"
        "    p.DatumPointByCoordinate(coords=(L_all/2, W_all/2, damp_thickness))\n"
    )
    new_points = (
        "    p.DatumPointByCoordinate(coords=(-damp_length/2.0, damp_width/2.0, damp_thickness))\n"
        "    p.DatumPointByCoordinate(coords=(-damp_length/2.0, -damp_width/2.0, damp_thickness))\n"
        "    p.DatumPointByCoordinate(coords=(damp_length/2.0, damp_width/2.0, damp_thickness))\n"
    )
    anchors = (signature, outer, inner, old_points)
    if any(source_text.count(anchor) != 1 for anchor in anchors):
        raise RuntimeError("Verified Shell_create.py does not match the rect_patch adapter.")
    return source_text.replace(signature, "damp_inward_length=40e-3, damp_length=None, damp_width=None, \n").replace(
        outer + inner,
        "    if damp_length is None or damp_width is None or damp_length <= 0.0 or damp_width <= 0.0:\n"
        "        raise ValueError('rect_patch requires positive damp_length and damp_width')\n"
        "    s.rectangle(point1=(-damp_length/2.0, -damp_width/2.0), point2=(damp_length/2.0, damp_width/2.0))\n"
        "    patch_hole_x = abh_length_left + equal_t_length + platform_length/2.0 - L_all/2.0\n"
        "    patch_hole_y = abh_length_down + equal_t_width + platform_width/2.0 - W_all/2.0\n"
        "    if hole_radius > 0.0:\n"
        "        intersects_hole = abs(patch_hole_x) - hole_radius < damp_length/2.0 and abs(patch_hole_y) - hole_radius < damp_width/2.0\n"
        "        contains_hole = abs(patch_hole_x) + hole_radius < damp_length/2.0 and abs(patch_hole_y) + hole_radius < damp_width/2.0\n"
        "        if intersects_hole and not contains_hole:\n"
        "            raise ValueError('rect_patch partially intersects mounting hole')\n"
        "        if contains_hole:\n"
        "            s.CircleByCenterPerimeter(center=(patch_hole_x, patch_hole_y), point1=(patch_hole_x + hole_radius, patch_hole_y))\n"
    ).replace(old_points, new_points)


def _load_rect_patch_verified_shell_module():
    with open(config.VERIFIED_SHELL_CREATE, "r", encoding="utf-8") as source_file:
        source_text = source_file.read()
    namespace = {"__name__": "verified_shell_create_rect_patch", "__file__": config.VERIFIED_SHELL_CREATE}
    exec(compile(_rect_patch_source_transform(source_text), config.VERIFIED_SHELL_CREATE, "exec"), namespace)
    return namespace


def _build_fresh_model(parameters, run_directory, job_name, geometry_mode="legacy_ring"):
    if geometry_mode == "rect_patch" or getattr(config, "RECT_PATCH_DISABLED", False):
        if geometry_mode == "rect_patch":
            raise RuntimeError("rect_patch is disabled: the active physical topology is V2_DOMAIN_SAFE_FRAME_XY")
    valid, failure_reason, mapping = validate_geometry(parameters)
    if not valid:
        raise ValueError(failure_reason)
    from abaqus import Mdb, mdb
    from abaqusConstants import COMPLETED

    if geometry_mode == "rect_patch":
        shell_module = _load_rect_patch_verified_shell_module()
        shell_builder = shell_module["create_shell"]
    elif mapping["legacy_compatible"]:
        shell_module = _load_verified_module("verified_shell_create", config.VERIFIED_SHELL_CREATE)
        shell_builder = shell_module.create_shell
    else:
        shell_module = _load_two_axis_verified_shell_module()
        shell_builder = shell_module["create_shell"]
    boundary_module = _load_verified_module("verified_boundary_set", config.VERIFIED_BOUNDARY_SETUP)
    Mdb()
    original_directory = os.getcwd()
    try:
        os.chdir(run_directory)
        legacy_parameters = dict(parameters)
        legacy_parameters.update(config.FIXED_MATERIAL_PARAMETERS)
        if geometry_mode == "rect_patch":
            pass
        elif mapping["legacy_compatible"]:
            legacy_parameters["damp_inward_length"] = mapping["legacy_damp_inward_length"]
        else:
            legacy_parameters["damp_inward_length_x"] = mapping["damp_inward_length_x"]
            legacy_parameters["damp_inward_length_y"] = mapping["damp_inward_length_y"]
        if geometry_mode != "rect_patch":
            for name in ("damp_length", "damp_width"):
                legacy_parameters.pop(name, None)
        shell_builder(model_name="Model-1", **legacy_parameters)
        boundary_module.setup_Boundary(
            model_name="Model-1", abh_shell_instance_name="ABH_Shell-1", job_name=job_name
        )
        model = mdb.models["Model-1"]
        masses = extract_attachment_masses(model)
        job = mdb.jobs[job_name]
        job.submit()
        job.waitForCompletion()
        odb_path = os.path.join(run_directory, job_name + ".odb")
        # Some Abaqus/CAE noGUI sessions expose ``None`` for the in-memory Job
        # status even after a completed Standard analysis.  Do not accept that
        # value blindly: verify the fresh ODB, released lock, and Standard
        # completion marker produced by this very job instead.
        status_is_complete = job.status == COMPLETED
        msg_path = os.path.join(run_directory, job_name + ".msg")
        lock_path = os.path.join(run_directory, job_name + ".lck")
        completion_marker = False
        if os.path.isfile(msg_path):
            with open(msg_path, "r") as message_file:
                message_text = message_file.read().upper()
            completion_marker = (
                "THE ANALYSIS HAS COMPLETED SUCCESSFULLY" in message_text or
                "THE ANALYSIS HAS BEEN COMPLETED" in message_text
            )
        if not status_is_complete and not (completion_marker and not os.path.exists(lock_path)):
            raise RuntimeError("Abaqus job did not complete; reported status: {0}".format(job.status))
        if not os.path.isfile(odb_path):
            raise RuntimeError("Fresh ODB was not created: {0}".format(odb_path))
        return odb_path, masses, mapping
    finally:
        os.chdir(original_directory)


def run_fresh_frequency_job(parameters, run_directory, job_name="validation_frequency", geometry_mode="legacy_ring", mode_count=6):
    """Run exactly one isolated job and return its fresh ODB-derived results."""
    if not os.path.isdir(run_directory):
        os.makedirs(run_directory)
    stale_artifacts = [
        name for name in os.listdir(run_directory)
        if name.startswith(job_name + ".") or name == "input_parameters.json"
    ]
    if stale_artifacts:
        raise RuntimeError(
            "Refusing to reuse a run directory containing old job artifacts: {0}".format(
                ", ".join(sorted(stale_artifacts))
            )
        )
    valid, reason, mapping = validate_geometry(parameters)
    if not valid:
        raise ValueError(reason)
    manifest = {
        "job_name": job_name, "input_parameters": parameters,
        "derived_dimensions": mapping, "damp_geometry_mode": geometry_mode,
        "material_status": config.CALIBRATION_STATUS,
    }
    with open(os.path.join(run_directory, config.MANIFEST_FILENAME), "w") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
    with open(os.path.join(run_directory, "input_parameters.json"), "w") as stream:
        json.dump(parameters, stream, indent=2, sort_keys=True)
    odb_path, masses, mapping = _build_fresh_model(parameters, run_directory, job_name, geometry_mode)
    frequencies = extract_first_positive_frequencies(odb_path, count=mode_count)
    return {"odb_path": odb_path, "frequencies_hz": frequencies,
            "total_mass": masses["Total_Attachment_Mass"], "masses": masses,
            "damping_mapping": mapping, "job_status": "COMPLETED"}


def run_real_sample(parameters, run_directory, sample_id):
    """Future single-sample entry point; never launches a batch."""
    valid, reason, _ = validate_geometry(parameters)
    if not valid:
        return {"Model_Valid": False, "Failure_Reason": reason, "Job_Status": "NOT_SUBMITTED",
                "Result_Source": "Not_Submitted"}
    try:
        result = run_fresh_frequency_job(parameters, run_directory, "sample_{0}".format(sample_id))
        if result.get("total_mass") is None:
            raise RuntimeError("Total mass could not be extracted from this Abaqus model.")
        result.update({"Model_Valid": True, "Failure_Reason": "", "Result_Source": "Abaqus"})
        return result
    except Exception as error:
        return {"Model_Valid": False, "Failure_Reason": str(error), "Job_Status": "FAILED",
                "Result_Source": "Abaqus_Attempted"}


def create_isolated_sample_directory(sample_id):
    """Return a new directory so no sample can consume an old ODB or DAT."""
    sample_root = os.path.join(config.PROJECT_ROOT, "results", "samples")
    if not os.path.isdir(sample_root):
        os.makedirs(sample_root)
    directory = os.path.join(sample_root, "{0}_{1}".format(sample_id, uuid.uuid4().hex))
    os.makedirs(directory)
    return directory


def run_and_record_real_sample(sample_id, abh_id, parameters, dataset_manager, run_directory=None):
    """Future one-sample workflow: geometry check -> Abaqus -> fresh ODB -> CSV."""
    actual_parameters = dict(config.FIXED_BASELINE_PARAMETERS)
    actual_parameters.update(parameters)
    if run_directory is None:
        run_directory = create_isolated_sample_directory(sample_id)
    result = run_real_sample(actual_parameters, run_directory, sample_id)
    return dataset_manager.append_real_abaqus_result(sample_id, abh_id, actual_parameters, result)
