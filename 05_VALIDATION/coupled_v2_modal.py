#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build-only / modal runner for the locked Fig.4.4 Host and a current V2 RABH.

Run this module with ``abaqus cae noGUI=coupled_v2_modal.py -- --request <json>``.
It deliberately does not import or edit the external reference project.  It
reuses its Plate builder as a read-only Host reference and uses the current
project's V2 shell builder for the candidate RABH.

The first supported operation is ``build_only``.  It creates a fresh CAE
model and INP, then verifies the named SNEG--SPOS interface before any modal
job is permitted.  ``submit_modal`` is explicit and intended to be enabled
only after a successful build-only regression.
"""
from __future__ import print_function

import csv
import imp
import json
import os
import re
import sys
import traceback


# Abaqus/CAE ``noGUI`` executes scripts through ``exec`` and does not expose
# ``__file__``.  The launcher must therefore pass the project root explicitly
# instead of guessing from the current job directory on E:.
PROJECT_ROOT = os.environ.get("RABH_FAST_PROJECT_ROOT", "").strip()
if not PROJECT_ROOT:
    raise RuntimeError("RABH_FAST_PROJECT_ROOT_ENVIRONMENT_VARIABLE_REQUIRED")
PROJECT_ROOT = os.path.abspath(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import project_bootstrap
project_bootstrap.activate()

import config
from damping_policy import fixed_band_width_to_rx_ry
from abaqus_runner import load_variable_power_shell_module
from frequency_extractor import extract_positive_frequencies


def _float(value, name):
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError("CANDIDATE_NUMERIC_FIELD_INVALID:%s" % name)


def _read_candidate(candidate_csv, candidate_id):
    with open(candidate_csv, "r") as stream:
        # Abaqus' bundled Python may not support ``utf-8-sig``.  Normalise
        # the BOM explicitly so a CSV created by Excel/Python is read the
        # same way as it is by the normal project interpreter.
        rows = []
        for raw in csv.DictReader(stream):
            normalised = {}
            for key, value in raw.items():
                key_text = str(key)
                # Depending on the Abaqus interpreter's locale, a UTF-8 BOM
                # can appear as raw bytes, U+FEFF, or the three mojibake
                # characters "ï»¿".  Canonicalise all three forms.
                key_text = key_text.replace("\xef\xbb\xbf", "")
                key_text = key_text.replace("\ufeff", "")
                key_text = key_text.replace("ï»¿", "")
                # CP936 decoding in Abaqus 2024 can render the same three
                # UTF-8 BOM bytes as ``锘縞``.  Candidate_ID is the only
                # required first CSV field, so canonicalise its suffix rather
                # than guessing a system text codec.
                if key_text.endswith("andidate_ID"):
                    key_text = "Candidate_ID"
                normalised[key_text] = value
            rows.append(normalised)
    matches = [row for row in rows if str(row.get("Candidate_ID", "")).strip() == candidate_id]
    if len(matches) != 1:
        first_keys = sorted(rows[0].keys()) if rows else []
        raise ValueError("CANDIDATE_ID_NOT_UNIQUE_OR_MISSING:%s:row_count=%d:keys=%s" %
                         (candidate_id, len(rows), first_keys))
    row = matches[0]
    required = ("Rou_2", "Rou_3", "Rou_4", "cutoff_thickness",
                "equal_t_length", "equal_t_width", "platform_height",
                "power_law", "theta")
    return dict((name, _float(row.get(name), name)) for name in required)


def _candidate_parameters(candidate_csv, candidate_id):
    parameters = dict(config.FRAME_FIXED_PARAMETERS)
    parameters.update(_read_candidate(candidate_csv, candidate_id))
    parameters.update(config.FIXED_MATERIAL_PARAMETERS)
    band = fixed_band_width_to_rx_ry(parameters, 0.015, 0.015)
    parameters["damp_inward_x"] = band["damp_inward_x"]
    parameters["damp_inward_y"] = band["damp_inward_y"]
    return parameters, band


def _reference_plate_builder():
    plate_path = os.path.join(config.FIG44_HOST_REFERENCE_ROOT, "abaqus_module", "Plate_create.py")
    if not os.path.isfile(plate_path):
        raise RuntimeError("FIG44_REFERENCE_PLATE_BUILDER_MISSING:%s" % plate_path)
    return imp.load_source("fig44_reference_plate_builder", plate_path).create_plate


def _read_solver_text(path):
    """Read Abaqus diagnostics without depending on the Windows ACP."""
    if not path or not os.path.isfile(path):
        return ""
    with open(path, "r", encoding="latin-1", errors="replace") as stream:
        return stream.read()


def _connectivity_gate(dat_path, odb_path):
    """Return the submitted-model connection gate; never infer it from CAE.

    The final authority for a coupled calculation is Abaqus' own DAT tie
    diagnostic plus the fresh ODB spectrum.  ``RABH-Plate`` uses the current
    physical SNEG-to-SPOS tie; the pre-existing V2 DAMP-to-ABH tie remains
    separately visible so a floating damping assembly cannot be hidden.
    """
    text = _read_solver_text(dat_path)
    if not text:
        raise RuntimeError("COUPLED_MODAL_DAT_MISSING_OR_EMPTY:%s" % dat_path)
    rabh_labels = re.findall(
        r"SECONDARY\s+NODE\s+(\d+)\s+INSTANCE\s+ABH_SHELL-1\s+"
        r"WILL\s+NOT\s+BE\s+TIED", text, flags=re.I)
    damp_labels = re.findall(
        r"SECONDARY\s+NODE\s+(\d+)\s+INSTANCE\s+DAMP-1\s+"
        r"WILL\s+NOT\s+BE\s+TIED\s+TO\s+THE\s+MAIN\s+"
        r"SURFACE\s+ASSEMBLY_ABH_SHELL-1_", text, flags=re.I)
    all_untied = len(re.findall(r"WILL\s+NOT\s+BE\s+TIED", text, flags=re.I))
    other_untied = all_untied - len(rabh_labels) - len(damp_labels)
    frequencies = extract_positive_frequencies(odb_path, threshold=1.0e-12)
    near_zero = len([value for value in frequencies if value < 1.0])
    passed = (len(rabh_labels) == 0 and len(damp_labels) == 0 and
              near_zero == 0 and len(frequencies) > 0)
    return {
        "RABH_Plate_Untied_Node_Count": len(rabh_labels),
        "RABH_Plate_Untied_Node_Labels": rabh_labels,
        "DAMP_ABH_Untied_Node_Count": len(damp_labels),
        "DAMP_ABH_Untied_Node_Labels": damp_labels,
        "Other_Tie_Warning_Count": other_untied,
        "Near_Zero_Mode_Count_Below_1Hz": near_zero,
        "Positive_Frequency_Count": len(frequencies),
        "Coupled_Frequencies_0_300_Hz": [value for value in frequencies if value <= 300.0 + 1.0e-8],
        "Connectivity_Gate": "PASS" if passed else "FAIL",
    }


def _audit_modal_artifacts(result, dat_path, msg_path, odb_path):
    """Validate a completed modal run from solver evidence, not CAE status.

    Abaqus/CAE can expose ``job.status`` as ``None`` after a synchronous
    noGUI wait even though Standard has completed correctly.  The `.msg`
    completion records plus an unlocked fresh ODB are the authoritative
    artifacts.  This helper also makes a completed run safely auditable
    without submitting it again.
    """
    msg_text = _read_solver_text(msg_path).upper()
    evidence = {
        "Analysis_Completed_Message": "THE ANALYSIS HAS BEEN COMPLETED" in msg_text,
        "Lanczos_Completed_Message": "EIGENVALUE EXTRACTION USING THE LANCZOS EIGENSOLVER IS COMPLETE" in msg_text,
        "ODB_Exists": os.path.isfile(odb_path),
        "ODB_Lock_Exists": os.path.isfile(odb_path + ".lck"),
    }
    result["Solver_Evidence"] = evidence
    if (not evidence["Analysis_Completed_Message"] or
            not evidence["Lanczos_Completed_Message"] or
            not evidence["ODB_Exists"] or evidence["ODB_Lock_Exists"]):
        result["Status"] = "MODAL_NOT_COMPLETED"
        return result
    result.update(_connectivity_gate(dat_path, odb_path))
    result["Status"] = ("MODAL_COMPLETED_CONNECTIVITY_PASS" if
                        result["Connectivity_Gate"] == "PASS" else
                        "MODAL_COMPLETED_CONNECTIVITY_FAIL")
    return result


def _create_modal_boundary(model, parameters):
    """Create only the locked Host BC and zero-gap V2 modal interface."""
    from abaqusConstants import (COMPUTED, ENCASTRE, LANCZOS, MASS, OFF, ON)

    host = config.COUPLED_V2_HOST_REFERENCE
    assembly = model.rootAssembly
    plate = model.parts["Plate"]
    assembly.Instance(name="Plate-1", part=plate, dependent=ON)

    h = float(parameters["platform_height"])
    translation = (float(host["plate_translation_x"]),
                   float(host["plate_translation_y"]), -h)
    assembly.translate(instanceList=("Plate-1",), vector=translation)

    # Preserve the verified Fig.4.4 fixed-end rule from Boundary_Set.py.
    edges = assembly.instances["Plate-1"].edges.getSequenceFromMask(mask=("[#42 ]",),)
    region = assembly.Set(edges=edges, name="HOST_FIXED_END")
    model.EncastreBC(name="BC-1", createStepName="Initial", region=region,
                     localCsys=None)

    policy = config.CONTROLLED_COUPLED_MODAL_POLICY
    # ``simLinearDynamics=OFF`` is intentional.  The project-wide solver
    # policy is Lanczos with SIM selective recovery disabled; leaving this
    # keyword at Abaqus' default would emit ``SIM`` in the input deck and
    # silently diverge from the validated isolated/modal chain.
    model.FrequencyStep(name="Step-1", previous="Initial",
                        maxEigen=float(policy["upper_frequency_hz"]),
                        numEigen=int(policy["requested_mode_count"]),
                        eigensolver=LANCZOS, normalization=MASS,
                        simLinearDynamics=OFF)
    model.fieldOutputRequests["F-Output-1"].setValues(variables=("U",))

    # The current V2 shell's physical platform bottom is SNEG at z=-H; the
    # translated plate's physical top is SPOS at exactly the same z.  The
    # platform centre itself is a mounting hole, so the probe must be inside
    # solid platform material rather than at its geometric centre.
    cx = -0.20 * float(parameters["platform_length"])
    cy = float(parameters["platform_width"]) / 2.0
    abh_face = assembly.instances["ABH_Shell-1"].faces.findAt(((cx, cy, 0.0),))
    plate_face = assembly.instances["Plate-1"].faces.findAt(((cx, cy, -h),))
    abh_selected_face_count = len(abh_face)
    plate_selected_face_count = len(plate_face)
    assembly.Surface(name="RABH_PLATFORM_INTERFACE", side2Faces=abh_face)
    assembly.Surface(name="PLATE_RABH_INTERFACE", side1Faces=plate_face)
    model.Tie(name="RABH-Plate", main=assembly.surfaces["PLATE_RABH_INTERFACE"],
              secondary=assembly.surfaces["RABH_PLATFORM_INTERFACE"],
              positionToleranceMethod=COMPUTED, adjust=OFF,
              tieRotations=ON, thickness=ON)
    assembly.regenerate()
    return {
        "Installation_XY_m": translation[:2],
        "Plate_Interface_Z_m": -h,
        "ABH_Interface_Z_m": -h,
        "Physical_Interface_Gap_m": 0.0,
        "PositionToleranceMethod": "COMPUTED",
        "Tie_Adjust": "OFF",
        "Platform_Surface_Probe_XY_m": (cx, cy),
        # Shell side surfaces can have an empty ``nodes`` repository before
        # input processing even when the selected face is valid and meshed.
        # Face selection is the build-only gate; DAT diagnostics after a real
        # modal run remain the authoritative untied-node gate.
        "ABH_Selected_Face_Count": abh_selected_face_count,
        "Plate_Selected_Face_Count": plate_selected_face_count,
        "ABH_Interface_Node_Count_PreInput": len(assembly.surfaces["RABH_PLATFORM_INTERFACE"].nodes),
        "Plate_Interface_Node_Count_PreInput": len(assembly.surfaces["PLATE_RABH_INTERFACE"].nodes),
    }


def build_coupled_modal_request(request):
    """Build a fresh V2 model and optionally submit its 0--300 Hz modal job."""
    try:
        from abaqus import Mdb, mdb
        from abaqusConstants import COMPLETED, OFF, PERCENTAGE
    except ImportError:
        raise RuntimeError("ABAQUS_CAE_REQUIRED")

    candidate_id = str(request["candidate_id"]).strip()
    run_directory = os.path.abspath(request["run_directory"])
    job_name = str(request.get("job_name", "coupled_" + candidate_id)).strip()
    operation = str(request.get("operation", "build_only")).strip().lower()
    if operation not in ("build_only", "submit_modal", "audit_existing"):
        raise ValueError("UNKNOWN_COUPLED_OPERATION:%s" % operation)
    if operation == "audit_existing":
        job_name = str(request["job_name"]).strip()
        result = {
            "Candidate_ID": candidate_id,
            "Operation": operation,
            "Job_Name": job_name,
            "ODB_Path": os.path.join(run_directory, job_name + ".odb"),
            "DAT_Path": os.path.join(run_directory, job_name + ".dat"),
            "MSG_Path": os.path.join(run_directory, job_name + ".msg"),
        }
        return _audit_modal_artifacts(result, result["DAT_Path"],
                                      result["MSG_Path"], result["ODB_Path"])
    if os.path.exists(run_directory) and os.listdir(run_directory):
        raise RuntimeError("COUPLED_RUN_DIRECTORY_NOT_EMPTY:%s" % run_directory)
    if not os.path.isdir(run_directory):
        os.makedirs(run_directory)

    parameters, band = _candidate_parameters(request["candidate_csv"], candidate_id)
    original_cwd = os.getcwd()
    try:
        os.chdir(run_directory)
        Mdb()
        shell = load_variable_power_shell_module(config.THICKNESS_FIELD_IMPLEMENTATION)["create_shell"]
        args = dict(parameters)
        args["damp_inward_length_x"] = args["damp_inward_x"]
        args["damp_inward_length_y"] = args["damp_inward_y"]
        args.pop("damp_inward_x", None)
        args.pop("damp_inward_y", None)
        args.pop("damp_length", None)
        args.pop("damp_width", None)
        shell(model_name="Model-1", **args)
        _reference_plate_builder()(model_name="Model-1",
                                  plate_length=config.COUPLED_V2_HOST_REFERENCE["plate_length"],
                                  plate_width=config.COUPLED_V2_HOST_REFERENCE["plate_width"],
                                  plate_height=config.COUPLED_V2_HOST_REFERENCE["plate_height"])
        model = mdb.models["Model-1"]
        interface = _create_modal_boundary(model, parameters)
        if interface["ABH_Selected_Face_Count"] <= 0 or interface["Plate_Selected_Face_Count"] <= 0:
            raise RuntimeError("EMPTY_HOST_RABH_INTERFACE_SURFACE:%s" % interface)
        job = mdb.Job(name=job_name, model="Model-1", memory=90,
                      memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True,
                      numCpus=1, numDomains=1)
        job.writeInput(consistencyChecking=OFF)
        inp_path = os.path.join(run_directory, job_name + ".inp")
        if not os.path.isfile(inp_path) or os.path.getsize(inp_path) <= 0:
            raise RuntimeError("COUPLED_INPUT_WRITE_FAILED")
        result = {
            "Status": "BUILD_ONLY_PASS" if operation == "build_only" else "SUBMITTED",
            "Candidate_ID": candidate_id,
            "Operation": operation,
            "Job_Name": job_name,
            "INP_Path": inp_path,
            "V2_Geometry_Builder": config.GEOMETRY_BUILDER_VERSION,
            "Thickness_Field": config.THICKNESS_FIELD_IMPLEMENTATION,
            "Interface_Rule": config.V2_HOST_RABH_INTERFACE_RULE,
            "Damping_Configuration": band["Damping_Configuration"],
            "Damping_Inward_X_m": band["damp_inward_x"],
            "Damping_Inward_Y_m": band["damp_inward_y"],
            "Host_Reference": dict(config.COUPLED_V2_HOST_REFERENCE),
        }
        result.update(interface)
        if operation == "submit_modal":
            job.submit()
            job.waitForCompletion()
            result["Job_Status"] = str(job.status)
            odb_path = os.path.join(run_directory, job_name + ".odb")
            dat_path = os.path.join(run_directory, job_name + ".dat")
            msg_path = os.path.join(run_directory, job_name + ".msg")
            result["ODB_Path"] = odb_path
            result["DAT_Path"] = dat_path
            result["MSG_Path"] = msg_path
            _audit_modal_artifacts(result, dat_path, msg_path, odb_path)
        return result
    finally:
        os.chdir(original_cwd)


def main():
    if "--request" not in sys.argv:
        raise SystemExit("Usage: abaqus cae noGUI=coupled_v2_modal.py -- --request REQUEST.json")
    index = sys.argv.index("--request")
    request_path = os.path.abspath(sys.argv[index + 1])
    with open(request_path, "r") as stream:
        request = json.load(stream)
    result_path = os.path.abspath(request["result_path"])
    result = {"Status": "FAILED"}
    try:
        result = build_coupled_modal_request(request)
    except Exception as exc:
        result.update({"Error_Type": type(exc).__name__, "Error": str(exc),
                       "Traceback": traceback.format_exc()})
    with open(result_path, "w") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    print("COUPLED_V2_RESULT=" + result_path)
    if result.get("Status") == "FAILED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
