"""Robust read-only Abaqus DAT parser for frequency and modal-mass tables."""
from __future__ import print_function
import math
import os
import re


def _value(value):
    return float(value.replace("D", "E").replace("d", "e"))


def _letters(line):
    return re.sub(r"[^A-Z]", "", line.upper())


def parse_modal_dat(path):
    result = {"frequencies": {}, "generalized_mass": {}, "participation": {}, "effective_mass": {},
              "total_mass": float("nan"), "source": "DAT", "path": path, "errors": []}
    if not path or not os.path.isfile(path):
        result["errors"].append("DAT_NOT_FOUND")
        return result
    try:
        with open(path, "r", encoding="latin-1", errors="ignore") as handle:
            lines = handle.readlines()
    except Exception as exc:
        result["errors"].append("DAT_READ_ERROR:%s" % exc)
        return result
    joined = "".join(lines)
    mass = re.search(r"TOTAL\s+MASS\s+OF\s+MODEL\s+([+-]?[0-9.EDed+-]+)", joined, re.S)
    if mass:
        try: result["total_mass"] = _value(mass.group(1))
        except ValueError: result["errors"].append("TOTAL_MASS_PARSE_ERROR")
    section = None
    for line in lines:
        label = _letters(line)
        if "PARTICIPATIONFACTORS" in label:
            section = "participation"; continue
        if "EFFECTIVEMASS" in label:
            section = "effective"; continue
        if "MODENO" in label and "EIGENVALUE" in label and "FREQUENCY" in label:
            section = "frequency"; continue
        tokens = line.split()
        if section is None or len(tokens) < 2 or not tokens[0].isdigit():
            continue
        mode = int(tokens[0])
        try:
            if section == "frequency" and len(tokens) >= 6:
                # Abaqus DAT table: mode, eigenvalue, angular frequency,
                # cyclic frequency (Hz), generalized mass, modal damping.
                result["frequencies"][mode] = _value(tokens[3])
                result["generalized_mass"][mode] = _value(tokens[4])
            elif section == "participation" and len(tokens) >= 7:
                result["participation"][mode] = [_value(item) for item in tokens[1:7]]
            elif section == "effective" and len(tokens) >= 7:
                result["effective_mass"][mode] = [_value(item) for item in tokens[1:7]]
        except ValueError:
            result["errors"].append("NUMERIC_PARSE_ERROR_MODE_%d" % mode)
    return result


def positive_frequencies_0_500(parsed):
    return [(mode, frequency) for mode, frequency in sorted(parsed["frequencies"].items())
            if math.isfinite(frequency) and frequency > 1.0e-8 and frequency <= 500.0 + 1.0e-7]


def extract_host_rabh_mode_characters(odb_path, step_name, mode_indices,
                                      host_instance_token="PLATE-1"):
    """Read simple per-mode Host/RABH displacement character diagnostics.

    The metric is intentionally descriptive, not a modal-energy or damping
    measure: RMS nodal displacement across the Host instance is compared with
    the RMS displacement across the two RABH instances.  It is evaluated from
    the submitted coupled ODB only, after modal solution, and makes no mode
    order assumption.
    """
    try:
        from odbAccess import openOdb
    except ImportError:
        raise RuntimeError("ODB_ACCESS_REQUIRED_FOR_MODE_CHARACTER")
    requested = set(int(mode) for mode in mode_indices)
    result = dict((mode, {
        "Host_RMS_U": None,
        "RABH_RMS_U": None,
        "Host_RABH_Response_Ratio": None,
        "Mode_Character": "UNKNOWN",
        "Character_Status": "U_OUTPUT_NOT_AVAILABLE",
    }) for mode in requested)
    if not requested:
        return result
    odb = openOdb(path=odb_path, readOnly=True)
    try:
        if step_name not in odb.steps:
            raise RuntimeError("MODE_CHARACTER_STEP_NOT_FOUND:%s" % step_name)
        assembly = odb.rootAssembly
        instances = assembly.instances
        # Abaqus Repository objects implement ``keys`` and item access but
        # not the normal Python ``get`` / ``items`` mapping helpers.
        instance_names = list(instances.keys())
        host_name = next((name for name in instance_names
                          if name.upper() == host_instance_token.upper()), None)
        host = instances[host_name] if host_name is not None else None
        rabhs = [instances[name] for name in instance_names
                 if name.upper() in ("ABH_SHELL-1", "DAMP-1")]
        if host is None or len(rabhs) != 2:
            raise RuntimeError("MODE_CHARACTER_INSTANCE_SET_INVALID")
        for frame in odb.steps[step_name].frames:
            mode = getattr(frame, "mode", None)
            if mode not in requested or "U" not in frame.fieldOutputs:
                continue
            field = frame.fieldOutputs["U"]

            def rms_for(regions):
                squared, count = 0.0, 0
                for region in regions:
                    for value in field.getSubset(region=region).values:
                        data = value.data
                        squared += sum(component * component for component in data)
                        count += len(data)
                return math.sqrt(squared / count) if count else None

            host_rms = rms_for((host,))
            rabh_rms = rms_for(tuple(rabhs))
            entry = result[mode]
            entry["Host_RMS_U"] = host_rms
            entry["RABH_RMS_U"] = rabh_rms
            if host_rms is None or rabh_rms is None or rabh_rms <= 0.0:
                entry["Character_Status"] = "INSUFFICIENT_U_VALUES"
                continue
            ratio = host_rms / rabh_rms
            entry["Host_RABH_Response_Ratio"] = ratio
            if ratio >= 1.5:
                entry["Mode_Character"] = "HOST_DOMINATED"
            elif ratio <= (1.0 / 1.5):
                entry["Mode_Character"] = "RABH_DOMINATED"
            else:
                entry["Mode_Character"] = "MIXED"
            entry["Character_Status"] = "SUCCESS_SIMPLE_RMS_DIAGNOSTIC"
    finally:
        odb.close()
    return result
