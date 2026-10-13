from __future__ import print_function
import json, os
from abaqus_runner import _load_verified_module, _two_axis_source_transform, _continuous_power_safe_source_transform, load_variable_power_shell_module
from frequency_extractor import extract_first_positive_frequencies, extract_positive_frequencies
from mass_extractor import extract_odb_attachment_masses
from frame_geometry import validate_frame
import config


def _load_two_axis_mesh_shell_module(mesh_level, thickness_implementation):
    if thickness_implementation != config.THICKNESS_FIELD_IMPLEMENTATION:
        raise RuntimeError("UNKNOWN_THICKNESS_FIELD_IMPLEMENTATION: {0}".format(thickness_implementation))
    level=config.MESH_LEVELS[mesh_level]
    with open(config.VERIFIED_SHELL_CREATE, "r", encoding="utf-8") as handle:
        source=_continuous_power_safe_source_transform(handle.read())
    old_shell="p.seedPart(size=0.0035, deviationFactor=0.1, minSizeFactor=0.1)"
    old_damp="p.seedPart(size=0.003, deviationFactor=0.1, minSizeFactor=0.1)"
    new_shell="p.seedPart(size={0!r}, deviationFactor=0.1, minSizeFactor=0.1)".format(level["shell_mesh_size"])
    new_damp="p.seedPart(size={0!r}, deviationFactor=0.1, minSizeFactor=0.1)".format(level["damp_mesh_size"])
    if source.count(old_shell)!=1 or source.count(old_damp)!=1:
        raise RuntimeError("Verified mesh anchors changed; refusing mesh override")
    namespace={"__name__":"verified_shell_create_two_axis_mesh", "__file__":config.VERIFIED_SHELL_CREATE}
    exec(compile(source.replace(old_shell,new_shell).replace(old_damp,new_damp), config.VERIFIED_SHELL_CREATE, "exec"), namespace)
    return namespace["create_shell"]


def run_frame(parameters, run_directory, job_name, split_xy=False, mesh_level=None, mode_count=8, material_overrides=None, field_implementation=None, dataset_source="", submit_job=True, resource_policy=None):
    field_implementation = field_implementation or config.THICKNESS_FIELD_IMPLEMENTATION
    ok, reason, derived = validate_frame(parameters)
    if not ok: raise ValueError(reason)
    if os.path.exists(run_directory) and os.listdir(run_directory):
        raise RuntimeError("Run directory is not empty")
    if not os.path.isdir(run_directory): os.makedirs(run_directory)
    with open(os.path.join(run_directory, config.MANIFEST_FILENAME), "w") as f:
        json.dump({"job_name": job_name, "input_parameters": parameters, "derived_dimensions": derived,
                   "mode": "frame_xy" if split_xy else "legacy_frame", "mesh_level": mesh_level or "verified_medium",
                   "calibration_status": config.CALIBRATION_STATUS,
                   "material_overrides": material_overrides or {}, "field_implementation": field_implementation,
                   "Thickness_Field_Implementation": field_implementation,
                   "Geometry_Builder_Version": config.GEOMETRY_BUILDER_VERSION,
                   "Dataset_Source": dataset_source, "Resource_Policy": resource_policy}, f, indent=2)
    from abaqus import Mdb, mdb
    from abaqusConstants import COMPLETED
    Mdb(); cwd=os.getcwd()
    try:
        os.chdir(run_directory)
        if split_xy:
            builder=_load_two_axis_mesh_shell_module(mesh_level, field_implementation) if mesh_level else load_variable_power_shell_module(field_implementation)["create_shell"]
        else:
            builder=_load_verified_module("legacy_shell", config.VERIFIED_SHELL_CREATE).create_shell
        args=dict(parameters); args.update(config.FIXED_MATERIAL_PARAMETERS)
        if material_overrides: args.update(material_overrides)
        if split_xy:
            args["damp_inward_length_x"]=parameters["damp_inward_x"]; args["damp_inward_length_y"]=parameters["damp_inward_y"]
        else: args["damp_inward_length"]=parameters["damp_inward_x"]
        for key in ("damp_inward_x", "damp_inward_y", "damp_length", "damp_width"): args.pop(key, None)
        builder(model_name="Model-1", **args)
        boundary=_load_verified_module("legacy_boundary", config.VERIFIED_BOUNDARY_SETUP)
        boundary.setup_Boundary(model_name="Model-1", abh_shell_instance_name="ABH_Shell-1", job_name=job_name)
        model=mdb.models["Model-1"]
        damp_part=model.parts["DAMP"]
        section_names=[item.sectionName for item in damp_part.sectionAssignments]
        damp_instance=model.rootAssembly.instances["DAMP-1"]
        slave_faces=damp_instance.surfaces["Surf-1"].faces
        try:
            tie_area=sum(face.getSize(printResults=False) for face in slave_faces)
        except Exception:
            tie_area=0.0
        damp_elastic=model.materials["Damp_Material"].elastic.table[0][0]
        constrained_elastic=model.materials["Constrained_Layer_Material"].elastic.table[0][0]
        checks={"Material_Valid": abs(damp_elastic-args["damp_elastic"])/args["damp_elastic"]<1.0e-12 and abs(constrained_elastic-args["constrained_elastic"])/args["constrained_elastic"]<1.0e-12,
                "Section_Valid": set(section_names)==set(["Damp","Constrained"]),
                "Tie_Valid": "Constraint-1" in model.constraints and len(slave_faces)>0 and tie_area>0.0,
                "BC_Valid": "BC-1" in model.boundaryConditions,
                "Mesh_Valid": len(damp_part.elements)>0 and len(model.parts["ABH_Shell"].elements)>0}
        if not all(checks.values()): raise RuntimeError("Model checks failed: {0}".format(checks))
        if not submit_job:
            inp=os.path.join(run_directory,job_name+".inp")
            if not os.path.isfile(inp) or os.path.getsize(inp)==0:
                raise RuntimeError("BUILD_ONLY_INP_MISSING")
            return dict(derived, Geometry_Valid=True, Thickness_Field_Valid=True,
                        INP_Path=inp, Frequency_Upper_Limit_Hz=float(model.steps["Step-1"].maxEigen),
                        Element_Count=sum(len(i.elements) for i in model.rootAssembly.instances.values()),
                        Node_Count=sum(len(i.nodes) for i in model.rootAssembly.instances.values()),
                        Thickness_Field_Implementation=field_implementation,
                        Geometry_Builder_Version=config.GEOMETRY_BUILDER_VERSION,
                        Dataset_Source=dataset_source, **checks)
        job=mdb.jobs[job_name]
        if resource_policy:
            from abaqusConstants import MEGA_BYTES, OFF
            cpus=int(resource_policy["cpus"]); memory_mb=int(resource_policy["memory_mb"])
            experimental=(resource_policy.get("experiment")=="SINGLE_R_MEMORY6GB")
            if cpus not in (1,2,4) or not 512<=memory_mb<=(6144 if experimental else 4096):
                raise RuntimeError("INVALID_EXPLICIT_RESOURCE_POLICY")
            if experimental and (cpus!=1 or memory_mb!=6144 or mesh_level is not None):
                raise RuntimeError("EXPERIMENT_REQUIRES_1CPU_6144MB_ORIGINAL_MESH")
            job.setValues(numCpus=cpus,numDomains=cpus,memory=memory_mb,
                          memoryUnits=MEGA_BYTES,getMemoryFromAnalysis=OFF)
            effective={"memory_mb":job.memory,"numCpus":job.numCpus,
                       "memoryUnits":str(job.memoryUnits),"getMemoryFromAnalysis":bool(job.getMemoryFromAnalysis),
                       "maxEigen":float(model.steps["Step-1"].maxEigen)}
            if (effective["memory_mb"]!=memory_mb or effective["numCpus"]!=cpus or
                    effective["memoryUnits"]!=str(MEGA_BYTES) or effective["getMemoryFromAnalysis"]):
                raise RuntimeError("JOB_RESOURCE_POLICY_NOT_APPLIED")
            with open(os.path.join(run_directory,"effective_job_resources.json"),"w") as handle:
                json.dump(effective,handle,indent=2)
            if experimental:
                import hashlib
                expected=resource_policy.get("expected_inp_sha256")
                actual=hashlib.sha256(open(job_name+".inp","rb").read()).hexdigest()
                if not expected or actual!=expected or effective["maxEigen"]!=500.0:
                    raise RuntimeError("EXPERIMENT_PHYSICAL_INPUT_CHANGED")
        job.submit(); job.waitForCompletion()
        odb=os.path.join(run_directory, job_name+".odb"); msg=os.path.join(run_directory, job_name+".msg")
        # Abaqus 2024 may write UTF-8 bytes to MSG files even on a Chinese
        # Windows installation whose default text codec is GBK.  Solver
        # completion must never be misclassified because report text cannot
        # be decoded by the host default codec.
        text=(open(msg, "r", encoding="utf-8", errors="replace").read().upper()
              if os.path.exists(msg) else "")
        if resource_policy and resource_policy.get("experiment")=="SINGLE_R_MEMORY6GB":
            if (job.status!=COMPLETED or "THE ANALYSIS HAS BEEN COMPLETED" not in text or
                    "EIGENVALUE EXTRACTION USING THE LANCZOS EIGENSOLVER IS COMPLETE" not in text):
                raise RuntimeError("EXPERIMENT_JOB_NOT_COMPLETED_SUCCESSFULLY")
        if not os.path.isfile(odb) or os.path.exists(os.path.join(run_directory, job_name+".lck")) or "THE ANALYSIS HAS" not in text:
            raise RuntimeError("Abaqus job did not complete")
        masses=extract_odb_attachment_masses(odb)
        # Frequency_1..8 are the primary Pilot outputs.  Modes 9..12 are a
        # tracking diagnostic and can be truncated by the verified legacy
        # FrequencyStep's 500 Hz maxEigen ceiling without invalidating an
        # otherwise stable, positive first-eight result.
        all_positive=extract_positive_frequencies(odb)
        primary_required=min(8, mode_count)
        if len(all_positive)<primary_required:
            raise RuntimeError("Expected at least {0} positive frequencies, found {1}.".format(primary_required, len(all_positive)))
        values=all_positive[:mode_count]
        tracking_complete=len(all_positive)>=mode_count
        rho_d=args["damp_density"]; rho_c=args["constrained_density"]
        estimate=rho_d*derived["A_damp"]*0.003+rho_c*derived["A_damp"]*0.0001
        mass_error=abs(masses["DAMP_Mass"]-estimate)/estimate
        if mass_error>0.02: raise RuntimeError("DAMP mass analytical mismatch: {0}".format(mass_error))
        element_count=sum(len(model.rootAssembly.instances[name].elements) for name in ("ABH_Shell-1", "DAMP-1"))
        node_count=sum(len(model.rootAssembly.instances[name].nodes) for name in ("ABH_Shell-1", "DAMP-1"))
        return dict(derived, frequencies_hz=values, masses=masses, ODB_Path=odb, Tie_Area=tie_area, Element_Count=element_count, Node_Count=node_count, DAMP_Mass_Estimate=estimate,
                    DAMP_Mass_Relative_Error=mass_error, Geometry_Valid=True, ODB_Valid=True,
                    Primary_Frequency_Valid=True, Tracking_Frequency_Complete=tracking_complete,
                    Positive_Frequency_Count=len(all_positive),
                    Frequency_Upper_Limit_Hz=float(model.steps["Step-1"].maxEigen),
                    Resource_Policy=resource_policy,
                    Job_Status=str(job.status),
                    Frequency_Warning="" if tracking_complete else "ONLY_%d_POSITIVE_MODES" % len(all_positive),
                    Thickness_Field_Implementation=field_implementation,
                    Geometry_Builder_Version=config.GEOMETRY_BUILDER_VERSION, Dataset_Source=dataset_source, **checks)
    finally: os.chdir(cwd)
