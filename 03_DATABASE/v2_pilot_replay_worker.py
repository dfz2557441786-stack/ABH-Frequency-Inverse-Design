"""One V2 replay of an old Joint Pilot coordinate, isolated from old outputs."""
from __future__ import print_function
import argparse, csv, json, math, os, sys, uuid
from datetime import datetime
if os.getcwd() not in sys.path: sys.path.insert(0,os.getcwd())
from project_bootstrap import activate
activate()
import config
import eleven_d_worker as base
from frame_runner import run_frame
from modal_frequency_tools import parse_modal_dat, positive_frequencies_0_500

ROOT=config.PROJECT_ROOT; R=config.RESULTS_DIR
CANDIDATES=os.path.join(R,"pilot20_v2_replay_candidates.csv")
RUNROOT=os.environ.get("ABH_V2_REPLAY_RUNROOT",os.path.join(R,"pilot20_v2_replay_runs"))
CANONICAL_RUNROOT=os.path.join(R,"pilot20_v2_replay_runs")
CHECKPOINT=os.path.join(R,"pilot20_v2_replay_checkpoint.csv")
LONG=os.path.join(R,"pilot20_v2_replay_all_modes_effective_mass_long.csv")
GENERAL=base.GENERAL


def _generated_design_parameters(request):
    """Return the complete V2 parameter vector for a generated candidate.

    Fixed-band candidates store only the nine geometry variables.  Their
    physical damping bands are deliberately fixed in metres and must be
    converted to V2's geometry-dependent dimensionless ``rx``/``ry`` here.
    """
    candidate_row = request["Candidate"]
    if candidate_row.get("Validation_Model") != "FIXED_BAND_15MM":
        return {key: float(candidate_row[key]) for key in GENERAL}, {}

    from damping_policy import fixed_band_width_to_rx_ry
    from target_retrieval import FIXED_BAND_9D
    geometry = {key: float(candidate_row[key]) for key in FIXED_BAND_9D}
    band_x = float(candidate_row["Damping_Inward_X_m"])
    band_y = float(candidate_row["Damping_Inward_Y_m"])
    damping = fixed_band_width_to_rx_ry(geometry, band_x, band_y)
    # The physical 15 mm bands, not these derived ratios, are the study input.
    geometry.update({"rx": damping["rx"], "ry": damping["ry"]})
    return geometry, damping

def read(path):
    if not os.path.isfile(path):return []
    with open(path,"r",newline="",encoding="utf-8-sig") as h:return list(csv.DictReader(h))
def write(path,rows):
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields:fields.append(key)
    with open(path,"w",newline="",encoding="utf-8") as h:w=csv.DictWriter(h,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(rows)
def truth(v):return v is True or str(v).lower()=="true"
def num(row,key):
    try:return float(row[key])
    except (ValueError,TypeError,KeyError):return float("nan")
def candidate(replay_id):
    found=[r for r in read(CANDIDATES) if r.get("New_Replay_ID")==replay_id]
    if not found:raise RuntimeError("V2_REPLAY_CANDIDATE_NOT_FOUND")
    row=found[0]
    if not truth(row.get("Current_V2_Geometry_Valid")):raise RuntimeError("V2_REPLAY_CANDIDATE_INVALID")
    if truth(row.get("Duplicate_With_Formal60")):raise RuntimeError("V2_REPLAY_DUPLICATE_WITH_FORMAL60")
    return row
def completed(replay_id):
    rows=[r for r in read(CHECKPOINT) if r.get("Sample_ID")==replay_id]
    return bool(rows and rows[0].get("Status")=="SUCCESS" and truth(rows[0].get("Result_Valid")))
def update(item):
    rows=[r for r in read(CHECKPOINT) if r.get("Sample_ID")!=item["Sample_ID"]];rows.append(item);write(CHECKPOINT,rows)
def save_rows(sample,result,parsed,canonical):
    modes=positive_frequencies_0_500(parsed);official=bool(modes and parsed["participation"] and parsed["effective_mass"]);tm=parsed["total_mass"];rows=[]
    for mode,freq in modes:
        pf=parsed["participation"].get(mode,[float("nan")]*6);em=parsed["effective_mass"].get(mode,[float("nan")]*6)
        row={"Sample_ID":sample["New_Replay_ID"],"Old_Pilot_ID":sample["Old_Pilot_ID"],"ABH_ID":sample["ABH_ID"],"Damping_ID":"JOINT_REPLAY","Source_Batch":"V2_REPLAY_PILOT","Mode_Index":mode,"Frequency_Hz":freq,
             "Participation_Factor_X":pf[0],"Participation_Factor_Y":pf[1],"Participation_Factor_Z":pf[2],"Generalized_Mass":parsed["generalized_mass"].get(mode,float("nan")),
             "Effective_Modal_Mass_X":em[0],"Effective_Modal_Mass_Y":em[1],"Effective_Modal_Mass_Z":em[2],"Effective_Mass_Ratio_X":em[0]/tm if tm>0 else float("nan"),"Effective_Mass_Ratio_Y":em[1]/tm if tm>0 else float("nan"),"Effective_Mass_Ratio_Z":em[2]/tm if tm>0 else float("nan"),
             "Total_Mass":result["masses"]["Total_Attachment_Mass"],"Total_Model_Mass":tm,"Footprint":num(sample,"Footprint_Area"),"Extraction_Status":"SUCCESS" if official else "OFFICIAL_MODAL_TABLES_UNAVAILABLE"}
        row.update(dict((k,float(sample[k])) for k in GENERAL));rows.append(row)
    old=[r for r in read(LONG) if r.get("Sample_ID")!=sample["New_Replay_ID"]];write(LONG,old+rows)
    return len(modes),official
def main(replay_id):
    sample=candidate(replay_id)
    if completed(replay_id):print(json.dumps({"Sample_ID":replay_id,"Status":"REUSED"}));return
    if not os.path.isdir(RUNROOT):os.makedirs(RUNROOT)
    folder=replay_id+"_"+datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex[:10];run=os.path.join(RUNROOT,folder);canonical=os.path.join(CANONICAL_RUNROOT,folder)
    try:
        general=dict((k,float(sample[k])) for k in GENERAL)
        result=run_frame(base._full(general)[0],run,"v2r_"+uuid.uuid4().hex[:10],split_xy=True,mode_count=999,field_implementation="DOMAIN_SAFE_CONTINUOUS_POWER_V2",dataset_source="V2_REPLAY_PILOT")
        odb=os.path.join(canonical,os.path.basename(result["ODB_Path"]));dat=os.path.splitext(odb)[0]+".dat";count,official=save_rows(sample,result,parse_modal_dat(dat),canonical)
        # The run remains physically source-independent; these fields are only
        # provenance for an intentionally near/repeated-coordinate replay.
        manifest_path=os.path.join(run,config.MANIFEST_FILENAME)
        if os.path.isfile(manifest_path):
            with open(manifest_path,"r") as h:manifest=json.load(h)
            manifest.update({"Sample_ID":replay_id,"Old_Pilot_ID":sample["Old_Pilot_ID"],"ABH_ID":sample["ABH_ID"],"Dataset_Source":"V2_REPLAY_PILOT","Nearest_Formal60_Distance":sample.get("Nearest_Formal60_Distance",""),"Duplicate_With_Formal60":sample.get("Duplicate_With_Formal60",""),"Thickness_Field_Implementation":"DOMAIN_SAFE_CONTINUOUS_POWER_V2","Geometry_Builder_Version":"VERIFIED_SHELL_CREATE_TWO_AXIS_DOMAIN_SAFE_V2"})
            with open(manifest_path,"w") as h:json.dump(manifest,h,indent=2,sort_keys=True)
        status={"Sample_ID":replay_id,"Old_Pilot_ID":sample["Old_Pilot_ID"],"ABH_ID":sample["ABH_ID"],"Status":"SUCCESS" if official else "FAILED","Result_Valid":official,"Failure_Reason":"" if official else "OFFICIAL_MODAL_TABLES_UNAVAILABLE","Mode_Count_0_500Hz":count,"Run_Directory":canonical,"ODB_Path":odb,"DAT_Path":dat,"Manifest_Path":os.path.join(canonical,config.MANIFEST_FILENAME),"Thickness_Field_Implementation":"DOMAIN_SAFE_CONTINUOUS_POWER_V2","Geometry_Builder_Version":"VERIFIED_SHELL_CREATE_TWO_AXIS_DOMAIN_SAFE_V2"};update(status);print(json.dumps(status))
    except Exception as exc:
        status={"Sample_ID":replay_id,"Old_Pilot_ID":sample.get("Old_Pilot_ID",""),"ABH_ID":sample.get("ABH_ID",""),"Status":"FAILED","Result_Valid":False,"Failure_Reason":str(exc),"Mode_Count_0_500Hz":0,"Thickness_Field_Implementation":"DOMAIN_SAFE_CONTINUOUS_POWER_V2","Geometry_Builder_Version":"VERIFIED_SHELL_CREATE_TWO_AXIS_DOMAIN_SAFE_V2"};update(status);print(json.dumps(status))
def extract_staged_modal(request):
    """ODB-only stage. Never builds a model or submits a solver."""
    import re
    from odbAccess import openOdb
    from mass_extractor import extract_odb_attachment_masses
    odb_path=request["ODB_Path"]; stem=os.path.splitext(odb_path)[0]
    if request.get("Submitted_Epoch") and os.path.getmtime(odb_path)<request["Submitted_Epoch"]:
        raise RuntimeError("ODB_PREDATES_THIS_SUBMISSION")
    # `interactive` command-line submissions can write completion evidence to
    # the controller console instead of creating a per-job .log file.  The
    # solver console is owned by this staged job and is an equivalent record;
    # never treat the optional .log filename alone as a solver failure.
    log_path=stem+".log"
    if not os.path.isfile(log_path):
        log_path=os.path.join(os.path.dirname(os.path.dirname(odb_path)),"solver_console.log")
    if not os.path.isfile(log_path):
        raise RuntimeError("SOLVER_LOG_EVIDENCE_MISSING")
    log=open(log_path,encoding="utf-8",errors="replace").read()
    msg=open(stem+".msg",encoding="utf-8",errors="replace").read()
    if (" COMPLETED" not in log or "THE ANALYSIS HAS BEEN COMPLETED" not in msg or
            "EIGENVALUE EXTRACTION USING THE LANCZOS EIGENSOLVER IS COMPLETE" not in msg):
        raise RuntimeError("SOLVER_COMPLETION_EVIDENCE_MISSING")
    odb=openOdb(odb_path,readOnly=True)
    try:
        modes=[(int(f.mode),float(f.frequency)) for f in odb.steps["Step-1"].frames if getattr(f,"mode",0)>0]
        if any(not math.isfinite(f) or f<1 for _,f in modes):raise RuntimeError("ABNORMAL_MODAL_FREQUENCY")
        modes=[(m,f) for m,f in modes if 0<f<=500.00001]
        if len(modes)<8:raise RuntimeError("INSUFFICIENT_POSITIVE_MODES")
        counts={name:{"nodes":len(i.nodes),"elements":len(i.elements)} for name,i in odb.rootAssembly.instances.items()}
    finally:odb.close()
    masses=extract_odb_attachment_masses(odb_path)
    if not all(math.isfinite(m) and m>0 for m in masses.values()):raise RuntimeError("INVALID_MASS")
    # Use the same accepted frame analytical solid-mass comparison.
    from frame_geometry import validate_frame
    general, _ = _generated_design_parameters(request)
    params=base._full(general)[0]
    valid,reason,geom=validate_frame(params)
    if not valid:raise RuntimeError(reason)
    materials=config.FIXED_MATERIAL_PARAMETERS
    expected=geom["A_damp"]*(materials["damp_density"]*0.003+materials["constrained_density"]*0.0001)
    mass_error=abs(masses["DAMP_Mass"]-expected)/expected
    if mass_error>0.02:raise RuntimeError("DAMP_MASS_CHECK_FAILED")
    def metric(pattern):
        found=re.findall(pattern,msg)
        return float(found[-1]) if found else None
    node_count = 0
    element_count = 0
    for values in counts.values():
        node_count += values["nodes"]
        element_count += values["elements"]
    result=dict(request.get("Build_Result",{}))
    result.update(Status="SUCCESS",Extraction_Schema=2,FE_SOLVER_SUCCESS=True,ODB_Path=odb_path,ODB_Valid=True,
        masses=masses,DAMP_Mass_Relative_Error=mass_error,Geometry_Valid=True,
        All_Positive_Frequencies_Hz=[f for _,f in modes],Raw_Mode_Indices=[m for m,_ in modes],
        Instance_Counts=counts,Node_Count=node_count,
        Element_Count=element_count,
        Solver_Wallclock_Seconds=metric(r'WALLCLOCK TIME \(SEC\)\s*=\s*([\d.E+\-]+)'),
        Solver_Memory_Peak_GB=metric(r'MEMORY PEAK \(GB\)\s*=\s*([\d.E+\-]+)'),
        Analysis_Warnings=metric(r'(\d+)\s+WARNING MESSAGES DURING ANALYSIS'),
        Input_Warnings=metric(r'(\d+)\s+WARNING MESSAGES DURING USER INPUT PROCESSING'),
        Error_Count=metric(r'(\d+)\s+ERROR MESSAGES'),Frequency_Upper_Limit_Hz=500.)
    if result["Error_Count"]!=0:raise RuntimeError("SOLVER_ERROR_COUNT_NOT_ZERO")
    return result


def run_generated_design(request_path):
    """One isolated generated geometry; reuse the exact accepted V2 replay chain."""
    import traceback
    from paths import FREQUENCY9D_RUNS_DIR
    request=json.load(open(request_path))
    run=os.path.realpath(request["Run_Directory"])
    allowed=os.path.realpath(FREQUENCY9D_RUNS_DIR)
    if not os.path.normcase(run).startswith(os.path.normcase(allowed+os.sep)):
        raise RuntimeError("GENERATED_RUN_OUTSIDE_AUTHORIZED_ROOT")
    result={"Status":"FAILED","Sample_ID":request["Candidate"]["Candidate_ID"]}
    try:
        if request.get("Stage")=="EXTRACT":
            result=extract_staged_modal(request)
            with open(request["Result_Path"],"w") as handle:json.dump(result,handle,indent=2)
            return result
        general, damping = _generated_design_parameters(request)
        result=run_frame(base._full(general)[0],run,request["Job_Name"],split_xy=True,
                         mode_count=999,field_implementation=config.THICKNESS_FIELD_IMPLEMENTATION,
                         dataset_source="FREQUENCY_9D_GENERATED_VALIDATION",
                         resource_policy=request.get("Resource_Policy"),
                         submit_job=request.get("Stage")!="BUILD")
        if request.get("Stage")=="BUILD":
            result["Status"]="INP_READY"
            with open(request["Result_Path"],"w") as handle:json.dump(result,handle,indent=2)
            return result
        from frequency_extractor import extract_positive_frequencies
        frequencies=extract_positive_frequencies(result["ODB_Path"])
        if len(frequencies)<8 or not all(math.isfinite(float(f)) and f>0 for f in frequencies):
            raise RuntimeError("INVALID_PRIMARY_FREQUENCY_EXPORT")
        if not all(math.isfinite(float(m)) and m>0 for m in result["masses"].values()):
            raise RuntimeError("INVALID_ATTACHMENT_MASS_EXPORT")
        from odbAccess import openOdb
        odb=openOdb(result["ODB_Path"],readOnly=True)
        try:
            raw=[float(f.frequency) for step in odb.steps.values() for f in step.frames if getattr(f,"mode",0)>0]
        finally:odb.close()
        if any(f<1.0 for f in raw):raise RuntimeError("UNEXPLAINED_NEAR_ZERO_OR_NEGATIVE_MODE")
        result.update({"Status":"SUCCESS","Sample_ID":request["Candidate"]["Candidate_ID"],
                       "All_Positive_Frequencies_Hz":[float(f) for f in frequencies],
                       "Independent_Validation":True,"Training_Use":False})
        if damping:
            result.update({
                "Damping_Configuration": damping["Damping_Configuration"],
                "Damping_Inward_X_m": damping["Fixed_Damping_Band_Width_X_m"],
                "Damping_Inward_Y_m": damping["Fixed_Damping_Band_Width_Y_m"],
                "rx_for_v2_builder": damping["rx"],
                "ry_for_v2_builder": damping["ry"],
            })
    except Exception as exc:
        result.update({"Status":"FAILED","Failure_Reason":str(exc),"Traceback":traceback.format_exc()})
    temporary=request["Result_Path"]+".pending"
    with open(temporary,"w") as handle:json.dump(result,handle,indent=2,default=lambda v:v.item())
    os.replace(temporary,request["Result_Path"])
    return result

if __name__=="__main__":
    if "--design-request" in sys.argv:
        run_generated_design(sys.argv[sys.argv.index("--design-request")+1]);sys.exit(0)
    parser=argparse.ArgumentParser();parser.add_argument("--sample",required=True);args=parser.parse_args(sys.argv[sys.argv.index("--sample"):] if "--sample" in sys.argv else None);main(args.sample)
