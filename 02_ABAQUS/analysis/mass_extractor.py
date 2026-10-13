"""Mass extraction for the attachment parts, never PLATE-1 only."""
from __future__ import division


def extract_attachment_masses(model):
    def part_mass(name):
        properties = model.parts[name].getMassProperties()
        mass = float(properties.get("mass", 0.0))
        if mass <= 0.0:
            raise RuntimeError("Non-positive mass for {0}".format(name))
        return mass
    abh_mass = part_mass("ABH_Shell")
    damp_mass = part_mass("DAMP")
    return {"ABH_Mass": abh_mass, "DAMP_Mass": damp_mass,
            "Total_Attachment_Mass": abh_mass + damp_mass}


def extract_odb_attachment_masses(odb_path):
    """Use this for shell analytical-thickness models after a fresh solve."""
    from odbAccess import openOdb
    odb = openOdb(path=odb_path, readOnly=False)
    try:
        assembly = odb.rootAssembly
        masses = {}
        for instance_name, label in (("ABH_SHELL-1", "ABH_Mass"), ("DAMP-1", "DAMP_Mass")):
            instance = assembly.instances[instance_name]
            set_name = "__mass_{0}__".format(instance_name)
            element_set = assembly.elementSets[set_name] if set_name in assembly.elementSets else assembly.ElementSet(
                name=set_name, elements=(instance.elements,))
            masses[label] = float(assembly.getMassProperties(elementSet=element_set)["mass"])
        masses["Total_Attachment_Mass"] = masses["ABH_Mass"] + masses["DAMP_Mass"]
        return masses
    finally:
        odb.close()
