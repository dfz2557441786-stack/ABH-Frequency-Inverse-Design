from math import sin, cos
from damping_mapping import calculate_envelope
import config
from numeric_audit import assert_real_finite
def validate_general(p):
    required=('Rou_2','Rou_3','Rou_4','theta','equal_t_length','equal_t_width','platform_height','cutoff_thickness','power_law','rx','ry')
    if any(k not in p for k in required): return False,'Missing general-design input',{}
    if any(p[key] <= 0 for key in ('Rou_2','Rou_3','Rou_4','equal_t_length','equal_t_width','platform_height','cutoff_thickness','power_law')):return False,'Non-positive ABH dimension',{}
    if p['Rou_4']**2-(p['Rou_3']*sin(p['theta']))**2<=0:return False,'Rou_4 radical invalid',{}
    if p['Rou_2']**2-(p['Rou_3']*cos(p['theta']))**2<=0:return False,'Rou_2 radical invalid',{}
    if not p['platform_height']>p['cutoff_thickness']>0 or p['power_law']<=0:return False,'Thickness/power-law invalid',{}
    if not 0<p['rx']<1 or not 0<p['ry']<1:return False,'rx/ry must be in (0,1)',{}
    full=dict(config.FRAME_FIXED_PARAMETERS,**p)
    if 2.0*full['hole_radius'] >= min(full['platform_length'],full['platform_width']):return False,'Hole does not fit fixed platform',{}
    try:
        env=calculate_envelope(full);dx=p['rx']*env['L_all']/2.;dy=p['ry']*env['W_all']/2.;li=env['L_all']-2*dx;wi=env['W_all']-2*dy
        for name,value in list(env.items())+[('dx',dx),('dy',dy),('L_inner',li),('W_inner',wi)]:assert_real_finite(name,value,'general_geometry_validator derived geometry')
    except ValueError as error:return False,str(error),{}
    if li<=0 or wi<=0:return False,'Inner rectangle invalid',{}
    area=env['L_all']*env['W_all']-li*wi;rho=config.FIXED_MATERIAL_PARAMETERS;mass=area*(rho['damp_density']*.003+rho['constrained_density']*.0001)
    # ABH mass changes with the nine geometry variables and is therefore taken
    # from the fresh ODB.  Only the damping-frame mass has an analytical form.
    return True,'',dict(env,dx=dx,dy=dy,L_inner=li,W_inner=wi,A_damp=area,
                         Footprint_Area=env['L_all']*env['W_all'],
                         DAMP_Mass_Estimate=mass)
