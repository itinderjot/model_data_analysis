import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
import datetime
import sys
import glob
import os
import hdf5plugin
import h5py
# have to install h5netcdf for xarray to read .h5 files
import pandas as pd
import xarray as xr
import cartopy.crs as crs

import matplotlib.axis as maxis
import matplotlib.path as mpath
import matplotlib.spines as mspines
import matplotlib.transforms as transforms
import metpy.calc as mpcalc
import sharppy.sharptab.interp as interp
import sharppy.sharptab.params as params
import sharppy.sharptab.profile as profile
import sharppy.sharptab.thermo as thermo
import sharppy.sharptab.utils as utils
import sharppy.sharptab.winds as winds
from matplotlib.axes import Axes
from matplotlib.projections import register_projection
from metpy.cbook import get_test_data
from metpy.plots import Hodograph, SkewT
from metpy.units import units
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#matplotlib.rcParams['axes.facecolor'] = 'white'

from cartopy.geodesic import Geodesic
import shapely.geometry as sgeom

from skimage.draw import ellipse
from skimage.measure import label, regionprops, regionprops_table
from skimage.transform import rotate
from skimage.morphology import closing, square
from skimage.segmentation import clear_border

import csv
import random
from multiprocessing import Pool, cpu_count
import os
import time


def create_indices(p, h, tc, td, u, v, WRITE_SRH_OBS=False, WRITE_ADVANCED_INDICES=False, L_OR_R=None, U_STORM_OBS=None, V_STORM_OBS=None):
    '''
    Parameters:
    Returns:
    ''' 
    #####    METPY CALCULATIONS #####
    rhum     = mpcalc.relative_humidity_from_dewpoint(tc, td).to('percent')
    mr       = mpcalc.mixing_ratio_from_relative_humidity(pressure=p, temperature=tc, relative_humidity=rhum)
    tv       = mpcalc.virtual_temperature(temperature=tc, mixing_ratio=mr).to('degC')

    lcl_pressure, lcl_temperature = mpcalc.lcl(p[0], tv[0], td[0])  # calc LCL
    lfc                           = mpcalc.lfc(p, tc, td, parcel_temperature_profile=None, which='top')
    #print('LFC:',lfc)
    wspeed   = mpcalc.wind_speed(u, v)
    wspeed   = wspeed.to('knots')
    wind_dir = mpcalc.wind_direction(u, v)
    
    pw = mpcalc.precipitable_water(p, td)
    # TCWV, low-level RH, mid-level RH, upper-level RH, mid-level lapse rate, freezing level height, 
    # CAPE, CIN, low-level shear, mid-level shear, upper-level shear, and 2m temperature. 
    #In all cases low-level is the average from the surface to 800 hPa, mid-level from 400 to 800 hPa, 
    # and upper-level from 100 to 400 hPa
    #print('h',h)
    #print('min h',min(h.magnitude)*units('m'))
    #print('max h',max(h.magnitude)*units('m'))
    #print('input u in create_indices',u)
    #print('input v in create_indices',v)
    u_0_1, v_0_1 = mpcalc.bulk_shear(p, u, v, height=h, depth=1000.0*units('m'))
    u_0_3, v_0_3 = mpcalc.bulk_shear(p, u, v, height=h, depth=3000.0*units('m'))
    u_0_6, v_0_6 = mpcalc.bulk_shear(p, u, v, height=h, depth=6000.0*units('m'))
    _1km_shear   = mpcalc.wind_speed(u_0_1, v_0_1).magnitude
    _3km_shear   = mpcalc.wind_speed(u_0_3, v_0_3).magnitude
    _6km_shear   = mpcalc.wind_speed(u_0_6, v_0_6).magnitude
    
    # calculate mean u and v over various heights using MetPy
    
    ################# 0 - 8 km #######################
    p_0_8,u_0_8 = mpcalc.get_layer(p, u, height=h, bottom=None, depth=8000.0 *units('m'), interpolate=True)
    p_0_8,v_0_8 = mpcalc.get_layer(p, v, height=h, bottom=None, depth=8000.0 *units('m'), interpolate=True)

    mean_u_0_8 = np.nanmean(u_0_8.magnitude)
    mean_v_0_8 = np.nanmean(v_0_8.magnitude)
    
    ################# 0 - 6 km #######################
    p_0_6,u_0_6 = mpcalc.get_layer(p, u, height=h, bottom=None, depth=6000.0 *units('m'), interpolate=True)
    p_0_6,v_0_6 = mpcalc.get_layer(p, v, height=h, bottom=None, depth=6000.0 *units('m'), interpolate=True)

    mean_u_0_6 = np.nanmean(u_0_6.magnitude)
    mean_v_0_6 = np.nanmean(v_0_6.magnitude)
    
    ################# 0 - 3 km #######################
    p_0_3,u_0_3 = mpcalc.get_layer(p, u, height=h, bottom=None, depth=3000.0 *units('m'), interpolate=True)
    p_0_3,v_0_3 = mpcalc.get_layer(p, v, height=h, bottom=None, depth=3000.0 *units('m'), interpolate=True)

    mean_u_0_3 = np.nanmean(u_0_3.magnitude)
    mean_v_0_3 = np.nanmean(v_0_3.magnitude)
    
    ################# 0 - 1 km #######################
    p_0_1,u_0_1 = mpcalc.get_layer(p, u, height=h, bottom=None, depth=1000.0 *units('m'), interpolate=True)
    p_0_1,v_0_1 = mpcalc.get_layer(p, v, height=h, bottom=None, depth=1000.0 *units('m'), interpolate=True)

    mean_u_0_1 = np.nanmean(u_0_1.magnitude)
    mean_v_0_1 = np.nanmean(v_0_1.magnitude)
    ################################################################
    p_sfc_800,rh_sfc_800 = mpcalc.get_layer(p, rhum, height=h, bottom=None, depth=200.0 *units('hPa'), interpolate=True)
    mean_rh_sfc_800      = np.nanmean(rh_sfc_800.magnitude)
    
    p_800_400,rh_800_400 = mpcalc.get_layer(p, rhum, height=h, bottom=800.0*units('hPa'),depth=400.0*units('hPa'), interpolate=True)
    mean_rh_800_400      = np.nanmean(rh_800_400.magnitude)
    
    p_400_100,rh_400_100 = mpcalc.get_layer(p, rhum, height=h, bottom=400.0*units('hPa'),depth=300.0*units('hPa'), interpolate=True)
    mean_rh_400_100      = np.nanmean(rh_400_100.magnitude)
    ################################################################


    #     print('Metpy shear')
    #     print('u,v,speed 0-1 km shear = ', u_0_1, v_0_1, _1km_shear)
    #     print('u,v,speed 0-3 km shear = ', u_0_3, v_0_3, _3km_shear)
    #     print('u,v,speed 0-6 km shear = ', u_0_6, v_0_6, _6km_shear)
    #     print(' ')

    ################################ SHARPPY ################################

    prof = profile.create_profile(profile='default', pres=np.array(p), hght=np.array(h), tmpc=np.array(tc),
                                  dwpc=np.array(td), wspd=np.array(wspeed), wdir=np.array(wind_dir), missing=-9999, strictQC=True)
#     print('LFC IS ', lfc)
#     msl_hght = prof.hght[prof.sfc]  # Grab the surface height value
#     print("SURFACE HEIGHT (m MSL):", msl_hght)
#     agl_hght = interp.to_agl(prof, msl_hght)  # Converts to AGL
#     print("SURFACE HEIGHT (m AGL):", agl_hght)
#     msl_hght = interp.to_msl(prof, agl_hght)  # Converts to MSL
#     print("SURFACE HEIGHT (m MSL):", msl_hght)

    sfcpcl = params.parcelx(prof, flag=1)  # Surface Parcel
    fcstpcl = params.parcelx(prof, flag=2)  # Forecast Parcel
    mupcl = params.parcelx(prof, flag=3)  # Most-Unstable Parcel
    mlpcl = params.parcelx(prof, flag=4)  # 100 mb Mean Layer Parcel

    sfc = prof.pres[prof.sfc]
    p3km = interp.pres(prof, interp.to_msl(prof, 3000.))
    p6km = interp.pres(prof, interp.to_msl(prof, 6000.))
    p1km = interp.pres(prof, interp.to_msl(prof, 1000.))
    mean_3km = winds.mean_wind(prof, pbot=sfc, ptop=p3km)
    sfc_6km_shear = winds.wind_shear(prof, pbot=sfc, ptop=p6km)
    sfc_3km_shear = winds.wind_shear(prof, pbot=sfc, ptop=p3km)
    sfc_1km_shear = winds.wind_shear(prof, pbot=sfc, ptop=p1km)
    #print('SharpPy shear')
    #print('sfc_6km_shear ',sfc_6km_shear)
    #print('sfc_3km_shear ',sfc_3km_shear)
    #print('sfc_1km_shear ',sfc_1km_shear)

    srwind = params.bunkers_storm_motion(prof)
    srh3km_r = winds.helicity(prof, 0, 3000., stu=srwind[0], stv=srwind[1])
    srh1km_r = winds.helicity(prof, 0, 1000., stu=srwind[0], stv=srwind[1])
    srh3km_l = winds.helicity(prof, 0, 3000., stu=srwind[2], stv=srwind[3])
    srh1km_l = winds.helicity(prof, 0, 1000., stu=srwind[2], stv=srwind[3])
    
    
    #bunkers_sharppy2 = winds.non_parcel_bunkers_motion(prof)
    #bunkers_sharppy3 = winds.non_parcel_bunkers_motion_experimental(prof)
    rm, lm, wm = mpcalc.bunkers_storm_motion(pressure=p, u=u, v=v, height=h)
    
    #print(' SharpPy Bunkers #1 : ', srwind)
    #print("Bunker's Storm Motion (right-mover):", srwind[0], srwind[1])
    #print("Bunker's Storm Motion (left-mover) :", srwind[2], srwind[3])
    #print(' ')
    #print(' SharpPy Bunkers #2 (non-parcel) : ', bunkers_sharppy2)
    #print("Bunker's Storm Motion (right-mover):", bunkers_sharppy2[0], bunkers_sharppy2[1])
    #print("Bunker's Storm Motion (left-mover) :", bunkers_sharppy2[2], bunkers_sharppy2[3])
    #print(' ')
    #print(' SharpPy Bunkers #3 (non-parcel experimental) : ', bunkers_sharppy3)
    #print("Bunker's Storm Motion (right-mover):", bunkers_sharppy3[0], bunkers_sharppy3[1])
    #print("Bunker's Storm Motion (left-mover) :", bunkers_sharppy3[2], bunkers_sharppy3[3])
    #print(' ')
    #print(' MetPy Bunkers: ', rm, lm)
    #print("Bunker's Storm Motion (right-mover):", rm)
    #print("Bunker's Storm Motion (left-mover) :", lm)

    #stp_fixed = params.stp_fixed(sfcpcl.bplus, sfcpcl.lclhght, srh1km[0], utils.comp2vec(sfc_6km_shear[0], sfc_6km_shear[1])[1])
    ship = params.ship(prof)
    eff_inflow = params.effective_inflow_layer(prof)
    #print('effective inflow from SharpPy:',eff_inflow)
    ebot_hght = interp.to_agl(prof, interp.hght(prof, eff_inflow[0]))
    etop_hght = interp.to_agl(prof, interp.hght(prof, eff_inflow[1]))
    eff_inflow_depth = etop_hght - ebot_hght # calculate EIL depth from SharpPy
    #print('EIL depth: ',eff_inflow_depth*units('m'))
    ################################################################
    ################################################################
    # calculate mean u and v over the inflow depth using MetPy
#     p_eil,u_eil = mpcalc.get_layer(p, u, height=h, bottom=None, depth=eff_inflow_depth*units('m'), interpolate=True)
#     p_eil,v_eil = mpcalc.get_layer(p, v, height=h, bottom=None, depth=eff_inflow_depth*units('m'), interpolate=True)
#     eil_mean_u = np.nanmean(u_eil.magnitude)
#     eil_mean_v = np.nanmean(v_eil.magnitude)
#     #print('mean u, v over EIL:',np.nanmean(u_eil),np.nanmean(v_eil))
    ################################################################
    ################################################################
    effective_srh_r = winds.helicity(prof, ebot_hght, etop_hght, stu=srwind[0], stv=srwind[1])
    effective_srh_l = winds.helicity(prof, ebot_hght, etop_hght, stu=srwind[2], stv=srwind[3])
    ebwd = winds.wind_shear(prof, pbot=eff_inflow[0], ptop=eff_inflow[1])
    ebwspd = utils.mag(ebwd[0], ebwd[1])

    if WRITE_SRH_OBS:
        
        U_STORM_OBS = U_STORM_OBS.to('knots')
        V_STORM_OBS = V_STORM_OBS.to('knots')
 
        #sharppy_srh1km_tot, sharppy_srh1km_pos, sharppy_srh1km_neg  = winds.helicity(
        #    prof, 0, 1000., stu=U_STORM_OBS.magnitude, stv=V_STORM_OBS.magnitude)
        #print('SharpPy 0-1 km observed SRH (pos, neg, total) ' , sharppy_srh1km_pos, sharppy_srh1km_neg, sharppy_srh1km_tot)

        #sharppy_srh3km_tot, sharppy_srh3km_pos, sharppy_srh3km_neg  = winds.helicity(
        #    prof, 0, 3000., stu=U_STORM_OBS.magnitude, stv=V_STORM_OBS.magnitude)
        #print('SharpPy 0-3 km observed SRH (pos, neg, total) ' , sharppy_srh3km_pos, sharppy_srh3km_neg, sharppy_srh3km_tot)

        metpy_srh1km_pos, metpy_srh1km_neg, metpy_srh1km_tot = mpcalc.storm_relative_helicity(h,u,v,1000.0*units('m'),bottom=0.0*units('m'), storm_u=U_STORM_OBS, storm_v=V_STORM_OBS)
        #print('MetPy 0-1 km observed SRH (pos, neg, total)' , metpy_srh1km_pos, metpy_srh1km_neg, metpy_srh1km_tot)

        metpy_srh3km_pos, metpy_srh3km_neg, metpy_srh3km_tot = mpcalc.storm_relative_helicity(h,u,v,3000.0*units('m'),bottom=0.0*units('m'), storm_u=U_STORM_OBS, storm_v=V_STORM_OBS)
        #print('MetPy 0-3 km observed SRH (pos, neg, total)' , metpy_srh3km_pos, metpy_srh3km_neg, metpy_srh3km_tot)

        # Storm-relative flow over the EIL
        sr_flow = np.sqrt( (eil_mean_u - U_STORM_OBS.magnitude)**2 + (eil_mean_v - V_STORM_OBS.magnitude)**2 )

    #scp = params.scp(mupcl.bplus, effective_srh[0], ebwspd)
    #stp_cin = params.stp_cin(mlpcl.bplus, effective_srh[0], ebwspd, mlpcl.lclhght, mlpcl.bminus)
    indices = {'LFC': [lfc[0].magnitude, 'hPa'],
               'LCL': [lcl_pressure.magnitude, 'hPa'],
               'SBCAPE': [sfcpcl.bplus, 'J/kg'],
               'SBCIN': [sfcpcl.bminus, 'J/kg'],
               'SBLCL': [sfcpcl.lclhght, 'm AGL'],
               'SBLFC': [sfcpcl.lfchght, 'm AGL'],
               'SBEL': [sfcpcl.elhght, 'm AGL'],
               'MLCAPE': [mlpcl.bplus, 'J/kg'],
               'MLCIN': [mlpcl.bminus, 'J/kg'],
               'MLLCL': [mlpcl.lclhght, 'm AGL'],
               'MLLFC': [mlpcl.lfchght, 'm AGL'],
               'MLEL': [mlpcl.elhght, 'm AGL'],
               'MUCAPE': [mupcl.bplus, 'J/kg'],
               'MUCIN': [mupcl.bminus, 'J/kg'],
               'MULCL': [mupcl.lclhght, 'm AGL'],
               'MULFC': [mupcl.lfchght, 'm AGL'],
               'MUEL': [mupcl.elhght, 'm AGL'],
               'EIL depth':[eff_inflow_depth,'m AGL'],
               'SFC-800 hPa RH':[mean_rh_sfc_800,'%'],
               '800-400 hPa RH':[mean_rh_800_400,'%'],
               '400-100 hPa RH':[mean_rh_400_100,'%'],
               'Precipitable Water':[pw.magnitude,'mm'],
               '0-1 km Shear SharpPy': [utils.comp2vec(sfc_1km_shear[0], sfc_1km_shear[1])[1], 'kts'],
               '0-3 km Shear SharpPy': [utils.comp2vec(sfc_3km_shear[0], sfc_3km_shear[1])[1], 'kts'],
               '0-6 km Shear SharpPy': [utils.comp2vec(sfc_6km_shear[0], sfc_6km_shear[1])[1], 'kts'],
               '0-1 km Shear': [_1km_shear, 'kts'],
               '0-3 km Shear': [_3km_shear, 'kts'],
               '0-6 km Shear': [_6km_shear, 'kts'],
               '0-1 km SRH (Parcel Bunkers LM)': [srh1km_l[0], '$m^{2}/s^{2}$'],
               '0-3 km SRH (Parcel Bunkers LM)': [srh3km_l[0], '$m^{2}/s^{2}$'],
               '0-1 km SRH (Parcel Bunkers RM)': [srh1km_r[0], '$m^{2}/s^{2}$'],
               '0-3 km SRH (Parcel Bunkers RM)': [srh3km_r[0], '$m^{2}/s^{2}$'],
               'Eff. SRH (Parcel Bunkers LM)': [effective_srh_l[0], '$m^{2}/s^{2}$'],
               'Eff. SRH (Parcel Bunkers RM)': [effective_srh_r[0], '$m^{2}/s^{2}$'],
               'Bunkers (MetPy) LM u': [lm[0].magnitude, '$kts$'],
               'Bunkers (MetPy) LM v': [lm[1].magnitude, '$kts$'],
               'Bunkers (MetPy) RM u': [rm[0].magnitude, '$kts$'],
               'Bunkers (MetPy) RM v': [rm[1].magnitude, '$kts$'],
               'Bunkers (SharpPy-parcel) LM u': [srwind[2], '$kts$'],
               'Bunkers (SharpPy-parcel) LM v': [srwind[3], '$kts$'],
               'Bunkers (SharpPy-parcel) RM u': [srwind[0], '$kts$'],
               'Bunkers (SharpPy-parcel) RM v': [srwind[1], '$kts$']}


    if WRITE_SRH_OBS:
        if L_OR_R == 'L':
            #SHARPPY
            # 0-1 km
            #indices['0-1 km SRH (Model-Obs pos SharpPy)'] = [np.int_(
            #    np.round(sharppy_srh1km_pos)), '$m^{2}/s^{2}$']
            #indices['0-1 km SRH (Model-Obs neg SharpPy)'] = [np.int_(
            #    np.round(sharppy_srh1km_neg)), '$m^{2}/s^{2}$']
            #indices['0-1 km SRH (Model-Obs tot SharpPy)'] = [np.int_(
            #    np.round(sharppy_srh1km_tot)), '$m^{2}/s^{2}$']
            # 0-3 km
            #indices['0-3 km SRH (Model-Obs pos SharpPy)'] = [np.int_(
            #    np.round(sharppy_srh3km_pos)), '$m^{2}/s^{2}$']
            #indices['0-3 km SRH (Model-Obs neg SharpPy)'] = [np.int_(
            #    np.round(sharppy_srh3km_neg)), '$m^{2}/s^{2}$']
            #indices['0-3 km SRH (Model-Obs tot SharpPy)'] = [np.int_(
            #    np.round(sharppy_srh3km_tot)), '$m^{2}/s^{2}$']
            
            #METPY
            # 0-1 km
            indices['0-1 km SRH (Model-Obs pos MetPy)'] = [np.int_(
                np.round(metpy_srh1km_pos)), '$m^{2}/s^{2}$']
            indices['0-1 km SRH (Model-Obs neg MetPy)'] = [np.int_(
                np.round(metpy_srh1km_neg)), '$m^{2}/s^{2}$']
            indices['0-1 km SRH (Model-Obs tot MetPy)'] = [np.int_(
                np.round(metpy_srh1km_tot)), '$m^{2}/s^{2}$']
            # 0-3 km
            indices['0-3 km SRH (Model-Obs pos MetPy)'] = [np.int_(
                np.round(metpy_srh3km_pos)), '$m^{2}/s^{2}$']
            indices['0-3 km SRH (Model-Obs neg MetPy)'] = [np.int_(
                np.round(metpy_srh3km_neg)), '$m^{2}/s^{2}$']
            indices['0-3 km SRH (Model-Obs tot MetPy)'] = [np.int_(
                np.round(metpy_srh3km_tot)), '$m^{2}/s^{2}$']
            
        elif L_OR_R == 'R':
            indices['0-1 km SRH (Model-Obs RM)'] = [np.int_(
                np.round(srh1km_r_obs[0])), '$m^{2}/s^{2}$']
            indices['0-3 km SRH (Model-Obs RM)'] = [np.int_(
                np.round(srh3km_r_obs[0])), '$m^{2}/s^{2}$']
        else:
            pass
        
        indices['Storm-relative flow (avgd. EIL)'] = [np.int_(sr_flow),      '$ksts$']
        indices['EIL depth']                       = [np.int_(eff_inflow_depth), '$m$']
        
    if WRITE_ADVANCED_INDICES:
        indices['0-1 km mean u'] = [np.int_(np.round(mean_u_0_1)), '$m/s^{-1}$']
        indices['0-1 km mean v'] = [np.int_(np.round(mean_v_0_1)), '$m/s^{-1}$']
        indices['0-3 km mean u'] = [np.int_(np.round(mean_u_0_3)), '$m/s^{-1}$']
        indices['0-3 km mean v'] = [np.int_(np.round(mean_v_0_3)), '$m/s^{-1}$']
        indices['0-6 km mean u'] = [np.int_(np.round(mean_u_0_6)), '$m/s^{-1}$']
        indices['0-6 km mean v'] = [np.int_(np.round(mean_v_0_6)), '$m/s^{-1}$']
        indices['0-8 km mean u'] = [np.int_(np.round(mean_u_0_8)), '$m/s^{-1}$']
        indices['0-8 km mean v'] = [np.int_(np.round(mean_v_0_8)), '$m/s^{-1}$']
        
        
        mean_wind_vector_0_6 = [mean_u_0_6,mean_v_0_6]
        storm_motion_vector  = [U_STORM_OBS.magnitude, V_STORM_OBS.magnitude]
        unit_vector_1 = mean_wind_vector_0_6   / np.linalg.norm(mean_wind_vector_0_6)
        unit_vector_2 = storm_motion_vector    / np.linalg.norm(storm_motion_vector)
        dot_product = np.dot(unit_vector_1, unit_vector_2)
        angle = np.arccos(dot_product)
        angle_of_deviation = math.degrees(angle)
        
        indices['angle of deviation'] = [np.int_(np.round(angle_of_deviation)), '$degrees$']
        

    # if U_STORM_OBS:
    #    indices['Storm relative flow']=

    # 'EBWD': [np.round(ebwspd), 'kts'],\
    # 'PWV': [np.round(params.precip_water(prof), 2), 'inch']}
    # 'K-index': [np.round(params.k_index(prof)), ''],\
    # 'STP(fix)': [np.round(stp_fixed, 1), ''],\
    # 'SHIP': [np.round(ship, 1), ''],\
    # 'SCP': [np.round(scp, 1), ''],\
    # 'STP(cin)': [np.round(stp_cin, 1), '']}
    # List the indices within the indices dictionary on the side of the plot.
    return indices



def plot_skewT(TEMP_DEGC,DEWPT_DEGC,U_MPS,V_MPS,H_M,PRES_HPA,TITLE, FILENAME,BARBS=True,HODOGRAPH=True,LABELS=None,BARB_INTERVAL=8,PLOT_PARCEL_PROFILE=None, PROF_TV=None, PRINT_INDICES=None, BUNKERS=None,
               WRITE_SRH_OBS=None, L_OR_R=None, U_STORM_OBS=None, V_STORM_OBS=None):

    ###############################################################
    # Create a new figure. The dimensions here give a good aspect ratio
    fig = plt.figure(figsize=(12, 12))
    skew = SkewT(fig, rotation=30)
    ###############################################################
    if isinstance(TEMP_DEGC,list):
        print('plotting',len(TEMP_DEGC),'soundings soundings')
        print('note: currently a maximum of 3 soundings are supported')
        
        if len(TEMP_DEGC) == 2:
            tc1 = TEMP_DEGC[0]  # .plot()
            td1 = DEWPT_DEGC [0] # .plot()
            u1 = U_MPS[0]
            u1 = u1.to('knots')
            v1 = V_MPS[0]
            v1 = v1.to('knots')
            h1 = H_M[0]
            p1 = PRES_HPA[0]

            tc2 = TEMP_DEGC [1] # .plot()
            td2 = DEWPT_DEGC[1]  # .plot()
            u2 = U_MPS[1]
            u2 = u2.to('knots')
            v2 = V_MPS[1]
            v2 = v2.to('knots')
            h2 = H_M[1]
            p2 = PRES_HPA[1]

            wspeed1   = mpcalc.wind_speed(u1, v1)
            wspeed1   = wspeed1.to('knots')
            wind_dir1 = mpcalc.wind_direction(u1, v1)

            wspeed2   = mpcalc.wind_speed(u2, v2)
            wspeed2   = wspeed2.to('knots')
            wind_dir2 = mpcalc.wind_direction(u2, v2)

            below_100hpa1   = np.where(np.array(p1) > 100.0)
            p_below_100hPa1 = p1[below_100hpa1]
            u_below_100hPa1 = u1[below_100hpa1]
            v_below_100hPa1 = v1[below_100hpa1]
            h_below_100hpa1 = h1[below_100hpa1]

            below_200hpa1   = np.where(np.array(p1) > 200.0)
            p_below_200hPa1 = p1[below_200hpa1]
            u_below_200hPa1 = u1[below_200hpa1]
            v_below_200hPa1 = v1[below_200hpa1]
            h_below_200hpa1 = h1[below_200hpa1]

            below_100hpa2   = np.where(np.array(p2) > 100.0)
            p_below_100hPa2 = p2[below_100hpa2]
            u_below_100hPa2 = u2[below_100hpa2]
            v_below_100hPa2 = v2[below_100hpa2]
            h_below_100hpa2 = h2[below_100hpa2]

            below_200hpa2 = np.where(np.array(p2) > 200.0)
            p_below_200hPa2 = p2[below_200hpa2]
            u_below_200hPa2 = u2[below_200hpa2]
            v_below_200hPa2 = v2[below_200hpa2]
            h_below_200hpa2 = h2[below_200hpa2]
            # print(wspeed)

            #plt.title('Modified Weisman-Klemp and \n'+os.path.splitext(os.path.basename(EOL_SOUNDING))[0])
            # Plot the DATA using normal plotting functions, in this case using
            # log scaling in Y, as dictated by the typical meteorological plot
            skew.plot(p1, tc1, 'r', linewidth=1.5,label=LABELS[0] if LABELS else None)
            skew.plot(p1, td1, 'r', linewidth=1.5)

            skew.plot(p2, tc2, 'g', linewidth=1.5,label=LABELS[1] if LABELS else None)
            skew.plot(p2, td2, 'g', linewidth=1.5)

        elif len(TEMP_DEGC) == 3:
            tc1 = TEMP_DEGC[0]  # .plot()
            td1 = DEWPT_DEGC [0] # .plot()
            u1 = U_MPS[0]
            u1 = u1.to('knots')
            v1 = V_MPS[0]
            v1 = v1.to('knots')
            h1 = H_M[0]
            p1 = PRES_HPA[0]

            tc2 = TEMP_DEGC [1] # .plot()
            td2 = DEWPT_DEGC[1]  # .plot()
            u2 = U_MPS[1]
            u2 = u2.to('knots')
            v2 = V_MPS[1]
            v2 = v2.to('knots')
            h2 = H_M[1]
            p2 = PRES_HPA[1]
            
            tc2 = TEMP_DEGC[2]  # .plot()
            td2 = DEWPT_DEGC [2] # .plot()
            u2 = U_MPS[2]
            u2 = u1.to('knots')
            v2 = V_MPS[2]
            v2 = v1.to('knots')
            h2 = H_M[2]
            p2 = PRES_HPA[2]


            wspeed1   = mpcalc.wind_speed(u1, v1)
            wspeed1   = wspeed1.to('knots')
            wind_dir1 = mpcalc.wind_direction(u1, v1)

            wspeed2   = mpcalc.wind_speed(u2, v2)
            wspeed2   = wspeed2.to('knots')
            wind_dir2 = mpcalc.wind_direction(u2, v2)
            
            wspeed3   = mpcalc.wind_speed(u3, v3)
            wspeed3   = wspeed3.to('knots')
            wind_dir3 = mpcalc.wind_direction(u3, v3)

            below_100hpa1   = np.where(np.array(p1) > 100.0)
            p_below_100hPa1 = p1[below_100hpa1]
            u_below_100hPa1 = u1[below_100hpa1]
            v_below_100hPa1 = v1[below_100hpa1]
            h_below_100hpa1 = h1[below_100hpa1]

            below_200hpa1   = np.where(np.array(p1) > 200.0)
            p_below_200hPa1 = p1[below_200hpa1]
            u_below_200hPa1 = u1[below_200hpa1]
            v_below_200hPa1 = v1[below_200hpa1]
            h_below_200hpa1 = h1[below_200hpa1]

            below_100hpa2   = np.where(np.array(p2) > 100.0)
            p_below_100hPa2 = p2[below_100hpa2]
            u_below_100hPa2 = u2[below_100hpa2]
            v_below_100hPa2 = v2[below_100hpa2]
            h_below_100hpa2 = h2[below_100hpa2]

            below_200hpa2 = np.where(np.array(p2) > 200.0)
            p_below_200hPa2 = p2[below_200hpa2]
            u_below_200hPa2 = u2[below_200hpa2]
            v_below_200hPa2 = v2[below_200hpa2]
            h_below_200hpa2 = h2[below_200hpa2]
            
            below_100hpa3   = np.where(np.array(p3) > 100.0)
            p_below_100hPa3 = p3[below_100hpa3]
            u_below_100hPa3 = u3[below_100hpa3]
            v_below_100hPa3 = v3[below_100hpa3]
            h_below_100hpa3 = h3[below_100hpa3]

            below_200hpa3 = np.where(np.array(p3) > 200.0)
            p_below_200hPa3 = p3[below_200hpa3]
            u_below_200hPa3 = u3[below_200hpa3]
            v_below_200hPa3 = v3[below_200hpa3]
            h_below_200hpa3 = h3[below_200hpa3]

            skew.plot(p1, tc1, 'r', linewidth=0.9,label=LABELS[0] if LABELS else None)
            skew.plot(p1, td1, 'r', linewidth=0.9)

            skew.plot(p2, tc2, 'g', linewidth=1.5,label=LABELS[1] if LABELS else None)
            skew.plot(p2, td2, 'g', linewidth=1.5)
            
            skew.plot(p3, tc3, 'g', linewidth=0.9,label=LABELS[2] if LABELS else None)
            skew.plot(p3, td3, 'g', linewidth=0.9)
            
        else:
            print('# soundings not supported:')
            #break
            
        if BARBS:
            print('barbs for the middle input will be plotted')
            nn = BARB_INTERVAL
            #skew.plot_barbs(p_below_100hPa1[::nn], u_below_100hPa1[::nn],
            #                v_below_100hPa1[::nn], lw=0.8, length=8, flip_barb=True,color='red')#, c = 'pink')
            
            skew.plot_barbs(p_below_100hPa2[::nn], u_below_100hPa2[::nn],
                            v_below_100hPa2[::nn], lw=0.8, length=8, flip_barb=True,color='green')#, c = 'pink')
            
            
        #skew.plot_barbs(p[::nn], u[::nn], v[::nn],lw=0.8,length=6,flip_barb=True)

        skew.ax.set_ylim(1000, 100)
        skew.ax.set_xlim(-55, 40)

        plt.rcParams['xtick.labelsize'] = 14
        plt.rcParams['ytick.labelsize'] = 14
        
        plt.legend(loc='upper right',fontsize=15)

    else:
        print('plotting single sounding')
        tc1 = TEMP_DEGC  # .plot()
        td1 = DEWPT_DEGC  # .plot()
        u1 = U_MPS
        u1 = u1.to('knots')
        v1 = V_MPS
        v1 = v1.to('knots')
        h1 = H_M
        p1 = PRES_HPA

        wspeed1 = mpcalc.wind_speed(u1, v1)
        wspeed1 = wspeed1.to('knots')
        # print(wspeed)
        wind_dir1 = mpcalc.wind_direction(u1, v1)

        below_100hpa1 = np.where(np.array(p1) > 100.0)
        p_below_100hPa1 = p1[below_100hpa1]
        u_below_100hPa1 = u1[below_100hpa1]
        v_below_100hPa1 = v1[below_100hpa1]
        h_below_100hpa1 = h1[below_100hpa1]

        below_200hpa1 = np.where(np.array(p1) > 200.0)
        p_below_200hPa1 = p1[below_200hpa1]
        u_below_200hPa1 = u1[below_200hpa1]
        v_below_200hPa1 = v1[below_200hpa1]
        h_below_200hpa1 = h1[below_200hpa1]
        # print(wspeed)

        #plt.title('Modified Weisman-Klemp and \n'+os.path.splitext(os.path.basename(EOL_SOUNDING))[0])
        # Plot the DATA using normal plotting functions, in this case using
        # log scaling in Y, as dictated by the typical meteorological plot
        skew.plot(p1, tc1, 'r', linewidth=1.9)
        skew.plot(p1, td1, 'g', linewidth=1.9)
        
        if BARBS:
            nn = BARB_INTERVAL
            skew.plot_barbs(p_below_100hPa1[::nn], u_below_100hPa1[::nn],
                            v_below_100hPa1[::nn], lw=0.8, length=8, flip_barb=True)
            #skew.plot_barbs(p[::nn], u[::nn], v[::nn],lw=0.8,length=6,flip_barb=True)

        skew.ax.set_ylim(1000, 100)
        skew.ax.set_xlim(-55, 40)

        plt.rcParams['xtick.labelsize'] = 14
        plt.rcParams['ytick.labelsize'] = 14
    ###############################################################
    if TITLE:
        plt.title(TITLE,fontsize=15)
    ###############################################################
    # PLOT PARCEL PROFILE
    if PLOT_PARCEL_PROFILE:
        rhum = mpcalc.relative_humidity_from_dewpoint(tc1, td1)
        print('sfc RH = '+str(rhum[0]))
        mr1 = mpcalc.mixing_ratio_from_relative_humidity(
            pressure=p1, temperature=tc1, relative_humidity=rhum1)
        print('sfc mr = '+str(mr1[0]))
        tv1 = mpcalc.virtual_temperature(temperature=tc1, mixing=mr1).to('degC')
        print('sfc Tv = '+str(tv[0]))

        if PROF_TV:
            lcl_pressure1, lcl_temperature1 = mpcalc.lcl(
                p1[0], tv1[0], td1[0])  # calc LCL
            # Calculate the parcel profile.
            parcel_prof = mpcalc.parcel_profile(p1, tv1[0], td1[0]).to('degC')
        else:
            lcl_pressure, lcl_temperature = mpcalc.lcl(p1[0], tc1[0], td1[0])
            # Calculate the parcel profile.
            parcel_prof = mpcalc.parcel_profile(p1, tc1[0], td1[0]).to('degC')

        print('LCL pressure is: '+str(lcl_pressure1))
        print('LCL temp is: '+str(lcl_temperature1))
        skew.plot(lcl_pressure1, lcl_temperature1, 'ko',
                  markerfacecolor='gray', markersize=2.6)  # plot LCL

        # Plot the parcel profile as a black line
        skew.plot(p1, parcel_prof1, 'gray', linewidth=1)
        # print(parcel_prof)

        # Shade areas of CAPE and CIN
        skew.shade_cin(p1, tc1, parcel_prof, edgecolors='gray',
                       facecolor='red', linewidth=1.0, alpha=0.1)
        skew.shade_cape(p1, tc1, parcel_prof, edgecolors='gray',
                        facecolor='blue', linewidth=1.0, alpha=0.1)
    #################################################################
    if PRINT_INDICES:
        indices = create_indices_inflow(p1, h1, tc1, td1, wspeed1, wind_dir1, u1, v1,
                                        WRITE_SRH_OBS, L_OR_R, U_STORM_OBS, V_STORM_OBS)
        string = ''
        for key in (indices.keys()):
            # print(key)
            string = string + key + ': ' + \
                str(indices[key][0]) + ' ' + indices[key][1] + '\n'
            plt.text(1.02, 1, string, verticalalignment='top',
                     transform=plt.gca().transAxes)
    ###############################################################     
    if HODOGRAPH:
        print('if multiple input soundings are provided as input, ')
        # Create an inset axes for a hodograph object that is 30% width and height of the
        # figure and put it in the upper right hand corner.
        ax_hod = inset_axes(skew.ax, '40%', '40%', loc=1, borderpad=1.5)

        # add labels to the x and y axes
        ax_hod.get_xaxis().labelpad = 0.0
        ax_hod.set_xlabel('knots', rotation=0, fontsize=10)
        ax_hod.get_yaxis().labelpad = 0.0
        ax_hod.set_ylabel('knots', rotation=90, fontsize=10)

        # Create a hodograph object
        hodo = Hodograph(ax_hod, component_range=75.)
        hodo.add_grid(increment=10, linestyle='-', linewidth=0.6)
        # Plot a line colored by wind speed
        aa1 = hodo.plot_colormapped(u_below_100hPa1, v_below_100hPa1,
                                   h_below_100hpa1/1000.0, cmap=plt.get_cmap('hsv'), linestyle='solid')
        if isinstance(TEMP_DEGC,list):
            aa2 = hodo.plot_colormapped(u_below_100hPa2, v_below_100hPa2,
                                       h_below_100hpa2/1000.0, cmap=plt.get_cmap('hsv'), linestyle='solid')
            
        # Create axes for the heights shown in the hodograph
        cbaxes = inset_axes(ax_hod,
                            width="100%",  # width = 100% of parent_bbox width
                            height="3%",   # height : 3%
                            loc='lower left',
                            bbox_to_anchor=(0.0, -0.19, 1, 1),
                            bbox_transform=ax_hod.transAxes,
                            borderpad=0,
                            )
        CB = fig.colorbar(aa1, shrink=1.0, extend='both',
                          orientation='horizontal', cax=cbaxes, ticks=[0, 3, 6, 9, 12])
        CB.ax.get_xaxis().labelpad = 1.0
        CB.ax.set_xlabel('km', rotation=0, fontsize=10)
    ###################### PLOT BUNKERS AND/OR OBSERVED/MODELED STORM MOTION ########################
    if BUNKERS:
        # print(h)
        rm, lm, wm = mpcalc.bunkers_storm_motion(
            pressure=p1, u=u1, v=v1, heights=h1)
        hodo.wind_vectors(rm[0], rm[1], color='red')  # ,markersize=52)
        hodo.wind_vectors(lm[0], lm[1], color='blue')  # ,markersize=52)
        # hodo.plot(np.array([25.]),np.array([25.]),color='blue',markersize=572)
        hodo.wind_vectors(wm[0], wm[1])
        print('right moving: ', rm)
        print('left moving: ', lm)
        print('mean wind: ', wm)

    if WRITE_SRH_OBS:
        U_STORM_OBS = U_STORM_OBS * units('m/s')
        U_STORM_OBS = U_STORM_OBS.to('knots')

        V_STORM_OBS = V_STORM_OBS * units('m/s')
        V_STORM_OBS = V_STORM_OBS.to('knots')

        hodo.wind_vectors(U_STORM_OBS, V_STORM_OBS, color='green')

        if L_OR_R == 'L':
            print('left moving (model/obs): ', U_STORM_OBS, V_STORM_OBS)
        elif L_OR_R == 'R':
            print('right moving (model/obs): ', U_STORM_OBS, V_STORM_OBS)
        else:
            pass
    ###############################################################
    # Plot a zero degree isotherm
    skew.ax.axvline(0, color='c', linestyle='--', linewidth=2)
    # Add the relevant special lines
    skew.plot_dry_adiabats(colors='grey', linestyles='-', linewidth=0.50)
    skew.plot_moist_adiabats(colors='red', linestyles='-', linewidth=0.50)
    skew.plot_mixing_lines(colors='green', linestyles='-', linewidth=0.50)
    # Save the skew-T file
    # FILENAME='cm1_snd_'+LOCATION+'_'+extract_CM1_time(DATA)[1]+'.png'
    plt.savefig(FILENAME, bbox_inches="tight", dpi=150)
    print(FILENAME)
    ###############################################################
    #return pd.DataFrame.from_dict(create_indices(p, h, tc, td, u, v, WRITE_SRH_OBS=False, WRITE_ADVANCED_INDICES=False, L_OR_R=None, U_STORM_OBS=None, V_STORM_OBS=None))
    # plt.close()
    
    
def plot_area_average_sounding_RAMS_around_point(DS,ZT,PT_X,PT_Y,EXPANSE,TITRE,NOM_DE_FICHIER,MASK=None):

    Y1 = PT_Y-EXPANSE
    Y2 = PT_Y+EXPANSE
    X1 = PT_X-EXPANSE
    X2 = PT_X+EXPANSE
    
    if isinstance(MASK,np.ndarray):
        RAMS_exner=np.nanmean((DS.PI[:,Y1:Y2,X1:X2].values/1004.0)*MASK[np.newaxis,:,:],axis=(1, 2))
        RAMS_theta=np.nanmean(DS.THETA[:,Y1:Y2,X1:X2].values*MASK[np.newaxis,:,:],axis=(1, 2))# Kelvin
        RAMS_temp_K=RAMS_exner*RAMS_theta*units('K') # Kelvin
        RAMS_temp_degC=RAMS_temp_K.to('degC')
        RAMS_pres_Pa = (p00*(RAMS_exner)**(Cp/Rd))*units('Pa')
        RAMS_pres_hPa = RAMS_pres_Pa.to('hPa')
        RAMS_RV = np.nanmean(DS.RV[:,Y1:Y2,X1:X2].values*MASK[np.newaxis,:,:],axis=(1, 2))*units('kg/kg') # kg/kg
        RAMS_sphum = mpcalc.specific_humidity_from_mixing_ratio(RAMS_RV)
        RAMS_dewpt = mpcalc.dewpoint_from_specific_humidity(RAMS_pres_Pa,RAMS_temp_degC,RAMS_sphum)
        RAMS_U=np.nanmean(DS.UP[:,Y1:Y2,X1:X2].values*MASK[np.newaxis,:,:],axis=(1, 2))*units('m/s')
        RAMS_V=np.nanmean(DS.VP[:,Y1:Y2,X1:X2].values*MASK[np.newaxis,:,:],axis=(1, 2))*units('m/s')
        RAMS_TER=np.nanmean(DS.TOPT[Y1:Y2,X1:X2].values*MASK,axis=(0, 1))
        RAMS_HGT_MSL =  RAMS_TER + ZT
    else:
        RAMS_exner=(DS.PI[:,Y1:Y2,X1:X2].values/1004.0).mean(axis=(1, 2))
        RAMS_theta=DS.THETA[:,Y1:Y2,X1:X2].values.mean(axis=(1, 2))# Kelvin
        RAMS_temp_K=RAMS_exner*RAMS_theta*units('K') # Kelvin
        RAMS_temp_degC=RAMS_temp_K.to('degC')
        RAMS_pres_Pa = (p00*(RAMS_exner)**(Cp/Rd))*units('Pa')
        RAMS_pres_hPa = RAMS_pres_Pa.to('hPa')
        RAMS_RV = DS.RV[:,Y1:Y2,X1:X2].values.mean(axis=(1, 2))*units('kg/kg') # kg/kg
        RAMS_sphum = mpcalc.specific_humidity_from_mixing_ratio(RAMS_RV)
        RAMS_dewpt = mpcalc.dewpoint_from_specific_humidity(RAMS_pres_Pa,RAMS_temp_degC,RAMS_sphum)
        RAMS_U=DS.UP[:,Y1:Y2,X1:X2].values.mean(axis=(1, 2))*units('m/s')
        RAMS_V=DS.VP[:,Y1:Y2,X1:X2].values.mean(axis=(1, 2))*units('m/s')
        RAMS_TER=DS.TOPT[Y1:Y2,X1:X2].values.mean(axis=(0, 1))
        RAMS_HGT_MSL =  RAMS_TER + ZT
    
    plot_skewT(RAMS_temp_degC,RAMS_dewpt,RAMS_U,RAMS_V,RAMS_HGT_MSL,RAMS_pres_hPa,TITRE, NOM_DE_FICHIER,\
           BARB_INTERVAL=4,PLOT_PARCEL_PROFILE=None, PROF_TV=None, PRINT_INDICES=None, BUNKERS=None,\
           WRITE_SRH_OBS=None, L_OR_R=None, U_STORM_OBS=None, V_STORM_OBS=None)
    