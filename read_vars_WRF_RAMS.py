import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import h5py
import hdf5plugin
import pandas as pd
import xarray as xr
from netCDF4 import Dataset
from wrf import getvar, interpz3d, cape_2d, rh, interplevel
import metpy.calc as mpcalc
from metpy.units import units
import re

def get_time_from_RAMS_file(INPUT_FILE):
    """
    Input:  1 String: Path to RAMS file
    Output: 2 Strings:
    the first one of the format 2019-11-10 21:34:45 for use in plot titles and 
    the second one of the format 20191110213445 for use in file names
    """
    cur_time = os.path.split(INPUT_FILE)[1][4:21] # Grab time string from RAMS file
    pd_time = pd.to_datetime(cur_time[0:10]+' '+cur_time[11:13]+":"+cur_time[13:15]+":"+cur_time[15:17])
    return [pd_time.strftime('%Y-%m-%d %H:%M:%S'), pd_time.strftime('%Y%m%d%H%M%S')]

def get_time_from_WRF_file(INPUT_FILE):
    """
    Input:  1 String: Path to RAMS file
    Output: 2 Strings:
    the first one of the format 2019-11-10 21:34:45 for use in plot titles and 
    the second one of the format 20191110213445 for use in file names
    """
    cur_time = os.path.split(INPUT_FILE)[1][11:30] # Grab time string from WRF file
    pd_time = pd.to_datetime(cur_time[0:10]+' '+cur_time[11:19])
    return [pd_time.strftime('%Y-%m-%d %H:%M:%S'), pd_time.strftime('%Y%m%d%H%M%S')]

def find_matching_RAMS_headfile(H5FILE):
    converted_string = re.sub(r'-g\d+.h5$', '-head.txt', H5FILE)
    if converted_string:
        print('found a header file: ',converted_string)
    return converted_string

# PJM added read_head from RAMS_Post_Process (https://github.com/CSU-INCUS/RAMS_Post_Process)
def read_head(headfile,h5file):
    
    # Function that reads header files from RAMS
    
    # Inputs:
    #   headfile: header file including full path in str format
    #   h5file: h5 datafile including full path in str format
    
    # Returns:
    #   zmn: height levels for momentum values (i.e., grid box upper and lower levels)
    #   ztn: height levels for thermodynaic values (i.e., grid box centers)
    #   nx:: the number of x points for the domain associated with the h5file
    #   ny: the number of y points for the domain associated with the h5file
    #   npa: the number of surface patches
    
    
    dom_num = h5file[h5file.index('.h5')-1] # Find index of .h5 to determine position showing which nest domain to use

    with open(headfile) as f:
        contents = f.readlines()
        
    idx_zmn = contents.index('__zmn0'+dom_num+'\n')
    nz_m = int(contents[idx_zmn+1])
    zmn = np.zeros(nz_m)
    for i in np.arange(0,nz_m):
        zmn[i] =  float(contents[idx_zmn+2+i])
    
    idx_ztn = contents.index('__ztn0'+dom_num+'\n')
    nz_t = int(contents[idx_ztn+1])
    ztn = np.zeros(nz_t)
    for i in np.arange(0,nz_t):
        ztn[i] =  float(contents[idx_ztn+2+i])
    
    ztop = np.max(ztn) # Model domain top (m)
    
    # Grad the size of the horizontal grid spacing
    idx_dxy = contents.index('__deltaxn\n')
    dxy = float(contents[idx_dxy+1+int(dom_num)].strip())

    idx_npatch = contents.index('__npatch\n')
    npa = int(contents[idx_npatch+2])
    
    idx_ny = contents.index('__nnyp\n')
    idx_nx = contents.index('__nnxp\n')
    ny = np.ones(int(contents[idx_ny+1]))
    nx = np.ones(int(contents[idx_ny+1]))
    for i in np.arange(0,len(ny)):
        nx[i] = int(contents[idx_nx+2+i])
        ny[i] = int(contents[idx_ny+2+i])

    ny_out = ny[int(dom_num)-1]
    nx_out = nx[int(dom_num)-1]

    return zmn, ztn, nx_out, ny_out, dxy, npa 

def read_variable(filepath,variable,model_type,output_height=False,interpolate=False,level=500,interptype='pressure'):
    """
    -----------------
    3 Inputs: 1. string: Complete path to RAMS or WRF file. Only one timestep per file is allowed.
              2. string: Variable name. Only the following 25 variable names are allowed:
                2.1  "U"         - zonal velocity (m/s)
                2.2  "V"         - meridional velocity (m/s)
                2.3  "W"         - vertical velocity (m/s)
                2.4  "THETA"     - potential temperature (K)
                2.5  "QV"        - water vapor mixing ratio (kg/kg)
                2.6  "QC"        - cloud water mixing ratio (kg/kg)
                2.7  "QS"        - cloud snow mixing ratio (kg/kg)
                2.8  "QI"        - cloud ice mixing ratio (kg/kg)
                2.9  "QH"        - hail mixing ratio (kg/kg)
                2.10 "QG"        - graupel mixing ratio (kg/kg)
                2.11 "T"         - temperature (K)
                2.12 "RH"        - relative humidity (m/s)
                2.13 "P"         - pressure (Pa)
                2.14 "RHO"       - density (kg/m^3)
                2.15 "PCP_RATE"  - precipitation rate (mm/hr)
                2.16 "PCP_ACC"   - accumulated precipitation (mm)
                2.17 "QR"        - rain mixing ratio (kg/kg) 
                2.18 "QA"        - aggregate mixing ratio (kg/kg)
                2.19 "QTW"       - total water mixing ratio (kg/kg)
                2.20 "QTC"       - total condensate mixing ratio (kg/kg)
                2.21 "QTF"       - total frozen condensate mixing ratio (kg/kg)
                2.22 "PW"        - precipitable water (mm)
                2.23 "IWP"       - ice water path (mm) 
                2.24 "LWP"       - liquid water path (mm)
                2.25 "ITC"       - integrated condensate (mm)
              3. string: Model type. Only two values are allowed: "RAMS" or "WRF"
    ------------------
    5 Outputs:1. np array: 3D (or 2D) of the output variable 
              2. np array: 3D array of height of model grid points
              3. string: name of the variable 
              4. string: units of the variable 
              5. string: time in UTC
              
    example usage: read_variable('/monsoon/LES_MODEL_DATA/DRC1.1-R/G3/out_30s/a-L-2016-12-30-112430-g3.h5','RH','RAMS')
    to get 3D relative humidity from a RAMS file
    """
    
    Cp=1004.
    Rd=287.0
    p00 = 100000.0
    #print('variable: ',variable)
    if model_type=='RAMS':
        #print('RAMS file: ',filepath)
        
        # load the RAMS file 
        #da = xr.open_dataset(filepath,engine="h5netcdf",phony_dims='sort') # xarray: does not always work
        da = h5py.File(filepath,"r")                                      # h5py: alternate method
        var_time = get_time_from_RAMS_file(filepath)[1]

        if output_height:
            rams_header_file = sorted(glob.glob(os.path.dirname(filepath)+'/a-*head.txt'))[0]
        
            if os.path.isfile(rams_header_file):
                #print('Using header file: ',rams_header_file)
                zm, zt, nx, ny, dxy, npa = read_head(rams_header_file,filepath)
                rams_topo = da['TOPT'][:] #.values
                height_3d_array = rams_topo[np.newaxis,:,:]+zt[:,np.newaxis,np.newaxis]
            else:
                print('could not find a header file in the directory of the .h5 file. Exiting...')
                return

        if variable=='LAT':
            var_name   = 'latitude'
            var_units  = 'degrees'
            output_var = da['GLAT'][:]
        if variable=='LON':
            var_name   = 'longitude'
            var_units  = 'degrees'
            output_var = da['GLON'][:]
        if variable=='U':
            var_name   = 'zonal velocity'
            var_units  = 'm/s'
            output_var = da['UP'][:]
        if variable=='V':
            var_name   = 'meridional velocity'
            var_units  = 'm/s'
            output_var = da['VP'][:]
        if variable=='WSPD':
            var_name   = 'horizontal wind speed'
            var_units  = 'm/s'
            output_var = np.sqrt(da['UP'][:]**2 + da['VP'][:]**2)
        if variable=='W':
            var_name   = 'vertical velocity'
            var_units  = 'm/s'
            output_var = da['WP'][:]   
        if variable=='MAXCOL_W':
            var_name   = 'max column vertical velocity'
            var_units  = 'm/s'
            output_var = np.max(da['WP'][:],axis=0)
        if variable=='THETA':
            var_name   = 'potential temperature'
            var_units  = 'K'
            output_var = da['THETA'][:]
        if variable=='THETAV':
            var_name   = 'virtual potential temperature'
            var_units  = 'K'
            theta      = da['THETA'][:]
            qv         = da['RV'][:]
            ql         = da['RCP'][:]+da['RRP'][:]+da['RDP'][:]
            output_var = theta*(1.0+0.61*qv-ql)
        if variable=='QV':
            var_name   = 'water vapor mixing ratio'
            var_units  = 'kg/kg'
            output_var = da['RV'][:]
        if variable=='QC':
            var_name   = 'cloud water mixing ratio'
            var_units  = 'kg/kg'
            output_var = da['RCP'][:]
        if variable=='QS':
            var_name   = 'cloud snow mixing ratio'
            var_units  = 'kg/kg'
            output_var = da['RSP'][:]
        if variable=='QI':
            var_name   = 'cloud ice mixing ratio'
            var_units  = 'kg/kg'
            output_var = da['RPP'][:]
        if variable=='QH':
            var_name   = 'hail mixing ratio'
            var_units  = 'kg/kg'
            output_var = da['RHP'][:]
        if variable=='QG':
            var_name   = 'graupel mixing ratio'
            var_units  = 'kg/kg'
            output_var = da['RGP'][:]
        if variable=='QA':
            var_name   = 'aggregate mixing ratio'
            var_units  = 'kg/kg'
            output_var = da['RAP'][:]
        if variable=='QSA':
            var_name   = 'aggregate + snow mixing ratio'
            var_units  = 'kg/kg'
            output_var = da['RAP'][:] + da['RSP'][:]
        if variable=='QISA':
            var_name   = 'ice + aggregate + snow mixing ratio'
            var_units  = 'kg/kg'
            output_var = da['RPP'][:] + da['RAP'][:] + da['RSP'][:]
        if variable=='QGH':
            var_name   = 'graupel + hail mixing ratio'
            var_units  = 'kg/kg'
            output_var = da['RGP'][:] + da['RHP'][:]
        if variable=='QR':
            var_name   = 'rain mixing ratio'
            var_units  = 'kg/kg'
            output_var = da['RRP'][:]
        if variable=='QTW':
            var_name   = 'total water mixing ratio'
            var_units  = 'kg/kg'
            output_var = da['RTP'][:]
        if variable=='QTC':
            var_name   = 'total condensate mixing ratio'
            var_units  = 'kg/kg'
            rtp        = da['RTP'][:]
            rtp[rtp<=0.0] = 0.000000000000001   
            rv         = da['RV'][:]
            rv[rv<=0.0]= 0.0
            output_var = rtp-rv
            output_var[output_var<=0.0]= 0.0
            del(rtp,rv)
        if variable=='QTF':
            var_name   = 'total frozen condensate mixing ratio'
            var_units  = 'kg/kg'
            output_var = da['RTP'][:]-da['RV'][:]-da['RCP'][:]-da['RDP'][:]-da['RRP'][:]
        if variable=='QTL':
            var_name   = 'total liquid condensate mixing ratio'
            var_units  = 'kg/kg'
            output_var = da['RCP'][:]+da['RRP'][:]+da['RDP'][:]
        if variable=='QNICE_per_kg':
            var_name   = 'ice number concentration'
            var_units  = '#/kg'
            output_var = da['CPP'][:]
        if variable=='QNCLOUD_per_kg':
            var_name   = 'cloud number concentration'
            var_units  = '#/kg'
            output_var = da['CCP'][:]
        if variable=='QNDRIZZLE_per_kg':
            var_name   = 'drizzle number concentration'
            var_units  = '#/kg'
            output_var = da['CDP'][:]
        if variable=='QNRAIN_per_kg':
            var_name   = 'rain number concentration'
            var_units  = '#/kg'
            output_var = da['CRP'][:]
        if variable=='QNSNOW_per_kg':
            var_name   = 'snow number concentration'
            var_units  = '#/kg'
            output_var = da['CSP'][:]
        if variable=='QNAGG_per_kg':
            var_name   = 'aggregates number concentration'
            var_units  = '#/kg'
            output_var = da['CAP'][:]
        if variable=='QNGRAUPEL_per_kg':
            var_name   = 'graupel number concentration'
            var_units  = '#/kg'
            output_var = da['CGP'][:]
        if variable=='QNHAIL_per_kg':
            var_name   = 'hail number concentration'
            var_units  = '#/kg'
            output_var = da['CHP'][:]
        if variable=='QNICE_per_m3':
            var_name   = 'ice number concentration'
            var_units  = '#/m^3'
            th         = da['THETA'][:]
            pi         = da['PI'][:]
            rv         = da['RV'][:]
            pres       = np.power((pi/Cp),Cp/Rd)*p00
            temp       = th*(pi/Cp)
            del(th,pi)
            density    = pres/(Rd*temp*(1+0.61*rv))
            del(pres,temp,rv)
            output_var = da['CPP'][:]*density
        if variable=='QNCLOUD_per_m3':
            var_name   = 'cloud number concentration'
            var_units  = '#/m^3'
            th         = da['THETA'][:]
            pi         = da['PI'][:]
            rv         = da['RV'][:]
            pres       = np.power((pi/Cp),Cp/Rd)*p00
            temp       = th*(pi/Cp)
            del(th,pi)
            density    = pres/(Rd*temp*(1+0.61*rv))
            del(pres,temp,rv)
            output_var = da['CCP'][:]*density
        if variable=='QNDRIZZLE_per_m3':
            var_name   = 'drizzle number concentration'
            var_units  = '#/m^3'
            th         = da['THETA'][:]
            pi         = da['PI'][:]
            rv         = da['RV'][:]
            pres       = np.power((pi/Cp),Cp/Rd)*p00
            temp       = th*(pi/Cp)
            del(th,pi)
            density    = pres/(Rd*temp*(1+0.61*rv))
            del(pres,temp,rv)
            output_var = da['CDP'][:]*density
        if variable=='QNRAIN_per_m3':
            var_name   = 'rain number concentration'
            var_units  = '#/m^3'
            th         = da['THETA'][:]
            pi         = da['PI'][:]
            rv         = da['RV'][:]
            pres       = np.power((pi/Cp),Cp/Rd)*p00
            temp       = th*(pi/Cp)
            del(th,pi)
            density    = pres/(Rd*temp*(1+0.61*rv))
            del(pres,temp,rv)
            output_var = da['CRP'][:]*density
        if variable=='QNSNOW_per_m3':
            var_name   = 'snow number concentration'
            var_units  = '#/m^3'
            th         = da['THETA'][:]
            pi         = da['PI'][:]
            rv         = da['RV'][:]
            pres       = np.power((pi/Cp),Cp/Rd)*p00
            temp       = th*(pi/Cp)
            del(th,pi)
            density    = pres/(Rd*temp*(1+0.61*rv))
            del(pres,temp,rv)
            output_var = da['CSP'][:]*density
        if variable=='QNAGG_per_m3':
            var_name   = 'aggregates number concentration'
            var_units  = '#/m^3'
            th         = da['THETA'][:]
            pi         = da['PI'][:]
            rv         = da['RV'][:]
            pres       = np.power((pi/Cp),Cp/Rd)*p00
            temp       = th*(pi/Cp)
            del(th,pi)
            density    = pres/(Rd*temp*(1+0.61*rv))
            del(pres,temp,rv)
            output_var = da['CAP'][:]*density
        if variable=='QNGRAUPEL_per_m3':
            var_name   = 'graupel number concentration'
            var_units  = '#/m^3'
            th         = da['THETA'][:]
            pi         = da['PI'][:]
            rv         = da['RV'][:]
            pres       = np.power((pi/Cp),Cp/Rd)*p00
            temp       = th*(pi/Cp)
            del(th,pi)
            density    = pres/(Rd*temp*(1+0.61*rv))
            del(pres,temp,rv)
            output_var = da['CGP'][:]*density
        if variable=='QNHAIL_per_m3':
            var_name   = 'hail number concentration'
            var_units  = '#/m^3'
            th         = da['THETA'][:]
            pi         = da['PI'][:]
            rv         = da['RV'][:]
            pres       = np.power((pi/Cp),Cp/Rd)*p00
            temp       = th*(pi/Cp)
            del(th,pi)
            density    = pres/(Rd*temp*(1+0.61*rv))
            del(pres,temp,rv)
            output_var = da['CHP'][:]*density
        if variable=='CMMDIA':
            var_name   = 'cloud mean mass diameter'
            var_units  = 'mm'
            qr         = da['RCP'][:]
            nr         = da['CCP'][:]
            alpha      = 524. 
            beta       = 3.   
            output_var = ((qr / (nr * alpha))**(1./beta))*1000.0
        if variable=='DMMDIA':
            var_name   = 'drizzle mean mass diameter'
            var_units  = 'mm'
            qr         = da['RDP'][:]
            nr         = da['CDP'][:]
            alpha      = 524. 
            beta       = 3.   
            output_var = ((qr / (nr * alpha))**(1./beta))*1000.0
        if variable=='RMMDIA':
            var_name   = 'rain mean mass diameter'
            var_units  = 'mm'
            qr         = da['RRP'][:]
            nr         = da['CRP'][:]
            alpha      = 524. 
            beta       = 3.   
            output_var = ((qr / (nr * alpha))**(1./beta))*1000.0
        if variable=='SMMDIA':
            var_name   = 'snow (col) mean mass diameter'
            var_units  = 'mm'
            qr         = da['RSP'][:]
            nr         = da['CSP'][:]
            alpha      = 2.739e-3
            beta       = 1.74
            output_var = ((qr / (nr * alpha))**(1./beta))*1000.0
        if variable=='IMMDIA':
            var_name   = 'ice (col) mean mass diameter'
            var_units  = 'mm'
            qr         = da['RPP'][:]
            nr         = da['CPP'][:]
            alpha      = 110.8
            beta       = 2.91
            output_var = ((qr / (nr * alpha))**(1./beta))*1000.0
        if variable=='AMMDIA':
            var_name   = 'aggregates mean mass diameter'
            var_units  = 'mm'
            qr         = da['RAP'][:]
            nr         = da['CAP'][:]
            alpha      = .496   
            beta       = 2.4
            output_var = ((qr / (nr * alpha))**(1./beta))*1000.0
        if variable=='GMMDIA':
            var_name   = 'graupel mean mass diameter'
            var_units  = 'mm'
            qr         = da['RGP'][:]
            nr         = da['CGP'][:]
            alpha      = 157.   
            beta       = 3.   
            output_var = ((qr / (nr * alpha))**(1./beta))*1000.0
        if variable=='HMMDIA':
            var_name   = 'hail mean mass diameter'
            var_units  = 'mm'
            qr         = da['RHP'][:]
            nr         = da['CHP'][:]
            alpha      = 471.
            beta       = 3.   
            output_var = ((qr / (nr * alpha))**(1./beta))*1000.0
        if variable=='Tk':
            var_name   = 'temperature'
            var_units  = 'K'
            th         = da['THETA'][:]
            pi         = da['PI'][:]
            pres       = np.power((pi/Cp),Cp/Rd)*p00
            output_var = th*(pi/Cp)
            del(th,pi,pres)
        if variable=='Tc':
            var_name   = 'temperature'
            var_units  = 'K'
            th         = da['THETA'][:]
            pi         = da['PI'][:]
            pres       = np.power((pi/Cp),Cp/Rd)*p00
            output_var = th*(pi/Cp) - 273.15
            del(th,pi,pres)
        if variable=='RH':
            var_name   = 'relative humidity'
            var_units  = '%'
            th         = da['THETA'][:]
            pi         = da['PI'][:]
            rv         = da['RV'][:]
            pres       = np.power((pi/Cp),Cp/Rd)*p00
            temp       = th*(pi/Cp)
            # rh_with_units = mpcalc.relative_humidity_from_mixing_ratio(pres*units.Pa,temp*units.K,rv*units('kg/kg'))
            # del(th,pi,rv,pres,temp)
            # output_var = rh_with_units.magnitude
            # print('max, min are: ',np.max(output_var),np.min(output_var))
            output_var = rh(rv, pres, temp, meta=False)
            #print('shape of output_var: ',np.shape(output_var))
            #print('max, min RH are: ',np.max(output_var),np.min(output_var))
        if variable=='P':
            var_name   = 'pressure'
            var_units  = 'Pa'
            pi         = da['PI'][:]
            output_var = np.power((pi/Cp),Cp/Rd)*p00
        if variable=='RHO':
            var_name   = 'density'
            var_units  = 'kg/m^3'
            th         = da['THETA'][:]
            pi         = da['PI'][:]
            rv         = da['RV'][:]
            pres       = np.power((pi/Cp),Cp/Rd)*p00
            temp       = th*(pi/Cp)
            del(th,pi)
            output_var = pres/(Rd*temp*(1+0.61*rv))
            del(pres,temp,rv)
        if variable=='LHF':
            var_name   = 'latent heat flux'
            var_units  = 'W/m^2'
            output_var = da['SFLUX_R'][:]*2.5e6	
        if variable=='SHF':
            var_name   = 'sensible heat flux'
            var_units  = 'W/m^2'
            output_var = da['SFLUX_T'][:]*1004.
        if variable=='TERR_HGT':
            var_name   = 'terrain height'
            var_units  = 'm'
            output_var = da['TOPT'][:]
        if variable=='TOP_SOIL_MOISTURE':
            var_name   = 'vol. soil moisture'
            var_units  = 'm^3 m^-3'
            soil_water = da['SOIL_WATER'].values
            output_var = np.zeros_like(da['TOPT'])
            patch_area = da['PATCH_AREA'].values
            for npa in np.arange(0,np.shape(patch_area)[0]):
                output_var = output_var + patch_area[npa,:,:] * soil_water[npa,-1,:,:]
        if variable=='MCAPE':
            var_name   = 'MCAPE'
            var_units  = 'J/kg'
            th         = da['THETA'][:]
            pi         = da['PI'][:]
            rv         = da['RV'][:]
            pres       = np.power((pi/Cp),Cp/Rd)*p00
            temp       = th*(pi/Cp)
            rams_header_file = sorted(glob.glob(os.path.dirname(filepath)+'/a-*head.txt'))[0]
            if os.path.isfile(rams_header_file):
                #print('Using header file: ',rams_header_file)
                zm, zt, nx, ny, dxy, npa = read_head(rams_header_file,filepath)
                rams_topo = da['TOPT'][:] #.values
                height_3d_array = rams_topo[np.newaxis,:,:]+zt[:,np.newaxis,np.newaxis]
            cape, cin, lcl, lfc = cape_2d(pres/100.0, temp, rv, height_3d_array, rams_topo, pres[0,:,:], True)
            output_var = cape.values
            del(pres,temp,rv,th,pi)
        if variable=='MCIN':
            var_name   = 'MCIN'
            var_units  = 'J/kg'
            th         = da['THETA'][:]
            pi         = da['PI'][:]
            rv         = da['RV'][:]
            pres       = np.power((pi/Cp),Cp/Rd)*p00
            temp       = th*(pi/Cp)
            rams_header_file = sorted(glob.glob(os.path.dirname(filepath)+'/a-*head.txt'))[0]
            if os.path.isfile(rams_header_file):
                #print('Using header file: ',rams_header_file)
                zm, zt, nx, ny, dxy, npa = read_head(rams_header_file,filepath)
                rams_topo = da['TOPT'][:] #.values
                height_3d_array = rams_topo[np.newaxis,:,:]+zt[:,np.newaxis,np.newaxis]
            cape, cin, lcl, lfc = cape_2d(pres/100.0, temp, rv, height_3d_array, rams_topo, pres[0,:,:], True)
            output_var = cin.values
            del(pres,temp,rv,th,pi)
        if variable=='LCL':
            var_name   = 'LCL'
            var_units  = 'm'
            th         = da['THETA'][:]
            pi         = da['PI'][:]
            rv         = da['RV'][:]
            pres       = np.power((pi/Cp),Cp/Rd)*p00
            temp       = th*(pi/Cp)
            rams_header_file = sorted(glob.glob(os.path.dirname(filepath)+'/a-*head.txt'))[0]
            if os.path.isfile(rams_header_file):
                #print('Using header file: ',rams_header_file)
                zm, zt, nx, ny, dxy, npa = read_head(rams_header_file,filepath)
                rams_topo = da['TOPT'][:] #.values
                height_3d_array = rams_topo[np.newaxis,:,:]+zt[:,np.newaxis,np.newaxis]
            cape, cin, lcl, lfc = cape_2d(pres/100.0, temp, rv, height_3d_array, rams_topo, pres[0,:,:], True)
            output_var = lcl.values
            del(pres,temp,rv,th,pi)
        if variable=='LFC':
            var_name   = 'LFC'
            var_units  = 'm'
            th         = da['THETA'][:]
            pi         = da['PI'][:]
            rv         = da['RV'][:]
            pres       = np.power((pi/Cp),Cp/Rd)*p00
            temp       = th*(pi/Cp)
            rams_header_file = sorted(glob.glob(os.path.dirname(filepath)+'/a-*head.txt'))[0]
            if os.path.isfile(rams_header_file):
                #print('Using header file: ',rams_header_file)
                zm, zt, nx, ny, dxy, npa = read_head(rams_header_file,filepath)
                rams_topo = da['TOPT'][:] #.values
                height_3d_array = rams_topo[np.newaxis,:,:]+zt[:,np.newaxis,np.newaxis]
            cape, cin, lcl, lfc = cape_2d(pres/100.0, temp, rv, height_3d_array, rams_topo, pres[0,:,:], True)
            output_var = lfc.values
            del(pres,temp,rv,th,pi)
        if variable=='PCP_RATE':
            var_name   = 'precipitation rate'
            var_units  = 'mm/hr'
            output_var = da['PCPRR'][:]*3600.0
        if variable=='PCP_RATE_3D':
            var_name   = '3D precipitation rate'
            var_units  = 'mm/hr'
            output_var = da['PCPVR'][:]*3600.0    
        if variable=='PCP_ACC':
            var_name   = 'accumulated precipitation'
            var_units  = 'mm'
            output_var = da['ACCPR'][:]
        if variable=='PW':
            var_name   = 'precipitable water'
            var_units  = 'mm'
            rv         = da['RV'][:]
            # Load variables needed to calculate density
            th = da['THETA'][:]
            nx = np.shape(th)[2]
            ny = np.shape(th)[1]
            pi = da['PI'][:]
            # Convert RAMS native variables to temperature and pressure
            pres = np.power((pi/Cp),Cp/Rd)*p00
            temp = th*(pi/Cp)
            del(th,pi)
            # Calculate atmospheric density
            dens = pres/(Rd*temp*(1+0.61*rv))
            del(pres,temp)
            # Difference in heights (dz)    
            diff_zt_3D = np.tile(np.diff(zt),(int(ny),int(nx),1))
            diff_zt_3D = np.moveaxis(diff_zt_3D,2,0)
            pw                        = np.nansum(rv[1:,:,:]*dens[1:,:,:]*diff_zt_3D,axis=0) 
            output_var                = pw/997.0*1000 # precipitable water in mm
            output_var[output_var<=0] = 0.001
        if variable=='IWP':
            var_name   = 'ice water path'
            var_units  = 'mm'
            frozen_condensate = da['RPP'][:]+da['RSP'][:]+da['RAP'][:]+da['RGP'][:]+da['RHP'][:]
            # Load variables needed to calculate density
            th = da['THETA'][:]
            nx = np.shape(th)[2]
            ny = np.shape(th)[1]
            pi = da['PI'][:]
            rv = da['RV'][:]
            # Convert RAMS native variables to temperature and pressure
            pres = np.power((pi/Cp),Cp/Rd)*p00
            temp = th*(pi/Cp)
            del(th,pi)
            # Calculate atmospheric density
            dens = pres/(Rd*temp*(1+0.61*rv))
            del(pres,temp,rv)
            # Difference in heights (dz)    
            diff_zt_3D = np.tile(np.diff(zt),(int(ny),int(nx),1))
            diff_zt_3D = np.moveaxis(diff_zt_3D,2,0)
            # Calculate IWP (ice water path)
            iwp                       = np.nansum(frozen_condensate[1:,:,:]*dens[1:,:,:]*diff_zt_3D,axis=0) 
            output_var                = iwp/997.0*1000 # integrated total frozen condensate in mm
            output_var[output_var<=0] = 0.001
        if variable=='LWP':
            var_name   = 'liquid water path'
            var_units  = 'mm'
            liquid_condensate = da['RCP'][:]+da['RDP'][:]+da['RRP'][:]
            # Load variables needed to calculate density
            th = da['THETA'][:]
            nx = np.shape(th)[2]
            ny = np.shape(th)[1]
            pi = da['PI'][:]
            rv = da['RV'][:]
            # Convert RAMS native variables to temperature and pressure
            pres = np.power((pi/Cp),Cp/Rd)*p00
            temp = th*(pi/Cp)
            del(th,pi)
            # Calculate atmospheric density
            dens = pres/(Rd*temp*(1+0.61*rv))
            del(pres,temp,rv)
            # Difference in heights (dz)    
            diff_zt_3D = np.tile(np.diff(zt),(int(ny),int(nx),1))
            diff_zt_3D = np.moveaxis(diff_zt_3D,2,0)
            lwp                       = np.nansum(liquid_condensate[1:,:,:]*dens[1:,:,:]*diff_zt_3D,axis=0) 
            output_var                = lwp/997.0*1000 # liquid water path in mm
            output_var[output_var<=0] = 0.001
        if variable=='ITC':
            header_file = find_matching_RAMS_headfile(filepath)
            zm, zt, nx, ny, dxy, npa = read_head(header_file,filepath)
            var_name   = 'integrated total condensate'
            var_units  = 'mm'
            th = da['THETA'][:]
            nx = np.shape(th)[2]
            ny = np.shape(th)[1]
            rtp = da['RTP'][:] - da['RV'][:]
            pi = da['PI'][:]
            rv = da['RV'][:]
            # Convert RAMS native variables to temperature and pressure
            pres = np.power((pi/Cp),Cp/Rd)*p00
            temp = th*(pi/Cp)
            del(th,pi)
            # Calculate atmospheric density
            dens = pres/(Rd*temp*(1+0.61*rv))
            del(pres,temp,rv)
            # Difference in heights (dz)    
            diff_zt_3D = np.tile(np.diff(zt),(int(ny),int(nx),1))
            diff_zt_3D = np.moveaxis(diff_zt_3D,2,0)
            # Calculate integrated condensate
            itc = np.nansum(rtp[1:,:,:]*dens[1:,:,:]*diff_zt_3D,axis=0) # integrated total condensate in kg
            output_var = itc/997.0*1000.0 # integrated total condensate in mm
            output_var[output_var<=0] = 0.0001         
            
    if model_type=='WRF':
        # load the WRF file 
        da = Dataset(filepath) #,engine="h5netcdf",phony_dims='sort')
        # We are using the netCDF Python Dataset function instead of Xarray because
        # it works well with the WRF-Python package
        var_time = get_time_from_WRF_file(filepath)[1]
        
        if output_height:
            height_3d_array = getvar(da, "height").values
        
        if variable=='TERR_HGT':
            var_name   = 'terrain height'
            var_units  = 'm'
            output_var = getvar(da, "HGT").values
        if variable=='U':
            var_name   = 'zonal velocity'
            var_units  = 'm/s'
            output_var = getvar(da, "ua").values
        if variable=='V':
            var_name   = 'meridional velocity'
            var_units  = 'm/s'
            output_var = getvar(da, "va").values
        if variable=='WSPD':
            var_name   = 'horizontal wind speed'
            var_units  = 'm/s'
            output_var = np.sqrt(getvar(da, "ua").values**2 + getvar(da, "va").values**2)
        if variable=='W':
            var_name   = 'vertical velocity'
            var_units  = 'm/s'
            output_var = getvar(da, "wa").values
        if variable=='THETA':
            var_name   = 'potential temperature'
            var_units  = 'K'
            output_var = getvar(da, "theta").values
        if variable=='THETAV':
            var_name   = 'virtual potential temperature'
            var_units  = 'K'
            theta      = getvar(da, "theta").values
            qv         = getvar(da, "QVAPOR").values
            ql         = getvar(da, "QCLOUD").values + getvar(da, "QRAIN").values
            output_var = theta*(1.0+0.61*qv-ql)
        if variable=='LHF':
            var_name   = 'latent heat flux'
            var_units  = 'W/m^2'
            output_var = getvar(da, "LH").values
        if variable=='SHF':
            var_name   = 'sensible heat flux'
            var_units  = 'W/m^2'
            output_var = getvar(da, "HFX").values
        if variable=='TOP_SOIL_MOISTURE':
            var_name   = 'vol. soil moisture'
            var_units  = 'm^3 m^-3'
            output_var = getvar(da, "SMOIS").values[0,:,:]
        if variable=='QV':
            var_name   = 'water vapor mixing ratio'
            var_units  = 'kg/kg'
            output_var = getvar(da, "QVAPOR").values
        if variable=='QC':
            var_name   = 'cloud water mixing ratio'
            var_units  = 'kg/kg'
            output_var = getvar(da, "QCLOUD").values
        if variable=='QS':
            var_name   = 'cloud snow mixing ratio'
            var_units  = 'kg/kg'
            output_var = getvar(da, "QSNOW").values    
        if variable=='QSA':
            var_name   = 'cloud snow mixing ratio'
            var_units  = 'kg/kg'
            output_var = getvar(da, "QSNOW").values
        if variable=='QISA':
            var_name   = 'cloud ice + snow mixing ratio'
            var_units  = 'kg/kg'
            output_var = getvar(da, "QSNOW").values + getvar(da, "QICE").values
        if variable=='QGH':
            var_name   = 'graupel mixing ratio'
            var_units  = 'kg/kg'
            output_var = getvar(da, "QGRAUP").values
        if variable=='QI':
            var_name   = 'cloud ice mixing ratio'
            var_units  = 'kg/kg'
            output_var = getvar(da, "QICE").values
        if variable=='QH':
            var_name   = 'hail mixing ratio'
            var_units  = 'kg/kg'
            output_var = getvar(da, "QHAIL").values
        if variable=='QG':
            var_name   = 'graupel mixing ratio'
            var_units  = 'kg/kg'
            output_var = getvar(da, "QGRAUP").values
        if variable=='QR':
            var_name   = 'rain mixing ratio'
            var_units  = 'kg/kg'
            output_var = getvar(da, "QRAIN").values
        if variable=='QTC':
            var_name   = 'total condensate mixing ratio'
            var_units  = 'kg/kg'
            output_var = getvar(da, "QCLOUD").values + getvar(da, "QSNOW").values + getvar(da, "QICE").values + \
            getvar(da, "QGRAUP").values  + getvar(da, "QRAIN").values  
        if variable=='QTF':
            var_name   = 'total frozen condensate mixing ratio'
            var_units  = 'kg/kg'
            output_var = getvar(da, "QSNOW").values + getvar(da, "QICE").values + getvar(da, "QGRAUP").values
        if variable=='QTL':
            var_name   = 'total liquid condensate mixing ratio'
            var_units  = 'kg/kg'
            output_var = getvar(da, "QCLOUD").values + getvar(da, "QRAIN").values
        if variable=='QTW':
            var_name   = 'total water mixing ratio'
            var_units  = 'kg/kg'
            output_var = getvar(da, "QSNOW").values + getvar(da, "QICE").values + getvar(da, "QGRAUP").values + \
            getvar(da, "QCLOUD").values + getvar(da, "QRAIN").values + getvar(da, "QVAPOR").values
        if variable=='QNICE_per_kg':
            var_name   = 'ice number concentration'
            var_units  = '#/kg'
            output_var = getvar(da, "QNICE").values
        if variable=='QNCLOUD_per_kg':
            print('Error: this variable is not available in WRF output: outputting NaNs')
            var_name   = 'cloud number concentration'
            var_units  = '#/kg'
            temp_array = getvar(da, "QVAPOR").values
            output_var = np.full_like(temp_array, np.nan)
        if variable=='QNDRIZZLE_per_kg':
            print('Error: this variable is not available in WRF output: outputting NaNs')
            var_name   = 'drizzle number concentration'
            var_units  = '#/kg'
            temp_array = getvar(da, "QVAPOR").values
            output_var = np.full_like(temp_array, np.nan)
        if variable=='QNRAIN_per_kg':
            var_name   = 'rain number concentration'
            var_units  = '#/kg'
            output_var = getvar(da, "QNRAIN").values
        if variable=='QNSNOW_per_kg':
            var_name   = 'snow number concentration'
            var_units  = '#/kg'
            try: 
                output_var = getvar(da, "QNSNOW").values
            except ValueError:
                print('Error: this variable is not available in WRF output: outputting NaNs')
                temp_array = getvar(da, "QVAPOR").values
                output_var = np.full_like(temp_array, np.nan)    
        if variable=='QNAGG_per_kg':
            print('Error: this variable is not available in WRF output: outputting NaNs')
            var_name   = 'aggregates number concentration'
            var_units  = '#/kg'
            temp_array = getvar(da, "QVAPOR").values
            output_var = np.full_like(temp_array, np.nan)
        if variable=='QNGRAUPEL_per_kg':
            var_name   = 'graupel number concentration'
            var_units  = '#/kg'
            try:
                output_var = getvar(da, "QNGRAUPEL").values
            except ValueError:
                print('Error: this variable is not available in WRF output: outputting NaNs')
                temp_array = getvar(da, "QVAPOR").values
                output_var = np.full_like(temp_array, np.nan)
        if variable=='QNHAIL_per_kg':
            print('Error: this variable is not available in WRF output: outputting NaNs')
            var_name   = 'hail number concentration'
            var_units  = '#/kg'
            temp_array = getvar(da, "QVAPOR").values
            output_var = np.full_like(temp_array, np.nan)
        if variable=='QNICE_per_m3':
            var_name   = 'ice number concentration'
            var_units  = '#/m^3'
            pres       = getvar(da, "pres")
            temp       = getvar(da, "tk")
            rv         = getvar(da, "QVAPOR")
            density    = pres/(Rd*temp*(1+0.61*rv))
            output_var = getvar(da, "QNICE").values*density.values
        if variable=='QNCLOUD_per_m3':
            print('Error: this variable is not available in WRF output: outputting NaNs')
            var_name   = 'cloud number concentration'
            var_units  = '#/m^3'
            temp_array = getvar(da, "QVAPOR").values
            output_var = np.full_like(temp_array, np.nan)
        if variable=='QNDRIZZLE_per_m3':
            print('Error: this variable is not available in WRF output: outputting NaNs')
            var_name   = 'drizzle number concentration'
            var_units  = '#/m^3'
            temp_array = getvar(da, "QVAPOR").values
            output_var = np.full_like(temp_array, np.nan)
        if variable=='QNRAIN_per_m3':
            var_name   = 'rain number concentration'
            var_units  = '#/m^3'
            pres       = getvar(da, "pres")
            temp       = getvar(da, "tk")
            rv         = getvar(da, "QVAPOR")
            density    = pres/(Rd*temp*(1+0.61*rv))
            output_var = getvar(da, "QNRAIN").values*density.values
        if variable=='QNSNOW_per_m3':
            var_name   = 'snow number concentration'
            var_units  = '#/m^3'
            pres       = getvar(da, "pres")
            temp       = getvar(da, "tk")
            rv         = getvar(da, "QVAPOR")
            density    = pres/(Rd*temp*(1+0.61*rv))
            output_var = getvar(da, "QNSNOW").values*density.values
        if variable=='QNAGG_per_m3':
            print('Error: this variable is not available in WRF output: outputting NaNs')
            var_name   = 'aggregates number concentration'
            var_units  = '#/m^3'
            temp_array = getvar(da, "QVAPOR").values
            output_var = np.full_like(temp_array, np.nan)
        if variable=='QNGRAUPEL_per_m3':
            var_name   = 'graupel number concentration'
            var_units  = '#/m^3'
            pres       = getvar(da, "pres")
            temp       = getvar(da, "tk")
            rv         = getvar(da, "QVAPOR")
            density    = pres/(Rd*temp*(1+0.61*rv))
            output_var = getvar(da, "QNGRAUPEL").values*density.values
        if variable=='QNHAIL_per_m3':
            print('Error: this variable is not available in WRF output: outputting NaNs')
            var_name   = 'hail number concentration'
            var_units  = '#/m^3'
            temp_array = getvar(da, "QVAPOR").values
            output_var = np.full_like(temp_array, np.nan)
        if variable=='CMMDIA':
            print('Error: this variable is not available in WRF output: outputting NaNs')
            var_name   = 'cloud mean mass diameter'
            var_units  = 'mm'
            temp_array = getvar(da, "QVAPOR").values
            output_var = np.full_like(temp_array, np.nan)
        if variable=='DMMDIA':
            print('Error: this variable is not available in WRF output: outputting NaNs')
            var_name   = 'drizzle mean mass diameter'
            var_units  = 'mm'
            temp_array = getvar(da, "QVAPOR").values
            output_var = np.full_like(temp_array, np.nan)
        if variable=='RMMDIA':
            var_name   = 'rain mean mass diameter'
            var_units  = 'mm'
            qr         = getvar(da, "QRAIN").values
            nr         = getvar(da, "QNRAIN").values
            alpha      = 524. 
            beta       = 3.   
            output_var = ((qr / (nr * alpha))**(1./beta))*1000.0
        if variable=='SMMDIA':
            var_name   = 'snow (col) mean mass diameter'
            var_units  = 'mm'
            try:
                nr         = getvar(da, "QNSNOW").values
                qr         = getvar(da, "QSNOW").values
                alpha      = 2.739e-3
                beta       = 1.74
                output_var = ((qr / (nr * alpha))**(1./beta))*1000.0
            except ValueError:
                print('Error: this variable is not available in WRF output: outputting NaNs')
                temp_array = getvar(da, "QVAPOR").values
                output_var = np.full_like(temp_array, np.nan)
        if variable=='IMMDIA':
            var_name   = 'ice (col) mean mass diameter'
            var_units  = 'mm'
            qr         = getvar(da, "QICE").values
            nr         = getvar(da, "QNICE").values
            alpha      = 110.8
            beta       = 2.91
            output_var = ((qr / (nr * alpha))**(1./beta))*1000.0
        if variable=='AMMDIA':
            print('Error: this variable is not available in WRF output: outputting NaNs')
            var_name   = 'aggregates mean mass diameter'
            var_units  = 'mm'
            alpha      = .496   
            beta       = 2.4
            temp_array = getvar(da, "QVAPOR").values
            output_var = np.full_like(temp_array, np.nan) #((qr / (nr * alpha))**(1./beta))*1000.0
        if variable=='GMMDIA':
            var_name   = 'graupel mean mass diameter'
            var_units  = 'mm'
            try:
                nr         = getvar(da, "QNGRAUPEL").values
                qr         = getvar(da, "QGRAUPEL").values
                alpha      = 157.   
                beta       = 3.   
                output_var = ((qr / (nr * alpha))**(1./beta))*1000.0
            except ValueError:
                print('Error: this variable is not available in WRF output: outputting NaNs')
                temp_array = getvar(da, "QVAPOR").values
                output_var = np.full_like(temp_array, np.nan)
        if variable=='HMMDIA':
            print('Error: this variable is not available in WRF output: outputting NaNs')
            var_name   = 'hail mean mass diameter'
            var_units  = 'mm'
            #qr         = getvar(da, "QHAIL")
            #nr         = getvar(da, "QNHAIL")
            alpha      = 471.
            beta       = 3.   
            temp_array = getvar(da, "QVAPOR").values
            output_var = np.full_like(temp_array, np.nan) #((qr / (nr * alpha))**(1./beta))*1000.0
        if variable=='Tc':
            var_name   = 'temperature'
            var_units  = 'deg C'
            # thm_pert   = getvar(da, "THM")
            # base_temp  = 290. #getvar(da, "T00")
            # thm        = thm_pert + base_temp
            # qv         = getvar(da, "QVAPOR")
            # theta      = thm/(1+1.61*qv)
            # pres       = getvar(da, "pres")
            # tk         = theta/((100000.0/pres)**0.286) 
            tk           = getvar(da, "tk")
            output_var = tk.values - 273.15
        if variable=='Tk':
            var_name   = 'temperature'
            var_units  = 'K'
            # thm_pert   = getvar(da, "THM")
            # base_temp  = 290. #getvar(da, "T00")
            # thm        = thm_pert + base_temp
            # qv         = getvar(da, "QVAPOR")
            # theta      = thm/(1+1.61*qv)
            # pres       = getvar(da, "pres")
            # tk         = theta/((100000.0/pres)**0.286) 
            tk         = getvar(da, "tk")
            output_var = tk.values
        if variable=='RH':
            var_name   = 'relative humidity'
            var_units  = '%'
            # thm_pert   = getvar(da, "THM")
            # base_temp  = getvar(da, "T00")
            # thm        = thm_pert + base_temp
            # print('max, min of thm are: ',np.max(thm),np.min(thm))
            # qv         = getvar(da, "QVAPOR")
            # theta      = thm/(1+1.61*qv)
            # print('max, min of theta are: ',np.max(theta),np.min(theta))
            # pres       = getvar(da, "pres")
            # print('max, min of pres are: ',np.max(pres),np.min(pres))
            # tk         = theta/((p00/pres)**0.286)
            # print('max, min of tk are: ',np.max(tk),np.min(tk))
            # output_var = mpcalc.relative_humidity_from_mixing_ratio(pres.values*units('Pa'),tk.values*units('K'),qv.values*units('kg/kg')).magnitude
            # print('max, min are: ',np.max(output_var),np.min(output_var))
            # del(thm_pert,base_temp,thm,theta,pres,tk,qv)
            #rh(qv, pres, tkel, meta=True)
        if variable=='P':
            var_name   = 'pressure'
            var_units  = 'Pa'
            output_var = getvar(da, "pres").values
        if variable=='RHO':
            var_name   = 'density'
            var_units  = 'kg/m^3'
            pres       = getvar(da, "pres")
            temp       = getvar(da, "tk")
            rv         = getvar(da, "QVAPOR")
            output_var = pres/(Rd*temp*(1+0.61*rv))
            del(pres,temp,rv)
        if variable=='PCP_RATE':
            var_name   = 'precipitation rate'
            var_units  = 'mm/hr'
            previous_file_time = (pd.to_datetime(get_time_from_WRF_file(filepath)[0]) - pd.Timedelta(30, "second")).strftime('%Y-%m-%d_%H:%M:%S')
            print('previous file time is : ',previous_file_time)
            previous_file = os.path.dirname(filepath)+'/wrfout_d03_'+previous_file_time
            if os.path.isfile(previous_file):
                print('previous file is : ',previous_file)
                da1 = Dataset(previous_file)
                output_var = ((getvar(da,  "RAINC").values + getvar(da,  "RAINNC").values + getvar(da,  "RAINSH").values) -\
                              (getvar(da1, "RAINC").values + getvar(da1, "RAINNC").values + getvar(da1, "RAINSH").values))/(30.0*(1./3600.))
            else:
                output_var = np.nan
                print('this might be the first file... cannot compute rain rate!')
        if variable=='PCP_ACC':
            var_name   = 'accumulated precipitation'
            var_units  = 'mm'
            output_var = getvar(da, "RAINC").values + getvar(da, "RAINNC").values + getvar(da, "RAINSH").values
        
    if interpolate:
        if model_type=='RAMS':
            if interptype=='pressure':
                print('getting '+variable+' data on pressure level ',level,' hPa')
                pi         = da['PI'][:]
                vert = np.power((pi/Cp),Cp/Rd)*p00/100.
                #output_var = interpz3d(output_var, vert, level)
                output_var = interplevel(output_var, vert, level, meta=False)
            if interptype=='height':
                print('getting '+variable+' data on height level ',level,' m')
                pi         = da['PI'][:]
                vert = np.power((pi/Cp),Cp/Rd)*p00
                #output_var = interpz3d(output_var, vert, level)
                output_var = interplevel(output_var, vert, level, meta=False)
            if interptype=='model':
                print('getting '+variable+' data on model level ',level)
                output_var = output_var[level,:,:]
        if model_type=='WRF':
            if interptype=='pressure':
                print('getting '+variable+' data on pressure level ',level,' hPa')
                vert = getvar(da, "pressure").values
                #output_var = interpz3d(output_var, vert, level)
                output_var = interplevel(output_var, vert, level, meta=False)
            if interptype=='height':
                print('getting '+variable+' data on height level ',level,' m')
                vert = getvar(da, "pres").values
                #output_var = interpz3d(output_var, vert, level)
                output_var = interplevel(output_var, vert, level, meta=False)
            if interptype=='model':
                print('getting '+variable+' data on model level ',level)
                output_var = output_var[level,:,:]
        
        #print('sanity check...')
        #print('shape of field to be interpolated... ',np.shape(output_var))
        #print('shape of the vertical coordinate field... ',np.shape(vert))
       
    
    if output_height:
        return output_var, height_3d_array, var_name, var_units, var_time
    else:
        return output_var, var_name, var_units, var_time