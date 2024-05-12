
import numpy as np
import matplotlib.pyplot as plt
import sys
import glob
import os
from multiprocessing import Pool, cpu_count
import time
import h5py
import hdf5plugin
import pandas as pd
import xarray as xr
from netCDF4 import Dataset
import cartopy.crs as crs
import random
plt.style.use('ggplot')
from pprint import pprint
import seaborn as sns
import matplotlib.ticker as ticker
import read_vars_WRF_RAMS

def get_time_from_RAMS_file(INPUT_FILE):
    cur_time = os.path.split(INPUT_FILE)[1][4:21] # Grab time string from RAMS file
    pd_time = pd.to_datetime(cur_time[0:10]+' '+cur_time[11:13]+":"+cur_time[13:15]+":"+cur_time[15:17])
    return pd_time.strftime('%Y-%m-%d %H:%M:%S'), pd_time.strftime('%Y%m%d%H%M%S')
   
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

simulations=['USA1.1-R','ARG1.1-R_old','PHI1.1-R','PHI2.1-R','WPO1.1-R','BRA1.1-R','DRC1.1-R','AUS1.1-R']
domain='3'

def save_storm_mask_ITC(simulation_name):
    Cp=1004.
    Rd=287.0
    p00 = 100000.0
    path = '/monsoon/MODEL/LES_MODEL_DATA/'+simulation_name+'/G3/out_30s/'
    # Grab all the rams files 
    h5filepath = path+'a-L*g3.h5'
    h5files1 = sorted(glob.glob(h5filepath))
    hefilepath = path+'a-L*head.txt'
    hefiles1 = sorted(glob.glob(hefilepath))
    print('    first file: ',h5files1[0])
    print('    last file: ',h5files1[-1])
    middle_file = h5files1[int(len(h5files1)/2)]
    timestr = get_time_from_RAMS_file(middle_file)[1]
    da=xr.open_dataset(middle_file,engine='h5netcdf', phony_dims='sort')
    domain_z_dim,domain_y_dim,domain_x_dim=np.shape(da.WP)
    print('    ',domain_z_dim)
    print('    ',domain_y_dim)
    print('    ',domain_x_dim)

    zm, zt, nx, ny, dxy, npa = read_head(hefiles1[0],h5files1[0])

    print('    calculating ITC...')
    total_condensate = da['RTP']-da['RV']
    # Load variables needed to calculate density
    th = da['THETA']
    nx = np.shape(th)[2]
    ny = np.shape(th)[1]
    pi = da['PI']
    rv = da['RV']
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
    itc                       = np.nansum(total_condensate[1:,:,:]*dens[1:,:,:]*diff_zt_3D,axis=0) 
    output_var                = itc/997.0*1000 # integrated total frozen condensate in mm
    output_var_mod = np.where(output_var > 2, 1 , np.nan)
    plt.imshow(output_var_mod,cmap='GnBu')
    plt.colorbar()
    print('    done estimating ITC for ',simulation_name)
    output_var_mask = np.where(output_var > 2.0, 1.0 , np.nan)
    storm_mask_filename = 'storm_mask_'+simulation_name+'_'+timestr+'.npy'
    print('    saving the storm mask to ',storm_mask_filename)
    with open(storm_mask_filename, 'wb') as f:
        np.save(f, output_var_mask)
    return
    
def save_storm_mask_surface_precip_w(WHICH_TIME, SIMULATIONS, DOMAIN, PLOT):
    print('creating a storm mask')
    if DOMAIN=='1':
        dx = 1.6
    if DOMAIN=='2':
        dx=0.4
    if DOMAIN=='3':
        dx=0.1
    
    for ii,simulation in enumerate(SIMULATIONS):   
        print('    working on simulation: ',simulation)
        if DOMAIN=='1' or DOMAIN =='2':
            rams_files=sorted(glob.glob('/monsoon/MODEL/LES_MODEL_DATA/'+simulation+'/G'+DOMAIN+'/out/'+'a-A-*g'+DOMAIN+'.h5'))# CSU machine
        if DOMAIN=='3':
            rams_files=sorted(glob.glob('/monsoon/MODEL/LES_MODEL_DATA/'+simulation+'/G'+DOMAIN+'/out_30s/'+'a-L-*g3.h5'))# CSU machine
        print('        total # files = ',len(rams_files))
        print('        first file is ',rams_files[0])
        print('        last file is ',rams_files[-1])
        if WHICH_TIME=='start':
            rams_fil    = rams_files[0]
        if WHICH_TIME=='middle':
            rams_fil    = rams_files[int(len(rams_files)/2)]
        if WHICH_TIME=='end':
            rams_fil    = rams_files[-1]
        print('        choosing the middle file: ',rams_fil)
       
        da     = xr.open_dataset(rams_fil,engine='h5netcdf', phony_dims='sort')
        precip = da['PCPRR'].values  # kgm**-2
        #print('        shape of precip rate is ',np.shape(z))
        w      = da['WP'].max(dim='phony_dim_3')
        print('        precip rate min = ',np.nanmin(precip),' max = ',np.nanmax(precip))
        #print('        max w is ',np.nanmax(w))
        # create a mask
        output_var_mask = np.where(precip > 0.0001, 1.0 , np.nan)#*np.where(qtc > 0.0001, 1.0 , np.nan)
        storm_mask_filename = 'storm_mask_precip_'+simulation+'_'+get_time_from_RAMS_file(rams_fil)[1]+'.npy'
        print('        saving the storm mask to ',storm_mask_filename)
        with open(storm_mask_filename, 'wb') as f:
            np.save(f, output_var_mask)
         
        if PLOT:
            fig    = plt.figure(figsize=(8,8))
            mask_contours = plt.imshow(output_var_mask)#,cmap='GnBu')
            plt.contour(w,levels=np.arange(5,60,10),colors='k')
            timestep_string = get_time_from_RAMS_file(rams_fil)[0]
            #timestep_string     = pd.to_datetime(z_time,format='%Y%m%d%H%M%S').strftime('%Y-%m-%d %H:%M:%S')
            plt.title('storm_mask_surface_precip'+simulation+'\n'+timestep_string)
            plt.colorbar(mask_contours)
            plt.savefig('storm_mask_surface_precip'+simulation+'_'+get_time_from_RAMS_file(rams_fil)[1]+'.png')

def save_storm_mask_upper_level_precip_w_qtc(WHICH_TIME, LEVEL, LEVEL_TYPE, SIMULATIONS, DOMAIN, PLOT, MASK_CRITERIA):
    print('creating a storm mask for ',LEVEL_TYPE,' level ',LEVEL)
    if DOMAIN=='1':
        dx = 1.6
    if DOMAIN=='2':
        dx=0.4
    if DOMAIN=='3':
        dx=0.1
    
    for ii,simulation in enumerate(SIMULATIONS):   
        print('    working on simulation: ',simulation)
        if DOMAIN=='1' or DOMAIN =='2':
            rams_files=sorted(glob.glob('/monsoon/MODEL/LES_MODEL_DATA/'+simulation+'/G'+DOMAIN+'/out/'+'a-A-*g'+DOMAIN+'.h5'))# CSU machine
        if DOMAIN=='3':
            rams_files=sorted(glob.glob('/monsoon/MODEL/LES_MODEL_DATA/'+simulation+'/G'+DOMAIN+'/out_30s/'+'a-L-*g3.h5'))# CSU machine
        print('        total # files = ',len(rams_files))
        print('        first file is ',rams_files[0])
        print('        last file is ',rams_files[-1])
        if WHICH_TIME=='start':
            rams_fil    = rams_files[0]
        if WHICH_TIME=='middle':
            rams_fil    = rams_files[int(len(rams_files)/2)]
        if WHICH_TIME=='end':
            rams_fil    = rams_files[-1]
        print('        choosing the ',WHICH_TIME,' file: ',rams_fil)
       
        w, w_name, w_units, w_time = read_vars_WRF_RAMS.read_variable(rams_fil,'W','RAMS',output_height=False,interpolate=True,level=LEVEL,interptype=LEVEL_TYPE)
        qtc, qtc_name, qtc_units, qtc_time = read_vars_WRF_RAMS.read_variable(rams_fil,'QTC','RAMS',output_height=False,interpolate=True,level=LEVEL,interptype=LEVEL_TYPE)
        #precip, precip_name, precip_units, precip_time = read_vars_WRF_RAMS.read_variable(rams_fil,'PCP_RATE_3D','RAMS',output_height=False,interpolate=True,level=LEVEL,interptype=LEVEL_TYPE)
        print('        w min = ',np.nanmin(w),' max = ',np.nanmax(w))
        print('        qtc min = ',np.nanmin(qtc),' max = ',np.nanmax(qtc))
        #print('        precip min = ',np.nanmin(precip),' max = ',np.nanmax(precip))
        # create a mask
        
        if MASK_CRITERIA=='qtc_0.00001_w_2':
            output_var_mask = np.where(qtc > 0.00001, 1.0 , np.nan)*np.where(w > 2.0, 1.0 , np.nan)
            storm_mask_filename = 'storm_mask_'+MASK_CRITERIA+'_'+LEVEL_TYPE+'_level_'+str(int(LEVEL))+'_'+simulation+'_'+get_time_from_RAMS_file(rams_fil)[1]+'.npy'
            print('        saving the storm mask to ',storm_mask_filename)
            with open(storm_mask_filename, 'wb') as f:
                np.save(f, output_var_mask)
        if MASK_CRITERIA=='qtc_0.00001_w_1':
            output_var_mask = np.where(qtc > 0.00001, 1.0 , np.nan)*np.where(w > 1.0, 1.0 , np.nan)
            storm_mask_filename = 'storm_mask_'+MASK_CRITERIA+'_'+LEVEL_TYPE+'_level_'+str(int(LEVEL))+'_'+simulation+'_'+get_time_from_RAMS_file(rams_fil)[1]+'.npy'
            print('        saving the storm mask to ',storm_mask_filename)
            with open(storm_mask_filename, 'wb') as g:
                np.save(g, output_var_mask)
        if MASK_CRITERIA=='qtc_0.00001':
            output_var_mask = np.where(qtc > 0.00001, 1.0 , np.nan)
            storm_mask_filename = 'storm_mask_'+MASK_CRITERIA+'_'+LEVEL_TYPE+'_level_'+str(int(LEVEL))+'_'+simulation+'_'+get_time_from_RAMS_file(rams_fil)[1]+'.npy'
            print('        saving the storm mask to ',storm_mask_filename)
            with open(storm_mask_filename, 'wb') as h:
                np.save(h, output_var_mask)

        if PLOT:
            fig    = plt.figure(figsize=(8,8))
            mask_contours = plt.imshow(output_var_mask)#,cmap='GnBu')
            plt.contour(w,levels=np.arange(5,60,10),colors='k')
            timestep_string = get_time_from_RAMS_file(rams_fil)[0]
            #timestep_string     = pd.to_datetime(z_time,format='%Y%m%d%H%M%S').strftime('%Y-%m-%d %H:%M:%S')
            plt.title('storm_mask: '+MASK_CRITERIA+' ;'+simulation+'\n'+timestep_string)
            plt.savefig('storm_mask_'+MASK_CRITERIA+'_'+simulation+'_'+w_time+'.png',dpi=150)
            plt.colorbar(mask_contours)
 
#save_storm_mask_surface_precip_w('end',simulations,domain,False)

# for pressure_lev in [750,500,200]:
#     #for mask_type in ['qtc_0.00001_w_2','qtc_0.00001_w_1','qtc_0.00001']:
#     for mask_criteria in ['qtc_0.00001']:
#         for sim_time in ['start','end']:
#             save_storm_mask_upper_level_precip_w_qtc('start', pressure_lev, 'pressure', simulations, domain, True, mask_criteria)
#             print('-----------\n\n')

#save_storm_mask_surface_precip_w('middle', simulations, domain, True)
# sims=['RSA1.1-R']
# for sim in sims:
#     print('working on simulation: ',sim)
#     save_storm_mask(sim)

#save_variogram_masked_domain('not_near_storm','middle',variables[0], simulations, sample_size, domain, nsamples, colors, True)                                        
                                         
print('working on domain' ,domain)
#Running on the terminal in parallel
argument = []
for sim_time in ['start','middle','end']:
    argument = argument + [(sim_time, simulations, domain, False)]

print('length of argument is: ',len(argument))


# # ############################### FIRST OF ALL ################################
cpu_count1 = 37 #cpu_count()
print('number of cpus: ',cpu_count1)
# # #############################################################################

def main(FUNCTION, ARGUMENT):
    start_time = time.perf_counter()
    with Pool(processes = (cpu_count1-1)) as pool:
        data = pool.starmap(FUNCTION, ARGUMENT)
    finish_time = time.perf_counter()
    print(f"Program finished in {finish_time-start_time} seconds")
    #df_all = pd.concat(data, ignore_index=True)
    #thermo_indices_data_csv_file = csv_folder+'thermodynamic_indices_' + DOMAIN + '_comb_track_filt_01_02_50_02_sr5017_setpos.csv'
    #print('saving thermodynamic indices to the file: ',thermo_indices_data_csv_file)
    #df_all.to_csv(thermo_indices_data_csv_file)  # sounding data
    
if __name__ == "__main__":
    main(save_storm_mask_surface_precip_w, argument)
