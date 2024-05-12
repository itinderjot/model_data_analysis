
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
import skgstat as skg
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

simulations=['ARG1.1-R_old','PHI1.1-R','PHI2.1-R','WPO1.1-R','BRA1.1-R','USA1.1-R','DRC1.1-R','AUS1.1-R']
domain='3'
nsamples=1
sample_size = 15000
variables = [['Tk', 0, 'model', '$T_{sfc}^{2} (K^{2})$','precip']             , ['THETA', 0, 'model', '$Theta_{sfc}^{2} (K^{2})$','precip'],\
             ['QV', 0, 'model', '$Qvapor_{sfc}^{2} (kg^{2}kg^{-2})$','precip'], ['RH', 0, 'model', '$RH_{sfc}^{2} (percent^{2})$','precip'],\
             ['U', 0, 'model', '$U_{sfc}^{2} (m^{2}s^{-2})$','precip']        , ['V', 0, 'model', '$V_{sfc}^{2} (m^{2}s^{-2})$','precip'],\
             ['WSPD', 0, 'model', '$WSPD_{sfc}^{2} (m^{2}s^{-2})$','precip']  , ['W', 0, 'model', '$W_{sfc}^{2} (m^{2}s^{-2})$','precip'],\
             ['MCAPE', -999, None, '$MCAPE^{2} (J^{2}kg^{-2})$','precip']     , ['MCIN', -999, None, '$MCIN^{2} (J^{2}kg^{-2})$','precip'], \
             ['Tk', 750, 'pressure', '$T_{750}^{2} (K^{2})$','qtc_0.00001_w_1']        , ['THETA', 750, 'pressure', '$Theta_{750}^{2} (K^{2})$','qtc_0.00001_w_1'],\
             ['QV', 750, 'pressure', '$Qvapor_{750}^{2} (kg^{2}kg^{-2})$','qtc_0.00001_w_1'], ['RH', 750, 'pressure', '$RH_{750}^{2} (percent^{2})$','qtc_0.00001_w_1'],\
             ['U', 750, 'pressure', '$U_{750}^{2} (m^{2}s^{-2})$','qtc_0.00001_w_1']   , ['V', 750, 'pressure', '$V_{750}^{2} (m^{2}s^{-2})$','qtc_0.00001_w_1'],\
             ['WSPD', 750, 'pressure', '$WSPD_{750}^{2} (m^{2}s^{-2})$','qtc_0.00001_w_1'], ['W', 750, 'pressure', '$W_{750}^{2} (m^{2}s^{-2})$','qtc_0.00001_w_1'],\
             ['Tk', 500, 'pressure', '$T_{500}^{2} (K^{2})$','qtc_0.00001_w_1']        , ['THETA', 500, 'pressure', '$Theta_{500}^{2} (K^{2})$','qtc_0.00001_w_1'],\
             ['QV', 500, 'pressure', '$Qvapor_{500}^{2} (kg^{2}kg^{-2})$','qtc_0.00001_w_1'], ['RH', 500, 'pressure', '$RH_{500}^{2} (percent^{2})$','qtc_0.00001_w_1'],\
             ['U', 500, 'pressure', '$U_{500}^{2} (m^{2}s^{-2})$','qtc_0.00001_w_1']   , ['V', 500, 'pressure', '$V_{500}^{2} (m^{2}s^{-2})$','qtc_0.00001_w_1'],\
             ['WSPD', 500, 'pressure', '$WSPD_{500}^{2} (m^{2}s^{-2})$','qtc_0.00001_w_1'], ['W', 500, 'pressure', '$W_{500}^{2} (m^{2}s^{-2})$','qtc_0.00001_w_1'],\
             ['Tk', 200, 'pressure', '$T_{200}^{2} (K^{2})$','qtc_0.00001_w_1']        , ['THETA', 200, 'pressure', '$Theta_{200}^{2} (K^{2})$','qtc_0.00001_w_1'],\
             ['QV', 200, 'pressure', '$Qvapor_{200}^{2} (kg^{2}kg^{-2})$','qtc_0.00001_w_1'], ['RH', 200, 'pressure', '$RH_{200}^{2} (percent^{2})$','qtc_0.00001_w_1'],\
             ['U', 200, 'pressure', '$U_{200}^{2} (m^{2}s^{-2})$','qtc_0.00001_w_1']   , ['V', 200, 'pressure', '$V_{200}^{2} (m^{2}s^{-2})$','qtc_0.00001_w_1'], \
             ['WSPD', 200, 'pressure', '$WSPD_{200}^{2} (m^{2}s^{-2})$','qtc_0.00001_w_1'], ['W', 200, 'pressure', '$W_{200}^{2} (m^{2}s^{-2})$','qtc_0.00001_w_1']]

colors    =  ['#000000','#E69F00','#56B4E9','#009E73','#F0E442','#0072B2','#D55E00','#CC79A7']

def save_variogram_masked_domain(MASK_AREA,WHICH_TIME, VARIABLE, SIMULATIONS, SAMPLE_SIZE, DOMAIN, NSAMPLES, COLORS, PLOT):
    if DOMAIN=='1':
        dx = 1.6
    if DOMAIN=='2':
        dx=0.4
    if DOMAIN=='3':
        dx=0.1
        
    print('working on ',VARIABLE,'\n')
    MASK_TYPE= 'precip' #VARIABLE[4]
    print('mask type is ',MASK_TYPE)
    
    if PLOT:
        fig    = plt.figure(figsize=(8,8))
        
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
        
        ######################################################
        # if (VARIABLE[1]<0) or (VARIABLE[1]==0):
        #     mask_search_string = 'storm_mask_'+MASK_TYPE+'_'+simulation+'_'+get_time_from_RAMS_file(rams_fil)[1]+'.npy'
        #     print(        'mask search string: ',mask_search_string)
        #     mask_name = sorted(glob.glob(mask_search_string))
        # else:
        #     mask_search_string = 'storm_mask_'+MASK_TYPE+'_'+VARIABLE[2]+'_level_'+str(VARIABLE[1])+'_'+simulation+'_'+get_time_from_RAMS_file(rams_fil)[1]+'.npy'
        #     print('        mask search string: ',mask_search_string)
        #     mask_name = sorted(glob.glob(mask_search_string))
        
        mask_search_string = 'storm_mask_'+MASK_TYPE+'_'+simulation+'_'+get_time_from_RAMS_file(rams_fil)[1]+'.npy'
        mask_name = sorted(glob.glob(mask_search_string))
        print('        <<< found mask file>>>: ',mask_name)

        if MASK_AREA=='near_storm':
            mask    = np.load(mask_name[0])
        elif MASK_AREA=='not_near_storm':
            mask    = np.load(mask_name[0])
            mask    = np.where((mask < 1.1) & (mask > 0.9), np.nan , 1.0)
        else:
            print('please provide a corect value of area type')
        ###############################################################
        

        da     = xr.open_dataset(rams_fil,engine='h5netcdf', phony_dims='sort')
        z_temp = da['TOPT'].values
        y_dim, x_dim     = np.shape(z_temp)
        print('        shape of the arrays is ',y_dim,'x',x_dim)
        x      = np.arange(0,x_dim)
        y      = np.arange(0,y_dim)
        # # full coordinate arrays
        xx, yy = np.meshgrid(x, y)*mask
        coords_tuples_2d = np.vstack(([yy.T], [xx.T])).T
        print('        shape of combined coords matrix: ',np.shape(coords_tuples_2d))
        coords_all_np = coords_tuples_2d.reshape(-1, 2)#.tolist()
        print('        shape of 1d list of coords: ',np.shape(coords_all_np))
        # Create a boolean mask for rows with NaN values
        nan_rows_mask = np.any(np.isnan(coords_all_np), axis=1)
        # Use the mask to select rows without NaN values
        coords_all_np = coords_all_np[~nan_rows_mask]
        coords_all    = coords_all_np.tolist()
        #print('first 10 coords are: ',coords_all[0:11])

        z, z_name, z_units, z_time = read_vars_WRF_RAMS.read_variable(rams_fil,VARIABLE[0],'RAMS',output_height=False,interpolate=VARIABLE[1]>-1,level=VARIABLE[1],interptype=VARIABLE[2])
        print('        got the data... min = ',np.nanmin(z),' max = ',np.nanmax(z))
        print('        percentage of nans is ',np.count_nonzero(np.isnan(z))/len(z.flatten()))
        print('        choosing ',min(SAMPLE_SIZE,len(coords_all)),' random points...')
        coords = random.sample(coords_all,min(SAMPLE_SIZE,len(coords_all)))
        #print('first 10 randomly selected coords are: ',coords[0:11])
        print('        get field values from these points...')
        values = np.fromiter((z[int(c[0]), int(c[1])] for c in coords), dtype=float)
        # Remove nan values
        print('        Removing nan values and the corresponding coordinates...')
        nan_mask = ~np.isnan(values)
        print('        # non-nan values',np.count_nonzero(nan_mask))
        values   = values[nan_mask]
        sampled_coords_array = np.array(coords)
        coords   = sampled_coords_array[nan_mask].tolist()
        print('        final shape of coords is ',np.shape(coords))
        print('        final shape of values is ',np.shape(values))
        V        = skg.Variogram(coords, values,n_lags=200,bin_func='even')
        print('        creating variogram...')
        
        bins = V.bins*dx # convert from ineteger coordinates to physical coordinates (km)
        exp_variograms =  V.experimental

        matrix_for_saving = np.array([bins,exp_variograms]).T
        
        if VARIABLE[2]:
            data_file = 'experimental_variogram_RAMS_mask_'+MASK_AREA+'_criteria_'+MASK_TYPE+'_'+VARIABLE[0]+'_levtype_'+VARIABLE[2]+'_lev_'+str(int(VARIABLE[1]))+'_'+z_time+'_'+simulation+'_1_sample_d0'+DOMAIN+'.npy'
        else:
            data_file = 'experimental_variogram_RAMS_mask_'+MASK_AREA+'_criteria_'+MASK_TYPE+'_'+VARIABLE[0]+'_levtype_'+'None'+'_lev_'+'None'+'_'+z_time+'_'+simulation+'_1_sample_d0'+DOMAIN+'.npy'

        with open(data_file, 'wb') as f:
            np.save(f, matrix_for_saving)
            np.save(f, matrix_for_saving)

        print('        saving variogram data to ',data_file)
        print('    ------\n')
        
        if PLOT:
            plt.plot(bins,exp_variograms,label=simulation, color=COLORS[ii])
        
    if PLOT:
        if VARIABLE[2]:
            title_string = 'Variogram for masked ('+MASK_AREA+'; '+MASK_TYPE+') '+VARIABLE[0]+' at '+VARIABLE[2]+' level '+str(int(VARIABLE[1]))+' for d0'+DOMAIN+'\nmid-simulation'
        else:
            title_string = 'Variogram for masked ('+MASK_AREA+'; '+MASK_TYPE+') '+VARIABLE[0]+' for d0'+DOMAIN+'\nmid-simulation'    
        plt.title(title_string)
        plt.xlabel('distance (km)')
        plt.ylabel(VARIABLE[3])
        plt.legend()
        if VARIABLE[2]:
            filename = 'experimental_variogram_8_simulations_RAMS_mask_'+MASK_AREA+'_criteria_'+MASK_TYPE+'_'+VARIABLE[0]+'_levtype_'+VARIABLE[2]+'_lev_'+str(int(VARIABLE[1]))+'_1_sample_d0'+DOMAIN+'_mid-simulation.png'
        else:
            filename = 'experimental_variogram_8_simulations_RAMS_mask_'+MASK_AREA+'_criteria_'+MASK_TYPE+'_'+VARIABLE[0]+'_levtype_'+'None'+'_lev_'+'None'+'_1_sample_d0'+DOMAIN+'_mid-simulation.png'
        print('saving to file: ',filename)
        plt.savefig(filename,dpi=150)
        print('\n\n')

                                         
#save_variogram_masked_domain('not_near_storm','middle',variables[0], simulations, sample_size, domain, nsamples, colors, False)                                        
                                         
print('working on domain' ,domain)
#Running on the terminal in parallel
argument = []
for var in variables:
    for sim_time in ['start','end','middle']:
        argument = argument + [('near_storm',sim_time,var, simulations, sample_size, domain, nsamples, colors, False)]

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
    main(save_variogram_masked_domain, argument)
