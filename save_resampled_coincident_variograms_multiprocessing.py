
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
from pathlib import Path


def get_time_from_RAMS_file(INPUT_FILE):
    cur_time = os.path.split(INPUT_FILE)[1][4:21] # Grab time string from RAMS file
    pd_time = pd.to_datetime(cur_time[0:10]+' '+cur_time[11:13]+":"+cur_time[13:15]+":"+cur_time[15:17])
    return pd_time.strftime('%Y-%m-%d %H:%M:%S'), pd_time.strftime('%Y%m%d%H%M%S'), pd_time

def find_closest_datetime_index(datetime_list, target_datetime):
    """
    Find the index of the closest datetime in the datetime_list to the target_datetime.
    """
    closest_datetime = min(datetime_list, key=lambda x: abs(x - target_datetime))
    closest_index = datetime_list.index(closest_datetime)
    return closest_index
   
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

def grab_intersection_gbig_gsmall(VARIABLE,RAMS_G1_or_G2_FILE,RAMS_G3_FILE):
    z, z_name, z_units, z_time = read_vars_WRF_RAMS.read_variable(RAMS_G1_or_G2_FILE,VARIABLE[0],'RAMS',output_height=False,interpolate=VARIABLE[1]>-1,level=VARIABLE[1],interptype=VARIABLE[2])
    #print(np.min(z))
    #print(np.max(z))
    #z2, z_name2, z_units2, z_time2 = read_vars_WRF_RAMS.read_variable(RAMS_G3_FILE,VARIABLE[0],'RAMS',output_height=False,interpolate=VARIABLE[1]>-1,level=VARIABLE[1],interptype=VARIABLE[2])
    print('        done getting the variable ',VARIABLE[0],' with shape: ',np.shape(z),'\n')
    print('        subsetting the larger domain...\n')
    # read the variables for which you want the variogram
    ds_big   = xr.open_dataset(RAMS_G1_or_G2_FILE,engine='h5netcdf',phony_dims='sort')[['GLAT','GLON']]
    ds_small = xr.open_dataset(RAMS_G3_FILE,engine='h5netcdf',phony_dims='sort')[['GLAT','GLON']]
    dim1, dim2 = ds_big.GLAT.dims
    #print(ds_big)
    #print(ds_small)
    #ds_big = ds_big.rename_dims({'phony_dim_0': 'y','phony_dim_1': 'x'})
    #ds_small = ds_small.rename_dims({'phony_dim_0': 'y','phony_dim_1': 'x'})
    min_lat_big = ds_big.GLAT.min().values
    max_lat_big = ds_big.GLAT.max().values
    min_lon_big = ds_big.GLON.min().values
    max_lon_big = ds_big.GLON.max().values
    print('        min and max lat for big domain = ',min_lat_big,' ',max_lat_big)
    print('        min and max lon for big domain = ',min_lon_big,' ',max_lon_big)
    print('        ----')
    min_lat_small = ds_small.GLAT.min().values
    max_lat_small = ds_small.GLAT.max().values
    min_lon_small = ds_small.GLON.min().values
    max_lon_small = ds_small.GLON.max().values
    print('        min and max lat for small domain = ',min_lat_small,' ',max_lat_small)
    print('        min and max lon for small domain = ',min_lon_small,' ',max_lon_small)
    print('        ----')
    #subset by lat/lon - used so only region covered by inner grid is compared
    ds = xr.Dataset({VARIABLE[0]: xr.DataArray(data   = z,  dims   = [dim1,dim2])})
    ds = ds.assign(GLAT=ds_big.GLAT)
    ds = ds.assign(GLON=ds_big.GLON)
    #print(ds)
    ds = ds.where((ds.GLAT>=min_lat_small) & (ds.GLAT<=max_lat_small) & (ds.GLON>=min_lon_small) & (ds.GLON<=max_lon_small), drop=True)
    #print(ds)
    min_lat = ds.GLAT.min().values
    max_lat = ds.GLAT.max().values
    min_lon = ds.GLON.min().values
    max_lon = ds.GLON.max().values
    
    print('        min and max lat for modified domain = ',min_lat,' ',max_lat)
    print('        min and max lon for modified domain = ',min_lon,' ',max_lon)
    print('        ----')
    
    #print(ds)
    print('        shape of small domain: ',np.shape(ds_small.GLAT))
    print('        shape of big domain: ',np.shape(ds_big.GLAT))
    print('        shape of modified domain: ',np.shape(ds.GLAT))
    #return z, z_name, z_units, z_time
    return ds.variables[VARIABLE[0]].values, z_name, z_units, z_time

def produce_random_coords(X_DIM,Y_DIM,SAMPLE_SIZE):
    print('getting a random sample of coordinates...')
    print('        shape of the arrays is ',Y_DIM,'x',X_DIM)
    x      = np.arange(0,X_DIM)
    y      = np.arange(0,Y_DIM)
    # # full coordinate arrays
    xx, yy = np.meshgrid(x, y)
        
    coords_tuples_2d = np.vstack(([yy.T], [xx.T])).T
    print('        shape of combined coords matrix: ',np.shape(coords_tuples_2d))
    coords_all = coords_tuples_2d.reshape(-1, 2).tolist()
    print('        shape of 1d list of coords: ',np.shape(coords_all))
    
    if SAMPLE_SIZE>=(X_DIM*Y_DIM):
        print('        sample larger than the population; choosing all points')
        coords = coords_all 
    else:
        coords = random.sample(coords_all,SAMPLE_SIZE)
        
    return coords
    
def get_values_at_random_coords(TWOD_FIELD, COORDS):
    print('getting values at the chosen coordinates...')
    print('        got the data... min = ',np.nanmin(TWOD_FIELD),' max = ',np.nanmax(TWOD_FIELD))
    print('        percentage of nans is ',np.count_nonzero(np.isnan(TWOD_FIELD))/len(TWOD_FIELD.flatten()))
    print('        choosing '+str(len(COORDS))+' random points...')
    print('        get field values from these points...')
    values = np.fromiter((TWOD_FIELD[c[0], c[1]] for c in COORDS), dtype=float)
    # Remove nan values
    print('        Removing nan values and the corresponding coordinates...')
    nan_mask = ~np.isnan(values)
    print('        # non-nan values',np.count_nonzero(nan_mask))
    values   = values[nan_mask]
    sampled_coords_array = np.array(COORDS)
    coords   = sampled_coords_array[nan_mask].tolist()
    print('        final shape of coords is ',np.shape(coords))
    print('        final shape of values is ',np.shape(values))
    return coords, values

def make_variogram(COORDS, VALUES, NBINS, DX):
    print('        creating variogram...')
    V        = skg.Variogram(COORDS, VALUES,n_lags=NBINS,bin_func='even')
    bins = V.bins*DX # convert from ineteger coordinates to physical coordinates (km)
    exp_variogram =  V.experimental
    matrix_for_saving = np.array([bins,exp_variogram]).T
    return bins, exp_variogram, matrix_for_saving

def save_resampled_coincident_variograms(WHICH_TIME, VARIABLE, SIMULATIONS, SAMPLE_SIZE, DOMAINS, COINCIDENT, NSAMPLES, COLORS, PLOT):
    
    print('working on ',VARIABLE,'\n')

    for ii,simulation in enumerate(SIMULATIONS): 
        print('  <<working on simulation: ',simulation,'>>\n')
        if PLOT:
            fig    = plt.figure(figsize=(8,8))
        for DOMAIN in DOMAINS:
            print('    <<working on domain: ',DOMAIN,'>>\n')
            if DOMAIN==1:
                dx=1.6
            if DOMAIN==2:
                dx=1.6
            if DOMAIN==3:
                dx=1.6
            
            if simulation=='PHI2.1-R':
                rams_g3_files=sorted(glob.glob('/monsoon/MODEL/LES_MODEL_DATA/V0/'+simulation+'-V0/G'+'3'+'/out_30s/Lite/'+'a-L-*g3.h5'))# CSU machine
            else:
                rams_g3_files=sorted(glob.glob('/monsoon/MODEL/LES_MODEL_DATA/V0/'+simulation+'-V0/G'+'3'+'/out_30s/'+'a-L-*g3.h5'))# CSU machine
            print('        total # files = ',len(rams_g3_files))
            print('        first file is ',rams_g3_files[0])
            print('        last file is ',rams_g3_files[-1])

            if WHICH_TIME=='start':
                rams_g3_fil    = rams_g3_files[0]
                print('        choosing the start file: ',rams_g3_fil)
            if WHICH_TIME=='middle':
                rams_g3_fil    = rams_g3_files[int(len(rams_g3_files)/2)]
                print('        choosing the middle file: ',rams_g3_fil)
            if WHICH_TIME=='end':
                rams_g3_fil    = rams_g3_files[-1]
                print('        choosing the end file: ',rams_g3_fil)

            if DOMAIN==1 or DOMAIN ==2:
                print('searching for domain ',DOMAIN,' file for the same time in the directory: ',\
                      '/monsoon/MODEL/LES_MODEL_DATA/V0/'+simulation+'-V0/G'+str(DOMAIN)+'/out/')
                g3_time = get_time_from_RAMS_file(rams_g3_fil)[2]
                print('        time in G3 file ',g3_time)
                if simulation=='PHI2.1-R':
                    rams_g1_or_g2_files=sorted(glob.glob('/monsoon/MODEL/LES_MODEL_DATA/V0/'+simulation+'-V0/G3/out_30s/Lite/'+'a-L-*g'+str(DOMAIN)+'.h5'))# CSU machine
                else:
                    rams_g1_or_g2_files=sorted(glob.glob('/monsoon/MODEL/LES_MODEL_DATA/V0/'+simulation+'-V0/G3/out_30s/'+'a-L-*g'+str(DOMAIN)+'.h5'))# CSU machine
                print('        found total ',len(rams_g1_or_g2_files),' files for domain ',DOMAIN,' for this simulation')
                list_of_times = [get_time_from_RAMS_file(fil)[2] for fil in rams_g1_or_g2_files]
                ind_g1_or_g2_file_index = find_closest_datetime_index(list_of_times, g3_time)
                rams_larger_grid_file = rams_g1_or_g2_files[ind_g1_or_g2_file_index]
                print('        found the file ',rams_larger_grid_file)


            if DOMAIN<3:
                print('Domain intersection portion...')
                z, z_name, z_units, z_time = grab_intersection_gbig_gsmall(VARIABLE,rams_larger_grid_file,rams_g3_fil)
            else:
                z, z_name, z_units, z_time = read_vars_WRF_RAMS.read_variable(rams_g3_fil,VARIABLE[0],'RAMS',output_height=False,interpolate=VARIABLE[1]>-1,level=VARIABLE[1],interptype=VARIABLE[2])

            
            if DOMAIN==2:
                # Reshape and average to reduce size by 4 in each dimension
                print('Resampling the domain 2 array...')
                new_shape = (z.shape[0] // 4, z.shape[1] // 4)
                z_temp = z[:new_shape[0]*4, :new_shape[1]*4].reshape(new_shape[0], 4, new_shape[1], 4)
                print('        shape of the temporary reshaped array is ',z_temp.shape)
                z = np.nanmean(z_temp,axis=(1, 3))
                print('        shape of the smaller array is ',z.shape)
            if DOMAIN==3:
                print('        Resampling the domain 3 array...')
                new_shape = (z.shape[0] // 16, z.shape[1] // 16)
                z_temp = z[:new_shape[0]*16, :new_shape[1]*16].reshape(new_shape[0], 16, new_shape[1], 16)
                print('        shape of the temporary reshaped array is ',z_temp.shape)
                z = np.nanmean(z_temp, axis=(1, 3))
                print('        shape of the smaller array is ',z.shape)
                
            # read the file to get coordinates
            y_dim, x_dim     = np.shape(z)
            coords = produce_random_coords(x_dim,y_dim,SAMPLE_SIZE)                           
            # produce a random sample of coordinates
            nonnan_coords, nonnan_values = get_values_at_random_coords(z, coords)
            # get the values of the field at the random coordinates
            bins, exp_variogram, matrix_for_saving = make_variogram(nonnan_coords, nonnan_values, 200, dx)
            # create a variogram and save bin and variogram values in a matrix for saving

            if VARIABLE[2]:
                data_file = 'experimental_resampled_variogram_RAMS_coincident_'+VARIABLE[0]+'_levtype_'+VARIABLE[2]+'_lev_'+str(int(VARIABLE[1]))+'_'+z_time+'_'+simulation+'_1_sample_no_mask_d0'+str(DOMAIN)+'.npy'
            else:
                data_file = 'experimental_resampled_variogram_RAMS_coincident_'+VARIABLE[0]+'_levtype_'+'None'+'_lev_'+'None'+'_'+z_time+'_'+simulation+'_1_sample_no_mask_d0'+str(DOMAIN)+'.npy'

            with open(data_file, 'wb') as f:
                np.save(f, matrix_for_saving)

            print('        saving variogram data to ',data_file)
            print('        ------\n')

            if DOMAIN==1:
                linestyle = '-'
            if DOMAIN==2:
                linestyle = '--'
            if DOMAIN==3:
                linestyle = ':'

            if PLOT:
                plt.plot(bins[bins<150],exp_variogram[bins<150],label=simulation+' d0'+str(DOMAIN), color=COLORS[ii],linestyle=linestyle)

        if PLOT:
            if VARIABLE[2]:
                title_string = 'Variogram (resampled) for '+VARIABLE[0]+' at '+VARIABLE[2]+' level '+str(int(VARIABLE[1]))+'\nat mid-simulation'
            else:
                title_string = 'Variogram (resampled) for '+VARIABLE[0]+'\nat mid-simulation'    
            plt.title(title_string)
            plt.xlabel('distance (km)')
            plt.ylabel(VARIABLE[3])
            plt.legend()
            if VARIABLE[2]:
                filename = 'experimental_resampled_variogram_RAMS_'+simulation+'_coincident_'+VARIABLE[0]+'_levtype_'+VARIABLE[2]+'_lev_'+str(int(VARIABLE[1]))+'_1_sample_no_mask_mid-simulation.png'
            else:
                filename = 'experimental_resampled_variogram_RAMS_'+simulation+'_coincident_'+VARIABLE[0]+'_levtype_'+'None'+'_lev_'+'None'+'_1_sample_no_mask_mid-simulation.png'
            print('        saving plot to file: ',filename)
            plt.savefig(filename,dpi=150)
            #print('\n\n')
        print('------------------------------------------------------------------------\n')

# PARAMETERS
simulations=['WPO1.1-R']#['BRA1.1-R','PHI2.1-R','DRC1.1-R','PHI1.1-R','WPO1.1-R','USA1.1-R','AUS1.1-R']
domains=[1,2,3]
nsamples=1
sample_size = 15000
variables = [['THETA', 0, 'model', '$Theta_{sfc}^{2} (K^{2})$'], ['QV', 0, 'model', '$Qvapor_{sfc}^{2} (kg^{2}kg^{-2})$'],\
             ['W', 750, 'pressure', '$W_{750}^{2} (m^{2}s^{-2})$']]#,['MCAPE', -999, None, '$MCAPE^{2} (J^{2}kg^{-2})$'] ]             
# [['TOP_SOIL_MOISTURE', -999, None, '$SM^{2} (m^{3} m^{3})$'], ['LHF', -999, None, '$LHF^{2} (Wm^{-2})$'],\
#              ['SHF', -999, None, '$SHF^{2} (Wm^{-2})$'],\
#              ['Tk', 0, 'model', '$T_{sfc}^{2} (K^{2})$']             , ['THETA', 0, 'model', '$Theta_{sfc}^{2} (K^{2})$'],\
#              ['QV', 0, 'model', '$Qvapor_{sfc}^{2} (kg^{2}kg^{-2})$'], ['RH', 0, 'model', '$RH_{sfc}^{2} (percent^{2})$'],\
#              ['U', 0, 'model', '$U_{sfc}^{2} (m^{2}s^{-2})$']        , ['V', 0, 'model', '$V_{sfc}^{2} (m^{2}s^{-2})$'],\
#              ['WSPD', 0, 'model', '$WSPD_{sfc}^{2} (m^{2}s^{-2})$']  , ['W', 0, 'model', '$W_{sfc}^{2} (m^{2}s^{-2})$'],\
#              ['MCAPE', -999, None, '$MCAPE^{2} (J^{2}kg^{-2})$']     , ['MCIN', -999, None, '$MCIN^{2} (J^{2}kg^{-2})$'], \
#              ['Tk', 750, 'pressure', '$T_{750}^{2} (K^{2})$']        , ['THETA', 750, 'pressure', '$Theta_{750}^{2} (K^{2})$'],\
#              ['QV', 750, 'pressure', '$Qvapor_{750}^{2} (kg^{2}kg^{-2})$'], ['RH', 750, 'pressure', '$RH_{750}^{2} (percent^{2})$'],\
#              ['U', 750, 'pressure', '$U_{750}^{2} (m^{2}s^{-2})$']   , ['V', 750, 'pressure', '$V_{750}^{2} (m^{2}s^{-2})$'],\
#              ['WSPD', 750, 'pressure', '$WSPD_{750}^{2} (m^{2}s^{-2})$'], ['W', 750, 'pressure', '$W_{750}^{2} (m^{2}s^{-2})$'],\
#              ['Tk', 500, 'pressure', '$T_{500}^{2} (K^{2})$']        , ['THETA', 500, 'pressure', '$Theta_{500}^{2} (K^{2})$'],\
#              ['QV', 500, 'pressure', '$Qvapor_{500}^{2} (kg^{2}kg^{-2})$'], ['RH', 500, 'pressure', '$RH_{500}^{2} (percent^{2})$'],\
#              ['U', 500, 'pressure', '$U_{500}^{2} (m^{2}s^{-2})$']   , ['V', 500, 'pressure', '$V_{500}^{2} (m^{2}s^{-2})$'],\
#              ['WSPD', 500, 'pressure', '$WSPD_{500}^{2} (m^{2}s^{-2})$'], ['W', 500, 'pressure', '$W_{500}^{2} (m^{2}s^{-2})$'],\
#              ['Tk', 200, 'pressure', '$T_{200}^{2} (K^{2})$']        , ['THETA', 200, 'pressure', '$Theta_{200}^{2} (K^{2})$'],\
#              ['QV', 200, 'pressure', '$Qvapor_{200}^{2} (kg^{2}kg^{-2})$'], ['RH', 200, 'pressure', '$RH_{200}^{2} (percent^{2})$'],\
#              ['U', 200, 'pressure', '$U_{200}^{2} (m^{2}s^{-2})$']   , ['V', 200, 'pressure', '$V_{200}^{2} (m^{2}s^{-2})$'], \
#              ['WSPD', 200, 'pressure', '$WSPD_{200}^{2} (m^{2}s^{-2})$'], ['W', 200, 'pressure', '$W_{200}^{2} (m^{2}s^{-2})$']]
colors    =  ['#000000','#E69F00','#56B4E9','#009E73','#F0E442','#0072B2','#D55E00','#CC79A7']

# for variable in variables:
#     save_resampled_coincident_variograms('middle', variable, simulations, 15000, domains, False, 1, colors, True)
#     print('=====================================================================================\n\n\n\n')
          
          
#Running on the terminal in parallel
argument = []
for variable in variables:
    argument = argument + [('middle', variable, simulations, 15000, domains, False, 1, colors, True)]

print('length of argument is: ',len(argument))

# # ############################### FIRST OF ALL ################################
cpu_count1 = 18 #cpu_count()
print('number of cpus: ',cpu_count1)
# # #############################################################################

def main(FUNCTION, ARGUMENT):
    start_time = time.perf_counter()
    with Pool(processes = (cpu_count1-1)) as pool:
        data = pool.starmap(FUNCTION, ARGUMENT)
    finish_time = time.perf_counter()
    print(f"Program finished in {finish_time-start_time} seconds")
      
if __name__ == "__main__":
    main(save_resampled_coincident_variograms, argument)
