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
import variogram_helper_functions
 
def save_variogram(WHICH_TIME, VARIABLE, SIMULATIONS, SAMPLE_SIZE, DOMAIN, COLORS, SAVE_NPY, PLOT):
    print('working on domain' ,domain)
    if DOMAIN=='1':
        dx = 1.6
    if DOMAIN=='2':
        dx=0.4
    if DOMAIN=='3':
        dx=0.1
    print('working on ',VARIABLE,'\n')
    if PLOT:
        fig    = plt.figure(figsize=(8,8))
        
    for ii,simulation in enumerate(SIMULATIONS):   
        print('\n    working on simulation: ',simulation)
        
        if simulation[7]=='W':
            model_name = 'WRF'
            microphysics_scheme = simulation[8]
        elif simulation[7]=='R':
            model_name = 'RAMS'
        else:
            print('!!!!!issues with identifying model_name!!!!!')
            break
        print('        model name: ',model_name)

        if model_name=='RAMS':   
            selected_fil = variogram_helper_functions.find_RAMS_file(SIMULATION=simulation,DOMAIN=DOMAIN,WHICH_TIME=WHICH_TIME)
            
        if model_name=='WRF':
            selected_fil =  variogram_helper_functions.find_WRF_file(SIMULATION=simulation,DOMAIN=DOMAIN,WHICH_TIME=WHICH_TIME)
        
        
        #### MAIN PART ####
        z, z_name, z_units, z_time = read_vars_WRF_RAMS.read_variable(selected_fil,VARIABLE[0],model_name,output_height=False,interpolate=VARIABLE[1]>-1,level=VARIABLE[1],interptype=VARIABLE[2])
        # read the file to get coordinates
        y_dim, x_dim     = np.shape(z)
        coords = variogram_helper_functions.produce_random_coords(x_dim,y_dim,SAMPLE_SIZE)                           
        # produce a random sample of coordinates
        nonnan_coords, nonnan_values = variogram_helper_functions.get_values_at_random_coords(z, coords)
        # get the values of the field at the random coordinates
        max_lag = np.sqrt(x_dim**2 + y_dim**2)/2.0# in grid points
        num_lag_classses = int(max_lag*dx/5.0)
        # create a variogram and save bin and variogram values in a matrix for saving
        _ , bins, exp_variogram, matrix_for_saving = variogram_helper_functions.make_variogram(nonnan_coords, nonnan_values,num_lag_classses,MAXLAG=max_lag,DX=dx)
        ##########################
        
        if SAVE_NPY:
            if VARIABLE[2]:
                data_file = 'experimental_variogram_'+simulation+'_'+VARIABLE[0]+'_levtype_'+VARIABLE[2]+'_lev_'+str(int(VARIABLE[1]))+'_'+z_time+'_'+simulation+'_1_sample_no_mask_d0'+DOMAIN+'.npy'
            else:
                data_file = 'experimental_variogram_'+simulation+'_'+VARIABLE[0]+'_levtype_'+'None'+'_lev_'+'None'+'_'+z_time+'_'+simulation+'_1_sample_no_mask_d0'+DOMAIN+'.npy'

            with open(data_file, 'wb') as f:
                np.save(f, matrix_for_saving)

            print('        saving variogram data to ',data_file)
            print('    ------\n')
        
        if PLOT:
            plt.plot(bins,exp_variogram,label=simulation, color=COLORS[ii])
        
        
    if PLOT:
        if VARIABLE[2]:
            title_string = 'Variogram for '+VARIABLE[0]+' at '+VARIABLE[2]+' level '+str(int(VARIABLE[1]))+' for d0'+DOMAIN+'\nmid-simulation'
        else:
            title_string = 'Variogram for '+VARIABLE[0]+' for d0'+DOMAIN+'\nmid-simulation'    
        plt.title(title_string)
        plt.xlabel('distance (km)')
        plt.ylabel(VARIABLE[3])
        #plt.yscale("log") 
        plt.legend()
        # if VARIABLE[2]:
        #     filename = 'experimental_variogram_'+simulation[0:6]+'_'+VARIABLE[0]+'_levtype_'+VARIABLE[2]+'_lev_'+str(int(VARIABLE[1]))+'_1_sample_no_mask_d0'+DOMAIN+'_mid-simulation.png'
        # else:
        #     filename = 'experimental_variogram_'+simulation[0:6]+'_'+VARIABLE[0]+'_levtype_'+'None'+'_lev_'+'None'+'_1_sample_no_mask_d0'+DOMAIN+'_mid-simulation.png'
        # print('saving to file: ',filename)
        if VARIABLE[2]:
            filename = 'experimental_variogram_all_RAMS_sims_'+VARIABLE[0]+'_levtype_'+VARIABLE[2]+'_lev_'+str(int(VARIABLE[1]))+'_1_sample_no_mask_d0'+DOMAIN+'_'+WHICH_TIME+'-simulation.png'
        else:
            filename = 'experimental_variogram_all_RAMS_sims_'+VARIABLE[0]+'_levtype_'+'None'+'_lev_'+'None'+'_1_sample_no_mask_d0'+DOMAIN+'_'+WHICH_TIME+'-simulation.png'
        print('saving to file: ',filename)
        plt.savefig(filename,dpi=150)
        print('\n\n')

#simulations_all=[['AUS1.1-WT','AUS1.1-WM','AUS1.1-R'],                                     \
#                 ['DRC1.1-WT','DRC1.1-WM','DRC1.1-R'],['PHI1.1-WT','PHI1.1-WM','PHI1.1-R'],\
#                 ['USA1.1-WT','USA1.1-WM','USA1.1-R'],['WPO1.1-WT','WPO1.1-WM','WPO1.1-R'],\
#                 ['PHI2.1-R'],['BRA1.1-R'],['BRA1.2-R'],['RSA1.1-R'],['ARG1.1-R'],['ARG1.2-R']]
simulations=['AUS1.1-R','DRC1.1-R','PHI1.1-R','USA1.1-R','WPO1.1-R','PHI2.1-R','BRA1.1-R','BRA1.2-R','RSA1.1-R','ARG1.1-R','ARG1.2-R']

#['ARG1.2-WT','ARG1.2-WM','ARG1.2-R']# ARG1.2-R has no G1 folder

#'PHI1.1-R','PHI2.1-R','WPO1.1-R','BRA1.1-R','USA1.1-R','DRC1.1-R','AUS1.1-R','ARG1.1-R_old']
domain='3'
variables = [['THETAV', 0, 'model', '$Theta_{v-sfc}^{2} (K^{2})$'],['THETAV', 0, 'model', '$Theta_{v-sfc}^{2} (K^{2})$'],\
             ['QV', 0, 'model', '$Qvapor_{sfc}^{2} (kg^{2}kg^{-2})$'],\
             ['W', 750, 'pressure', '$W_{750}^{2} (m^{2}s^{-2})$'],['WSPD', 0, 'model', '$WSPD_{sfc}^{2} (m^{2}s^{-2})$'],\
             ['SHF', -999, None, '$SHF^{2} (Wm^{-2})$']           ,['LHF', -999, None, '$LHF^{2} (Wm^{-2})$']]
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

#colors    =  ['#000000','#E69F00','#56B4E9','#009E73','#F0E442','#0072B2','#D55E00','#CC79A7']
colors     = ['#000000','#377eb8', '#56B4E9','#ff7f00', '#4daf4a','#f781bf', '#a65628', '#984ea3','#999999', '#e41a1c', '#dede00']

# simulations_all=[['AUS1.1-WT','AUS1.1-WM','AUS1.1-R']] # use for testing
#variables = [['LHF', -999, None, '$LHF^{2} (Wm^{-2})$']]# use for testing
# for simulations in simulations_all:
# for variable in variables:
#     save_variogram('middle', variable, simulations, 150, domain, colors, False, True)
#     print('=====================================================================================\n\n\n\n')

#Running on the terminal in parallel
argument = []


for variable in variables:
    argument = argument + [('middle', variable, simulations, 15000, domain, colors, False, True)]

# for simulations in simulations_all:
#     for variable in variables:
#         argument = argument + [('middle', variable, simulations, 15000, domain, colors, False, True)]

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
    main(save_variogram, argument)
