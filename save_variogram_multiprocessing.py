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
import matplotlib

matplotlib.rcParams["font.family"] = "Roboto"
matplotlib.rcParams["font.sans-serif"] = ["Roboto"]  # For non-unicode text
#matplotlib.rcParams['axes.facecolor'] = [0.9,0.9,0.9]
matplotlib.rcParams['axes.labelsize'] = 18
matplotlib.rcParams['axes.titlesize'] = 18
matplotlib.rcParams['xtick.labelsize'] = 18
matplotlib.rcParams['ytick.labelsize'] = 18
matplotlib.rcParams['legend.fontsize'] = 18
#matplotlib.rcParams['legend.facecolor'] = 'w'
 
def save_variogram(WHICH_TIME, VARIABLE, SIMULATION, CONDITION_INFO, SAMPLE_SIZE, DOMAIN, SAVE_CSV):
    print('working on domain' ,domain)
    print('working on ',VARIABLE)
    print('working on simulation: ',SIMULATION)
    
    # provide grid spacings in km to multiply with the variogram bin distances
    if DOMAIN=='1':
        dx = 1.6 # km
    if DOMAIN=='2':
        dx=0.4   # km
    if DOMAIN=='3':
        dx=0.1   # km
        
    
    # which model are we working on; need this for the read_RAMS_WRF_data_file
    if SIMULATION[7]=='W':
        model_name = 'WRF'
        microphysics_scheme = SIMULATION[8]
    elif SIMULATION[7]=='R':
        model_name = 'RAMS'
    else:
        print('!!!!!issues with identifying model_name!!!!!')

    print('        model name: ',model_name)

    # grab the file needed
    if model_name=='RAMS':   
        selected_fil = variogram_helper_functions.find_RAMS_file(SIMULATION=SIMULATION,DOMAIN=DOMAIN,WHICH_TIME=WHICH_TIME)

    if model_name=='WRF':
        selected_fil =  variogram_helper_functions.find_WRF_file(SIMULATION=SIMULATION,DOMAIN=DOMAIN,WHICH_TIME=WHICH_TIME)

    timestring = variogram_helper_functions.get_time_from_RAMS_file(selected_fil)[0]
    #### MAIN PART ####
    z, z_name, z_units, z_time = read_vars_WRF_RAMS.read_variable(selected_fil,VARIABLE[0],model_name,output_height=False,interpolate=VARIABLE[1]>-1,level=VARIABLE[1],interptype=VARIABLE[2])
    # read the file to get coordinates
    y_dim, x_dim     = np.shape(z)

    if CONDITION_INFO[0]=='environment':
        print('getting random coordinates over ',CONDITION_INFO[0],' points with threshold ',CONDITION_INFO[1])
        print('        getting total condensate for conditional variogram')
        if VARIABLE[1]<0:
            conditional_field, _, _, _ = read_vars_WRF_RAMS.read_variable(selected_fil,'QTC',model_name,output_height=False,interpolate=True,level=0,interptype='model')
        else:
            conditional_field, _, _, _ = read_vars_WRF_RAMS.read_variable(selected_fil,'QTC',model_name,output_height=False,interpolate=VARIABLE[1]>-1,level=VARIABLE[1],interptype=VARIABLE[2])
        print('        min, max for the condensate field is ',np.min(conditional_field),' ',np.max(conditional_field))
        coords = variogram_helper_functions.produce_random_coords_conditional(SAMPLE_SIZE, conditional_field, CONDITION_STATEMENT=lambda x: x < CONDITION_INFO[1])
    if CONDITION_INFO[0]=='storm': 
        print('getting random coordinates over ',CONDITION_INFO[0],' points with threshold ',CONDITION_INFO[1])
        print('        getting total condensate for conditional variogram')
        if VARIABLE[1]<0:
            conditional_field, _, _, _ = read_vars_WRF_RAMS.read_variable(selected_fil,'QTC',model_name,output_height=False,interpolate=True,level=0,interptype='model')
        else:
            conditional_field, _, _, _ = read_vars_WRF_RAMS.read_variable(selected_fil,'QTC',model_name,output_height=False,interpolate=VARIABLE[1]>-1,level=VARIABLE[1],interptype=VARIABLE[2])
        print('        min, max for the condensate field is ',np.min(conditional_field),' ',np.max(conditional_field))
        coords = variogram_helper_functions.produce_random_coords_conditional(SAMPLE_SIZE, conditional_field, CONDITION_STATEMENT=lambda x: x >= CONDITION_INFO[1])
    if CONDITION_INFO[0]=='all':
        print('getting random coordinates over ',CONDITION_INFO[0],' points')
        coords = variogram_helper_functions.produce_random_coords(x_dim,y_dim,SAMPLE_SIZE)   

    # produce a random sample of coordinates
    nonnan_coords, nonnan_values = variogram_helper_functions.get_values_at_random_coords(z, coords)
    # get the values of the field at the random coordinates
    max_lag = np.sqrt(x_dim**2 + y_dim**2)/2.0# in grid points
    num_lag_classses = int(max_lag*dx/5.0)
    # create a variogram and save bin and variogram values in a matrix for saving
    V , bins, exp_variogram = variogram_helper_functions.make_variogram(nonnan_coords, nonnan_values,num_lag_classses,MAXLAG=max_lag,DX=dx)
    bins_middle_points, counts, widths = variogram_helper_functions.retrieve_histogram(V,DX=dx)
    ##########################

    if SAVE_CSV:
        savecsv = '/home/isingh/code/variogram_data/'+SIMULATION+'/'+'G'+DOMAIN+'/CSVs'
        if not os.path.exists(savecsv):
            os.makedirs(savecsv)
        if VARIABLE[2]:
            data_file = savecsv+'/experimental_variogram_'+SIMULATION+'_G'+DOMAIN+'_'+CONDITION_INFO[0]+'_points_threshold_'+str(CONDITION_INFO[1])+'_'+VARIABLE[0]+'_levtype_'+VARIABLE[2]+'_lev_'+str(int(VARIABLE[1]))+'_'+z_time+'.csv'
        else:
            data_file = savecsv+'/experimental_variogram_'+SIMULATION+'_G'+DOMAIN+'_'+CONDITION_INFO[0]+'_points_threshold_'+str(CONDITION_INFO[1])+'_'+VARIABLE[0]+'_levtype_'+'None'+'_lev_'+'None'+'_'+z_time+'.csv'

        data_matrix = np.column_stack((bins, counts, widths, exp_variogram))
        np.savetxt(data_file, data_matrix, delimiter=',', header='bins,counts,widths,exp_variogram', comments='')

        print('        saving variogram data to ',data_file)
        #print('    ------\n')


#------------------------------------------------------------------------------------------------------------------------

variables =[ ['QV', 0, 'model', '$q_{v}$', '$kg^{2}kg^{-2}$']     ,['THETAV', 0, 'model','${\Theta}_{v}$', '$K^{2}$'],\
             ['QV', 750, 'pressure', '$q_{v}$', '$kg^{2}kg^{-2}$'],['THETAV', 750, 'pressure','${\Theta}_{v}$', '$K^{2}$'],\
             ['W', 750, 'pressure', '$w$', '$m^{2}s^{-2}$']       ,['WSPD', 0, 'model', '$wspd$', '$m^{2}s^{-2}$'],\
             ['SHF', -999, None, '$SHF$', '$W^{2}m^{-4}$']        ,['LHF', -999, None, '$LHF$' , '$W^{2}m^{-4}$'],\
             ['ITC',-999, None, '$ITC$', '$mm^{2}$']]

colors      = ['#000000','#377eb8', '#56B4E9','#ff7f00', '#4daf4a','#f781bf', '#a65628', '#984ea3','#999999', '#e41a1c', '#dede00']
simulations = ['AUS1.1-R','DRC1.1-R','PHI1.1-R','USA1.1-R','WPO1.1-R','PHI2.1-R','BRA1.1-R','BRA1.2-R','RSA1.1-R','ARG1.1-R','ARG1.2-R']
#simulations_wrf=['AUS1.1-WT','AUS1.1-WM','DRC1.1-WT','DRC1.1-WM','USA1.1-WT','USA1.1-WM','PHI1.1-WT','PHI1.1-WM','WPO1.1-WT','WPO1.1-WM']
domain='1'
thresholds = [0.0000001,0.000001,0.00001,0.0001]

# for simulation in simulations:
#     for variable in variables:
#         for threshold in thresholds:
#             for partition in ['storm','environment','all']:
#                 save_variogram('middle', variable, simulation, [partition, threshold] , 15000, domain, True)
#                 print('=====================================================================================\n\n\n\n')

                
# Running on the terminal in parallel
argument = []


# for variable in variables:
#     argument = argument + [('middle', variable, simulations, 15000, domain, colors, False, True)]

for simulation in simulations:
    for variable in variables:
        for threshold in thresholds:
            for partition in ['storm','environment','all']:
                #save_variogram('middle', variable, simulation, [partition, threshold] , 15000, domain, True)
                #print('=====================================================================================\n\n\n\n')
                argument = argument + [('middle', variable, simulation, [partition, threshold] , 15000, domain, True)]

print('length of argument is: ',len(argument))


# # ############################### FIRST OF ALL ################################
cpu_count1 = 20 #cpu_count()
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
