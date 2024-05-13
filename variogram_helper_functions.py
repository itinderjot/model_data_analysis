

import numpy as np
import matplotlib.pyplot as plt
import sys
import glob
import os
import time
import h5py
import hdf5plugin
import pandas as pd
import xarray as xr
from netCDF4 import Dataset
import random
import skgstat as skg
from pprint import pprint
import seaborn as sns
import matplotlib.ticker as ticker
import read_vars_WRF_RAMS
from libpysal.weights.distance import DistanceBand
import libpysal 
from esda.moran import Moran

def find_WRF_file(SIMULATION,DOMAIN,WHICH_TIME):
    print('/monsoon/MODEL/LES_MODEL_DATA/V0/'+SIMULATION+'-V0/G'+DOMAIN+'/wrfout*')
    wrf_files=sorted(glob.glob('/monsoon/MODEL/LES_MODEL_DATA/V0/'+SIMULATION+'-V0/G'+DOMAIN+'/wrfout*'))# CSU machine
    print('        total # files = ',len(wrf_files))
    print('        first file is ',wrf_files[0])
    print('        last file is ',wrf_files[-1])
    if WHICH_TIME=='start':
        selected_fil    = wrf_files[0]
    if WHICH_TIME=='middle':
        selected_fil    = wrf_files[int(len(wrf_files)/2)]
    if WHICH_TIME=='end':
        selected_fil    = wrf_files[-1]
    print('        choosing the middle file: ',selected_fil)

    return selected_fil

def find_RAMS_file(SIMULATION, DOMAIN, WHICH_TIME):
    if DOMAIN=='1' or DOMAIN =='2':
        try:
            rams_files=sorted(glob.glob('/monsoon/MODEL/LES_MODEL_DATA/V0/'+SIMULATION+'-V0/G'+DOMAIN+'/out/'+'a-A-*g'+DOMAIN+'.h5'))# CSU machine
        except (IndexError, FileNotFoundError):
            print("No files found or folder does not exist. Trying a different folder...")
            # Change directory to a different folder and try again
            if os.path.isdir('/monsoon/MODEL/LES_MODEL_DATA/V0/'+SIMULATION+'-V0/G3/out_30s/'+'a-L-*g1.h5'):
                return find_RAMS_file(SIMULATION, DOMAIN, WHICH_TIME)  # Recursive call with a different folder
            else:
                print("Alternate folder does not exist. Exiting function.")
        
    if DOMAIN=='3':
        rams_files=sorted(glob.glob('/monsoon/MODEL/LES_MODEL_DATA/V0/'+SIMULATION+'-V0/G'+DOMAIN+'/out_30s/'+'a-L-*g3.h5'))# CSU machine
    
    print('        total # files = ',len(rams_files))
    print('        first file is ',rams_files[0])
    print('        last file is ',rams_files[-1])
    if WHICH_TIME=='start':
        selected_fil    = rams_files[0]
    if WHICH_TIME=='middle':
        selected_fil    = rams_files[int(len(rams_files)/2)]
    if WHICH_TIME=='end':
        selected_fil    = rams_files[-1]
    print('        choosing the middle file: ',selected_fil)

    return selected_fil
   
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
    
def produce_random_coords(X_DIM,Y_DIM,SAMPLE_SIZE,COORDS_RETURN_TYPE='list'):
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
    
    if COORDS_RETURN_TYPE=='tuple':
        coords_all =  [tuple(sublist) for sublist in coords_all]
    
    if SAMPLE_SIZE>=(X_DIM*Y_DIM):
        print('        sample = or > than the population; choosing all points')
        coords = coords_all 
    else:
        coords = random.sample(coords_all,SAMPLE_SIZE)
        
    return coords

def produce_random_coords_conditional(SAMPLE_SIZE,TWOD_CONDITIONAL_FIELD, CONDITION_STATEMENT=lambda x: x != np.nan,COORDS_RETURN_TYPE='list'):
    print('getting a random sample of coordinates where ',CONDITION_STATEMENT)
    print('        shape of the 2D condition field is ',np.shape(TWOD_CONDITIONAL_FIELD))
    
    def indices_where_condition_met(array, condition):
        indices = np.where(condition(array))
        return list(zip(indices[0], indices[1]))

    # Get indices where condition is met
    coords_all = indices_where_condition_met(TWOD_CONDITIONAL_FIELD, CONDITION_STATEMENT)
    print('length of all coordinates where condition is met is ',len(coords_all),' about ',int(len(coords_all)*100.0/TWOD_CONDITIONAL_FIELD.size), ' percent of the total grid points')

    if COORDS_RETURN_TYPE=='list':
        coords_all =  [list(sublist) for sublist in coords_all]
    if COORDS_RETURN_TYPE=='tuple':
        pass

    print('        shape of 1d list of coords: ',np.shape(coords_all))
    
    if SAMPLE_SIZE>=(np.shape(TWOD_CONDITIONAL_FIELD)[0]*np.shape(TWOD_CONDITIONAL_FIELD)[1]):
        print('        sample = or > than the population; choosing all points')
        coords = coords_all 
    if SAMPLE_SIZE>len(coords_all):
        coords = coords_all 
    else:
        coords = random.sample(coords_all,SAMPLE_SIZE)
    return coords
    
def get_values_at_random_coords(TWOD_FIELD, COORDS, COORDS_RETURN_TYPE='list'):
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
    
    if COORDS_RETURN_TYPE=='tuple':
        coords =  [tuple(sublist) for sublist in coords]
        
    print('        final shape of coords is ',np.shape(coords))
    print('        final shape of values is ',np.shape(values))
    return coords, values

def make_variogram(COORDS, VALUES, NBINS, MAXLAG, DX=1.0, BIN_FUNCTION='even',ESTIMATOR='matheron'):
    """
    Estimator options:
    1. matheron [Matheron, default]
    2. cressie [Cressie-Hawkins]
    3. dowd [Dowd-Estimator]
    4. genton [Genton]
    5. minmax [MinMax Scaler]
    6. entropy [Shannon Entropy]
    """
    print('        creating variogram...')
    print('        MAXLAG= ',MAXLAG,'grid points')
    V        = skg.Variogram(COORDS, VALUES,n_lags=NBINS,maxlag = MAXLAG, bin_func=BIN_FUNCTION,estimator=ESTIMATOR)
    bins     = V.bins*DX # convert from integer coordinates to physical coordinates (km)
    #print('        upper edges of bins: ',bins,'\n')
    bins = np.subtract(bins, np.diff([0] + bins.tolist()) / 2)
    #print('        mid points of bins: ',bins)
    exp_variogram =  V.experimental
    matrix_for_saving = np.array([bins,exp_variogram]).T
    return V , bins, exp_variogram, matrix_for_saving
    
def retrieve_histogram(VARIOGRAM,DX=1.0):
    print('        retreiving counts of pairwise obs per lag class ...')
    bins_upper_edges = VARIOGRAM.bins
    counts = np.fromiter((g.size for g in VARIOGRAM.lag_classes()), dtype=int)
    widths = np.diff([0] + bins_upper_edges.tolist())
    bins_middle_points   = np.subtract(bins_upper_edges, np.diff([0] + bins_upper_edges.tolist()) / 2)*DX
    print('        widths of lag classes are: ',widths)
    
    return bins_middle_points, counts, widths

def grab_intersection_gbig_gsmall_RAMS(VARIABLE,RAMS_G1_or_G2_FILE,RAMS_G3_FILE):
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

def compute_moran(DISTANCE_INTERVAL, COORDS, VALUES):
    # Create binary spatial weights matrix based on distance interval
    w = libpysal.weights.DistanceBand(COORDS, threshold=DISTANCE_INTERVAL, binary=True, silence_warnings=True)
    # Compute Moran's I
    moran = Moran(VALUES, w)
    return DISTANCE_INTERVAL, moran.I, moran.EI, moran.VI_norm, moran.p_norm, moran.z_norm


def arrange_images_with_wildcard(input_folder, output_file, wildcard_pattern, non_target_string):
    # Get a list of PNG images in the input folder matching the wildcard pattern
    if non_target_string:
        image_files = sorted([f for f in glob.glob(os.path.join(input_folder, wildcard_pattern)) if f.lower().endswith('.png') and non_target_string not in f])[1::2]
    else:
        image_files = sorted([f for f in glob.glob(os.path.join(input_folder, wildcard_pattern)) if f.lower().endswith('.png')])[1::2]

    print('found ',len(image_files),' images')
    for fil in image_files:
        print(fil)
    # Check if there are any matching images
    if not image_files:
        print(f"Error: No PNG images matching the wildcard pattern '{wildcard_pattern}' found in the folder.")
        return

    # Calculate the number of rows and columns for the matrix
    num_images = len(image_files)
    num_cols = int(math.sqrt(num_images))
    num_rows = math.ceil(num_images / num_cols)

    # Create a new image with dimensions for the matrix and reduced white space
    img_width, img_height = Image.open(image_files[0]).size
    margin = 60  # Adjust this value to control the margin
    result_image = Image.new('RGB', (num_cols * (img_width - margin), num_rows * (img_height - margin)))

    # Loop through the matching images and paste them onto the result image with reduced white space
    for i in range(num_images):
        img = Image.open(image_files[i])

        # Calculate the position with margin to paste the image
        col = i % num_cols
        row = i // num_cols
        position = (col * (img_width - margin), row * (img_height - margin))

        # Paste the image onto the result image
        result_image.paste(img, position)

    # Save the result image
    result_image.save(output_file)
