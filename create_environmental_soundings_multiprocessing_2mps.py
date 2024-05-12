# masking of other updrafts
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

from RAMS_functions import read_head, get_time_from_RAMS_file
from sounding_functions import create_indices, plot_skewT, plot_area_average_sounding_RAMS_around_point

def create_circular_mask(h, w, keep_in_or_out, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    if keep_in_or_out=='in':
        mask = np.where(dist_from_center <= radius, 1.0, np.nan)
    elif keep_in_or_out=='out':
        mask = np.where(dist_from_center >= radius, 1.0, np.nan)
    else:
        print('Please provide <<in>> or <<out>> values for the --keep_in_or_out-- argument')
    return mask

def create_environmental_soundings_parallel(DOMAIN,DXY,DF_CELL,CELL_NO,ZM,ZT,PLOTTING_RANGE,FIXED_THREHOLD,MASK_OTHER_UPDRAFTS):
    
    # declare function-wide parameters
    Cp=1004.0
    Rd=287.0
    p00 = 100000.0
    ##############
    # Retreive all positions of this cell
    print('\n\n=============================================================')
    print('working on cell # ',CELL_NO)
    print('this cell has '+str(len(DF_CELL))+' time steps -- lifetime of ',len(DF_CELL)*0.5,' mins')
    xpos_all_times         = DF_CELL.X.values.astype(int)
    ypos_all_times         = DF_CELL.Y.values.astype(int)
    zpos_all_times         = DF_CELL.zmn.values.astype(int)
    cell_lat_all_times     = DF_CELL.lat.values
    cell_lon_all_times     = DF_CELL.lon.values
    times_tracked          = DF_CELL.timestr.values
    thresholds_all_times   = DF_CELL.threshold_value.values
    print('x-positions: ',xpos_all_times)
    print('y-positions: ',ypos_all_times)
    print('z-positions: ',zpos_all_times)
    print('threshold for this cell: ',thresholds_all_times)
    print('times for this cell: ',times_tracked)
    
    for counter,(tim,xpos,ypos,zpos,cell_lat,cell_lon,tobac_threshold) in enumerate(zip(times_tracked,xpos_all_times,ypos_all_times,zpos_all_times,cell_lat_all_times,cell_lon_all_times,thresholds_all_times)):
        print('\n\n-----------------------')
        print('timestep '+str(counter)+': '+tim)
        tim_pd = pd.to_datetime(tim)
        #rams_fil=glob.glob('/nobackup/pmarines/DATA_FM/'+DOMAIN+'/LES_data/'+'a-L-'+tim_pd.strftime("%Y-%m-%d-%H%M%S")+'-g3.h5')[0] # Pleiades
        rams_fil=glob.glob('/monsoon/LES_MODEL_DATA/'+DOMAIN+'/G3/out_30s/'+'a-L-'+tim_pd.strftime("%Y-%m-%d-%H%M%S")+'-g3.h5')[0] # CSU machine
        print('    RAMS date file: ',rams_fil)
        da = xr.open_dataset(rams_fil,engine='h5netcdf', phony_dims='sort')
        #da = read_file_h5py(h5files1[-1])
        ############################### WP SNAPSHOTS ################################
        rams_lats=da['GLAT'][ypos-PLOTTING_RANGE:ypos+PLOTTING_RANGE,xpos-PLOTTING_RANGE:xpos+PLOTTING_RANGE].values
        rams_lons=da['GLON'][ypos-PLOTTING_RANGE:ypos+PLOTTING_RANGE,xpos-PLOTTING_RANGE:xpos+PLOTTING_RANGE].values
        
        RAMS_closest_level = np.argmin(np.abs(ZM-zpos))
        print('    RAMS closest vertical level to the thermal centroid is ',RAMS_closest_level)
        vertical_vel = da['WP'][RAMS_closest_level,ypos-PLOTTING_RANGE:ypos+PLOTTING_RANGE,xpos-PLOTTING_RANGE:xpos+PLOTTING_RANGE].values
        print('    dimensions of plotted field is ',np.shape(vertical_vel))  
        
        # 2D segmentation of the updraft
        IMAGE       = vertical_vel*create_circular_mask(np.shape(vertical_vel)[0],np.shape(vertical_vel)[1], 'in', center=None, radius=60)
        bw          = closing(IMAGE > FIXED_THREHOLD, square(3)) # apply fixed threshold
        cleared     = clear_border(bw)             # remove artifacts connected to image border
        label_image = label(cleared)           # label image regions
        regions     = [rr for rr in regionprops(label_image)]
        #print('regions: ',regions)
        #print('length of regions list ',len(regions))
        if len(regions)==0:
            print('regionprops could not find any updraft!!!\n\n\n')
            continue
            
        region_choice_criteria='closest'
        # choose the region whose centroid is closest to the centroid of tobac cell in question
        if region_choice_criteria=='closest':
            centroids = [r.centroid for r in regions] #  y0, x0 = props.centroid; note the order 'y' and 'x'
            areas     = [r.area for r in regions]
            dist      = []  # calculate distances of the centroids of ud from the given point
            for centt in centroids:
                centt = np.array((centt))  # convert tuple to np array
                dist.append(np.linalg.norm(centt - np.array([np.shape(vertical_vel)[0]//2,np.shape(vertical_vel)[1]//2])))
            
            ind_closest_ud = np.argmin(np.array(dist))
            
            if areas[ind_closest_ud] <0.5:
                print('area of the selected updraft is too small... moving on to the next timestep\n*\n*\n*\n')
                continue
            
            print('distances of all the detected updrafts from the tobac thermal centroid are: ', dist)
            
            cell_area          = areas[ind_closest_ud]
            chosen_region      = regions[ind_closest_ud]
            chosen_centroid    = centroids[ind_closest_ud]
            chosen_centroid    = list(chosen_centroid) 
            chosen_centroid[0] = int(chosen_centroid[0])   # convert to integer # Y
            chosen_centroid[1] = int(chosen_centroid[1])                        # X
            print('    index of chosen region is ',     ind_closest_ud)
            print('    centroid of chosen region is ', chosen_centroid)
            
        elif region_choice_criteria=='largest':
            areas = [r.area for r in regions]
            centroids = [r.centroid for r in regions]
            max_area=max(np.array(areas))
            max_area_ind= areas.index(max(areas))
            chosen_region=regions[max_area_ind]
            chosen_centroid = centroids[max_area_ind]
            chosen_centroid = list(chosen_centroid) 
            chosen_centroid[0] = int(chosen_centroid[0])   # convert to integer
            chosen_centroid[1] = int(chosen_centroid[1])
            print('    index of chosen region is ', max_area_ind)
            print('    centroid of chosen region is ', chosen_centroid)

        else:
            print('    closest or max are the only options for choosing areas')
            
        chosen_area_label_coords_array = chosen_region.coords

        rows = [uu[0] for uu in chosen_area_label_coords_array]
        cols = [uu[1] for uu in chosen_area_label_coords_array]

        segmented_arr = np.zeros_like(label_image)  # *np.nan
        segmented_arr[rows, cols] = 1111111.0

        updraft_radius = chosen_region.axis_minor_length*0.75 + chosen_region.axis_major_length*0.25
        print('    found an updraft with radius = ',updraft_radius,' gridpoints or ',updraft_radius*dxy/1000.,' km')
        print('    using it to create updraft mask')
        
        mask_center_string = 'detected_updraft_center' # 'tobac_cell_centroid'
        
        if mask_center_string   == 'detected_updraft_center':
            mask_center          = [chosen_centroid[1],chosen_centroid[0]] # has to X, Y
            circle_center_latlon = [rams_lats[chosen_centroid[0],chosen_centroid[1]],rams_lons[chosen_centroid[0],chosen_centroid[1]]]
            print('    lat-lon of updated cell position is : ',circle_center_latlon)
        elif mask_center_string == 'tobac_cell_centroid':
            mask_center          = [int(np.shape(vertical_vel)[1]/2), int(np.shape(vertical_vel)[0]/2)]
            circle_center_latlon = [cell_lat,cell_lon]

        # mask out the main updraft with a circle surrounding it
        updraft_mask1_2D = create_circular_mask(np.shape(vertical_vel)[0],np.shape(vertical_vel)[1], 'out', center=mask_center, radius=int(updraft_radius))
        vertical_vel_3D  = da['WP'][:,ypos-PLOTTING_RANGE:ypos+PLOTTING_RANGE,xpos-PLOTTING_RANGE:xpos+PLOTTING_RANGE].values
        
        # mask out all the updrafts in 3d 
        if MASK_OTHER_UPDRAFTS:
            all_uds_mask_3D  = np.where(vertical_vel_3D<=2.0,1.0,np.nan)
            masking_filename_label='other_uds_masked'
        else:
            all_uds_mask_3D  = np.ones_like(vertical_vel_3D)
            masking_filename_label=''
            
        #env radius dependent on updraft radius
        env_radius_array = np.arange(10,210,10) # in units of pixels
        
        for env_radius in env_radius_array:
            print('    \n---- working on radius ---- : ',env_radius ,'grid points or '+str(int(env_radius*DXY))+' m')
            # create mask for this environmental width
            env_mask_2D = create_circular_mask(np.shape(vertical_vel)[0],np.shape(vertical_vel)[1], 'in', center=mask_center, radius=env_radius+updraft_radius)#*\
            total_3D_mask=updraft_mask1_2D[np.newaxis,:,:]*env_mask_2D[np.newaxis,:,:]*all_uds_mask_3D
            print('    shape  of 3d mask: ',np.shape(total_3D_mask))
            ######### PLOT WP SNAPSHOT and ENV ANNULUS FOR THIS CELL AT THIS TIME #########
            plot_snapshots=True
            current_cmap = plt.get_cmap('bwr').copy()
            if env_radius == 70:
                if plot_snapshots:
                    coords = 'latlon'
                    if coords=='cartesian':
                        fig = plt.figure(figsize=(8,8))
                        ax = plt.gca()
                        ax.axis('equal')
                        C111 = ax.contourf(vertical_vel,levels=np.arange(-20,21,1),cmap=current_cmap,extend='both')#,colors=PLOT_ANOTHER_VAR_CONT[7],linestyles=np.where(levels >= 0, "-", "--"),linewidths=PLOT_ANOTHER_VAR_CONT[8])
                        plt.colorbar(C111,shrink=0.7, pad=0.02,fraction=0.11)
                        ax.set_title('Plan view of vertical velocity at height '+str(zpos)+' m AGL; Cell#'+str(CELL_NO)+'\n'+get_time_from_RAMS_file(rams_fil)[0])

                        # for zoomed in 
                        tobac_features_scatter = ax.scatter(ypos,xpos,label='cell#'+str(CELL_NO),marker='.',s=85.5,c='k')

                    elif coords=='latlon':
                        fig = plt.figure(figsize=(8,8))
                        ax = fig.add_subplot(1, 1, 1, projection=crs.PlateCarree(),facecolor='lightgray')
                        # plot vertical velocity of the environment
                        C111 = ax.contourf(rams_lons ,rams_lats, vertical_vel*total_3D_mask[RAMS_closest_level,:,:],levels=np.arange(-20,21,1),cmap=current_cmap,extend='both',transform=crs.PlateCarree())#,colors=PLOT_ANOTHER_VAR_CONT[7],linestyles=np.where(levels >= 0, "-", "--"),linewidths=PLOT_ANOTHER_VAR_CONT[8])
                        # plot the segmented main updrafts
                        C7 = ax.contour(rams_lons ,rams_lats, segmented_arr, levels=[1111111.0], colors='darkblue', linewidths=1.1, linestyles='--')
                        # plot the centroid the segmented updraft
                        ax.scatter(rams_lons[chosen_centroid[0],chosen_centroid[1]],rams_lats[chosen_centroid[0],chosen_centroid[1]],marker='^',s=55.5,c='fuchsia',)

                        plt.colorbar(C111,shrink=0.7, pad=0.02,fraction=0.11)
                        ax.set_title('Plan view of vertical velocity at height '+str(zpos)+' m AGL; Cell#'+str(CELL_NO)+'\n'+get_time_from_RAMS_file(rams_fil)[0])

                        # plot the cell point
                        tobac_features_scatter = ax.scatter(cell_lon,cell_lat,label='cell#'+str(CELL_NO),marker='.',s=55.5,c='k',transform=crs.PlateCarree())
                        gd = Geodesic()
                        env_circle = gd.circle(lon=circle_center_latlon[1], lat=circle_center_latlon[0], radius=(env_radius+updraft_radius)*dxy)
                        updraft_circle = gd.circle(lon=circle_center_latlon[1], lat=circle_center_latlon[0], radius=updraft_radius*dxy)
                        ax.add_geometries([sgeom.Polygon(env_circle)], crs=crs.PlateCarree(), edgecolor='green', facecolor="none")
                        ax.add_geometries([sgeom.Polygon(updraft_circle)], crs=crs.PlateCarree(), edgecolor='maroon', facecolor="none")

                        gl = ax.gridlines()
                        ax.coastlines(resolution='50m')
                        gl.xlines = True
                        gl.ylines = True
                        LATLON_LABELS=True
                        if LATLON_LABELS:
                            #print('LATLON labels are on')
                            gl.xlabels_top = True
                            gl.ylabels_right = False
                            gl.ylabels_left = True
                            gl.ylabels_bottom = True
                        else:
                            gl.xlabels_top = False
                            gl.ylabels_right = False
                            gl.ylabels_left = False
                            gl.ylabels_bottom = True

                        gl.xlabel_style = {'size': 13, 'color': 'gray'}
                        gl.ylabel_style = {'size': 13, 'color': 'gray'}

                        import matplotlib.transforms as transforms
                        trans = transforms.blended_transform_factory(ax.transAxes,ax.transData)
                        props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
                        ax.text(0.52, 0.94, 'cell radius = '+str(np.round(updraft_radius*dxy/1000.,1))+\
                                ' km'+'\n'+'tobac cell detection threshold ='+str(int(tobac_threshold))+' m/s\n'+\
                                'w threshold for sounding ='+str(int(FIXED_THREHOLD))+' m/s\n'+\
                                'environment width = '+str(int(env_radius*DXY/1000.0))+' km',
                                fontsize=9,verticalalignment='top',\
                                bbox=props,transform=ax.transAxes)

                    else:
                        print('coords option for w snapshots has to be <cartesian> or <latlon>')

                #snapshot_png_saving_directory = '/nobackupp11/isingh2/tobac_plots/sounding_csvs_and_WP_snapshots/' #Pleaides
                #snapshot_png_saving_directory = '/Users/isingh/SVH/SVH_paper1/scratch/'  # Personal computer
                snapshot_png_saving_directory  = '/home/isingh/code/scratch/environmental_assessment/' # CSU machine
                
                wp_snapshot_with_environment_annulus_png=snapshot_png_saving_directory + 'WP_snapshot_cellcenter_'+mask_center_string+'_envwidth'+str(int(env_radius*DXY))+'m_cell'+str(CELL_NO)+'_xpos'+str(xpos)+'_ypos'+str(ypos)+'_zpos'+str(zpos)+'_'+get_time_from_RAMS_file(rams_fil)[1]+'_'+DOMAIN+'_comb_track_filt_01_02_50_02_sr5017_setpos_'+masking_filename_label+'.png'
                print('    saving plan view wp snapshot with name: \n',wp_snapshot_with_environment_annulus_png)
                plt.savefig(wp_snapshot_with_environment_annulus_png,dpi=75)
                plt.close()
            ########### ################## GET SOUNDING DATA ############################# 
            save_soundings = True
            
            if save_soundings:
                #sounding_saving_directory = '/nobackupp11/isingh2/tobac_plots/sounding_csvs_and_WP_snapshots/' # Pleaides
                #sounding_saving_directory = '/Users/isingh/SVH/SVH_paper1/scratch/' # personal computer
                sounding_saving_directory =  '/home/isingh/code/scratch/environmental_assessment/' # CSU machine
                print('    saving soundings...\n')

                rams_exner       = da['PI']   [:,ypos-PLOTTING_RANGE:ypos+PLOTTING_RANGE,xpos-PLOTTING_RANGE:xpos+PLOTTING_RANGE].values/1004.0*total_3D_mask
                rams_theta       = da['THETA'][:,ypos-PLOTTING_RANGE:ypos+PLOTTING_RANGE,xpos-PLOTTING_RANGE:xpos+PLOTTING_RANGE].values*total_3D_mask# Kelvin
                rams_temp_K      = rams_exner*rams_theta*units('K') # Kelvin
                rams_temp_degC   = rams_temp_K.to('degC')
                rams_pres_Pa     = (p00*(rams_exner)**(Cp/Rd))*units('Pa')
                rams_pres_hPa    = rams_pres_Pa.to('hPa')
                rams_rv          = da['RV'][:,ypos-PLOTTING_RANGE:ypos+PLOTTING_RANGE,xpos-PLOTTING_RANGE:xpos+PLOTTING_RANGE].values*total_3D_mask*units('kg/kg') # kg/kg
                rams_sphum       = mpcalc.specific_humidity_from_mixing_ratio(rams_rv)
                rams_dewpt       = mpcalc.dewpoint_from_specific_humidity(rams_pres_Pa,rams_temp_degC,rams_sphum)
                rams_u           = da['UP'][:,ypos-PLOTTING_RANGE:ypos+PLOTTING_RANGE,xpos-PLOTTING_RANGE:xpos+PLOTTING_RANGE].values*total_3D_mask*units('m/s')
                rams_v           = da['VP'][:,ypos-PLOTTING_RANGE:ypos+PLOTTING_RANGE,xpos-PLOTTING_RANGE:xpos+PLOTTING_RANGE].values*total_3D_mask*units('m/s')
                rams_ter         = np.nanmean(da['TOPT'][ypos-PLOTTING_RANGE:ypos+PLOTTING_RANGE,xpos-PLOTTING_RANGE:xpos+PLOTTING_RANGE].values*env_mask_2D,axis=(0,1))
                rams_hgt_msl     = (rams_ter + ZT)*units('meter')

                #print('shape of rams_hgt_msl: ', np.shape(rams_hgt_msl))

                output_median = False
                output_mean   = True
                output_std    = True
                
                if output_median: 
                    print('    saving median sounding')
                    rams_temp_degC_median      = np.nanmedian(rams_temp_degC.magnitude,axis=(1,2))
                    rams_dewpt_median          = np.nanmedian(rams_dewpt.magnitude,axis=(1,2))
                    rams_u_median              = np.nanmedian(rams_u.magnitude,axis=(1,2))
                    rams_v_median              = np.nanmedian(rams_v.magnitude,axis=(1,2))
                    rams_pres_hPa_median       = np.nanmedian(rams_pres_hPa.magnitude,axis=(1,2))
                    csv_sounding_file_name_mean= sounding_saving_directory +'area_avgd_annulus_envwidth_'+str(int(env_radius*DXY))+'m_2mps_'+'median_sounding_cell_'  +str(CELL_NO)+'_'+get_time_from_RAMS_file(rams_fil)[1]+'_'+DOMAIN+'_comb_track_filt_01_02_50_02_sr5017_setpos_'+masking_filename_label+'.csv'
                    csv_median_sounding_df     = pd.DataFrame(data={'height_m':rams_hgt_msl,'pressure_hPa':rams_pres_hPa_median,'temp_degC':rams_temp_degC_median,'dewpt_degC':rams_dewpt_median,'uwnd_mps':rams_u_median,'vwnd_mps':rams_v_median})
                    csv_median_sounding_df.to_csv(csv_sounding_file_name_median,index=False)
                    print('    writing median sounding csv file for cell#',CELL_NO,' for time ',tim,' : <<<',csv_sounding_file_name_median,'>>>')
                    
                if output_mean: 
                    print('    saving mean sounding')
                    rams_temp_degC_mean        = np.nanmean(rams_temp_degC.magnitude,axis=(1,2))
                    rams_dewpt_mean            = np.nanmean(rams_dewpt.magnitude,axis=(1,2))
                    rams_u_mean                = np.nanmean(rams_u.magnitude,axis=(1,2))
                    rams_v_mean                = np.nanmean(rams_v.magnitude,axis=(1,2))
                    rams_pres_hPa_mean         = np.nanmean(rams_pres_hPa.magnitude,axis=(1,2))
                    csv_sounding_file_name_mean= sounding_saving_directory +'area_avgd_annulus_envwidth_'+str(int(env_radius*DXY))+'m_2mps_'+'mean_sounding_cell_'  +str(CELL_NO)+'_'+get_time_from_RAMS_file(rams_fil)[1]+'_'+DOMAIN+'_comb_track_filt_01_02_50_02_sr5017_setpos_'+masking_filename_label+'.csv'
                    csv_mean_sounding_df = pd.DataFrame(data={'height_m':rams_hgt_msl,'pressure_hPa':rams_pres_hPa_mean,'temp_degC':rams_temp_degC_mean,'dewpt_degC':rams_dewpt_mean,'uwnd_mps':rams_u_mean,'vwnd_mps':rams_v_mean})
                    csv_mean_sounding_df.to_csv(csv_sounding_file_name_mean,index=False)
                    print('    writing mean sounding csv file for cell#',CELL_NO,' for time ',tim,' : <<<',csv_sounding_file_name_mean,'>>>')
                    
                if output_std:
                    print('    saving std sounding')
                    rams_temp_degC_std         = np.nanstd(rams_temp_degC.magnitude,axis=(1,2))
                    rams_dewpt_std             = np.nanstd(rams_dewpt.magnitude,axis=(1,2))
                    rams_u_std                 = np.nanstd(rams_u.magnitude,axis=(1,2))
                    rams_v_std                 = np.nanstd(rams_v.magnitude,axis=(1,2))
                    rams_pres_hPa_std          = np.nanstd(rams_pres_hPa.magnitude,axis=(1,2))
                    csv_sounding_file_name_std = sounding_saving_directory +'area_avgd_annulus_envwidth_'+str(int(env_radius*DXY))+'m_2mps_'+'std_sounding_cell_'   +str(CELL_NO)+'_'+get_time_from_RAMS_file(rams_fil)[1]+'_'+DOMAIN+'_comb_track_filt_01_02_50_02_sr5017_setpos_'+masking_filename_label+'.csv'
                    csv_std_sounding_df = pd.DataFrame(data={'height_m':rams_hgt_msl,'pressure_hPa':rams_pres_hPa_std,'temp_degC':rams_temp_degC_std,'dewpt_degC':rams_dewpt_std,'uwnd_mps':rams_u_std,'vwnd_mps':rams_v_std})
                    csv_std_sounding_df.to_csv(csv_sounding_file_name_std,index=False)
                    print('    writing std sounding csv file for cell#',CELL_NO,' for time ',tim,' : <<<',csv_sounding_file_name_std,'>>>')

    print('====================== END OF CELL ======================\n\n')
    

############################################################################
# Paths to model data and where to save data

domain='DRC1.1-R'

#path = '/nobackup/pmarines/DATA_FM/'+domain+'/LES_data/'     # Pleiades
#path = '/Users/isingh/SVH/INCUS/sample_LES_data/'+domain+'/' # personal macbook
path = '/monsoon/LES_MODEL_DATA/'+domain+'/G3/out_30s/'         # CSU machine

#savepath = './'
#tobac_data='/nobackup/pmarines/DATA_FM/'+domain+'/tobac_data/'                                 # Pleiades
#tobac_data='/Users/isingh/SVH/INCUS/jupyter_nbks/tobac_thermals/peter_tobac_output/'+domain+'/'# personal macbook
tobac_data='/monsoon/pmarin/Tracking/Updrafts/'+domain+'/tobac_data/'                           # CSU machine

#tobac_filename = 'comb_track_filt_01_02_05_10_20.p'
tobac_filename = 'comb_track_filt_01_02_50_02_sr5017_setpos.p'
tobac_filepath  = tobac_data+tobac_filename


# Grab all the rams files 
h5filepath = path+'a-L*g3.h5'
h5files1 = sorted(glob.glob(h5filepath))
hefilepath = path+'a-L*head.txt'
hefiles1 = sorted(glob.glob(hefilepath))
#print(h5files1)
start_time=get_time_from_RAMS_file(h5files1[0])[0]
end_time=get_time_from_RAMS_file(h5files1[-1])[0]
print('Simulation name: ',domain)
print('starting time of the simulation: ',start_time)
print('ending time of the simulation: ',end_time)

ds=xr.open_dataset(h5files1[-1],engine='h5netcdf', phony_dims='sort')
wp = ds['WP']
#ds = h5py.File(h5files1[-1], 'r')
#wp = read_var_h5py(h5files1[-1],'WP')
#ds#.TOPT.values

domain_z_dim,domain_y_dim,domain_x_dim=np.shape(wp)
print('domain_z_dim: ',domain_z_dim)
print('domain_y_dim: ',domain_y_dim)
print('domain_x_dim: ',domain_x_dim)

zm, zt, nx, ny, dxy, npa = read_head(hefiles1[0],h5files1[0])

#******************
plotting_range = 250 # 25 km # need to know now for filtering cells close to edges
#******************

##### read in tobac data #####
print('reading ',tobac_filepath)
tdata = pd.read_pickle(tobac_filepath)

def filter_cells(g):
    return ((g.zmn.max() >= 2000.0) & (g.zmn.min() <= 15000.) & (g.X.max() <= domain_x_dim-plotting_range-1) & (g.X.min() >= plotting_range+1) &
            (g.Y.max() <= domain_y_dim-plotting_range-1) & (g.Y.min() >= plotting_range+1) & (g.threshold_value.count() >= 5)  \
             & (pd.to_datetime(g.timestr).min() > pd.to_datetime(start_time)) & (pd.to_datetime(g.timestr).max() <  pd.to_datetime(end_time)))

tdata_temp=tdata.groupby('cell').filter(filter_cells)
#print(tdata_temp)
all_cells = tdata_temp.cell.unique().tolist()

print('number of unique cells identified: ',len(all_cells))
#print('these cells are: ',all_cells)
############################################################################

already_done_cells = [8961, 9687, 11239, 11776, 11816, 11910, 12508, 12517, 14048, 14080, 14305, 14935, 14947, 15001, 15047, 15635, 15650, 15669, 15752, 15772, 16404, 17050, 17052, 17160, 17165, 17193, 17860, 17862, 18431, 20083, 22353, 22905, 23153, 24569, 25444, 25475, 28322, 28327, 28435, 28545, 29100, 29621, 29833, 29837, 29889, 30520, 31830, 34238, 34318, 34933, 34939, 35735, 36254, 37121, 40050, 40123, 41566, 43635, 44436, 45892, 46530, 46674, 47321, 47915, 48039, 48708, 49257, 52425, 54908, 56310, 56979, 59846, 62013, 62830, 66438, 67063, 69909, 70551, 70563, 70681, 73919, 74684, 74712, 75306, 76540, 76725, 78575, 79269, 79361, 79617, 80186, 81956, 81959, 82133, 83435, 83535, 84105, 86891, 88337, 90184]

all_cells = [cc for cc in all_cells if cc not in already_done_cells] 
print('removed already done cells: new # of all cells is ',len(all_cells))
# Running in the notebook
# cn = random.choice(all_cells)
# print('randomly chosen cell#: ',cn)
# tdata_subset=tdata_temp[tdata_temp['cell']==cn]
# #print(tdata_subset)
# create_environmental_soundings_parallel(domain,dxy,tdata_subset,cn,zm,zt,plotting_range,2.0,True)

#Running on the terminal in parallel
selected_cells = random.sample(all_cells,100)
print('100 randomly selected cells are: ',selected_cells)
argument = []
#for cn in all_cells:
for cn in selected_cells:
    tdata_subset=tdata_temp[tdata_temp['cell']==cn]
    argument = argument + [(domain,dxy,tdata_subset,cn,zm,zt,plotting_range,2.0,True)]

print('length of argument is: ',len(argument))

############################## FIRST OF ALL ################################
cpu_count1 = 32 #int(cpu_count()/2)
print('number of cpus: ',cpu_count1)
############################################################################


def main(FUNCTION, ARGUMENT):
    pool = Pool(cpu_count1)
    start_time = time.perf_counter()
    results = pool.starmap(FUNCTION, ARGUMENT)
    finish_time = time.perf_counter()
    print(f"Program finished in {finish_time-start_time} seconds")

if __name__ == "__main__":
    main(create_environmental_soundings_parallel, argument)
