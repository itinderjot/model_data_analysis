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
import pandas as pd
import xarray as xr
import cartopy.crs as crs
#matplotlib.rcParams['axes.facecolor'] = 'white'
#plt.rcParams["font.family"] = "helvetica"

from cartopy.geodesic import Geodesic
import shapely.geometry as sgeom

from skimage.draw import ellipse
from skimage.measure import label, regionprops, regionprops_table
from skimage.transform import rotate
from skimage.morphology import closing, square
from skimage.segmentation import clear_border

from multiprocessing import Pool, cpu_count
import os
import time
import random
#import istarmap
#import tqdm

def radar_colormap():
    nws_reflectivity_colors = [
    "#646464", # ND
    "#ccffff", # -30
    "#cc99cc", # -25
    "#996699", # -20
    "#663366", # -15
    "#cccc99", # -10
    "#999966", # -5
    "#646464", # 0
    "#04e9e7", # 5
    "#019ff4", # 10
    "#0300f4", # 15
    "#02fd02", # 20
    "#01c501", # 25
    "#008e00", # 30
    "#fdf802", # 35
    "#e5bc00", # 40
    "#fd9500", # 45
    "#fd0000", # 50
    "#d40000", # 55
    "#bc0000", # 60
    "#f800fd", # 65
    "#9854c6", # 70
    "#fdfdfd" # 75
    ]

    return mpl.colors.ListedColormap(nws_reflectivity_colors)


cma1=plt.get_cmap('bwr')
cma2=radar_colormap()
cma3=plt.get_cmap('tab20c')
import nclcmaps as ncm
cma4=ncm.cmap("WhiteBlueGreenYellowRed")
cma5=plt.get_cmap('gray_r')
cma6=plt.get_cmap('rainbow')
cma7=plt.get_cmap('Oranges')
cma8=plt.get_cmap('coolwarm')
#cma9=cma4.reversed()
cma10=plt.get_cmap('gist_yarg')

####################################################################################

def get_time_from_RAMS_file(INPUT_FILE):
    cur_time = os.path.split(INPUT_FILE)[1][4:21] # Grab time string from RAMS file
    pd_time = pd.to_datetime(cur_time[0:10]+' '+cur_time[11:13]+":"+cur_time[13:15]+":"+cur_time[15:17])
    return pd_time.strftime('%Y-%m-%d %H:%M:%S'), pd_time.strftime('%Y%m%d%H%M%S')

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


def get_cell_W_area_rad_itc_iwp_rain(DOMAIN,DXY,UPDRAFT_DETECTION_THRESHOLD,TOBAC_CELL_DF,CELLNO,ZM,ZT,PLOTTING_RANGE,UPDRAFT_CONTAMINATION_CHECK,PLOT_QUANTITIES):
    # make a copy of the dataframe to work on
    # this dataframe is a subset of the larger dataframe and contains data for only one cell 
    g                       = TOBAC_CELL_DF.copy()
    #print('the DATAFRAME: ',g,'\n\n')
    print('working on cell#',CELLNO)
    print('this cell has '+str(len(g))+' time steps -- lifetime of ',len(g)*0.5,' mins')
    xpos_all_times          = g.X.values.astype(int)
    ypos_all_times          = g.Y.values.astype(int)
    zpos_all_times          = g.zmn.values.astype(int)
    cell_lat_all_times      = g.lat.values
    cell_lon_all_times      = g.lon.values
    times_tracked           = g.timestr.values
    thresholds_all_times    = g.threshold_value.values
    #print('x-positions: ',xpos)
    #print('y-positions: ',ypos)
    #print('z-positions: ',zpos)
    print('thresholds for this cell: ',thresholds_all_times)
    print('times for this cells: '    ,times_tracked)


    for counter, (tim,xpos,ypos,zpos,cell_lat,cell_lon,tobac_threshold) in enumerate(zip(times_tracked,xpos_all_times,ypos_all_times,zpos_all_times,cell_lat_all_times,cell_lon_all_times,thresholds_all_times)): # loop over timesteps of this cell 
        print('------------------------------------------------')
        print('timestep '+str(counter)+': '+tim)
        tim_pd   = pd.to_datetime(tim)
        # Pleiades
        #rams_fil=glob.glob('/nobackup/pmarines/DATA_FM/'+domain+'/LES_data/a-L-'+tim_pd.strftime("%Y-%m-%d-%H%M%S")+'-g3.h5')[0]
        # CSU machine
        rams_fil = glob.glob('/monsoon/LES_MODEL_DATA/'+DOMAIN+'/G3/out_30s/'+'a-L-'+tim_pd.strftime("%Y-%m-%d-%H%M%S")+'-g3.h5')[0]

        print('RAMS file for this timestep: ',rams_fil)

        # use xarray like here or h5py
        da = xr.open_dataset(rams_fil,engine='h5netcdf', phony_dims='sort')
        rams_lats=da['GLAT'][ypos-PLOTTING_RANGE:ypos+PLOTTING_RANGE,xpos-PLOTTING_RANGE:xpos+PLOTTING_RANGE].values
        rams_lons=da['GLON'][ypos-PLOTTING_RANGE:ypos+PLOTTING_RANGE,xpos-PLOTTING_RANGE:xpos+PLOTTING_RANGE].values
        RAMS_closest_level = np.argmin(np.abs(ZM-zpos))
        print('RAMS closest vertical level to the thermal centroid is ',RAMS_closest_level)
        # get the maximum value of w in the vicinity of the tobacl cell centroid
        vertical_vel_centroid = np.nanmax(da.WP[RAMS_closest_level-3:RAMS_closest_level+3,ypos-3:ypos+3,xpos-3:xpos+3].values)
        print('vertical velocity at the centroid is ',vertical_vel_centroid,' m/s')
        # 2D vertical velocity at the level of the tobac cell centroid
        vertical_vel = da['WP'][RAMS_closest_level,ypos-PLOTTING_RANGE:ypos+PLOTTING_RANGE,xpos-PLOTTING_RANGE:xpos+PLOTTING_RANGE]#.values
        
        ########## 2D SEGMENTATION HERE #############
        IMAGE   = vertical_vel*create_circular_mask(np.shape(vertical_vel)[0],np.shape(vertical_vel)[1], 'in', center=None, radius=PLOTTING_RANGE-1)
        bw      = closing(IMAGE >= UPDRAFT_DETECTION_THRESHOLD, square(3)) # apply fixed threshold
        cleared = clear_border(bw)                                         # remove artifacts connected to image border
        label_image = label(cleared)                                       # label image regions
        regions = [rr for rr in regionprops(label_image)]                  # use regionprops to get regions of w with the goven value
        #print('regions: ',regions)
        #print('length of regions list ',len(regions))
        if len(regions)==0:
            print('regionprops could not find any updraft!!!\n\n\n')
            g.loc[g['timestr'] == tim, 'area_m2'] = np.nan
            g.loc[g['timestr'] == tim, 'calcnumpixels']  = np.nan
            g.loc[g['timestr'] == tim, 'perimeter_scipy_default'] = np.nan
            g.loc[g['timestr'] == tim, 'areaconvex_scipy_default'] = np.nan
            g.loc[g['timestr'] == tim, 'eqdiameterareascipy_m'] = np.nan
            g.loc[g['timestr'] == tim, 'calcellipseradius_m'] = np.nan
            g.loc[g['timestr'] == tim, 'minoraxis_m'] = np.nan
            g.loc[g['timestr'] == tim, 'majoraxis_m'] = np.nan
            g.loc[g['timestr'] == tim, 'eccentricity'] = np.nan
            g.loc[g['timestr'] == tim, 'wcentroid_mps'] = np.nan
            g.loc[g['timestr'] == tim, 'IWP_mm'] = np.nan
            g.loc[g['timestr'] == tim, 'ITC_mm'] = np.nan
            g.loc[g['timestr'] == tim, 'rainmax_mm_per_hr'] = np.nan
            g.loc[g['timestr'] == tim, 'rain_mean'] = np.nan
            g.loc[g['timestr'] == tim, 'updraftcontamination_fraction'] = np.nan
            continue # to the next timestep 
        else:
            centroids = [r.centroid for r in regions]
            areas     = [r.area for r in regions]
            dist      = []  # calculate distances of the centroids of ud from the tobac cell centroid (which is at the center of the 
                            # subdomain
            for centt in centroids:
                centt = np.array((centt))  # convert tuple to np array
                dist.append(np.linalg.norm(centt - np.array([np.shape(vertical_vel)[0]//2,np.shape(vertical_vel)[1]//2])))

            print('distances of all the detected updrafts from the tobac thermal centroid are: ', dist)

            ind_closest_ud = np.argmin(np.array(dist))
            print('index of chosen region is ', ind_closest_ud)
            print('checking if the area of the appropriate region is > 0.5')

            if areas[ind_closest_ud] <0.5:
                g.loc[g['timestr'] == tim, 'area_m2'] = np.nan
                g.loc[g['timestr'] == tim, 'calcnumpixels']  = np.nan
                g.loc[g['timestr'] == tim, 'perimeter_scipy_default'] = np.nan
                g.loc[g['timestr'] == tim, 'areaconvex_scipy_default'] = np.nan
                g.loc[g['timestr'] == tim, 'eqdiameterareascipy_m'] = np.nan
                g.loc[g['timestr'] == tim, 'calcellipseradius_m'] = np.nan
                g.loc[g['timestr'] == tim, 'minoraxis_m'] = np.nan
                g.loc[g['timestr'] == tim, 'majoraxis_m'] = np.nan
                g.loc[g['timestr'] == tim, 'eccentricity'] = np.nan
                g.loc[g['timestr'] == tim, 'wcentroid_mps'] = np.nan
                g.loc[g['timestr'] == tim, 'IWP_mm'] = np.nan
                g.loc[g['timestr'] == tim, 'ITC_mm'] = np.nan
                g.loc[g['timestr'] == tim, 'rainmax_mm_per_hr'] = np.nan
                g.loc[g['timestr'] == tim, 'rain_mean'] = np.nan
                g.loc[g['timestr'] == tim, 'updraftcontamination_fraction'] = np.nan
                print('area of the selected updraft is too small... moving on to the next timestep\n*\n*\n*\n')
                continue
            else:
                print('Yes, it does.')
                ################## AREA AND RADIUS #######################
                cell_area          = areas[ind_closest_ud]*DXY*DXY
                chosen_region      = regions[ind_closest_ud]
                chosen_centroid    = centroids[ind_closest_ud]
                chosen_centroid    = list(chosen_centroid) 
                chosen_centroid[0] = int(chosen_centroid[0])   # convert to integer
                chosen_centroid[1] = int(chosen_centroid[1])
                print('centroid of chosen updrafts is : ',chosen_centroid)
                updraft_radius = (chosen_region.axis_minor_length*0.75 + chosen_region.axis_major_length*0.25)*DXY
                print('the chosen updraft has radius = ',updraft_radius,' m')
                print('the chosen updraft has area = ',cell_area,' m^2')
                ####################### CREATE STORM MASK ###############################
                chosen_area_label_coords_array = chosen_region.coords
                rows = [uu[0] for uu in chosen_area_label_coords_array]
                cols = [uu[1] for uu in chosen_area_label_coords_array]
                storm_mask_zeros = np.zeros_like(label_image)#*np.nan  # *np.nan
                storm_mask_zeros[rows, cols] = 1.0
                storm_mask_nans = np.zeros_like(label_image)*np.nan  # *np.nan
                storm_mask_nans[rows, cols] = 1.0
                ########################################################################
                
                if len(regions) > 1:
                    if UPDRAFT_CONTAMINATION_CHECK:
                        overlap_list = []
                        storm_circular_mask_boolean= ~np.isnan(create_circular_mask(np.shape(vertical_vel)[0],np.shape(vertical_vel)[1], 'in', center=[chosen_centroid[1],chosen_centroid[0]], radius=updraft_radius/DXY))
                        regions_without_main_storm = regions.copy()
                        regions_without_main_storm.pop(ind_closest_ud)
                        print('original regions: ',regions)
                        print('regions_without_main_storm:', regions_without_main_storm)
                        radii_other_uds = [(r.axis_minor_length*0.75 + r.axis_major_length*0.25) for r in regions_without_main_storm]
                        centroids_other_uds = [r.centroid for r in regions_without_main_storm]
                        print('\n\n !!!!updraft contamination check!!!!')
                        print('centroids of all detection regions are: ',centroids_other_uds)
                        print('radii of all detection regions are: ',radii_other_uds)
                        overlap_fraction_list = []
                        iii = 0
                        for dd,ee in list(zip(centroids_other_uds,radii_other_uds)):
                            print('region#',iii,' : ',dd,ee)
                            print('centoid of storm is',chosen_centroid)
                            print('centroid is ',dd)
                            print('radius is ',ee)
                            adjacent_storm_circular_mask_boolean = ~np.isnan(create_circular_mask(np.shape(vertical_vel)[0],np.shape(vertical_vel)[1], 'in', center=[dd[1],dd[0]], radius=ee))
                            overlap = storm_circular_mask_boolean * adjacent_storm_circular_mask_boolean # Logical AND
                            IOS = np.count_nonzero(overlap)/np.count_nonzero(storm_circular_mask_boolean) 
                            print('overlap for this region is: ',IOS)
                            overlap_fraction_list.append(IOS)
                            iii = iii + 1
                            print('\n-----')
                        print('maximum overlap is ',max(overlap_fraction_list))
                        print('minimum overlap is ',min(overlap_fraction_list))
                        print('\n\n !!!!updraft contamination check OVER!!!!')
                        max_ud_contamination_fraction = max(overlap_fraction_list)
                else:
                    max_ud_contamination_fraction = 0.0
                ####################### IWP ###############################
                # Constants for calculating total integrated condensate
                cp  = 1004    # J/kg/K
                rd  = 287     # J/kg/K
                p00 = 100000  # Reference Pressure
                condensate = da['RTP'][:,ypos-PLOTTING_RANGE:ypos+PLOTTING_RANGE,xpos-PLOTTING_RANGE:xpos+PLOTTING_RANGE].values- \
                             da['RV'] [:,ypos-PLOTTING_RANGE:ypos+PLOTTING_RANGE,xpos-PLOTTING_RANGE:xpos+PLOTTING_RANGE].values

                frozen_condensate = da['RPP'][:,ypos-PLOTTING_RANGE:ypos+PLOTTING_RANGE,xpos-PLOTTING_RANGE:xpos+PLOTTING_RANGE].values+\
                                    da['RSP'][:,ypos-PLOTTING_RANGE:ypos+PLOTTING_RANGE,xpos-PLOTTING_RANGE:xpos+PLOTTING_RANGE].values+\
                                    da['RAP'][:,ypos-PLOTTING_RANGE:ypos+PLOTTING_RANGE,xpos-PLOTTING_RANGE:xpos+PLOTTING_RANGE].values+\
                                    da['RGP'][:,ypos-PLOTTING_RANGE:ypos+PLOTTING_RANGE,xpos-PLOTTING_RANGE:xpos+PLOTTING_RANGE].values+\
                                    da['RHP'][:,ypos-PLOTTING_RANGE:ypos+PLOTTING_RANGE,xpos-PLOTTING_RANGE:xpos+PLOTTING_RANGE].values

                # Load variables needed to calculate density
                th = da['THETA'][:,ypos-PLOTTING_RANGE:ypos+PLOTTING_RANGE,xpos-PLOTTING_RANGE:xpos+PLOTTING_RANGE].values
                nx = np.shape(th)[2]
                ny = np.shape(th)[1]
                pi = da['PI']   [:,ypos-PLOTTING_RANGE:ypos+PLOTTING_RANGE,xpos-PLOTTING_RANGE:xpos+PLOTTING_RANGE].values
                rv = da['RV']   [:,ypos-PLOTTING_RANGE:ypos+PLOTTING_RANGE,xpos-PLOTTING_RANGE:xpos+PLOTTING_RANGE].values
                # Convert RAMS native variables to temperature and pressure
                pres = np.power((pi/cp),cp/rd)*p00
                temp = th*(pi/cp)
                del(th,pi)
                # Calculate atmospheric density
                dens = pres/(rd*temp*(1+0.61*rv))
                del(pres,temp,rv)
                # Difference in heights (dz)    
                diff_zt_3D = np.tile(np.diff(zt),(int(ny),int(nx),1))
                diff_zt_3D = np.moveaxis(diff_zt_3D,2,0)
                # Calculate integrated condensate
                itc               = np.nansum(condensate[1:,:,:]*dens[1:,:,:]*diff_zt_3D,axis=0) # integrated total condensate in kg
                itc_mm            = itc/997.0*1000 # integrated total condensate in mm
                itc_mm[itc_mm<=0] = 0.001
                itc_max           = np.nanmax(storm_mask_zeros*itc_mm)
                # Calculate IWP (ice water path)
                iwp               = np.nansum(frozen_condensate[1:,:,:]*dens[1:,:,:]*diff_zt_3D,axis=0) 
                iwp_mm            = iwp/997.0*1000 # integrated total frozen condensate in mm
                iwp_mm[iwp_mm<=0] = 0.001
                iwp_max           = np.nanmax(storm_mask_zeros*iwp_mm)
                ####################### MAX SURFACE PRECIP RATE ###############################
                rain              = da['PCPRR'][ypos-PLOTTING_RANGE:ypos+PLOTTING_RANGE,xpos-PLOTTING_RANGE:xpos+PLOTTING_RANGE]*3600.0#.values
                rain_max          = np.nanmax(storm_mask_zeros*rain)
                rain_mean         = np.nanmean(storm_mask_nans*rain)
                #####################################################################
                # add all these variables calculated above to the dataframe 
                g.loc[g['timestr'] == tim, 'area_m2']                  = cell_area
                g.loc[g['timestr'] == tim, 'calcnumpixels']            = np.count_nonzero(~np.isnan(storm_mask_nans))
                #g.loc[tdata_subset['timestr'] == tim, 'numpixels_scipy_default'] = chosen_region.num_pixels
                g.loc[g['timestr'] == tim, 'perimeter_scipy_default']  = chosen_region.perimeter
                g.loc[g['timestr'] == tim, 'areaconvex_scipy_default'] = chosen_region.area_convex
                g.loc[g['timestr'] == tim, 'eqdiameterareascipy_m']    = chosen_region.equivalent_diameter_area*DXY/2.0
                g.loc[g['timestr'] == tim, 'calcellipseradius_m']      = updraft_radius
                g.loc[g['timestr'] == tim, 'minoraxis_m']              = chosen_region.axis_minor_length*DXY
                g.loc[g['timestr'] == tim, 'majoraxis_m']              = chosen_region.axis_major_length*DXY
                g.loc[g['timestr'] == tim, 'eccentricity']             = chosen_region.eccentricity
                g.loc[g['timestr'] == tim, 'wcentroid_mps']            = vertical_vel_centroid
                g.loc[g['timestr'] == tim, 'IWP_mm']                   = iwp_max
                g.loc[g['timestr'] == tim, 'ITC_mm']                   = itc_max
                g.loc[g['timestr'] == tim, 'rainmax_mm_per_hr']        = rain_max
                g.loc[g['timestr'] == tim, 'rain_mean']                = rain_mean
                if UPDRAFT_CONTAMINATION_CHECK:
                    g.loc[g['timestr'] == tim, 'updraftcontamination_fraction'] = max_ud_contamination_fraction
                else:
                    g.loc[g['timestr'] == tim, 'updraftcontamination_fraction'] = np.nan
                    
                ######################### PLOTTING #########################
                if PLOT_QUANTITIES:
                    print('plotting = True')

                    current_cmap = plt.get_cmap('bwr').copy()
                    mask_center_string = 'detected_updraft_center' # 'tobac_cell_centroid'

                    if mask_center_string == 'detected_updraft_center':
                        mask_center = chosen_centroid
                        circle_center_latlon = [rams_lats[chosen_centroid[0],chosen_centroid[1]],rams_lons[chosen_centroid[0],chosen_centroid[1]]]
                        print('lat-lon of updated cell position is : ',circle_center_latlon)
                    elif mask_center_string == 'tobac_cell_centroid':
                        mask_center = [int(np.shape(vertical_vel)[1]/2), int(np.shape(vertical_vel)[0]/2)]
                        circle_center_latlon = [cell_lat,cell_lon]

                    ############ WP #############
                    fig  = plt.figure(figsize=(8,8))
                    ax1  = fig.add_subplot(2, 2, 1, projection=crs.PlateCarree())
                    w_plotting =  da['WP'][RAMS_closest_level,ypos-PLOTTING_RANGE:ypos+PLOTTING_RANGE,xpos-PLOTTING_RANGE:xpos+PLOTTING_RANGE]#.values
                    C111 = ax1.contourf(rams_lons ,rams_lats, w_plotting,levels=np.arange(-20,21,1),cmap=current_cmap,extend='both',transform=crs.PlateCarree())#,colors=PLOT_ANOTHER_VAR_CONT[7],linestyles=np.where(levels >= 0, "-", "--"),linewidths=PLOT_ANOTHER_VAR_CONT[8])
                    C7   = ax1.contour(rams_lons ,rams_lats, storm_mask_zeros,  colors='k', linewidths=0.7, linestyles='-')
                    #levels=[0.9],
                    ax1.scatter(rams_lons[chosen_centroid[0],chosen_centroid[1]],rams_lats[chosen_centroid[0],chosen_centroid[1]],marker='^',s=55.5,color='limegreen')
                    tobac_features_scatter = ax1.scatter(cell_lon,cell_lat,label='cell#'+str(CELLNO),marker='.',s=55.5,c='k',transform=crs.PlateCarree())
                    #plt.colorbar(C111,shrink=0.7, pad=0.02,fraction=0.11)
                    ax1.set_title('w (m/s) at height '+str(zpos)+' m AGL; Cell#'+str(CELLNO)+'\n'+get_time_from_RAMS_file(rams_fil)[0])
                    #plot the cell point
                    gd = Geodesic()
                    print('plotting the updraft radius circle with radius: ',updraft_radius)
                    updraft_circle = gd.circle(lon=circle_center_latlon[1], lat=circle_center_latlon[0], radius=updraft_radius)
                    ax1.add_geometries([sgeom.Polygon(updraft_circle)], crs=crs.PlateCarree(), edgecolor='maroon', facecolor="none")
                    ########################################################################################
                    plot_all_other_detected_regions = False
                    if len(regions) > 1:
                        if plot_all_other_detected_regions:
                            for centt,radius,ovlap in list(zip(centroids_other_uds,radii_other_uds,overlap_fraction_list)):
                                centt_int = list(centt) 
                                centt_int[0] = int(centt_int[0])   # convert to integer
                                centt_int[1] = int(centt_int[1])
                                ax1.scatter(rams_lons[centt_int[0],centt_int[1]],rams_lats[centt_int[0],centt_int[1]],marker='^',s=10.5,color='lightgreen')
                                ud_center = [rams_lats[centt_int[0],centt_int[1]].values,rams_lons[centt_int[0],centt_int[1]].values]
                                adjacent_updraft_circle = gd.circle(lon=ud_center[1], lat=ud_center[0], radius=radius*DXY)
                                ax1.add_geometries([sgeom.Polygon(adjacent_updraft_circle)], crs=crs.PlateCarree(), edgecolor='indianred', facecolor="none")
                                ax1.text(ud_center[1],ud_center[0],str(np.round(ovlap,2)))
                    ########################################################################################
                    gl = ax1.gridlines()
                    ax1.coastlines(resolution='50m')
                    gl.xlines = True
                    gl.ylines = True
                    LATLON_LABELS=True
                    gl.xlabels_top = True
                    gl.ylabels_right = False
                    gl.ylabels_left = True
                    gl.ylabels_bottom = False
                    gl.xlabel_style = {'size': 9, 'color': 'gray'}
                    gl.ylabel_style = {'size': 9, 'color': 'gray'}
                    fig.colorbar(C111, ax=ax1, shrink=0.7)#,orientation='horizontal')
                    ############ ITC (from Marinescu) #############
                     # contour and colobar ticks and levels     
                    itc_lvls = np.arange(0.01,10.01,0.01) # Adjusted these levels, such that figure shows regions with at least 1 grid box with 0.1 g/kg of condensate
                    itc_cbar_ticks = np.log10(np.array([1,5,10]))
                    itc_cbar_ticklbls = np.array([1,5,10])
                    # Make new colorbar to blue (no condensate) to white (condensate)
                    from matplotlib.colors import LinearSegmentedColormap
                    colorlist=["darkblue", "lightsteelblue", "white"]
                    newcmp = LinearSegmentedColormap.from_list('testCmap', colors=colorlist, N=256)
                    # Scale size of figure based on dimensions of domain
                    max_dim = np.max([nx,ny])
                    ax2 = fig.add_subplot(2, 2, 2, projection=crs.PlateCarree())
                    C111 = ax2.contourf(rams_lons ,rams_lats, np.log10(itc_mm),levels = np.log10(itc_lvls),cmap=newcmp,extend='both',transform=crs.PlateCarree())#,colors=PLOT_ANOTHER_VAR_CONT[7],linestyles=np.where(levels >= 0, "-", "--"),linewidths=PLOT_ANOTHER_VAR_CONT[8])
                    C7   = ax2.contour (rams_lons ,rams_lats, storm_mask_zeros,  colors='k', linewidths=0.7, linestyles='-')
                    ax2.scatter(rams_lons[chosen_centroid[0],chosen_centroid[1]],rams_lats[chosen_centroid[0],chosen_centroid[1]],marker='^',s=55.5,color='limegreen')
                    tobac_features_scatter = ax2.scatter(cell_lon,cell_lat,label='cell#'+str(CELLNO),marker='.',s=55.5,c='k',transform=crs.PlateCarree())
                    ax2.add_geometries([sgeom.Polygon(updraft_circle)], crs=crs.PlateCarree(), edgecolor='maroon', facecolor="none")
                    #plt.colorbar(C111,shrink=0.7, pad=0.02,fraction=0.11)
                    ax2.set_title('ITC (mm) for Cell#'+str(CELLNO)+'\n'+get_time_from_RAMS_file(rams_fil)[0])
                    gl = ax2.gridlines()
                    ax2.coastlines(resolution='50m')
                    gl.xlines = True
                    gl.ylines = True
                    gl.xlabels_top = True
                    gl.ylabels_right = False
                    gl.ylabels_left = False
                    gl.ylabels_bottom = False
                    gl.xlabel_style = {'size': 9, 'color': 'gray'}
                    gl.ylabel_style = {'size': 9, 'color': 'gray'}
                    cbar = plt.colorbar(C111,ax=ax2,ticks=itc_cbar_ticks,shrink=0.7)#,orientation='horizontal')
                    cbar.ax.set_yticklabels(itc_cbar_ticklbls)
                    # plot the cell point
#                         updraft_circle = gd.circle(lon=circle_center_latlon[1], lat=circle_center_latlon[0], radius=updraft_radius*DXY)
#                         ax2.add_geometries([sgeom.Polygon(updraft_circle)], crs=crs.PlateCarree(), edgecolor='maroon', facecolor="none")
                    ############ IWP (from Marinescu) #############
                    ax3   = fig.add_subplot(2, 2, 3, projection=crs.PlateCarree())
                    C_iwp = ax3.contourf(rams_lons ,rams_lats, np.log10(iwp_mm),levels = np.log10(itc_lvls),cmap=newcmp,extend='both',transform=crs.PlateCarree())#,colors=PLOT_ANOTHER_VAR_CONT[7],linestyles=np.where(levels >= 0, "-", "--"),linewidths=PLOT_ANOTHER_VAR_CONT[8])
                    C7    = ax3.contour(rams_lons ,rams_lats, storm_mask_zeros,  colors='k', linewidths=0.7, linestyles='-')
                    ax3.scatter(rams_lons[chosen_centroid[0],chosen_centroid[1]],rams_lats[chosen_centroid[0],chosen_centroid[1]],marker='^',s=55.5,color='limegreen')
                    tobac_features_scatter = ax3.scatter(cell_lon,cell_lat,label='cell#'+str(CELLNO),marker='.',s=55.5,c='k',transform=crs.PlateCarree())
                    ax3.add_geometries([sgeom.Polygon(updraft_circle)], crs=crs.PlateCarree(), edgecolor='maroon', facecolor="none")
                    #plt.colorbar(C111,shrink=0.7, pad=0.02,fraction=0.11)
                    ax3.set_title('IWP (mm) for Cell#'+str(CELLNO)+'\n'+get_time_from_RAMS_file(rams_fil)[0])
                    gl = ax3.gridlines()
                    ax3.coastlines(resolution='50m')
                    gl.xlines = True
                    gl.ylines = True
                    LATLON_LABELS=True
                    gl.xlabels_top = False
                    gl.ylabels_right = False
                    gl.ylabels_left = True
                    gl.ylabels_bottom = True
                    gl.xlabel_style = {'size': 9, 'color': 'gray'}
                    gl.ylabel_style = {'size': 9, 'color': 'gray'}
                    cbar = plt.colorbar(C_iwp,ax=ax3,ticks=itc_cbar_ticks,shrink=0.7)#,orientation='horizontal')
                    cbar.ax.set_yticklabels(itc_cbar_ticklbls)
                    # plot the cell point
#                       updraft_circle = gd.circle(lon=circle_center_latlon[1], lat=circle_center_latlon[0], radius=updraft_radius*DXY)
#                       ax3.add_geometries([sgeom.Polygon(updraft_circle)], crs=crs.PlateCarree(), edgecolor='maroon', facecolor="none")
                    ############ RR# #############
                    ax4 = fig.add_subplot(2, 2, 4, projection=crs.PlateCarree())
                    C111 = ax4.contourf(rams_lons ,rams_lats, rain,extend='both',cmap=cma4,levels=np.arange(5,35.5,.5),transform=crs.PlateCarree())#,colors=PLOT_ANOTHER_VAR_CONT[7],linestyles=np.where(levels >= 0, "-", "--"),linewidths=PLOT_ANOTHER_VAR_CONT[8])
                    C7   = ax4.contour(rams_lons  ,rams_lats, storm_mask_zeros,  colors='k', linewidths=0.7, linestyles='-')
                    ax4.scatter(rams_lons[chosen_centroid[0],chosen_centroid[1]],rams_lats[chosen_centroid[0],chosen_centroid[1]],marker='^',s=55.5,color='limegreen')
                    tobac_features_scatter = ax4.scatter(cell_lon,cell_lat,label='cell#'+str(CELLNO),marker='.',s=55.5,c='k',transform=crs.PlateCarree())
                    ax4.add_geometries([sgeom.Polygon(updraft_circle)], crs=crs.PlateCarree(), edgecolor='maroon', facecolor="none")
                    #plt.colorbar(C111,shrink=0.7, pad=0.02,fraction=0.11)
                    ax4.set_title('Rain rate (mm/h) for Cell#'+str(CELLNO)+'\n'+get_time_from_RAMS_file(rams_fil)[0])
                    gl = ax4.gridlines()
                    ax4.coastlines(resolution='50m')
                    gl.xlines = True
                    gl.ylines = True
                    LATLON_LABELS=True
                    gl.xlabels_top = False
                    gl.ylabels_right = False
                    gl.ylabels_left = False
                    gl.ylabels_bottom = True
                    gl.xlabel_style = {'size': 9, 'color': 'gray'}
                    gl.ylabel_style = {'size': 9, 'color': 'gray'}
                    fig.colorbar(C111, ax=ax4, shrink=0.7)#,orientation='horizontal')
                    # Adjust spacing b/w subplots
                    plt.subplots_adjust(left=0.1,bottom=0.1,right=0.9,top=0.9,wspace=0.1,hspace=0.0001)
                    #png_save_folder ='/nobackupp11/isingh2/tobac_plots/sounding_csvs_and_WP_snapshots/'  # Pleaides
                    png_save_folder =  '/home/isingh/code/scratch/environmental_assessment/'              # CSU machine
                    pngfile = 'four_panel_wp_itc_iwp_rain_'+DOMAIN+'_cell'+str(CELLNO)+'_'+get_time_from_RAMS_file(rams_fil)[1]+'_comb_track_filt_01_02_50_02_sr5017_setpos.png'
                    print('saving image to file: ',png_save_folder+pngfile)
                    plt.savefig(png_save_folder+pngfile,dpi = 75)
                    plt.close()
                    #plt.tight_layout(pad=1.0)
#                         import matplotlib.transforms as transforms
#                         trans = transforms.blended_transform_factory(ax.transAxes,ax.transData)
#                         props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
#                         ax.text(0.52, 0.94, 'cell radius = '+str(np.round(updraft_radius*DXY/1000.,1))+\
#                                 ' km'+'\n'+'tobac cell detection threshold ='+str(int(thresholds[ii]))+' m/s\n'+\
#                                 'w threshold for sounding ='+str(int(FIXED_THREHOLD))+' m/s\n'+\
#                                 'environment width = '+str(int(env_radius*DXY/1000.0))+' km',
#                                 fontsize=9,verticalalignment='top',\
#                                 bbox=props,transform=ax.transAxes)


        print('===============================\n\n')
    
    print('saving cell properties to ',DOMAIN+'_cell_'+str(CELLNO)+'_properties_comb_track_filt_01_02_50_02_sr5017_setpos.csv')
    g.to_csv('/home/isingh/code/scratch/environmental_assessment/'+DOMAIN+'_cell_'+str(CELLNO)+'_properties_comb_track_filt_01_02_50_02_sr5017_setpos.csv')
    return g

############################################################################
############################################################################
############################################################################
# Paths to model data and where to save data
domain='DRC1.1-R'
#path = '/nobackup/pmarines/DATA_FM/'+domain+'/LES_data/'     # Pleiades
#path = '/Users/isingh/SVH/INCUS/sample_LES_data/'+domain+'/' # personal macbook
path ='/monsoon/LES_MODEL_DATA/'+domain+'/G3/out_30s/'        # CSU machine

#savepath = './'
#tobac_data='/nobackup/pmarines/DATA_FM/'+domain+'/tobac_data/'                                 # Pleiades
#tobac_data='/Users/isingh/SVH/INCUS/jupyter_nbks/tobac_thermals/peter_tobac_output/'+domain+'/'# personal macbook
tobac_data='/monsoon/pmarin/Tracking/Updrafts/'+domain+'/tobac_data/'                           # CSU machine

#tobac_filename  = 'comb_track_filt_01_02_05_10_20.p'
tobac_filename   = 'comb_track_filt_01_02_50_02_sr5017_setpos.p'
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

da_dummy = xr.open_dataset(h5files1[0],engine='h5netcdf', phony_dims='sort')

domain_z_dim,domain_y_dim,domain_x_dim=np.shape(da_dummy['WP'])
print('domain_z_dim: ',domain_z_dim)
print('domain_y_dim: ',domain_y_dim)
print('domain_x_dim: ',domain_x_dim)

zm, zt, nx, ny, dxy, npa = read_head(hefiles1[0],h5files1[0])

#******************
PLOTTING_RANGE = 75 # 25 km # need to know now for filtering cells close to edges
#******************

##### read in tobac data #####
print('reading ',tobac_filepath)
tdata = pd.read_pickle(tobac_filepath)

def filter_cells(g):
    return ((g.zmn.max() >= 2000.0) & (g.zmn.min() <= 15000.) & (g.X.max() <= domain_x_dim-PLOTTING_RANGE-1) & (g.X.min() >= PLOTTING_RANGE+1) &
            (g.Y.max() <= domain_y_dim-PLOTTING_RANGE-1) & (g.Y.min() >= PLOTTING_RANGE+1) & (g.threshold_value.count() >= 10)\
             & (pd.to_datetime(g.timestr).min() > pd.to_datetime(start_time)) & (pd.to_datetime(g.timestr).max() <  pd.to_datetime(end_time)) \
            )

tdata_temp=tdata.groupby('cell').filter(filter_cells)
#print(tdata_temp)

#all_cells = tdata_temp.cell.unique()
#print('number of unique cells identified: ',len(all_cells))
#print('these cells are: ',all_cells)
############################################################################


# # Running in the notebook
# cn = random.choice(all_cells)
# print('cell#: ',cn)
# tdata_subset=tdata_temp[tdata_temp['cell']==cn]
# #print(tdata_subset)
# output_df = get_cell_W_area_rad_itc_iwp_rain(domain,dxy,2.0,tdata_subset,cn,zm,zt,100,UPDRAFT_CONTAMINATION_CHECK=False,PLOT_QUANTITIES=True)#(domain,dxy,tdata_subset,cn,zm,zt,PLOTTING_RANGE,2.0)
#output_df.to_csv('./cell_properties_'+domain+'_cell'+str(cn)+'.csv')
# #

already_done_cells=[76725, 70551, 47915, 34238, 15669, 81956, 40050, 67063, 14305, 69909, 29100, 17193, 37121, 52425,\
                    78575, 17050, 79269, 14080, 41566, 62830, 15001, 14048, 56979, 14935, 44436, 86891, 70681, 82133,\
                    66438, 62013, 70563, 48039, 74684, 31830, 12517, 34318, 29621, 8961, 46674, 11239, 17860, 34939,\
                    75306, 59846, 22353, 17862, 17165, 23153, 80186, 28327, 18431, 9687, 17160, 11816, 83535, 29837,\
                    12508, 14947, 22905, 20083, 56310, 73919, 48708, 35735, 40123, 15772, 34933, 43635, 25444, 29833,\
                    28435, 49257]

total_cells= [8776, 8800, 8961, 9663, 9687, 10283, 11239, 11776, 11816, 11910, 11919, 12508, 12517, 12534, 13311, 14048,\
              14080, 14305, 14935, 14947, 14989, 15001, 15047, 15070, 15635, 15650, 15657, 15669, 15752, 15772, 16277, 16392,\
              16404, 16426, 16450, 17050, 17052, 17061, 17160, 17165, 17193, 17633, 17834, 17860, 17862, 18431, 19181, 19898,\
              20061, 20083, 21744, 22353, 22496, 22905, 23153, 23181, 24569, 25444, 25475, 26747, 26857, 26864, 26933, 27576,\
              27650, 28322, 28327, 28340, 28435, 28545, 29067, 29100, 29621, 29833, 29837, 29889, 30520, 30526, 31224, 31830,\
              32010, 32610, 32714, 33358, 34238, 34318, 34933, 34939, 35611, 35735, 36254, 36367, 37121, 37808, 38468, 38600,\
              40050, 40123, 40505, 41338, 41566, 42197, 42422, 43547, 43635, 43651, 44436, 45164, 45185, 45280, 45856, 45892,\
              45928, 46530, 46674, 47321, 47915, 48039, 48708, 48714, 49257, 49490, 51073, 51088, 51653, 51867, 52393, 52425,\
              52521, 53290, 54057, 54132, 54171, 54908, 56044, 56243, 56310, 56979, 57761, 59143, 59778, 59846, 59906, 60658,\
              61618, 62013, 62038, 62070, 62830, 62915, 63471, 64226, 64977, 65009, 65563, 65682, 66425, 66438, 67063, 67149,\
              69213, 69841, 69909, 70551, 70563, 70681, 71871, 72618, 73919, 73941, 74684, 74712, 75306, 75323, 75342, 76540,\
              76725, 77789, 78575, 79269, 79361, 79465, 79466, 79617, 80119, 80186, 80870, 81477, 81956, 81959, 82133, 82791,\
              83435, 83535, 84105, 85485, 86891, 87634, 88337, 90184]

all_cells = [cc for cc in total_cells if cc not in already_done_cells]
print('will work on <<',len(all_cells),'>>',all_cells)
#Running on the terminal in parallel
argument = []
for cn in all_cells:
    tdata_subset=tdata_temp[tdata_temp['cell']==cn]
    argument = argument + [(domain,dxy,2.0,tdata_subset,cn,zm,zt,100,False,True)]

print('length of argument is: ',len(argument))

# # ############################### FIRST OF ALL ################################
cpu_count1 = 4 #cpu_count()
print('number of cpus: ',cpu_count1)
# # #############################################################################

def main(DOMAIN, FUNCTION, ARGUMENT):
    start_time = time.perf_counter()
    with Pool(processes = cpu_count1) as pool:
        #for _ in tqdm.tqdm(pool.istarmap(FUNCTION, argument),total=len(argument)):
        #    pass
        data = pool.starmap(FUNCTION, ARGUMENT)
    #output_df = pd.concat(data, axis=0)
    #output_df.to_csv('/home/isingh/code/scratch/environmental_assessment/'+DOMAIN+'_cell_properties.csv')
    #print('saving cell properties to : ','/home/isingh/code/scratch/environmental_assessment/'+DOMAIN+'_cell_properties.csv')
    finish_time = time.perf_counter()
    print(f"Program finished in {finish_time-start_time} seconds")

if __name__ == "__main__":
    main(domain, get_cell_W_area_rad_itc_iwp_rain, argument)
