# parallel version
import datetime
import glob
import os
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker
import metpy.calc as metcalc
import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.dates import DateFormatter
from matplotlib.ticker import FormatStrFormatter, LinearLocator
import nclcmaps as ncm
import cartopy.crs as crs
import matplotlib.ticker as mticker
from cartopy.feature import NaturalEarthFeature
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from matplotlib.cm import get_cmap


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
cma4=ncm.cmap("WhiteBlueGreenYellowRed")
cma5=plt.get_cmap('gray_r')
cma6=plt.get_cmap('rainbow')
cma7=plt.get_cmap('Oranges')
cma8=plt.get_cmap('coolwarm')
cma9=cma4.reversed()
cma10=plt.get_cmap('gist_yarg')

from matplotlib.colors import LinearSegmentedColormap
colorlist=["darkblue", "lightsteelblue", "white"]
newcmp = LinearSegmentedColormap.from_list('testCmap', colors=colorlist, N=256)
                
#import rams_tools
import h5py
import hdf5plugin
from RAMS_Post_Process import fx_postproc_RAMS as RAMS_fx
#from memory_profiler import profile

from wrf import (
    CoordPair,
    GeoBounds,
    cartopy_xlim,
    cartopy_ylim,
    get_cartopy,
    getvar,
    interplevel,
    interpline,
    latlon_coords,
    ll_to_xy,
    smooth2d,
    to_np,
    vertcross,
    xy_to_ll,
    ll_to_xy_proj
)

from multiprocessing import Pool, cpu_count
import time
import random

#rc('mathtext', default='regular')
#matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams['axes.facecolor'] = [0.9,0.9,0.9]
matplotlib.rcParams['axes.labelsize'] = 18
matplotlib.rcParams['axes.titlesize'] = 18
matplotlib.rcParams['xtick.labelsize'] = 17
matplotlib.rcParams['ytick.labelsize'] = 17
matplotlib.rcParams['legend.fontsize'] = 18
matplotlib.rcParams['legend.facecolor'] = 'w'
matplotlib.rcParams['hatch.linewidth'] = 0.25
matplotlib.rcParams['font.family'] = 'Helvetica'

def get_time_from_RAMS_file(INPUT_FILE):
    cur_time = os.path.split(INPUT_FILE)[1][4:21] # Grab time string from RAMS file
    pd_time = pd.to_datetime(cur_time[0:10]+' '+cur_time[11:13]+":"+cur_time[13:15]+":"+cur_time[15:17])
    return pd_time.strftime('%Y-%m-%d %H:%M:%S'), pd_time.strftime('%Y%m%d%H%M%S')

def read_var(filename,varname):
    with h5py.File(filename,"r") as f:
        data_out = f[varname][:]
    return data_out

def read_3dvar_subset(filename,varname,X1,X2,Y1,Y2):
    with h5py.File(filename,"r") as f:
        data_out = f[varname][:,Y1:Y2,X1:X2]
    return data_out

def read_2dvar_subset(filename,varname,X1,X2,Y1,Y2):
    with h5py.File(filename,"r") as f:
        data_out = f[varname][Y1:Y2,X1:X2]
    return data_out

#@profile
def plot_w_itc_WRF_RAMS_comparison(WRF_FILENAME,RAMS_FILENAME,ZT,savepath):
    print('RAMS fileame: ',RAMS_FILENAME)
    print('WRF fileame: ',WRF_FILENAME)
    # Constants for calculating total integrated condensate
    cp = 1004; # J/kg/K
    rd = 287; # J/kg/K
    p00 = 100000; # Reference Pressure
    ########### WRF PORTION ###########
    print('calculating WRF ITC for file: ',WRF_FILENAME)
    #NC_FILE = Dataset(WRF_FILENAME)
    ncfile = Dataset(WRF_FILENAME)
    wrf_terr = getvar(ncfile,'ter')
    wrf_lats, wrf_lons = latlon_coords(wrf_terr)
    da_wrf = xr.open_dataset(WRF_FILENAME)
    qq =  da_wrf['QCLOUD'].squeeze() + da_wrf['QGRAUP'].squeeze() + da_wrf['QICE'].squeeze() + da_wrf['QRAIN'].squeeze() + da_wrf['QSNOW'].squeeze()
    moist_pot_temp_pert = da_wrf['THM'] .squeeze()
    base_temp = da_wrf['T00'].squeeze() 
    moist_pot_temp = moist_pot_temp_pert + base_temp
    qv   = da_wrf['QVAPOR'].squeeze()
    theta = moist_pot_temp/(1+1.61*qv)
    p = da_wrf['P'].squeeze() + da_wrf['PB'].squeeze()
    tk = theta/((100000.0/p)**0.286)
    heights = (da_wrf['PH'].squeeze()+da_wrf['PHB'].squeeze())/9.81
    rho = p/(281.0 * tk)
    del(theta,p,tk,qv,moist_pot_temp,moist_pot_temp_pert)
    dz =  heights[1:,:,:] - heights[:-1,:,:]
    itc_wrf = np.nansum(qq.values*rho.values*dz.values,axis=0) # integrated total condensate in kg
    del(rho,qq,dz)
    itc_mm_wrf = itc_wrf/997.0*1000.0 # integrated total condensate in mm
    #print(itc_mm_wrf)
    print('shape of ITC WRF is ',np.shape(itc_mm_wrf))
    print('done calculating ITC (mm) WRF')
    wrf_times = da_wrf['XTIME']
    cur_time_wrf = np.datetime_as_string(wrf_times.values, timezone='UTC',unit='m')
    #---------------#---------------#---------------#---------------#---------------#---------------
    
    ########### RAMS PORTION ###########
    print('calculating RAMS ITC for file: ',RAMS_FILENAME)
    cur_time = os.path.split(RAMS_FILENAME)[1][9:21]
    ds_rams=xr.open_dataset(RAMS_FILENAME,engine='h5netcdf', phony_dims='sort')
    rams_lats=ds_rams.GLAT.values
    rams_lons=ds_rams.GLON.values
    rams_terr=ds_rams.TOPT.values
    #zm, zt, nx, ny, dxy, npa = RAMS_fx.read_head(hefiles1[0],h5files1[0])
    rams_time, rams_time_savestr = get_time_from_RAMS_file(RAMS_FILENAME)
    print('Time in this file ',rams_time)

    wp = ds_rams['WP']
    nx = np.shape(wp)[2]
    ny = np.shape(wp)[1]
    rtp = ds_rams['RTP'] - ds_rams['RV']
    th = ds_rams['THETA']
    pi = ds_rams['PI']
    rv = ds_rams['RV']
    # Convert RAMS native variables to temperature and pressure
    pres = np.power((pi/cp),cp/rd)*p00
    temp = th*(pi/cp)
    del(th,pi)
    # Calculate atmospheric density
    dens = pres/(rd*temp*(1+0.61*rv))
    del(pres,temp,rv)
    # Difference in heights (dz)    
    diff_zt_3D = np.tile(np.diff(ZT),(int(ny),int(nx),1))
    diff_zt_3D = np.moveaxis(diff_zt_3D,2,0)
    # Calculate integrated condensate
    itc = np.nansum(rtp[1:,:,:]*dens[1:,:,:]*diff_zt_3D,axis=0) # integrated total condensate in kg
    itc_mm_rams = itc/997*1000 # integrated total condensate in mm
    itc_mm_rams[itc_mm_rams<=0] = 0.001
    print('shape of ITC RAMS is ',np.shape(itc_mm_rams))
    print('done calculating ITC (mm) RAMS')
    ######################## PLOTTING ##########################################
    itc_lvls = np.arange(0.01,10.01,0.01) # Adjusted these levels, such that figure shows regions with at least 1 grid box with 0.1 g/kg of condensate
    itc_cbar_ticks = np.log10(np.array([1,5,10]))
    itc_cbar_ticklbls = np.array([1,5,10])
    
    # Scale size of figure based on dimensions of domain
    max_dim = np.max([nx,ny])
    fs_scale = 9
    lw = 1.0
   
    # Plot Figure
    fig, axs = plt.subplots(nrows=2,ncols=1,subplot_kw={'projection': crs.PlateCarree()},figsize=(11,11))
    # axs is a 2 dimensional array of `GeoAxes`.  We will flatten it into a 1-D array
    axs=axs.flatten()
    
    itc_wrf_cont  = axs[0].contourf(wrf_lons ,wrf_lats ,np.log10(itc_mm_wrf), levels=np.log10(itc_lvls),transform=crs.PlateCarree(),cmap=newcmp,extend='both')
    wrf_terr      = axs[0].contour (wrf_lons ,wrf_lats ,wrf_terr,levels=[500.],transform=crs.PlateCarree(),linewidths=1.4,colors="saddlebrown")
    itc_rams_cont = axs[1].contourf(rams_lons,rams_lats,np.log10(itc_mm_rams),levels=np.log10(itc_lvls),transform=crs.PlateCarree(),cmap=newcmp,extend='both')
    rams_terr     = axs[1].contour (rams_lons,rams_lats,rams_terr,levels=[500.],transform=crs.PlateCarree(),linewidths=1.4,colors="saddlebrown")

    axs[0].set_title ('WRF: Int. Total Condensate (mm; shaded) at '+cur_time, size=12)
    axs[1].set_title('RAMS: Int. Total Condensate (mm; shaded) at '+cur_time, size=12)
    #----------
    gl1 = axs[0].gridlines()#color="gray",alpha=0.5, linestyle='--',draw_labels=True,linewidth=2)
    axs[0].coastlines(resolution='110m')
    gl1.xlines = True
    gl1.ylines = True
    LATLON_LABELS=True
    if LATLON_LABELS:
        print('LATLON labels are on')
        gl1.xlabels_top = True
        gl1.ylabels_right = False
        gl1.ylabels_left = True
        gl1.ylabels_bottom = True
    else:
        gl1.xlabels_top = False
        gl1.ylabels_right = False
        gl1.ylabels_left = False
        gl1.ylabels_bottom = True
    gl1.xlabel_style = {'size': 15, 'color': 'gray'}#, 'weight': 'bold'}
    gl1.ylabel_style = {'size': 15, 'color': 'gray'}#, 'weight': 'bold'}
    #----------
    gl2 = axs[1].gridlines()#color="gray",alpha=0.5, linestyle='--',draw_labels=True,linewidth=2)
    axs[1].coastlines(resolution='110m')
    gl2.xlines = True
    gl2.ylines = True
    LATLON_LABELS=True
    if LATLON_LABELS:
        print('LATLON labels are on')
        gl2.xlabels_top = True
        gl2.ylabels_right = False
        gl2.ylabels_left = True
        gl2.ylabels_bottom = True
    else:
        gl2.xlabels_top = False
        gl2.ylabels_right = False
        gl2.ylabels_left = False
        gl2.ylabels_bottom = True
    gl2.xlabel_style = {'size': 15, 'color': 'gray'}#, 'weight': 'bold'}
    gl2.ylabel_style = {'size': 15, 'color': 'gray'}#, 'weight': 'bold'}
    
    # COLORBAR
    # Add a colorbar axis at the bottom of the graph
    cbar_ax = fig.add_axes([0.2, -0.017, 0.6, 0.015])
    cbar=fig.colorbar(rams_terr, cax=cbar_ax,orientation='horizontal',ticks=itc_cbar_ticks)
    cbar.ax.set_yticklabels(itc_cbar_ticklbls)
    cbar.ax.set_ylabel('Integrated Total Condensate (mm)')

    plt.tight_layout()
    png_name = 'WRF_RAMS_D03_ITC_mm_comparison_cbar_'+cur_time+'.png'
    print('saving to file: ',png_name)
    fig.savefig(savepath+png_name,dpi=200)
    plt.close()
    return
###############################################################
domain_wrf = "d03"  #2019-04-21 13
domain_rams = "g3"

if domain_wrf=='d01':
    grid_spacing = 1600.
    winds_thin=25
elif domain_wrf=='d02':
    grid_spacing=400.
    winds_thin=30
else:
    print('provide correct value of grid spacing!!!')
    
directory_wrf="/nobackupp11/isingh2/WRF_final_for_testing/WRF/run/FIRST_RUN/"
#directory_rams="/nobackup/pmarines/PROD/AUS1.1-R/G12/out/"       #G1 and G2
directory_rams="/nobackupp12/pmarines/PROD/AUS1.1-R/G3/out_30s/"  #G3

start_time = '2006-01-23 11:40:00'
end_time   = '2006-01-23 11:49:30'

#date_range = pd.date_range(start_time,end_time,freq='30min')
date_range  = pd.date_range(start_time,end_time,freq='30S')

fi_list_wrf  = []
fi_list_rams = []


for times in date_range:
    file_finding_string_wrf = "wrfout_"+domain_wrf+"_"+times.strftime('%Y-%m-%d_%H:%M:%S')
    #print('file_finding_string_wrf: ',file_finding_string_wrf)
    fi_list_wrf.append(sorted(glob.glob(directory_wrf+file_finding_string_wrf))[0])
    
    #file_finding_string_rams = "a-A-"+times.strftime('%Y-%m-%d-%H%M%S')+"-"+domain_rams+".h5"
    file_finding_string_rams  = "a-L-"+times.strftime('%Y-%m-%d-%H%M%S')+"-"+domain_rams+".h5"
    #print('file_finding_string_rams: ',file_finding_string_rams)
    fi_list_rams.append(sorted(glob.glob(directory_rams+file_finding_string_rams))[0])#.pop()
    
print('number of WRF files found: ',len(fi_list_wrf))
print('WRF files: ',fi_list_wrf)
    
print('number of RAMS files found: ',len(fi_list_rams))
print('RAMS files: ',fi_list_rams)
print('------\n\n')
    
hefilepath = directory_rams+'a-A*head.txt'
hefiles1 = sorted(glob.glob(hefilepath))

zm, zt, nx, ny, dxy, npa = RAMS_fx.read_head(hefiles1[0],fi_list_rams[0]) 

argument = []
for ii in range(len(fi_list_wrf)):
    argument = argument + [(fi_list_wrf[ii],fi_list_rams[ii],zt,'./')]

#print(argument)
print('will make plots for ',len(argument),' filesx2','\n\n\n')
print('first argument is ',argument[0])

# single processor
#plot_w_itc_WRF_RAMS_comparison(*random.choice(argument))

#multiple processors
#cpu_count1 = cpu_count()

def main(FUNCTION, ARGUMENT):
    #pool = Pool(int(cpu_count1/8))
    pool = Pool(1)
    start_time = time.perf_counter()
    results = pool.starmap(FUNCTION, ARGUMENT)
    finish_time = time.perf_counter()
    print(f"Program finished in {finish_time-start_time} seconds")

if __name__ == "__main__":
    main(plot_w_itc_WRF_RAMS_comparison, argument)
