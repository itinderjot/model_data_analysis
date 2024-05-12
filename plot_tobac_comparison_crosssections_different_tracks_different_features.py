# Script created by:
# Itinderjot Singh
# Colorado State University
# itinder@colostate.edu
## plot features (besides the cell being tracked) in the plan view
import numpy as np
import matplotlib
import numpy.ma as ma
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Rectangle
import datetime
import sys
import glob
import os
import h5py
import hdf5plugin
import pandas as pd
import xarray as xr
from RAMS_Post_Process import fx_postproc_RAMS as RAMS_fx
#import cartopy.crs as crs

import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from matplotlib import ticker
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D


from multiprocessing import Pool, cpu_count
import os
import time
import random

cma1=plt.get_cmap('bwr')
cma3=plt.get_cmap('tab20c')
cma5=plt.get_cmap('gray_r')
cma6=plt.get_cmap('rainbow')
cma7=plt.get_cmap('Oranges')
cma8=plt.get_cmap('coolwarm')
#cma9=cma4.reversed()
cma10=plt.get_cmap('gist_yarg')

Cp=1004.
Rd=287.0
p00 = 100000.0


def get_time_from_RAMS_file(INPUT_FILE):
    cur_time = os.path.split(INPUT_FILE)[1][4:21] # Grab time string from RAMS file
    pd_time = pd.to_datetime(cur_time[0:10]+' '+cur_time[11:13]+":"+cur_time[13:15]+":"+cur_time[15:17])
    return pd_time.strftime('%Y-%m-%d %H:%M:%S'), pd_time.strftime('%Y%m%d%H%M%S')

def fig_process_vert(AX, TER, CONTOUR, XY_1D, XY1, XY2, Y_CROSS, PLOT_COLORBAR, CBAR_EXP, FONTSIZE, TITLESTRING, TIMESTRING, FILENAMESTRING, PROD, UNITS, VERT_CROSS, HEIGHT, YLABEL, IS_PANEL_PLOT):
    #F = plt.gcf()  # Gets the current figure
    #ax = plt.gca()  # Gets the current axes

    if IS_PANEL_PLOT == False:
        AX.set_title('%s (%s) \n %s' % (TITLESTRING, UNITS, TIMESTRING),
                  fontsize=FONTSIZE, stretch='normal')

    if VERT_CROSS == "zonal":
        AX.fill_between(XY_1D[XY1:XY2], 0, TER[int(Y_CROSS),
                        XY1:XY2]/1000.0, facecolor='wheat')
        AX.set_xlabel('x-distance (km)', fontsize=FONTSIZE)

    else:
        AX.fill_between(XY_1D[XY1:XY2], 0, TER[XY1:XY2,
                        int(Y_CROSS)]/1000.0, facecolor='wheat')
        AX.set_xlabel('y-distance (km)', fontsize=FONTSIZE)

    AX.patch.set_color("white")

    if YLABEL:
        AX.set_ylabel('Height (km)', fontsize=FONTSIZE)
    AX.set_ylim([0, HEIGHT])
    AX.set_xlim([XY1*100.0/1000.0, XY2*100.0/1000.0])

    class OOMFormatter(matplotlib.ticker.ScalarFormatter):
        def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
            self.oom = order
            self.fformat = fformat
            matplotlib.ticker.ScalarFormatter.__init__(
                self, useOffset=offset, useMathText=mathText)

        def _set_orderOfMagnitude(self, nothing):
            self.orderOfMagnitude = self.oom

        def _set_format(self, vmin, vmax):
            self.format = self.fformat
            if self._useMathText:
                self.format = '$%s$' % matplotlib.ticker._mathdefault(
                    self.format)

    if PLOT_COLORBAR:
        if IS_PANEL_PLOT == False:
            if abs(CBAR_EXP):
                divider = make_axes_locatable(AX)
                cax = divider.append_axes("bottom", size="2%", pad=0.6)
                bar = plt.colorbar(CONTOUR, cax=cax, orientation="horizontal",
                                   format=OOMFormatter(CBAR_EXP, mathText=False),extend='both')
                bar.ax.tick_params(labelsize=FONTSIZE-1)
                #file_id = '%s_%s' % (PROD, FILENAMESTRING)
                #filename = '%s.png' % (file_id)
                #print(filename)
                # Saves the figure with small margins
                #plt.savefig(filename, dpi=my_dpi, bbox_inches='tight')
            else:
                divider = make_axes_locatable(AX)
                cax = divider.append_axes("bottom", size="2%", pad=0.66)
                bar = plt.colorbar(CONTOUR, cax=cax, orientation="horizontal",extend='both')
                bar.ax.tick_params(labelsize=FONTSIZE-1)
                #file_id = '%s_%s' % (PROD, FILENAMESTRING)
                #filename = '%s.png' % (file_id)
                #print(filename)
                # Saves the figure with small margins
                #plt.savefig(filename, dpi=my_dpi, bbox_inches='tight')

        else:
            if abs(CBAR_EXP):
                divider = make_axes_locatable(AX)
                cax = divider.append_axes("bottom", size="2%", pad=0.4)
                bar = plt.colorbar(CONTOUR, cax=cax, orientation="horizontal",
                                   format=OOMFormatter(CBAR_EXP, mathText=False),extend='both')
                bar.ax.tick_params(labelsize=FONTSIZE-1)
            else:
                divider = make_axes_locatable(AX)
                cax = divider.append_axes("bottom", size="2%", pad=0.4)
                bar = plt.colorbar(CONTOUR, cax=cax, orientation="horizontal",extend='both')
                bar.ax.tick_params(labelsize=FONTSIZE-1)
        # plt.close() This should remain commented. plt should be closed in the panel plot function
        # if export_flag == 1:
        # Convert the figure to a gif file
        #os.system('convert -render -flatten %s %s.gif' % (filename, file_id))
        #os.system('rm -f %s' % filename)

def plot_zonal_vertcross(DOMAIN, DATA, DX, DY ,TERR, X_1D, Z_3D, Z_1D, DXY, VAR1, DESTAGGER, STAGGER_DIM, LEVELS_VAR1, CMAP_VAR1,
                       VAR2_XR, LEVELS_VAR2, VAR2_COLOR,
                       VAR3, LEVELS_VAR3, VAR3_COLOR,
                       VAR4, LEVELS_VAR4, VAR4_COLOR,
                       yy, x1, x2,
                       PLOT_WINDS, PANEL_PLOT, HEIGHT, AX, PLOT_CBAR, EXP_LABEL, PANEL_LABEL, XPOS, YPOS, ZPOS, ZPOS_GRID, CELL_DIM, FEATURE_NUM, PLOT_SEGMENTATION_OUTPUT, NUM_MODEL_LEVS, TITLETIME, FILENAMETIME):  # rcParams["contour.negative_linestyle"] = 'dashed'

    y1 = yy - 50
    y2 = yy + 50
       
    XV1, __ = np.meshgrid(X_1D[x1:x2]/1000.0, Z_1D/1000.0)
    z_3D_2D_slice    =  (TERR[yy,x1:x2] + Z_1D[:,np.newaxis])/1000.0

    if DESTAGGER:
        print(var1)
        var1 = destagger(var1, STAGGER_DIM, meta=True)
    else:
        print(' ')

    if (isinstance(LEVELS_VAR1, np.ndarray)):
        C1 = AX.contourf(XV1,z_3D_2D_slice, VAR2_XR[:, yy, x1:x2], levels=LEVELS_VAR1,
                           cmap=CMAP_VAR1, extend='both')  # Spectral for qv
    else:
        C1 = AX.contourf(XV1, z_3D_2D_slice, VAR2_XR[:, yy, x1:x2],
                           cmap=CMAP_VAR1, extend='both')
       
    if (isinstance(LEVELS_VAR2, np.ndarray)):
        C2 = AX.contour(XV1, z_3D_2D_slice, VAR2_XR[:, yy, x1:x2],
                             levels=LEVELS_VAR2, colors=VAR2_COLOR, linewidths=1., linestyles="-")
    else:
        C2 = AX.contour(XV1, z_3D_2D_slice, VAR2_XR[:, yy, x1:x2],
                             colors='k', linewidths=1., linestyles="--")
    AX.clabel(C2, inline=1, fontsize=10, fmt='%3.0f')

    if VAR3=='RTP-RV_g/kg':
        total_condensate_zonal = DATA["RTP"][:, yy, x1:x2]*1000.0 - DATA["RV"][:, yy, x1:x2]*1000.0
        if (isinstance(LEVELS_VAR3, np.ndarray)):
            C3 = AX.contour(XV1,z_3D_2D_slice, np.absolute(total_condensate_zonal),
                                 levels=LEVELS_VAR3, colors=VAR3_COLOR, linewidths=2.0, linestyles="--") 
            AX.clabel(C3, inline=1, fontsize=10, fmt='%3.3f')
            
        else:
            print('please provide levels!')
            
    elif VAR3=='RTP-RV_g/m3':
        th = DATA['THETA'][:, yy, x1:x2]
        pi = DATA['PI'][:, yy, x1:x2]
        rv = DATA['RV'][:, yy, x1:x2]
        pres = np.power((pi/Cp),Cp/Rd)*p00
        temp = th*(pi/Cp)
        del(th,pi)
        dens = pres/(Rd*temp*(1+0.61*rv))
        del(pres,temp,rv)
        
        total_condensate_zonal = (DATA["RTP"][:, yy, x1:x2]*1000.0 - DATA["RV"] [:, yy, x1:x2]*1000.0)*dens
        
        if (isinstance(LEVELS_VAR3, np.ndarray)):
            C3 = AX.contour(XV1,z_3D_2D_slice, np.absolute(total_condensate_zonal),
                                 levels=LEVELS_VAR3, colors=VAR3_COLOR, linewidths=2.0, linestyles="--") 
            AX.clabel(C3, inline=1, fontsize=10, fmt='%3.3f')
            
        else:
            print('please provide levels!')
            
    elif VAR3=='precipitating_condensate_g/m3':
        th = DATA['THETA'][:, yy, x1:x2]
        pi = DATA['PI'][:, yy, x1:x2]
        rv = DATA['RV'][:, yy, x1:x2]
        pres = np.power((pi/Cp),Cp/Rd)*p00
        temp = th*(pi/Cp)
        del(th,pi)
        dens = pres/(Rd*temp*(1+0.61*rv))
        del(pres,temp,rv)
        total_condensate_zonal = (DATA["RTP"][:, yy, x1:x2]*1000.0 - DATA["RV"] [:, yy, x1:x2]*1000.0 \
                                                                        - DATA["RCP"][:, yy, x1:x2]*1000.0 \
                                                                        - DATA["RPP"][:, yy, x1:x2]*1000.0)*dens
        print('shape of zonal crosssec total_condensate_zonal = ',np.shape(total_condensate_zonal))

        if (isinstance(LEVELS_VAR3, np.ndarray)):
            C3 = AX.contour(XV1,z_3D_2D_slice, np.absolute(total_condensate_zonal),
                                 levels=LEVELS_VAR3, colors=VAR3_COLOR, linewidths=2.0, linestyles="--")
            AX.clabel(C3, inline=1, fontsize=10, fmt='%3.3f')

        else:
            print('please provide levels!')
        
    else:
        print('please provide correct value of VAR3')

    if VAR4:
        if VAR4=='buoyancy':
            print('buoyancy') 

    C4zs = AX.plot(X_1D[x1:x2], TERR[yy, x1:x2]/1000., color='sienna', linewidth=3.6)
    
    if PLOT_SEGMENTATION_OUTPUT:
        seg_filename=glob.glob('/nobackupp11/isingh2/tobac_tracking-main/'+DOMAIN+'_segmentation_mask_box_threshold_2_'+pd.to_datetime(FILENAMETIME).strftime('%Y%m%d%H%M%S')+'.nc')
        print('segmentation file is: ',seg_filename[0])
        print('feature# ',FEATURE_NUM)
        segmentation_da = xr.open_dataset(seg_filename[0]).segmentation_mask
        C_seg = AX.contour(XV1,z_3D_2D_slice, np.where(segmentation_da[:, yy, x1:x2]==FEATURE_NUM,1.0,0.0), levels=[0.9],  colors='green', linewidths=1.0, linestyles="-") 
        AX.clabel(C_seg, inline=1, fontsize=13, fmt='%f')
            
    
    title = 'Vertical cross-section (zonal) of $w$'
    prodid = 'tests'+'_'+VAR1+'_th_vcross_zonal_'+'y' + \
        str(yy)+'_x1_'+str(x1)+'_x2_'+str(x2)+'.png'
    units = 'm/s'#var1.attrs['units']  # '$ x 10^{-5} $'+

    import matplotlib.transforms as transforms
    trans = transforms.blended_transform_factory(
        AX.transAxes, AX.transData)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
    AX.text(0.55, 0.94, EXP_LABEL, fontsize=12,
            verticalalignment='top', bbox=props, transform=AX.transAxes)
    
    if PANEL_LABEL:
        props1 = dict(boxstyle='round', facecolor='white', alpha=1)
        AX.text(0.04, 0.94, PANEL_LABEL, fontsize=18,
        verticalalignment='top', bbox=props1, transform=AX.transAxes)
        
    AX.scatter(XPOS*DXY/1000.0,TERR[YPOS,XPOS]/1000.+ZPOS/1000.,marker='+',color='k',s=130.5)
    
     # Draw box that is used for averaging of hydrometeor conc. for filtering:
    if CELL_DIM is not None:
        #print('cell dimension is ',CELL_DIM,' grid points')
        low_x_km  = np.max([(XPOS - (CELL_DIM/2.)),0.0])*DXY/1000.0
        box_width_km = CELL_DIM*DXY/1000.0
        #print('width of the box is ',box_width_km,' km')
        low_z_grid  = int(np.max([ZPOS_GRID - (CELL_DIM / 2.),0]))
        high_z_grid = int(np.min([ZPOS_GRID + (CELL_DIM / 2.),NUM_MODEL_LEVS]))
        box_height_km = (Z_1D[high_z_grid] - Z_1D[low_z_grid])/1000.0
        #print('width of the box is ',box_height_km,' km')
        low_z_km   = (TERR[YPOS,XPOS] + Z_1D[low_z_grid])/1000.0
        #print('bottom side of the box is ',low_z_km,' km high')
        #print('lower left corner of the box is x= ',low_x_km, ' z=',low_z_km)
        rect1  = AX.add_patch(Rectangle((low_x_km,low_z_km), box_width_km, box_height_km , color='green', fc = 'none',lw = 1.8))

    PLOT_INCUS_COLUMN_MAXIMA=False
    if PLOT_INCUS_COLUMN_MAXIMA:
        BOX_SCALING_FACTOR = 1.5
        y_small = int(np.max([YPOS - BOX_SCALING_FACTOR*(CELL_DIM/2.),0]))
        y_large = int(YPOS + BOX_SCALING_FACTOR*(CELL_DIM / 2.))
        x_small = int(np.max([XPOS - BOX_SCALING_FACTOR*(CELL_DIM/2.),0]))
        x_large = int(XPOS + BOX_SCALING_FACTOR*(CELL_DIM / 2.))
        low_z_grid  = int(np.max([ZPOS_GRID - BOX_SCALING_FACTOR*(CELL_DIM / 2.),0]))
        high_z_grid = int(np.min([ZPOS_GRID + BOX_SCALING_FACTOR*(CELL_DIM / 2.),NUM_MODEL_LEVS]))
        
        wp_array = VAR2_XR[low_z_grid:high_z_grid, y_small:y_large, x_small:x_large].values
        
        print('shape of cell is ',np.shape(wp_array))
        
        ind_max = np.unravel_index(np.nanargmax(wp_array),wp_array.shape)
        z_coord_wmax = ind_max[0]+low_z_grid
        y_coord_wmax = ind_max[1]+y_small
        x_coord_wmax = ind_max[2]+x_small
    
        print('index of w max is ',ind_max)
        print('z coordinate is :', z_coord_wmax)
        print('y coordinate is :', y_coord_wmax)
        print('x coordinate is :', x_coord_wmax)
        
        zpos_wmax    = (TERR[y_coord_wmax,x_coord_wmax] + Z_1D[z_coord_wmax,np.newaxis])/1000.0
        ypos_wmax    = y_coord_wmax*DXY/1000.0
        xpos_wmax    = x_coord_wmax*DXY/1000.0
        
        print('z position is :',zpos_wmax)
        print('y position is :',ypos_wmax)
        print('x position is :',xpos_wmax)
        

        th = DATA['THETA'][low_z_grid:high_z_grid, y_small:y_large, x_small:x_large]
        pi = DATA['PI'][low_z_grid:high_z_grid, y_small:y_large, x_small:x_large]
        rv = DATA['RV'][low_z_grid:high_z_grid, y_small:y_large, x_small:x_large]
        pres = np.power((pi/Cp),Cp/Rd)*p00
        temp = th*(pi/Cp)
        del(th,pi)
        dens = pres/(Rd*temp*(1+0.61*rv))
        del(pres,temp,rv)
        
        cond_array =  (DATA["RTP"][low_z_grid:high_z_grid, y_small:y_large, x_small:x_large]*1000.0 - DATA["RV"][low_z_grid:high_z_grid, y_small:y_large,x_small:x_large]*1000.0)*dens
        print('shape of cell is ',np.shape(cond_array))
        ind_max = np.unravel_index(np.nanargmax(cond_array),cond_array.shape)
        z_coord_condmax = ind_max[0]+low_z_grid
        y_coord_condmax = ind_max[1]+y_small
        x_coord_condmax = ind_max[2]+x_small
        
        print('index of cond max is ',ind_max)
        print('z coordinate is :', z_coord_condmax)
        print('y coordinate is :', y_coord_condmax)
        print('x coordinate is :', x_coord_condmax)
        zpos_condmax    =  (TERR[y_coord_wmax,x_coord_condmax] + Z_1D[z_coord_condmax,np.newaxis])/1000.0
        ypos_condmax = y_coord_condmax*DXY/1000.0
        xpos_condmax = x_coord_condmax*DXY/1000.0
        print('z position is :',zpos_condmax)
        print('y position is :',ypos_condmax)
        print('x position is :',xpos_condmax)
        
        AX.scatter(xpos_wmax,zpos_wmax[0],s=130.5,marker='+',color='fuchsia')#facecolors='none',edgecolors='blue')
        AX.scatter(xpos_condmax,zpos_condmax[0],s=130.5,marker='+',color='blue')#facecolors='none',edgecolors='blue')

    
    # plot 
    fig_process_vert(AX, TERR, C1, X_1D, x1, x2, yy, PLOT_CBAR, 0, 15, title, TITLETIME, FILENAMETIME, prodid, units, "zonal", HEIGHT, True, PANEL_PLOT)

def plot_meridional_vertcross(DOMAIN, DATA, DX, DY ,TERR, Y_1D, Z_3D, Z_1D, DXY, VAR1, DESTAGGER, STAGGER_DIM, LEVELS_VAR1, CMAP_VAR1,
                       VAR2_XR, LEVELS_VAR2, VAR2_COLOR,
                       VAR3, LEVELS_VAR3, VAR3_COLOR,
                       VAR4, LEVELS_VAR4, VAR4_COLOR,
                       xx, y1, y2,
                       PLOT_WINDS, PANEL_PLOT, HEIGHT, AX, PLOT_CBAR, EXP_LABEL, PANEL_LABEL, XPOS, YPOS, ZPOS, ZPOS_GRID, CELL_DIM, FEATURE_NUM, PLOT_SEGMENTATION_OUTPUT, NUM_MODEL_LEVS, TITLETIME, FILENAMETIME):  # rcParams["contour.negative_linestyle"] = 'dashed'
    
    x1 = xx - 50
    x2 = xx + 50
    
    YV1, __ = np.meshgrid(Y_1D[y1:y2]/1000.0, Z_1D/1000.0)
    z_3D_2D_slice    =  (TERR[y1:y2, xx] + Z_1D[:,np.newaxis])/1000.0

    if DESTAGGER:
        print(var1)
        var1 = destagger(var1, STAGGER_DIM, meta=True)
    else:
        print(' ')

    
    if (isinstance(LEVELS_VAR1, np.ndarray)):
        #C1 = AX.contourf(YV1, zh[:, y1:y2, xx]/1000.0, VAR2_XR[:, y1:y2, xx], levels=LEVELS_VAR1,
        #                  cmap=CMAP_VAR1, extend='both')  # Spectral for qv
        C1 = AX.contourf(YV1, z_3D_2D_slice, VAR2_XR[:, y1:y2, xx], levels=LEVELS_VAR1,
                          cmap=CMAP_VAR1, extend='both')  # Spectral for qv
    else:
        C1 = AX.contourf(YV1, z_3D_2D_slice, VAR2_XR[:, y1:y2, xx],
                          cmap=CMAP_VAR1, extend='both')

        
    #levels_th = np.arange(290.0, 690.0, 2.0)
    #C4 = plt.contour(XV1, zh[:, yy, x1:x2]/1000.0, DATA.THETA.values[:, yy, x1:x2], colors='k', levels=levels_th, axis=AX, linewidth=0.6)
    #plt.clabel(C4, inline=1, fontsize=14, fmt='%3.0f')

        
    if (isinstance(LEVELS_VAR2, np.ndarray)):
        C2 = AX.contour(YV1, z_3D_2D_slice, VAR2_XR[:, y1:y2, xx],
                             levels=LEVELS_VAR2, colors=VAR2_COLOR, linewidths=1., linestyles="-")
    else:
        C2 = AX.contour(YV1,z_3D_2D_slice, VAR2_XR[:, y1:y2, xx],
                             colors='k', linewidths=1., linestyles="--")
    AX.clabel(C2, inline=1, fontsize=10, fmt='%3.0f')


    if VAR3=='RTP-RV_g/kg':
        total_condensate_meridional = DATA["RTP"][:, y1:y2, xx]*1000.0 - DATA["RV"][:, y1:y2, xx]*1000.0
        if (isinstance(LEVELS_VAR3, np.ndarray)):
            C3 = AX.contour(YV1, z_3D_2D_slice, np.absolute(total_condensate_meridional),\
                                levels=LEVELS_VAR3, colors=VAR3_COLOR, linewidths=2.0, linestyles="--")
        else:
            print('please provide levels!')
            
    elif VAR3=='RTP-RV_g/m3':
        th = DATA['THETA'][:, y1:y2, xx]
        pi = DATA['PI'][:, y1:y2, xx]
        rv = DATA['RV'][:, y1:y2, xx]
        pres = np.power((pi/Cp),Cp/Rd)*p00
        temp = th*(pi/Cp)
        del(th,pi)
        dens = pres/(Rd*temp*(1+0.61*rv))
        del(pres,temp,rv)
        
        total_condensate_meridional = (DATA["RTP"][:, y1:y2, xx]*1000.0 - DATA["RV"][:, y1:y2, xx]*1000.0)*dens
        if (isinstance(LEVELS_VAR3, np.ndarray)):
            C3 = AX.contour(YV1, z_3D_2D_slice, np.absolute(total_condensate_meridional),\
                                levels=LEVELS_VAR3, colors=VAR3_COLOR, linewidths=2.0, linestyles="--")
            AX.clabel(C3, inline=1, fontsize=10, fmt='%3.3f')
        else:
            print('please provide levels!')
            
    elif VAR3=='precipitating_condensate_g/m3':
        th = DATA['THETA'][:, y1:y2, xx]
        pi = DATA['PI'][:, y1:y2, xx]
        rv = DATA['RV'][:, y1:y2, xx]
        pres = np.power((pi/Cp),Cp/Rd)*p00
        temp = th*(pi/Cp)
        del(th,pi)
        dens = pres/(Rd*temp*(1+0.61*rv))
        del(pres,temp,rv)
        
        total_condensate_meridional = (DATA["RTP"][:, y1:y2, xx]*1000.0 - DATA["RV"] [:, y1:y2, xx]*1000.0 \
                                                                        - DATA["RCP"][:, y1:y2, xx]*1000.0 \
                                                                        - DATA["RPP"][:, y1:y2, xx]*1000.0)*dens
        
        
        if (isinstance(LEVELS_VAR3, np.ndarray)):
            C3 = AX.contour(YV1, z_3D_2D_slice, np.absolute(total_condensate_meridional),\
                                levels=LEVELS_VAR3, colors=VAR3_COLOR, linewidths=2.0, linestyles="--")
            AX.clabel(C3, inline=1, fontsize=10, fmt='%3.3f')
        else:
            print('please provide levels!')
            
            
    else:
        print('please provide correct value of VAR3')
 
        #del total_condensate_meridional
           
    if VAR4:
        if VAR4=='buoyancy':
            print('buoyancy')    
#     if PLOT_WINDS:
#         winds_thin_x = 4
#         winds_thin_z = 4
#         YVwind, ZVwind = np.meshgrid(yh[y1:y2:winds_thin_x], z[::winds_thin_z])
#         v1 = DATA.variables["VP"][::winds_thin_z,y1:y2:winds_thin_x, xx]*1.94384
#         w1 = DATA.variables["WP"][::winds_thin_z,y1:y2:winds_thin_x, xx]*1.94384
#         #QV1 = AX.barbs(YVwind, zh[::winds_thin_z,y1:y2:winds_thin_x, xx]/1000.0,
#         #               v1, w1, length=7.2, pivot='middle', linewidth=0.60, flip_barb=True)
          

    C4zs = AX.plot(Y_1D[y1:y2], TERR[y1:y2, xx]/1000., color='sienna', linewidth=3.6)
    
    if PLOT_SEGMENTATION_OUTPUT:
        seg_filename=glob.glob('/nobackupp11/isingh2/tobac_tracking-main/'+DOMAIN+'_segmentation_mask_box_threshold_2_'+pd.to_datetime(FILENAMETIME).strftime('%Y%m%d%H%M%S')+'.nc')
        print('segmentation file is: ',seg_filename[0])
        print('feature# ',FEATURE_NUM)
        segmentation_da = xr.open_dataset(seg_filename[0]).segmentation_mask
        C_seg = AX.contour(YV1, z_3D_2D_slice, np.where(segmentation_da[:, y1:y2, xx]==FEATURE_NUM,1.0,0.0), levels=[0.9],colors='green', linewidths=1.0, linestyles="-")
        AX.clabel(C_seg, inline=1, fontsize=13, fmt='%f')
        
    title = 'Vertical cross-section (meridional) of $w$'
    prodid = 'tests'+'_'+VAR1+'_th_vcross_zonal_'+'xx' + \
        str(xx)+'_y1_'+str(y1)+'_y2_'+str(y2)+'.png'
    units = 'm/s'#var1.attrs['units']  # '$ x 10^{-5} $'+

    import matplotlib.transforms as transforms
    trans = transforms.blended_transform_factory(
        AX.transAxes, AX.transData)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
    AX.text(0.55, 0.94, EXP_LABEL, fontsize=12,
            verticalalignment='top', bbox=props, transform=AX.transAxes)
    
    if PANEL_LABEL:
        props1 = dict(boxstyle='round', facecolor='white', alpha=1)
        AX.text(0.04, 0.94, PANEL_LABEL, fontsize=18,
        verticalalignment='top', bbox=props1, transform=AX.transAxes)
    
    AX.scatter(YPOS*DXY/1000.0,TERR[YPOS,XPOS]/1000.+ZPOS/1000.,marker='+',color='k',s=130.5)
    
    # Draw box that is used for averaging of hydrometeor conc. for filtering:
    if CELL_DIM is not None:
        low_y_km  = np.max([(YPOS - (CELL_DIM/2.)),0.0])*DXY/1000.0
        box_width_km = CELL_DIM*DXY/1000.0
        low_z_grid  = int(np.max([ZPOS_GRID - (CELL_DIM / 2.),0]))
        high_z_grid = int(np.min([ZPOS_GRID + (CELL_DIM / 2.),NUM_MODEL_LEVS]))
        box_height_km = (Z_1D[high_z_grid] - Z_1D[low_z_grid])/1000.0
        low_z_km   = (TERR[YPOS,XPOS] + Z_1D[low_z_grid])/1000.0
        rect1  =  AX.add_patch(Rectangle((low_y_km,low_z_km), box_width_km, box_height_km , color='green', fc = 'none',lw = 1.8))

    PLOT_INCUS_COLUMN_MAXIMA=False
    if PLOT_INCUS_COLUMN_MAXIMA:
        BOX_SCALING_FACTOR = 1.5
        y_small = int(np.max([YPOS - BOX_SCALING_FACTOR*(CELL_DIM/2.),0]))
        y_large = int(YPOS + BOX_SCALING_FACTOR*(CELL_DIM / 2.))
        x_small = int(np.max([XPOS - BOX_SCALING_FACTOR*(CELL_DIM/2.),0]))
        x_large = int(XPOS + BOX_SCALING_FACTOR*(CELL_DIM / 2.))
        low_z_grid  = int(np.max([ZPOS_GRID - BOX_SCALING_FACTOR*(CELL_DIM / 2.),0]))
        high_z_grid = int(np.min([ZPOS_GRID + BOX_SCALING_FACTOR*(CELL_DIM / 2.),NUM_MODEL_LEVS]))
        
        wp_array = VAR2_XR[low_z_grid:high_z_grid, y_small:y_large, x_small:x_large].values
        
        print('shape of cell is ',np.shape(wp_array))
        
        ind_max = np.unravel_index(np.nanargmax(wp_array),wp_array.shape)
        z_coord_wmax = ind_max[0]+low_z_grid
        y_coord_wmax = ind_max[1]+y_small
        x_coord_wmax = ind_max[2]+x_small
    
        print('index of w max is ',ind_max)
        print('z coordinate is :', z_coord_wmax)
        print('y coordinate is :', y_coord_wmax)
        print('x coordinate is :', x_coord_wmax)
        
        zpos_wmax    = (TERR[y_coord_wmax,x_coord_wmax] + Z_1D[z_coord_wmax,np.newaxis])/1000.0
        ypos_wmax    = y_coord_wmax*DXY/1000.0
        xpos_wmax    = x_coord_wmax*DXY/1000.0
        
        print('z position is :',zpos_wmax)
        print('y position is :',ypos_wmax)
        print('x position is :',xpos_wmax)
        

        th = DATA['THETA'][low_z_grid:high_z_grid, y_small:y_large, x_small:x_large]
        pi = DATA['PI'][low_z_grid:high_z_grid, y_small:y_large, x_small:x_large]
        rv = DATA['RV'][low_z_grid:high_z_grid, y_small:y_large, x_small:x_large]
        pres = np.power((pi/Cp),Cp/Rd)*p00
        temp = th*(pi/Cp)
        del(th,pi)
        dens = pres/(Rd*temp*(1+0.61*rv))
        del(pres,temp,rv)
        
        cond_array =  (DATA["RTP"][low_z_grid:high_z_grid, y_small:y_large, x_small:x_large]*1000.0 - DATA["RV"][low_z_grid:high_z_grid, y_small:y_large,x_small:x_large]*1000.0)*dens
        print('shape of cell is ',np.shape(cond_array))
        ind_max = np.unravel_index(np.nanargmax(cond_array),cond_array.shape)
        z_coord_condmax = ind_max[0]+low_z_grid
        y_coord_condmax = ind_max[1]+y_small
        x_coord_condmax = ind_max[2]+x_small
        
        print('index of cond max is ',ind_max)
        print('z coordinate is :', z_coord_condmax)
        print('y coordinate is :', y_coord_condmax)
        print('x coordinate is :', x_coord_condmax)
        zpos_condmax    =  (TERR[y_coord_wmax,x_coord_condmax] + Z_1D[z_coord_condmax,np.newaxis])/1000.0
        ypos_condmax = y_coord_condmax*DXY/1000.0
        xpos_condmax = x_coord_condmax*DXY/1000.0
        print('z position is :',zpos_condmax)
        print('y position is :',ypos_condmax)
        print('x position is :',xpos_condmax)
        AX.scatter(ypos_wmax,zpos_wmax[0],s=130.5,marker='+',color='fuchsia')#facecolors='none',edgecolors='blue')
        AX.scatter(ypos_condmax,zpos_condmax[0],s=130.5,marker='+',color='blue')#facecolors='none',edgecolors='blue')
        #AX.axvline(x = y_small*DXY/1000.0, color = 'b')
        #AX.axvline(x = y_large*DXY/1000.0, color = 'b')

    fig_process_vert(AX, TERR, C1, Y_1D, y1, y2, xx, PLOT_CBAR, 0, 15, title, TITLETIME, FILENAMETIME, prodid, units, "meridional", HEIGHT, False, PANEL_PLOT)
 
def plot_plan_view_cell(DOMAIN, DATA, DX, DY, TERR, X_1D, Y_1D, ZM, XPOS, YPOS, ZPOS, DXY, VAR1, LEVELS_VAR1, SINGLE_LEVEL_VAR1, CMAP_VAR1, VAR2, LEVELS_VAR2, VAR2_COLOR, x1,x2, y1, y2, FEATURE_NUM, PLOT_SEGMENTATION_OUTPUT,
                    AX, PLOT_CBAR, EXP_LABEL, PANEL_LABEL,  TITLETIME, FILENAMETIME, XPOS_FEATURES=None,YPOS_FEATURES=None,XPOS_LIST=None,YPOS_LIST=None):  # rcParams["contour.negative_linestyle"] = 'dashed'
    
    XH1, YH1 = np.meshgrid(X_1D[x1:x2]/1000.0,Y_1D[y1:y2]/1000.0)
    vert_lev = np.argmin(np.abs(ZM-ZPOS))
    var_to_plotted=DATA.variables[VAR1][vert_lev,y1:y2,x1:x2].values
    C111   = AX.contourf(XH1 ,YH1,var_to_plotted ,levels=LEVELS_VAR1,cmap= CMAP_VAR1,extend='both')#,colors=PLOT_ANOTHER_VAR_CONT[7],linestyles=np.where(levels >= 0, "-", "--"),linewidths=PLOT_ANOTHER_VAR_CONT[8])
    
    if SINGLE_LEVEL_VAR1:
        C112   = AX.contour(XH1 ,YH1,var_to_plotted ,levels=SINGLE_LEVEL_VAR1,colors='k')#,colors=PLOT_ANOTHER_VAR_CONT[7],linestyles=np.where(levels >= 0, "-", "--"),linewidths=PLOT_ANOTHER_VAR_CONT[8])
        AX.clabel(C112, inline=1, fontsize=10, fmt='%3.0f')

    if XPOS_FEATURES is not None:
        tobac_features_scatter1 = AX.scatter(XPOS_FEATURES*DXY/1000.0,YPOS_FEATURES*DXY/1000.0,marker='^',s=100.5,c='green')#facecolors='none', edgecolors='green')
        
    if XPOS_LIST:
        print('plotting cell track...')
        AX.plot(np.array(XPOS_LIST)*DXY/1000.0,np.array(YPOS_LIST)*DXY/1000.0,color='k',linewidth=1.5,linestyle='--')
        
    if VAR2:
        if VAR2=='buoyancy':
            print('buoyancy')
            
    if XPOS:
        tobac_features_scatter = AX.scatter(XPOS*DXY/1000.0,YPOS*DXY/1000.0,marker='+',s=130.5,c='k')
    
    if PLOT_SEGMENTATION_OUTPUT:
        seg_filename=glob.glob('/nobackupp11/isingh2/tobac_tracking-main/'+DOMAIN+'_segmentation_mask_box_threshold_2_'+pd.to_datetime(FILENAMETIME).strftime('%Y%m%d%H%M%S')+'.nc')
        print('segmentation file is: ',seg_filename[0])
        print('feature# ',FEATURE_NUM)
        segmentation_da = xr.open_dataset(seg_filename[0]).segmentation_mask
        C_plan_seg = AX.contour(XH1 ,YH1, np.where(segmentation_da[vert_lev, y1:y2, x1:x2]==FEATURE_NUM,1.0,0.0), levels=[0.9],colors='green', linewidths=1.0, linestyles="-")
        AX.clabel(C_plan_seg, inline=1, fontsize=13, fmt='%f')

    import matplotlib.transforms as transforms
    trans = transforms.blended_transform_factory(
        AX.transAxes, AX.transData)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)

    if EXP_LABEL:
        AX.text(0.55, 0.94, EXP_LABEL, fontsize=12,
                verticalalignment='top', bbox=props, transform=AX.transAxes)

    AX.set_title('Vertical velocity at model level '+str(vert_lev)+'\n'+TITLETIME,fontsize=16)
    AX.set_xlabel('x-distance (km)',fontsize=16)
    AX.set_ylabel('y-distance (km)',fontsize=16)
    
    if PLOT_CBAR:
        print('not plotting colorbar for the plan view')
    if PANEL_LABEL:
        props1 = dict(boxstyle='round', facecolor='white', alpha=1)
        AX.text(0.04, 0.94, PANEL_LABEL, fontsize=18,
        verticalalignment='top', bbox=props1, transform=AX.transAxes)
        
    return C111

def plot_vert_zonal_meridional_crosssection_plan_view_tobac_comparison(DOMAIN,TOBAC_DF1,TOBAC_DF2,CELL_NO1,CELL_NO2,XH,YH,ZZ_M,ZM,TERR,DXY,CMAP,OUTPUT_DIR,EXPERIMENT_MARKER,TOBAC_FEATURES_DF1=None,TOBAC_FEATURES_DF2=None):
        print('cell in file#1: ',CELL_NO1,' and cell in file#2:',CELL_NO2)
        tdata_neu1=TOBAC_DF1[TOBAC_DF1['cell']==CELL_NO1]
        print('this cell has '+str(len(tdata_neu1))+' time steps')
        xpos1=list(tdata_neu1.X.values.astype(int))
        ypos1=list(tdata_neu1.Y.values.astype(int))
        zpos1=list(tdata_neu1.zmn.values.astype(int))    
        zpos_grid1=list(tdata_neu1.vdim.values.astype(int))
        features1=list(tdata_neu1.feature.values)
        cell_dim1 =list(np.array(tdata_neu1.num.values.astype(int))**(1/3))
        times_tracked1=tdata_neu1.timestr.values
        times_tracked_pd1 = pd.to_datetime(times_tracked1)
        thresholds1=tdata_neu1.threshold_value.values
        cell_labels1=['cell#'+str(CELL_NO1)+'\nfeature#'+str(features1[kk])+'\nxpos:'+str(xpos1[kk])+' gr pt'+'\nypos:'+str(ypos1[kk])+' gr pt'+'\nzpos:'+str(zpos1[kk])+' m' for kk in range(len(xpos1))]
        print('cell one times vary from ',min(times_tracked_pd1),' to ',max(times_tracked_pd1))
        
        tdata_neu2=TOBAC_DF2[TOBAC_DF2['cell']==CELL_NO2]
        print('this cell has '+str(len(tdata_neu2))+' time steps')
        xpos2=list(tdata_neu2.X.values.astype(int))
        ypos2=list(tdata_neu2.Y.values.astype(int))
        zpos2=list(tdata_neu2.zmn.values.astype(int))    
        zpos_grid2=list(tdata_neu2.vdim.values.astype(int))
        features2=list(tdata_neu2.feature.values)
        cell_dim2 =list(np.array(tdata_neu2.num.values.astype(int))**(1/3))
        times_tracked2=tdata_neu2.timestr.values
        times_tracked_pd2 = pd.to_datetime(times_tracked2)
        thresholds2=tdata_neu2.threshold_value.values
        cell_labels2=['cell#'+str(CELL_NO2)+'\nfeature#'+str(features2[kk])+'\nxpos:'+str(xpos2[kk])+' gr pt'+'\nypos:'+str(ypos2[kk])+' gr pt'+'\nzpos:'+str(zpos2[kk])+' m' for kk in range(len(xpos2))]
        print('cell two times vary from ',min(times_tracked_pd2),' to ',max(times_tracked_pd2))
              
        start_time = min(min(times_tracked_pd1),min(times_tracked_pd2)) 
        end_time = max(max(times_tracked_pd1),max(times_tracked_pd2))
        all_times = pd.date_range(start=start_time, end=end_time, freq='30S')
        print('\n=======================')
        print('plotting times vary from ',min(all_times),' to ',max(all_times))
        print('=======================\n')
        # ADDING TWO MINUTES BEFORE AND AFTER
        start_time = min(min(times_tracked_pd1),min(times_tracked_pd2)) - pd.Timedelta(minutes=2) 
        end_time = max(max(times_tracked_pd1),max(times_tracked_pd2))   + pd.Timedelta(minutes=2) 
        all_times = pd.date_range(start=start_time, end=end_time, freq='30S')
        print('\n=======================')
        print('plotting times vary from (after adding 2 minutes before and after) ',min(all_times),' to ',max(all_times))
        print('=======================\n')
        
        ii = 0 
        ii1 = 0
        ii2 = 0
        for tim_pd in all_times:
            print('-------------------------------------------------------')
            if all_times[ii] in times_tracked_pd1:
                switch_1=True
            else:
                switch_1=False
                
            if all_times[ii] in times_tracked_pd2:
                switch_2=True
            else:
                switch_2=False
                
            print('timestep '+str(ii)+': '+tim_pd.strftime("%Y-%m-%d-%H:%M:%S"))
            #tim_pd = pd.to_datetime(tim)
            rams_fil=simulation_base_folder+'a-L-'+tim_pd.strftime("%Y-%m-%d-%H%M%S")+'-g3.h5'
            print('RAMS date file: ',rams_fil)
            rams_fil_da=xr.open_dataset(rams_fil,engine='h5netcdf', phony_dims='sort')
            titletime= get_time_from_RAMS_file(rams_fil)[0]
            filenametime= get_time_from_RAMS_file(rams_fil)[1]
            wpp = rams_fil_da["WP"]
            
            #COMPARISON PLOTTING STARTS HERE
            fig = plt.figure(figsize=(15, 15), frameon=False)  # (16,11)
            ax1 = plt.subplot(2, 3, 1)
            ax1.set_aspect('equal', adjustable='box') # plot square plan view
            ax2 = plt.subplot(2, 3, 2)
            ax3 = plt.subplot(2, 3, 3)
            ax4 = plt.subplot(2, 3, 4)
            ax4.set_aspect('equal', adjustable='box') # plot square plan view
            ax5 = plt.subplot(2, 3, 5)
            ax6 = plt.subplot(2, 3, 6)
            
            # plot first row from tracking file#1
            if switch_1:
                print('\nii1 = ',ii1,'\n')
                if TOBAC_FEATURES_DF1 is not None:
                # plot features that are wiithin 1 km of the vertical level of the cell that is being plotted
                    tdata_feat1=TOBAC_FEATURES_DF1[(TOBAC_FEATURES_DF1['time']==tim_pd)   & (abs(TOBAC_FEATURES_DF1['zmn']-zpos1[ii1])<=2000.) & \
                                                 (TOBAC_FEATURES_DF1['X']>=xpos1[ii1]-50) & (TOBAC_FEATURES_DF1['X']<=xpos1[ii1]+50)          & \
                                                 (TOBAC_FEATURES_DF1['Y']>=ypos1[ii1]-50) & (TOBAC_FEATURES_DF1['Y']<=ypos1[ii1]+50)]
                    xpos_feat1=np.array((tdata_feat1.X.values.astype(int)))
                    ypos_feat1=np.array((tdata_feat1.Y.values.astype(int)))
                else:
                    xpos_feat1=None
                    ypos_feat1=None
               
            
                cbar_conts1 = plot_plan_view_cell      (DOMAIN, rams_fil_da, 100.0, 100.0, TERR, XH, YH, ZM, xpos1[ii1], ypos1[ii1], zpos1[ii1], DXY, 'WP', np.arange(-20.,20.1,.1),[thresholds1[ii1]], CMAP, None, [0.01], 'maroon', min(xpos1)-50, max(xpos1)+50,  min(ypos1)-50,  max(ypos1)+50, features1[ii1], False,
                                          ax1, False,cell_labels1[ii1], '(a)'  ,titletime, filenametime,xpos_feat1,ypos_feat1,xpos1,ypos1)
                print('plan view done #1')

                plot_zonal_vertcross     (DOMAIN, rams_fil_da,  100.0, 100.0, TERR, XH, ZZ_M, ZM, DXY, 'WP', False, 2, np.arange(-20.,20.1,.1), CMAP,
                                           wpp,np.array([thresholds1[ii1]]), 'k',
                                           'RTP-RV_g/m3', np.array([0.05]), 'purple',
                                           None,[0.01], 'maroon',
                                           ypos1[ii1], min(xpos1)-50, max(xpos1)+50 ,
                                           False, False, 15.0, ax2,False,cell_labels1[ii1],'(b)',xpos1[ii1],ypos1[ii1],zpos1[ii1],zpos_grid1[ii1],cell_dim1[ii1], features1[ii1], False, 231, titletime,filenametime)
                print('vertical zonal cross-section done #1')

                plot_meridional_vertcross(DOMAIN, rams_fil_da,  100.0, 100.0,TERR, YH, ZZ_M, ZM,DXY, 'WP', False, 2, np.arange(-20.,20.1,.1), CMAP,
                                          wpp,np.array([thresholds1[ii1]]), 'k',
                                          'RTP-RV_g/m3', np.array([0.05]), 'purple',
                                          None, [0.01], 'maroon',
                                          xpos1[ii1], min(ypos1)-50, max(ypos1)+50 ,
                                          False, False, 15.0, ax3,False,cell_labels1[ii1],'(c)',xpos1[ii1],ypos1[ii1],zpos1[ii1],zpos_grid1[ii1],cell_dim1[ii1], features1[ii1], False, 231,titletime,filenametime)
                print('vertical meridional cross-section done #1')
                ii1=ii1+1
                
            print('\n$$###########$$\n')
            #-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----
            # plot second row from tracking file#2
            if switch_2:
                print('\nii2 = ',ii2,'\n')
                if TOBAC_FEATURES_DF2 is not None:
                    # plot features that are wiithin 1 km of the vertical level of the cell that is being plotted
                    tdata_feat2=TOBAC_FEATURES_DF2[(TOBAC_FEATURES_DF2['time']==tim_pd)   & (abs(TOBAC_FEATURES_DF2['zmn']-zpos2[ii2])<=2000.) & \
                                                 (TOBAC_FEATURES_DF2['X']>=xpos2[ii2]-50) & (TOBAC_FEATURES_DF2['X']<=xpos2[ii2]+50)          & \
                                                 (TOBAC_FEATURES_DF2['Y']>=ypos2[ii2]-50) & (TOBAC_FEATURES_DF2['Y']<=ypos2[ii2]+50)]

                    xpos_feat2=np.array((tdata_feat2.X.values.astype(int)))
                    ypos_feat2=np.array((tdata_feat2.Y.values.astype(int)))
                else:
                    xpos_feat2=None
                    ypos_feat2=None
            
                cbar_conts1 = plot_plan_view_cell      (DOMAIN, rams_fil_da, 100.0, 100.0, TERR, XH, YH, ZM, xpos2[ii2], ypos2[ii2], zpos2[ii2], DXY, 'WP', np.arange(-20.,20.1,.1),[thresholds2[0]], CMAP, None, [0.01], 'maroon', min(xpos2)-50, max(xpos2)+50,  min(ypos2)-50,  max(ypos2)+50, features2[ii2], False,
                                          ax4, False,cell_labels2[ii2], '(d)'  ,titletime, filenametime,xpos_feat2,ypos_feat2,xpos2,ypos2)
                print('plan view done #2')

                plot_zonal_vertcross     (DOMAIN, rams_fil_da,  100.0, 100.0, TERR, XH, ZZ_M, ZM, DXY, 'WP', False, 2, np.arange(-20.,20.1,.1), CMAP,
                                           wpp,np.array([thresholds2[ii2]]), 'k',
                                           'RTP-RV_g/m3', np.array([0.05]), 'purple',
                                           None,[0.01], 'maroon',
                                           ypos2[ii2], min(xpos2)-50, max(xpos2)+50 ,
                                           False, False, 15.0, ax5,False,cell_labels2[ii2],'(e)',xpos2[ii2],ypos2[ii2],zpos2[ii2],zpos_grid2[ii2],cell_dim2[ii2], features2[ii2], False, 231, titletime,filenametime)
                print('vertical zonal cross-section done #2')

                plot_meridional_vertcross(DOMAIN, rams_fil_da,  100.0, 100.0,TERR, YH, ZZ_M, ZM,DXY, 'WP', False, 2, np.arange(-20.,20.1,.1), CMAP,
                                          wpp,np.array([thresholds2[ii2]]), 'k',
                                          'RTP-RV_g/m3', np.array([0.05]), 'purple',
                                          None, [0.01], 'maroon',
                                          xpos2[ii2], min(ypos2)-50, max(ypos2)+50 ,
                                          False, False, 15.0, ax6,False,cell_labels2[ii2],'(f)',xpos2[ii2],ypos2[ii2],zpos2[ii2],zpos_grid2[ii2],cell_dim2[ii2], features2[ii2], False, 231,titletime,filenametime)
                print('vertical meridional cross-section done #2')
                ii2=ii2+1

            if (switch_1==True) &  (switch_2==False): 
                req_index = max([(ii2-1),0])
                if TOBAC_FEATURES_DF2 is not None:
                    # plot features that are wiithin 1 km of the vertical level of the cell that is being plotted
                    tdata_feat2=TOBAC_FEATURES_DF2[(TOBAC_FEATURES_DF2['time']==tim_pd)     & (abs(TOBAC_FEATURES_DF2['zmn']-zpos2[req_index])<=2000.) & \
                                                   (TOBAC_FEATURES_DF2['X']>=xpos2[req_index]-50) & (TOBAC_FEATURES_DF2['X']<=xpos2[req_index]+50)          & \
                                                   (TOBAC_FEATURES_DF2['Y']>=ypos2[req_index]-50) & (TOBAC_FEATURES_DF2['Y']<=ypos2[req_index]+50)]

                    xpos_feat2=np.array((tdata_feat2.X.values.astype(int)))
                    ypos_feat2=np.array((tdata_feat2.Y.values.astype(int)))
                else:
                    xpos_feat2=None
                    ypos_feat2=None
                print('found cell in file 1 but not in file 2...\n plotting features from file 2')
                
                cbar_conts1 = plot_plan_view_cell      (DOMAIN, rams_fil_da, 100.0, 100.0, TERR, XH, YH, ZM, None, None, zpos2[req_index], DXY, 'WP', np.arange(-20.,20.1,.1),None, CMAP, None, [0.01], 'maroon', min(xpos2)-50, max(xpos2)+50,  min(ypos2)-50,  max(ypos2)+50, None, False,
                                          ax4, False,None, '(d)'  ,titletime, filenametime,xpos_feat2,ypos_feat2,xpos2,ypos2)
                print('plan view done #2')
                
            if (switch_1==False) &  (switch_2==True):
                req_index = max([ii1-1,0])
                if TOBAC_FEATURES_DF1 is not None:
                # plot features that are wiithin 1 km of the vertical level of the cell that is being plotted
                    tdata_feat1=TOBAC_FEATURES_DF1[(TOBAC_FEATURES_DF1['time']==tim_pd)     & (abs(TOBAC_FEATURES_DF1['zmn']-zpos1[req_index])<=2000.) & \
                                                   (TOBAC_FEATURES_DF1['X']>=xpos1[req_index]-50) &     (TOBAC_FEATURES_DF1['X']<=xpos1[req_index]+50)          & \
                                                   (TOBAC_FEATURES_DF1['Y']>=ypos1[req_index]-50) &     (TOBAC_FEATURES_DF1['Y']<=ypos1[req_index]+50)]
                    xpos_feat1=np.array((tdata_feat1.X.values.astype(int)))
                    ypos_feat1=np.array((tdata_feat1.Y.values.astype(int)))
                else:
                    xpos_feat1=None
                    ypos_feat1=None
                print('found cell in file 2 but not in file 1\n plotting features from file 1')
                
                cbar_conts1 = plot_plan_view_cell      (DOMAIN, rams_fil_da, 100.0, 100.0, TERR, XH, YH, ZM, None, None, zpos1[req_index], DXY, 'WP', np.arange(-20.,20.1,.1),None, CMAP, None, [0.01], 'maroon', min(xpos1)-50, max(xpos1)+50,  min(ypos1)-50,  max(ypos1)+50, None, False,
                                          ax1, False,None, '(a)'  ,titletime, filenametime,xpos_feat1,ypos_feat1,xpos1,ypos1)
                print('plan view done #1')
                
                
            if (switch_1==False) &  (switch_2==False):
                print('before and after part!!!')
                req_index1 = max([ii1-1,0])
                req_index2 = max([ii2-1,0])
                
                if TOBAC_FEATURES_DF1 is not None:
                # plot features that are wiithin 1 km of the vertical level of the cell that is being plotted
                    tdata_feat1=TOBAC_FEATURES_DF1[(TOBAC_FEATURES_DF1['time']==tim_pd)     & (abs(TOBAC_FEATURES_DF1['zmn']-zpos1[req_index1])<=2000.) & \
                                                   (TOBAC_FEATURES_DF1['X']>=xpos1[req_index1]-50) &     (TOBAC_FEATURES_DF1['X']<=xpos1[req_index1]+50)          & \
                                                   (TOBAC_FEATURES_DF1['Y']>=ypos1[req_index1]-50) &     (TOBAC_FEATURES_DF1['Y']<=ypos1[req_index1]+50)]
                    xpos_feat1=np.array((tdata_feat1.X.values.astype(int)))
                    ypos_feat1=np.array((tdata_feat1.Y.values.astype(int)))
                else:
                    xpos_feat1=None
                    ypos_feat1=None
                
                if TOBAC_FEATURES_DF2 is not None:
                    # plot features that are wiithin 1 km of the vertical level of the cell that is being plotted
                    tdata_feat2=TOBAC_FEATURES_DF2[(TOBAC_FEATURES_DF2['time']==tim_pd)     & (abs(TOBAC_FEATURES_DF2['zmn']-zpos2[req_index2])<=2000.) & \
                                                   (TOBAC_FEATURES_DF2['X']>=xpos2[req_index2]-50) & (TOBAC_FEATURES_DF2['X']<=xpos2[req_index2]+50)          & \
                                                   (TOBAC_FEATURES_DF2['Y']>=ypos2[req_index2]-50) & (TOBAC_FEATURES_DF2['Y']<=ypos2[req_index2]+50)]
                    xpos_feat2=np.array((tdata_feat2.X.values.astype(int)))
                    ypos_feat2=np.array((tdata_feat2.Y.values.astype(int)))
                else:
                    xpos_feat2=None
                    ypos_feat2=None

                cbar_conts1 = plot_plan_view_cell      (DOMAIN, rams_fil_da, 100.0, 100.0, TERR, XH, YH, ZM, None, None, zpos1[req_index1], DXY, 'WP', np.arange(-20.,20.1,.1),None, CMAP, None, [0.01], 'maroon', min(xpos1)-50, max(xpos1)+50,  min(ypos1)-50,  max(ypos1)+50, None, False,
                                          ax1, False,'No cell (2 extra mins)', '(a)'  ,titletime, filenametime,xpos_feat1,ypos_feat1,xpos1,ypos1)
                cbar_conts2 = plot_plan_view_cell      (DOMAIN, rams_fil_da, 100.0, 100.0, TERR, XH, YH, ZM, None, None, zpos2[req_index2], DXY, 'WP', np.arange(-20.,20.1,.1),None, CMAP, None, [0.01], 'maroon', min(xpos2)-50, max(xpos2)+50,  min(ypos2)-50,  max(ypos2)+50, None, False,
                                          ax4, False,'No cell (2 extra mins)', '(d)'  ,titletime, filenametime,xpos_feat2,ypos_feat2,xpos2,ypos2)
     
            cb_ax = fig.add_axes([0.2, 0.0001, 0.6, 0.02])  # two panels #[left, bottom, width, height]
            cbar  = fig.colorbar(cbar_conts1, cax=cb_ax, orientation = 'horizontal')
            plt.tight_layout()
            png_file=OUTPUT_DIR+'three_panel_'+DOMAIN+'_'+EXPERIMENT_MARKER+'_cell1_'+str(CELL_NO1)+'_cell2_'+str(CELL_NO2)+'_timestep'+tim_pd.strftime("%Y%m%d%H%M%S")+'.png'
            print('\n')
            print(png_file)
            plt.savefig(png_file,dpi=150)
            plt.close()
            ii = ii + 1
        print('============================================\n\n\n')
        
        
        
#plot tracking differences
# MAKE CHANGES IN THE LINES MARKED WITH <CHANGE HERE>

domain = 'AUS1.1-R' # simulation name <CHANGE HERE>
print('working on simulation: ',domain)
simulation_base_folder= '/nobackup/pmarines/DATA_FM/'+domain+'/LES_data/'
tobac_tracking_dirpath1 = '/nobackup/pmarines/DATA_FM/'+domain+'/tobac_data/'
tobac_tracking_dirpath2 = '/nobackup/pmarines/DATA_FM/'+domain+'/tobac_data/'
tobac_features_dirpath1 = '/nobackup/pmarines/DATA_FM/'+domain+'/tobac_data/'
tobac_features_dirpath2 = '/nobackup/pmarines/DATA_FM/'+domain+'/tobac_data/'
les_path = simulation_base_folder

tobac_tracking_filename1   = 'comb_track_01_01_50_sr5035_setpos.p' # name of the first tracking file <CHANGE HERE>
tobac_tracking_filename2   = 'comb_track_01_02_50_02_sr5035_setpos.p' # name of the second tracking file <CHANGE HERE>
tobac_features_filename1   = 'comb_df_01_01_50.p'
tobac_features_filename2   = 'comb_df_01_02_50_02.p'

tobac_tracking_filepath1  = tobac_tracking_dirpath1+tobac_tracking_filename1
tobac_tracking_filepath2  = tobac_tracking_dirpath2+tobac_tracking_filename2
tobac_features_filepath1  = tobac_features_dirpath1+tobac_features_filename1
tobac_features_filepath2  = tobac_features_dirpath2+tobac_features_filename2

# Grab all the rams files
h5filepath = les_path+'a-L*g3.h5'
h5files1 = sorted(glob.glob(h5filepath))
hefilepath = les_path+'a-L*head.txt'
hefiles1 = sorted(glob.glob(hefilepath))
#print(h5files1)
start_time_simulation=get_time_from_RAMS_file(h5files1[0])[0]
end_time_simulation=get_time_from_RAMS_file(h5files1[-1])[0]
print('starting time in simulations: ',start_time_simulation)
print('ending time in simulations: ',end_time_simulation)

#### read in RAMS data file to get parameters for plotting ####
rams_terr=xr.open_dataset(h5files1[0],engine='h5netcdf', phony_dims='sort').TOPT.values

zm, zt, nx, ny, dxy, npa = RAMS_fx.read_head(hefiles1[0],h5files1[0])

xh=np.arange(dxy/2,nx*dxy,dxy)
yh=np.arange(dxy/2,ny*dxy,dxy)

##### read in tobac data #####
print('reading tracking file1: ',tobac_tracking_filepath1)
tdata1          = pd.read_pickle(tobac_tracking_filepath1)

print('reading tracking file2: ',tobac_tracking_filepath2)
tdata2          = pd.read_pickle(tobac_tracking_filepath2)

print('reading features file1',tobac_features_filepath1)
tdata_features1 = pd.read_pickle(tobac_features_filepath1)

print('reading features file2',tobac_features_filepath2)
tdata_features2 = pd.read_pickle(tobac_features_filepath2)

print('number of unique cells identified in tracking file 1: ',len(tdata1.cell.unique()))
print('number of unique cells identified in tracking file 2: ',len(tdata2.cell.unique()),'\n')

find_matched_cells=False
if find_matched_cells:
    print('finding matched cells...\n')  
    def filt_high_thres(g):
        return ((g.threshold_value.max() >= 5.0) & (g.zmn.max() >= 2000.0) & (g.zmn.min() <= 8000.0) & (abs((pd.to_datetime((g.timestr.values)).min() - pd.to_datetime(start_time_simulation)).total_seconds()) >= 120) & \
                                                                                                       (abs((pd.to_datetime((g.timestr.values)).max() - pd.to_datetime(end_time_simulation  )).total_seconds()) >= 120))            
    tdata_high_thr1=tdata1.groupby('cell').filter(filt_high_thres)
    print('number of unique cells identified in filtered tracking file 1: ',len(tdata_high_thr1.cell.unique()))

    tdata_high_thr2=tdata2.groupby('cell').filter(filt_high_thres)
    print('number of unique cells identified in filtered tracking file 2: ',len(tdata_high_thr2.cell.unique()),'\n')

    first_tracking_file_cells = tdata_high_thr1.cell.unique()
    times_tracked=tdata_high_thr1.timestr.values
    times_tracked_pd = sorted(pd.to_datetime(times_tracked).unique())
    
    print('times vary from : ',min(times_tracked_pd))
    print('times vary from : ',max(times_tracked_pd),'\n\n')
    
    matched_cell_dictionary = {}
    ii = 1
    print('finding cells in tracking file#2 matching with the following cells in file#1: \n')
    for cl in first_tracking_file_cells:
        potential_cells = []
        matched_cell_dictionary.update({cl:[]})
        print('working on cell#: ',cl,' - ',ii,'/',len(first_tracking_file_cells))
        track_data_cell_subset = tdata_high_thr1[tdata_high_thr1.cell == cl]
        all_times = track_data_cell_subset.timestr.values
        all_xpos = track_data_cell_subset.X.values
        all_ypos = track_data_cell_subset.Y.values
        all_zpos = track_data_cell_subset.zmn.values
    
        for timestep,xpos,ypos,zpos in list(zip(all_times,all_xpos,all_ypos,all_zpos)):
            second_df_time_subset = tdata_high_thr2[(tdata_high_thr2.timestr.values == timestep)     & \
                                                    (tdata_high_thr2.X.values   > (xpos-0.25))       & \
                                                    (tdata_high_thr2.X.values   < (xpos+0.25))       & \
                                                    (tdata_high_thr2.Y.values   > (ypos-0.25))       & \
                                                    (tdata_high_thr2.Y.values   < (ypos+0.25))       & \
                                                    (tdata_high_thr2.zmn.values > (zpos-100.0))      & \
                                                    (tdata_high_thr2.zmn.values < (zpos+100.0))]
            # <CHANGE ABOVE> for two cells to be the same entity, they need to be within 0.25 grid point of each 
            # other for now. 
            if len(second_df_time_subset)>0:
                potential_cells.extend(second_df_time_subset.cell.unique())
           
        potential_cells = list(set(potential_cells))
        if len(potential_cells) > 0:
            for matched_cell in potential_cells:
                matched_cell_dictionary[cl].append(matched_cell) 

        ii = ii + 1
        
    csv_filename = domain+'_matching_cells_file1_'+tobac_tracking_filename1+'_file2_'+tobac_tracking_filename2+'_v2.csv'
    print('saving the matched cells to a csv file: ',csv_filename)
    pd.DataFrame.from_dict(matched_cell_dictionary,orient='index').to_csv(csv_filename)

read_matched_from_csv = True
if read_matched_from_csv:
    def filt_high_thres(g):
        return ((g.threshold_value.max() >= 5.0) & (g.zmn.max() >= 2000.0) & (g.zmn.min() <= 8000.0) & (abs((pd.to_datetime((g.timestr.values)).min() - pd.to_datetime(start_time_simulation)).total_seconds()) >= 120) & \
                                                                                                       (abs((pd.to_datetime((g.timestr.values)).max() - pd.to_datetime(end_time_simulation  )).total_seconds()) >= 120))            
    # <CHANGE HERE FOR CHANGING THE CRITERIA>
    ## HERE ONLY CELLS THAT GET STRONGER THAN 5 m/s ARE SELECTED ##
    tdata_high_thr1=tdata1.groupby('cell').filter(filt_high_thres)
    print('number of unique cells identified in filtered tracking file 1: ',len(tdata_high_thr1.cell.unique()))

    tdata_high_thr2=tdata2.groupby('cell').filter(filt_high_thres)
    print('number of unique cells identified in filtered tracking file 2: ',len(tdata_high_thr2.cell.unique()),'\n')

    first_tracking_file_cells = tdata_high_thr1.cell.unique()
    times_tracked=tdata_high_thr1.timestr.values
    times_tracked_pd = sorted(pd.to_datetime(times_tracked).unique())
    
    print('times vary from : ',min(times_tracked_pd))
    print('times vary from : ',max(times_tracked_pd),'\n\n')
    
    csv_filename = domain+'_matching_cells_file1_'+tobac_tracking_filename1+'_file2_'+tobac_tracking_filename2+'_v2.csv'
    print('reading matching cells from csv: ',csv_filename)
    df = pd.read_csv(csv_filename,index_col='Unnamed: 0')
    matched_cell_dictionary = df.T.to_dict('list')
    for cl1,cl2_list in matched_cell_dictionary.items():
        cl2_list_no_nans = [int(item) for item in cl2_list if not(pd.isnull(item)) == True]
        matched_cell_dictionary.update({cl1:cl2_list_no_nans})
    
print('\n\n')
#multiprocessing below
cpu_count1 = cpu_count()
argument = []
for cl1,cl2_list in matched_cell_dictionary.items():
    if len(cl2_list) > 0:
        for cl2 in cl2_list:
            # IN THE SECOND LINE OF THE ARGUMENT, CHANGE THIS PATH TO WHERE YOU WANT THE OUPUT PNGS <CHANGE HERE>
            argument = argument + [(domain,tdata_high_thr1,tdata_high_thr2,cl1,cl2,\
                                    xh,yh,None,zm,rams_terr,dxy,plt.get_cmap('bwr'),'/nobackupp11/isingh2/tobac_plots/',\
                                    'tracking_comparison_different_features_01_01_50_vs_01_02_50_02_before_after_2mins',tdata_features1,tdata_features2)]  # CHANGE THIS DESCRIPTIVE STRING - THIS WILL BE IN THE FILENAME <CHANGE HERE>

print('total argument length: ',len(argument),' cells')

#run with single processor on a random cell
#plot_vert_zonal_meridional_crosssection_plan_view_tobac_comparison(*random.choice(argument))

arguments_100 = random.sample(argument, 100)

def main(FUNCTION, ARGUMENT):
    print('using parallel processing to create plots for ',len(arguments_100),' cells')
    pool = Pool(cpu_count1-1)
    start_time_function = time.perf_counter()
    results = pool.starmap(FUNCTION, ARGUMENT)
    finish_time_function = time.perf_counter()
    print(f"Program finished in {finish_time_function-start_time_function} seconds")

if __name__ == "__main__":
    main(plot_vert_zonal_meridional_crosssection_plan_view_tobac_comparison, arguments_100)
