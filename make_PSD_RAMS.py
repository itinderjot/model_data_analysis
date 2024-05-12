import matplotlib.image as mpimg
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import glob
import os
from multiprocessing import Pool, cpu_count
import time
import h5py
import hdf5plugin
import pandas as pd
import xarray as xr
from netCDF4 import Dataset
plt.style.use('ggplot')
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

simulations=['PHI1.1-R','PHI2.1-R','WPO1.1-R','BRA1.1-R','DRC1.1-R','AUS1.1-R']
domain='3'
variables = [['Tk', 0, 'model', '$K$']             , ['THETA', 0, 'model', '$K$'],\
             ['QV', 0, 'model', '$kg kg^{-1}$'], ['RH', 0, 'model', '$RH_{sfc}^{2} (percent^{2})$'],\
             ['U', 0, 'model', '$m s^{-1}$']        , ['V', 0, 'model', '$m s^{-1}$'],\
             ['WSPD', 0, 'model', '$m s^{-1}$']  , ['W', 0, 'model', '$m s^{-1}$'],\
             ['MCAPE', -999, None, '$J^{2}kg^{-2})$']     , ['MCIN', -999, None, '$MCIN^{2} (J^{2}kg^{-2})$'], \
             ['Tk', 750, 'pressure', '$K$']        , ['THETA', 750, 'pressure', '$K$'],\
             ['QV', 750, 'pressure', '$kg kg^{-1}$'], ['RH', 750, 'pressure', '$RH_{750}^{2} (percent^{2})$'],\
             ['U', 750, 'pressure', '$m^{2}s^{-2})$']   , ['V', 750, 'pressure', '$m s^{-1}$'],\
             ['WSPD', 750, 'pressure', '$m s^{-1}$'], ['W', 750, 'pressure', '$m s^{-1}$'],\
             ['Tk', 500, 'pressure', '$K$']        , ['THETA', 500, 'pressure', '$K$'],\
             ['QV', 500, 'pressure', '$kg kg^{-1}$'], ['RH', 500, 'pressure', '$RH_{500}^{2} (percent^{2})$'],\
             ['U', 500, 'pressure', '$m s^{-1}$']   , ['V', 500, 'pressure', '$m s^{-1}$'],\
             ['WSPD', 500, 'pressure', '$m s^{-1}$'], ['W', 500, 'pressure', '$m s^{-1}$'],\
             ['Tk', 200, 'pressure', '$K$']        , ['THETA', 200, 'pressure', '$K$'],\
             ['QV', 200, 'pressure', '$kg kg^{-1}$'], ['RH', 200, 'pressure', '$percent$'],\
             ['U', 200, 'pressure', '$m s^{-1}$']   , ['V', 200, 'pressure', '$m s^{-1}$'], \
             ['WSPD', 200, 'pressure', '$m s^{-1}$'], ['W', 200, 'pressure', '$m s^{-1}$']]

units_dict = {'Tk':'$K$','QV':'$kg kg^{-1}$','RH':'percent','WSPD':'$m s^{-1}$','U':'$m s^{-1}$',\
              'V':'$m s^{-1}$','W':'$m s^{-1}$','MCAPE':'$J kg^{-1}$','MCIN':'$J kg^{-1}$','THETA':'$K$'}

colors    =  ['#000000','#E69F00','#56B4E9','#009E73','#F0E442','#0072B2','#D55E00','#CC79A7']

color_dict = {'ARG1.1-R_old':'#000000',\
              'PHI1.1-R':'#E69F00',\
              'PHI2.1-R':'#56B4E9',\
              'WPO1.1-R':'#009E73',\
              'BRA1.1-R':'#7F7F7F',\
              'USA1.1-R':'#0072B2',\
              'DRC1.1-R':'#D55E00',\
              'AUS1.1-R':'#CC79A7'}

def make_PSD_RAMS(WHICH_TIME, VARIABLE, SIMULATIONS, DOMAIN):

    print('working on ',VARIABLE,'\n')
    fig = plt.figure(figsize=(9,9))
    ax1 = fig.add_subplot(111)
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
        print('        choosing the '+WHICH_TIME+' file: ',rams_fil)
      
        z, z_name, z_units, z_time = read_vars_WRF_RAMS.read_variable(rams_fil,VARIABLE[0],'RAMS',output_height=False,interpolate=VARIABLE[1]>-1,level=VARIABLE[1],interptype=VARIABLE[2])
        y_dim,x_dim = np.shape(z)
        if DOMAIN=='1':
            dx=1.6
        if DOMAIN=='2':
            dx=0.4
        if DOMAIN=='3':
            dx=0.1
            
        # make array square
        # Get the dimensions of the image
        height, width = z.shape
        print('height of image: ',height)
        print('width of image: ',width)
        # Calculate the size of the square
        size = min(width, height)

        # Calculate the coordinates for cropping
        left = (width - size) // 2
        top = (height - size) // 2
        right = (width + size) // 2
        bottom = (height + size) // 2

        # Crop the image using NumPy slicing
        cropped_img_array = z[top:bottom, left:right]

        # fourier spectrum part
        image = cropped_img_array
        print('shape of input image: ',np.shape(image))
        npix = image.shape[0]

        fourier_image = np.fft.fftn(image)
        fourier_amplitudes = np.abs(fourier_image)**2

        kfreq = np.fft.fftfreq(npix) * npix
        kfreq2D = np.meshgrid(kfreq, kfreq)
        knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)
        #print('shape of knrm: ',np.shape(knrm))
        #print('shape of fourier_amplitudes: ',np.shape(fourier_amplitudes))
        knrm = knrm.flatten()
        fourier_amplitudes = fourier_amplitudes.flatten()
        #print('shape of knrm: ',np.shape(knrm))
        #print('shape of fourier_amplitudes: ',np.shape(fourier_amplitudes))
        kbins = np.arange(0.5, npix//2+1, 1.)
        #print(kbins)
        kvals = 0.5 * (kbins[1:] + kbins[:-1])
        Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                             statistic = "mean",
                                             bins = kbins)
        Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)

        # plotting part
        #ax1.semilogy(kvals, Abins  , color=color_dict[simulation], label=simulation)
        #ax1.plot(kvals, Abins  , color=color_dict[simulation], label=simulation)
        ax1.plot(kvals, Abins*kvals, color=color_dict[simulation], label=simulation)
        #ax1.semilogy(kvals, Abins*kvals, color=color_dict[simulation], label=simulation)

    ax1.set_xscale("log")
    #ax1.plot(kvals[25:-120],(25*kvals[25:-120])**(-5/3) , linestyle='--',label=r'$E(k) \propto k^{-5/3}$', linewidth=1.5, color='r')
    ax1.set_ylabel(r'power spectral density')# $(m^{3} s^{-2})$')
    ax1.set_xlabel(r'# waves/length of domain')
    #ax1.axvline(2*np.pi/11200., color='g', alpha=0.5,label='')
    #secax = ax1.secondary_xaxis('top', functions=(lambda x: 2*np.pi/x, lambda x: 2*np.pi/x))
    secax = ax1.secondary_xaxis('top', functions=(lambda x: npix*dx/x, lambda x: npix*dx/x))
    secax.set_xlabel(r"Wavelength (km)")
    plt.legend(loc=('upper right'))
    if VARIABLE[2]:
        title_string = 'Power spectrum of '+VARIABLE[0]+' at '+VARIABLE[2]+' level '+str(int(VARIABLE[1]))+' for d0'+DOMAIN+': '+WHICH_TIME+' of the simulation'
    else:
        title_string = 'Power spectrum of '+VARIABLE[0]+' for d0'+DOMAIN+': '+WHICH_TIME+' of the simulation'
    plt.title(title_string)
    #plt.title('KE (w) spectrum (1D, west to east) for WRF and RAMS at  model level '+str(lev_num))
    #plt.savefig('KE_uv_lev_'+str(lev_num)+'_west_to_east_1D_spectral_density_linear_detrend_then_window_WRF_RAMS.png',dpi=200)
    #plt.savefig('KE_uv_lev_'+str(lev_num)+'_west_to_east_1D_spectral_density_linear_detrend_then_window.png')
    print('-----\n')
    plt.tight_layout()
    if VARIABLE[2]:
        filename = 'power_spectrum_premultiplied_RAMS_'+simulation+'_'+VARIABLE[0]+'_levtype_'+VARIABLE[2]+'_lev_'+str(int(VARIABLE[1]))+'_d0'+DOMAIN+'_'+WHICH_TIME+'.png'
    else:
        filename = 'power_spectrum_premultiplied_RAMS_'+simulation+'_'+VARIABLE[0]+'_levtype_'+'None'+'_lev_'+'None'+'_d0'+DOMAIN+'_'+WHICH_TIME+'.png'
    print('saving to png file: ',filename)
    plt.savefig(filename, dpi = 150, bbox_inches = "tight")
    plt.close()
    
        
#make_PSD_RAMS('middle', variables[1], simulations, domain)

argument = []
for var in variables:
    argument = argument + [('middle',var, simulations, domain)]

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
    main(make_PSD_RAMS, argument)
