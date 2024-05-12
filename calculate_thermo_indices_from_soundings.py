
# PARALLEL

# Call all the modules used in sounding functions
import pandas as pd
import re, time, glob, os
from multiprocessing import Pool, cpu_count

from sounding_functions import create_indices, plot_skewT
# Metpy
from metpy.units import units as munits
import metpy.calc as mpcalc
from metpy.cbook import get_test_data
from metpy.plots import Hodograph, SkewT
from metpy.units import units
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

##################################################
# SharpPy: https://github.com/sharppy/SHARPpy/archive/refs/heads/master.zip
import sharppy.sharptab.profile as profile
import sharppy.sharptab.interp as interp
import sharppy.sharptab.winds as winds
import sharppy.sharptab.utils as utils
import sharppy.sharptab.params as params
import sharppy.sharptab.thermo as thermo

# Matplotlib helper modules/functions for skew-T
from matplotlib.axes import Axes
import matplotlib.transforms as transforms
import matplotlib.axis as maxis
import matplotlib.spines as mspines
import matplotlib.path as mpath
from matplotlib.projections import register_projection
##################################################

def create_thermodynamic_indices(CSVFILE,UNUSED):   
        
    fname = os.path.basename(CSVFILE)
    split_filename = fname.split("_")
    env_radius = re.sub('[^0-9]','', split_filename[4])
    cellno     = split_filename[12]
    timestr    = split_filename[13]
    #print('cell#: ',cellno)
    #print('env radius: ',env_radius)
    #print('time: ',timestr)
    df_inflow = pd.read_csv(CSVFILE)[1:]
    #print(df_inflow)
    #print('##########')
    # Drop any rows with all NaN values for T, Td, winds
    df_inflow = df_inflow.dropna(subset=('height_m', 'pressure_hPa', 'temp_degC', 'dewpt_degC', 'uwnd_mps', 'vwnd_mps'),
                                 how='all').reset_index(drop=True)
    p_inflow      = df_inflow['pressure_hPa'].values * units.hPa
    h_inflow      = df_inflow['height_m'].values     * units.m
    tc_inflow     = df_inflow['temp_degC'].values    * units.degC
    td_inflow     = df_inflow['dewpt_degC'].values   * units.degC
    u_inflow      = df_inflow['uwnd_mps'].values     * units('m/s')
    u_inflow      = u_inflow.to('knots')
    v_inflow      = df_inflow['vwnd_mps'].values     * units('m/s')
    v_inflow      = v_inflow.to('knots')
    inflow_params = create_indices(p_inflow, h_inflow,tc_inflow,td_inflow,u_inflow,v_inflow, WRITE_SRH_OBS=False, WRITE_ADVANCED_INDICES=False, L_OR_R=None, U_STORM_OBS=None, V_STORM_OBS=None)
    inflow_params_df_temp = pd.DataFrame.from_dict(inflow_params).drop(labels=1)#.iloc[0]
    inflow_params_df_temp['cell']=cellno
    inflow_params_df_temp['env_radius']=env_radius
    inflow_params_df_temp['time']=timestr
    #print(inflow_params_df_temp)
    return inflow_params_df_temp

###-------------------------------------------------###
domain='DRC1.1-R'
#csv_folder='/nobackupp11/isingh2/tobac_plots/sounding_csvs_and_WP_snapshots/'  # Pleaides
#csv_folder='/Users/isingh/SVH/SVH_paper1/scratch/'                             # Personal computer
csv_folder='/home/isingh/code/scratch/environmental_assessment/'                # CSU machine

good_cells = [9248, 11337, 11338, 12359, 12375, 12433, 13385, 18361, 18367, 22293, 22337, 23259, 23283, 24234, 25223, 25249, \
              25266, 26292, 26368, 30365, 30392, 31336, 35269, 36244, 38196, 39147, 40177, 40186, 41205, 42212, 46105, 46112, \
              47077, 47084, 47106, 48108, 50021, 50022, 50966, 53788, 53790, 53803, 53859, 57689, 58626, 58632, 58648, 59611,\
              59615, 61601, 62583, 62668, 69790, 69794, 69843, 70804, 75524, 75609, 77534, 81338, 82296, 84244, 85178, 86169,\
              87056, 88814, 88820, 91621, 97939, 97952, 100682, 101563, 104265, 105990, 106841, 107801, 110523, 111367, 112325]

csv_files=[]
for cell in good_cells:
    for fil in sorted(glob.glob(csv_folder+'area_avgd_annulus_envwidth_*_2mps_mean_sounding_nearby_uds_incl_cell_'+str(cell)+'_*_'+domain+'.csv')):
        csv_files.append(fil)

print('total #csv files: ',len(csv_files))
#csv_files=sorted(glob.glob(csv_folder+'area_avgd_annulus_envwidth_*_2mps_mean_sounding_nearby_uds_incl_cell*_*_'+domain+'.csv'))#

#Running in the notebook
# cf = random.choice(csv_files)
# print('sample csv file #: ',cf)
# sample_output_df = create_thermodynamic_indices(cf,None)
# sample_output_df.to_csv('sample_file_thermodynamic_indices'+'.csv')
# 

#Running on the terminal in parallel
argument = []
for fil in csv_files:
    argument = argument + [(fil,None)]

print('length of argument is: ',len(argument))


# # ############################### FIRST OF ALL ################################
cpu_count1 = 33 #cpu_count()
print('number of cpus: ',cpu_count1)
# # #############################################################################

def main(DOMAIN, FUNCTION, ARGUMENT):
    start_time = time.perf_counter()
    with Pool(processes = (cpu_count1-1)) as pool:
        data = pool.starmap(FUNCTION, ARGUMENT)
    finish_time = time.perf_counter()
    print(f"Program finished in {finish_time-start_time} seconds")
    df_all = pd.concat(data, ignore_index=True)
    thermo_indices_data_csv_file = csv_folder+'thermodynamic_indices_' + DOMAIN + '.csv'
    print('saving thermodynamic indices to the file: ',thermo_indices_data_csv_file)
    df_all.to_csv(thermo_indices_data_csv_file)  # sounding data
    
if __name__ == "__main__":
    main(domain, create_thermodynamic_indices, argument)
