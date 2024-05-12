
# PARALLEL

# Call all the modules used in sounding functions
import pandas as pd
import re, time, glob, os , tqdm 
#from tqdm.notebook import trange, tqdm
from multiprocessing import Pool, cpu_count
import istarmap 

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
    env_width = re.sub('[^0-9]','', split_filename[4])
    cellno     = split_filename[9]
    timestr    = split_filename[10]
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
    inflow_params_df_temp['env_width']=env_width
    inflow_params_df_temp['time']=timestr
    #print(inflow_params_df_temp)
    return inflow_params_df_temp

###-------------------------------------------------###
domain='DRC1.1-R'
#csv_folder='/nobackupp11/isingh2/tobac_plots/sounding_csvs_and_WP_snapshots/'  # Pleaides
#csv_folder='/Users/isingh/SVH/SVH_paper1/scratch/'                             # Personal computer
csv_folder='/home/isingh/code/scratch/environmental_assessment/'                # CSU machine

good_cells= [8776, 8800, 8961, 9663, 9687, 10283, 11239, 11776, 11816, 11910, 11919, 12508, 12517, 12534, 13311, 14048, 14080, 14305, 14935, 14947, 14989, 15001, 15047, 15070, 15635, 15650, 15657, 15669, 15752, 15772, 16277, 16392, 16404, 16426, 16450, 17050, 17052, 17061, 17160, 17165, 17193, 17633, 17834, 17860, 17862, 18431, 19181, 19898, 20061, 20083, 21744, 22353, 22496, 22905, 23153, 23181, 24569, 25444, 25475, 26747, 26857, 26864, 26933, 27576, 27650, 28322, 28327, 28340, 28435, 28545, 29067, 29100, 29621, 29833, 29837, 29889, 30520, 30526, 31224, 31830, 32010, 32610, 32714, 33358, 34238, 34318, 34933, 34939, 35611, 35735, 36254, 36367, 37121, 37808, 38468, 38600, 40050, 40123, 40505, 41338, 41566, 42197, 42422, 43547, 43635, 43651, 44436, 45164, 45185, 45280, 45856, 45892, 45928, 46530, 46674, 47321, 47915, 48039, 48708, 48714, 49257, 49490, 51073, 51088, 51653, 51867, 52393, 52425, 52521, 53290, 54057, 54132, 54171, 54908, 56044, 56243, 56310, 56979, 57761, 59143, 59778, 59846, 59906, 60658, 61618, 62013, 62038, 62070, 62830, 62915, 63471, 64226, 64977, 65009, 65563, 65682, 66425, 66438, 67063, 67149, 69213, 69841, 69909, 70551, 70563, 70681, 71871, 72618, 73919, 73941, 74684, 74712, 75306, 75323, 75342, 76540, 76725, 77789, 78575, 79269, 79361, 79465, 79466, 79617, 80119, 80186, 80870, 81477, 81956, 81959, 82133, 82791, 83435, 83535, 84105, 85485, 86891, 87634, 88337, 90184]

csv_files=[]
for cell in good_cells:
    for fil in sorted(glob.glob(csv_folder+'area_avgd_annulus_envwidth_*_2mps_mean_sounding_cell_'+str(cell)+'_*_'+domain+'_comb_track_filt_01_02_50_02_sr5017_setpos_other_uds_masked.csv')):
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
cpu_count1 = 4 #cpu_count()
print('number of cpus: ',cpu_count1)
# # #############################################################################

data=[]

def main(DOMAIN, FUNCTION, ARGUMENT):
    start_time = time.perf_counter()
    with Pool(processes = cpu_count1) as pool:
        for result in tqdm.tqdm(pool.istarmap(create_thermodynamic_indices, argument),total=len(argument)):
            data.append(result)
        #data = pool.starmap(FUNCTION, ARGUMENT)
    finish_time = time.perf_counter()
    print(f"Program finished in {finish_time-start_time} seconds")
    df_all = pd.concat(data, ignore_index=True)
    thermo_indices_data_csv_file = csv_folder+'thermodynamic_indices_' + DOMAIN + '_comb_track_filt_01_02_50_02_sr5017_setpos_100_cells_other_uds_masked.csv'
    print('saving thermodynamic indices to the file: ',thermo_indices_data_csv_file)
    df_all.to_csv(thermo_indices_data_csv_file)  # sounding data
    
if __name__ == "__main__":
    main(domain, create_thermodynamic_indices, argument)
