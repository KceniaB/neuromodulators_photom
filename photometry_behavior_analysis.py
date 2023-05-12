#%% 
#===========================================================================
#  ?                                ABOUT
#  @author         :  Kcenia Bougrova
#  @repo           :  KceniaB
#  @createdOn      :  2023-05-12 new file from the previous "photometry_processing_code.py"
#  @description    :  align the processed photometry data to the behavior 
#  @lastUpdate     :  2023-05-12
#===========================================================================

#%%
#===========================================================================
#                            1. IMPORTS
#===========================================================================
from photometry_processing_new_functions import *

#%%
#===========================================================================
#                            2. FILE PATHS
#===========================================================================
#* 2022-11-28 S6 cut the other half of the session, which is regarding the photometry system working for 3G but only recording for S10 4G who did 1h30
mouse = 'S6'
main_path ='/home/kcenia/Documents/Photometry_results/2022-11-28/'
session_path = main_path+'raw_photometry4.csv' 
io_path = main_path+'bonsai_DI04.csv' 
mouse = 'S6' 
session_day = main_path[-11:-1]
session_path_behav = main_path+mouse
region = 'Region3G'

df_PhotometryData = pd.read_csv(session_path) 



"""

TO BE EDITED FROM THE OTHER FILE

"""