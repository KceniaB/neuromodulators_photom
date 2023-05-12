#%% 
#===========================================================================
#  ?                                ABOUT
#  @author         :  Kcenia Bougrova
#  @repo           :  KceniaB
#  @createdOn      :  photometry_processing_new 05102022
#  @description    :  process the photometry data and align to the behavior 
#  @lastUpdate     :  2023-04-05
#===========================================================================

#%%
#===========================================================================
#                            1. IMPORTS
#===========================================================================
from photometry_processing_functions import *
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
#%% 
#===========================================================================
#      4. FUNCTIONS TO LOAD DATA AND ADD SOME VARIABLES (BEHAVIOR)
#===========================================================================
if 'LedState' in df_PhotometryData.columns:                         #newversion 
    df_PhotometryData = start_2_end_1(df_PhotometryData)
    df_PhotometryData = df_PhotometryData.reset_index(drop=True)
    df_PhotometryData = (change_flags(df_PhotometryData))
else:                                                               #oldversion
    df_PhotometryData = start_17_end_18(df_PhotometryData) 
    df_PhotometryData = df_PhotometryData.reset_index(drop=True) 
    df_PhotometryData = (change_flags(df_PhotometryData))
    df_PhotometryData["LedState"] = df_PhotometryData["Flags"]

""" 4.1.2 Check for LedState/previous Flags bugs """ 
""" 4.1.2.1 Length """
# Verify the length of the data of the 2 different LEDs
df_470, df_415 = verify_length(df_PhotometryData)
""" 4.1.2.2 Verify if there are repeated flags """ 
verify_repetitions(df_PhotometryData["LedState"])
""" 4.1.3 Remove "weird" data (flag swap, huge signal) """ 
plot_outliers(df_470,df_415,region,mouse,session_day)
#================================================

#%% 
""" remove weird signal / cut session """
init_idx = 1000 #1000
end_idx = len(df_PhotometryData)
#end_idx = 300000 #180000
df_PhotometryData_1 = df_PhotometryData[init_idx:end_idx] 

#470nm 
df_470 = df_PhotometryData_1[df_PhotometryData_1.LedState==2] 
# df_470 = df_470.reset_index(drop=True)
#415nm 
df_415 = df_PhotometryData_1[df_PhotometryData_1.LedState==1] 
# df_415 = df_415.reset_index(drop=True)
print("470 = ",df_470.LedState.count()," 415 = ",df_415.LedState.count())

plt.rcParams["figure.figsize"] = (8,5)  
plt.plot(df_470[region],c='#279F95',linewidth=0.5)
plt.plot(df_415[region],c='#803896',linewidth=0.5) 
plt.legend(["GCaMP","isosbestic"],frameon=False)
sns.despine(left = False, bottom = False) 
plt.show()

# %% 
df_PhotometryData = df_PhotometryData_1.reset_index(drop=True)  
df_470 = df_PhotometryData[df_PhotometryData.LedState==2] 
df_470 = df_470.reset_index(drop=True)
df_415 = df_PhotometryData[df_PhotometryData.LedState==1] 
df_415 = df_415.reset_index(drop=True) 
#================================================
""" 4.1.4 FRAME RATE """ 
acq_FR = find_FR(df_470["Timestamp"]) 
#================================================
#%%
#===========================================================================
#                       4.2 TTL from Neurophotometrics
#===========================================================================
df_raw_phdata_DI0_true = import_DI(io_path)
#================================================
#%%
#===========================================================================
#                           4.3 BEHAVIOR Bpod data
#===========================================================================
# * * * * * * * * * * LOAD BPOD DATA * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
#! NOT REALLY WORKING 
behav_path = session_path_behav+"/raw_behavior_data/_iblmic_audioOnsetGoCue.times_mic.npy"
# behav_path=test
if path.exists(behav_path)==True: 
    from ibllib.io.extractors.training_trials import extract_all 
    df_alldata = extract_behav_t(session_path_behav) 
    print("trainingCW")
else: 
    from ibllib.io.extractors.biased_trials import extract_all 
    df_alldata = extract_behav_b(session_path_behav)
    #print("biasedCW")

#%% 
#================================================
#* contrasts column (negative values are when the stim appeared in the Left side) 
df_alldata = all_contrasts(df_alldata) 
#================================================ 
#* creating stim_Right variable for each side: stim_left is -1 and stim_right is 1 
#the contrast side is the df_alldata.position 
#positive is contrastRight
#================================================
# creating reaction and response time variables 
    #reaction = first mov after stim onset 
df_alldata = new_time_vars(df_alldata,new_var="reactionTime",second_action="firstMovement_times",first_action = "stimOn_times")
    #response = time for the decision/wheel movement, from stim on until choice 
df_alldata = new_time_vars(df_alldata,new_var="responseTime",second_action="response_times",first_action = "stimOn_times")
    #response_mov = time for the decision/wheel movement, from the 1st mov on until choice 
df_alldata = new_time_vars(df_alldata,new_var="responseTime_mov",second_action="response_times",first_action = "firstMovement_times")

fig, axs = plt.subplots(3, 2,figsize=(15,10)) 
axs[0, 0].hist(df_alldata.reactionTime, bins = 50,alpha=0.8, color='#4B8EB3')
axs[0, 0].set_title('Reaction times = 1stMov - stimOn')
axs[0, 1].hist(df_alldata.reactionTime, bins = 1000,alpha=0.8,color='#4B8EB3') 
axs[0, 1].set_title('Reaction times = 1stMov - stimOn zoomed in')
axs[0, 1].set_xlim([0,1.2])
axs[1, 0].hist(df_alldata.responseTime, bins = 50,alpha=0.8, color='#f9a620')
axs[1, 0].set_title('Response times = response - stimOn')
axs[1, 1].hist(df_alldata.responseTime, bins = 1000,alpha=0.8, color='#f9a620') 
axs[1, 1].set_title('Response times = response - stimOn zoomed in')
axs[1, 1].set_xlim([0,1.2])
axs[2, 0].hist(df_alldata.responseTime_mov, bins = 50,alpha=0.8, color='#7cb518')
axs[2, 0].set_title('Response times since mov = response - 1stmov')
axs[2, 1].hist(df_alldata.responseTime_mov, bins = 1000,alpha=0.8, color='#7cb518') 
axs[2, 1].set_title('Response times since mov = response - 1stmov zoomed in')
axs[2, 1].set_xlim([0,1.2])
for ax in axs.flat:
    ax.set(xlabel='time (s)') 
# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
fig.tight_layout()
plt.show()

#================================================
df_alldata = new_time_vars_c_inc(df_alldata,new_var="reactionTime") 
df_alldata = new_time_vars_c_inc(df_alldata,new_var="responseTime") 
df_alldata = new_time_vars_c_inc(df_alldata,new_var="responseTime_mov") 

fig, axs = plt.subplots(3, 2,figsize=(15,10)) 
axs[0, 0].hist(df_alldata.reactionTime_c, bins = 500,alpha=0.5)
axs[0, 0].hist(df_alldata.reactionTime_inc, bins = 500,alpha=0.5,color='red')
axs[0, 0].set_title('Reaction times = 1stMov - stimOn')
axs[0, 0].set_ylim([0,5])
axs[0, 1].hist(df_alldata.reactionTime_c, bins = 500,alpha=0.5)
axs[0, 1].hist(df_alldata.reactionTime_inc, bins = 500,alpha=0.5,color='red')
axs[0, 1].set_title('Reaction times = 1stMov - stimOn')
axs[0, 1].set_xlim([0,1.2])
axs[1, 0].hist(df_alldata.responseTime_c, bins = 500,alpha=0.5)
axs[1, 0].hist(df_alldata.responseTime_inc, bins = 500,alpha=0.5,color='red')
axs[1, 0].set_title('Response times = response - stimOn')
axs[1, 0].set_ylim([0,5])
axs[1, 1].hist(df_alldata.responseTime_c, bins = 500,alpha=0.5)
axs[1, 1].hist(df_alldata.responseTime_inc, bins = 500,alpha=0.5,color='red')
axs[1, 1].set_title('Response times = response - stimOn')
axs[1, 1].set_xlim([0,1.2])
axs[2, 0].hist(df_alldata.responseTime_mov_c, bins = 500,alpha=0.5)
axs[2, 0].hist(df_alldata.responseTime_mov_inc, bins = 500,alpha=0.5,color='red')
axs[2, 0].set_title('Response times from mov to choice = response - 1stMov')
axs[2, 0].set_ylim([0,5])
axs[2, 1].hist(df_alldata.responseTime_mov_c, bins = 500,alpha=0.5)
axs[2, 1].hist(df_alldata.responseTime_mov_inc, bins = 500,alpha=0.5,color='red')
axs[2, 1].set_title('Response times from mov to choice = response - 1stMov')
axs[2, 1].set_xlim([0,1.2])
for ax in axs.flat:
    ax.set(xlabel='time (s)') 
# Hide x labels and tick labels for top plots and y ticks for right plots.
# for ax in axs.flat:
#     ax.label_outer()
fig.tight_layout()
plt.show()

show_plot(df_alldata)
#================================================


# %%
#===========================================================================
#                   5. INTERPOLATE BPOD AND NPH TIME
#===========================================================================

""" 5.1 Check the same length of the events from BPOD (Behav) & TTL (Nph) """
df_raw_phdata_DI0_true = import_DI_seconds(io_path) 
# load the bpod event times - it was the TTL that was sent to the nph system
bpod_sync = bpod_sync_f(bpod_event = "stimOnTrigger_times") #"feedback_times" "goCue_times" 
# load the received TTL event times - it was the TTL that was received by the nph system
nph_sync = nph_sync_f(nph_col = "Seconds")

#to test if they have the same length 
print(len(bpod_sync),len(nph_sync))
bpod_diff = np.diff(bpod_sync)
nph_diff = np.diff(nph_sync)
print("Bpod first ", bpod_diff[0:5], "\n\n Nph first ", nph_diff[0:5], "\n\n \n\n Bpod last ", bpod_diff[(len(bpod_diff)-5):(len(bpod_diff))], "\n\n Nph last ", nph_diff[(len(nph_diff)-5):(len(nph_diff))]) 
#================================================


#%%
""" 5.2 Assert & reVerify the length from BPOD (Behav) & TTL (Nph) """ 
def assert_length(x,y): 
    """
    If the length is different only by 1 value, run this function (after checking the difference of consecutive values to figure out if it's one more value at the end and not beginning)
    x = nph_sync
    y = bpod_sync
    """ 
    if len(x)-1 == len(y): 
        x = x[0:len(y)]
        print("Option 1: nph_sync had 1 more value")
    else: 
        print("Option 2: EQUAL VALUES OR SOMETHING IS WRONG!")
    return x
nph_sync = assert_length(nph_sync,bpod_sync) 
assert len(bpod_sync) == len(nph_sync), "sync arrays are of different length" 

#$$
def verify_length(x,y): 
    """
    If the length is different only by 1 value, run this function (after checking the difference of consecutive values to figure out if it's one more value at the end and not beginning)
    x = nph_sync
    y = bpod_sync
    """ 
    if len(x) == len(y): 
        print("Option 1: same length :)")
    else: 
        print("Option 2: SOMETHING IS WRONG! Different len's")
verify_length(nph_sync,bpod_sync) 
#================================================


#%%
""" 5.3 INTERPOLATE """
# * * * * * * * * * * INTERPOLATE * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 

def interpolate_times(x,y): 
    """
    matches the values
    x and y should be the same length 
    x = nph_sync 
    y = bpod_sync
    """
    nph_to_bpod_times = interp1d(x, y, fill_value='extrapolate') 
    nph_to_bpod_times
    nph_frame_times = df_PhotometryData['Timestamp']
    frame_times = nph_to_bpod_times(nph_frame_times)   # use interpolation function returned by `interp1d`
    plt.plot(x, y, 'o', nph_frame_times, frame_times, '-')
    plt.show() 
    return frame_times


df_PhotometryData["bpod_frame_times_feedback_times"] = interpolate_times(nph_sync,bpod_sync)  
df_470 = df_PhotometryData[df_PhotometryData.LedState==2] 
df_470 = df_470.reset_index(drop=True)
df_415 = df_PhotometryData[df_PhotometryData.LedState==1] 
df_415 = df_415.reset_index(drop=True)
#================================================



#%% 
#===========================================================================
#                       6. PHOTOMETRY SIGNAL PROCESSING
#===========================================================================
#===========================================================================
# *                            INFO HEADER
#   I should have: 
#       GCaMP, 
#       isosbestic, 
#       times in nph, 
#       times in bpod, 
#       ttl in nph, 
#       ttl in bpod  
#===========================================================================

raw_reference = df_415[region]#[1:] #isosbestic
raw_signal = df_470[region]#[1:] #GCaMP signal 
# raw_reference = df_415['Region5G']#[1:] #isosbestic
# raw_signal = df_470['Region5G']#[1:] #GCaMP signal
#raw_reference = df_415['Region7G']#[1:] #isosbestic
#raw_signal = df_470['Region7G']#[1:] #GCaMP signal
raw_timestamps_bpod = df_470["bpod_frame_times_feedback_times"]#[1:] 
raw_timestamps_nph_470 = df_470["Timestamp"]#[1:] 
raw_timestamps_nph_415 = df_415["Timestamp"]#[1:] 
raw_TTL_bpod = bpod_sync
raw_TTL_nph = nph_sync

plt.plot(raw_signal[:])
plt.plot(raw_reference[:])
plt.show() 
#================================================


#%%
plt.rcParams["figure.figsize"] = (12,8)  
fig, ax1 = plt.subplots()

color = 'tab:green'
ax1.set_xlabel('frames')
ax1.set_ylabel('GCaMP', color=color)
ax1.plot(raw_signal, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:purple'
ax2.set_ylabel('sin', color=color)  # we already handled the x-label with ax1
ax2.plot(raw_reference, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.set_dpi(100)
fig.tight_layout() 
plt.show()
#================================================


#%% 
plt.rcParams["figure.figsize"] = (20,3)  
fig, ax1 = plt.subplots()
color = 'tab:green'
ax1.set_xlabel('frames')
ax1.set_ylabel('GCaMP', color=color)
ax1.plot(raw_signal, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:purple'
ax2.set_ylabel('isosbestic', color=color)  # we already handled the x-label with ax1
ax2.plot(raw_reference, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.set_dpi(100)
fig.tight_layout() 
plt.show() 
#================================================


# %%
#===========================================================================
#      6.1 PHOTOMETRY SIGNAL PROCESSING - according to the GitHub code
#===========================================================================
""" 
1. Smooth
""" 
smooth_win = 10
smooth_reference = smooth_signal(raw_reference, smooth_win)
smooth_signal = smooth_signal(raw_signal, smooth_win) 

fig = plt.figure(figsize=(16, 10))
ax1 = fig.add_subplot(211)
ax1.plot(smooth_signal,'blue',linewidth=1.5)
ax2 = fig.add_subplot(212)
ax2.plot(smooth_reference,'purple',linewidth=1.5) 
#===========================

#%% 
""" 
2. Find the baseline
""" 
lambd = 5e4 # Adjust lambda to get the best fit
porder = 1
itermax = 50
r_base=airPLS(smooth_reference.T,lambda_=lambd,porder=porder,itermax=itermax)
s_base=airPLS(smooth_signal,lambda_=lambd,porder=porder,itermax=itermax)

fig = plt.figure(figsize=(16, 10))
ax1 = fig.add_subplot(211)
ax1.plot(smooth_signal,'blue',linewidth=1.5)
ax1.plot(s_base,'black',linewidth=1.5)
ax2 = fig.add_subplot(212)
ax2.plot(smooth_reference,'purple',linewidth=1.5)
ax2.plot(r_base,'black',linewidth=1.5) 
#===========================

#%% 
""" 
3. Remove the baseline and the beginning of the recordings
""" 
remove=500
reference = (smooth_reference[remove:] - r_base[remove:])
signal = (smooth_signal[remove:] - s_base[remove:]) 
timestamps_bpod = raw_timestamps_bpod[remove:]
timestamps_nph_470 = raw_timestamps_nph_470[remove:] 
timestamps_nph_415 = raw_timestamps_nph_415[remove:]
#KB ADDED 

fig = plt.figure(figsize=(16, 10))
ax1 = fig.add_subplot(211)
ax1.plot(signal,'blue',linewidth=1.5)
ax2 = fig.add_subplot(212)
ax2.plot(reference,'purple',linewidth=1.5) 
#===========================

#%% 
""" 
4. Standardize signals
""" 
z_reference = (reference - np.median(reference)) / np.std(reference)
z_signal = (signal - np.median(signal)) / np.std(signal) 

fig = plt.figure(figsize=(16, 10))
ax1 = fig.add_subplot(211)
ax1.plot(z_signal,'blue',linewidth=1.5)
ax2 = fig.add_subplot(212)
ax2.plot(z_reference,'purple',linewidth=1.5) 
#===========================

#%% 
""" 
5. Fit reference signal to calcium signal using linear regression
""" 
from sklearn.linear_model import Lasso
lin = Lasso(alpha=0.0001,precompute=True,max_iter=1000,
            positive=True, random_state=9999, selection='random')
n = len(z_reference)
lin.fit(z_reference.reshape(n,1), z_signal.reshape(n,1))
#===========================

#%% 
""" 
6. Align reference to signal
""" 
z_reference_fitted = lin.predict(z_reference.reshape(n,1)).reshape(n,) 

fig = plt.figure(figsize=(16, 8))
ax1 = fig.add_subplot(111)
ax1.plot(z_reference,z_signal,'b.')
ax1.plot(z_reference,z_reference_fitted, 'r--',linewidth=1.5) 

#%%
fig = plt.figure(figsize=(16, 8))
ax1 = fig.add_subplot(111)
ax1.plot(z_signal,'blue')
ax1.plot(z_reference_fitted,'purple') 
#===========================

#%% 
""" 
7. Calculate z-score dF/F 
"""
zdFF = (z_signal - z_reference_fitted)

fig = plt.figure(figsize=(16, 8))
ax1 = fig.add_subplot(111)
ax1.plot(zdFF,'black')

""" """ """ """ """ """ """ """ """ """ """ """ """ """ """ """ """ """ """ """
#================================================


# %% 
#===========================================================================
#                            7. Joining data
#===========================================================================
""" 7.1 join the timestamp already transformed with the zdFF """
timestamps_nph_470 = timestamps_nph_470.reset_index(drop=True)
timestamps_nph_415 = timestamps_nph_415.reset_index(drop=True)
timestamps_bpod=timestamps_bpod.reset_index(drop=True)
df = pd.DataFrame(timestamps_nph_470)
df = df.rename(columns={'Timestamp': 'timestamps_nph_470'})
df = df.reset_index() 
df["timestamps_nph_415"] = timestamps_nph_415
df["timestamps_bpod"] = timestamps_bpod 
df["zdFF"] = zdFF 
#================================================


#%% 
""" 7.2 adding the raw_reference and the raw_signal """
raw_reference=raw_reference[remove:len(raw_reference)]
raw_signal=raw_signal[remove:len(raw_signal)]
df["raw_reference"] = raw_reference
df["rawsignal"] = raw_signal 
#================================================


#%%
""" 7.3 Removing extra TTLs (which are there due to the photometry signal removal during the processing) """
""" 7.3.1 check the data """ 
df_ttl = pd.DataFrame(raw_TTL_nph, columns=["ttl_nph"])
df_ttl["ttl_bpod"] = raw_TTL_bpod 

xcoords = df_ttl['ttl_bpod']
for xc in zip(xcoords):
    plt.axvline(x=xc, color='blue')
plt.rcParams.update({'font.size': 22})
plt.plot(df['timestamps_bpod'],df['zdFF'], color = 'g') 
plt.rcParams["figure.figsize"] = (20,10)
plt.show()
#================================================


#%% 
""" 7.3.2 join the rest of the events to this TTL """
df_events = pd.concat([df_ttl, df_alldata], axis=1)
xcoords = df_events['ttl_bpod']
for xc in zip(xcoords):
    plt.axvline(x=xc, color='blue')
plt.plot(df['timestamps_bpod'],df['zdFF'], color = 'g') 
plt.rcParams["figure.figsize"] = (20,10)
plt.show()
#================================================


# %%
""" 7.3.3 remove the TTLs that happen before the photometry signal """ 
def crop_around_TTLs(x,y): 
    """
    remove the TTLs that happened before and after the photonetry signal 
    
    x = 
    y = 
    
    output: 
    """

valuez_bef = df["timestamps_bpod"][0] #when I start the photometry signal (already cut from the pre-processing)
this_indexz = 0 
for i in range(0,len(df_events)): 
    if df_events.ttl_bpod[i] < valuez_bef: 
        this_valuez = df_events.ttl_bpod[i]
        this_indexz = i
df_events = df_events[int((this_indexz)+1):(len(df_events.ttl_bpod)-1)] 
df_events = df_events.reset_index(drop=True) 

xcoords = df_events['ttl_bpod']
for xc in zip(xcoords):
    plt.axvline(x=xc, color='blue')
#plt.xlim([1275,1300])
plt.plot(df['timestamps_bpod'],df['zdFF'], color = 'g') 
plt.rcParams["figure.figsize"] = (20,10)
#plt.xlim(1900, 1935)
plt.show() 
#================================================ 


#%% remove the TTLs that happen after the photometry signal 
""" RECHECK FOR THE FREEZING EVENTS 
#! WARNING %%%%%%% WRONG """
# valuez_aft = df["timestamps_bpod"][-1:]
# valuez_aft = valuez_aft.reset_index()
# valuez_aft = valuez_aft.timestamps_bpod[0] 
# for i in range(0,len(df_events)): 
#     if df_events.ttl_bpod[i] < valuez_aft: 
#         this_valuez = df_events.ttl_bpod[i]
#         this_indexz = i 
# df_events = df_events[0:int(this_indexz)] #gives an error Nov012021 #solved! -5669 above
# #df_events = df_events.reset_index(drop=True)

# xcoords = df_events['ttl_bpod']
# for xc in zip(xcoords):
#     plt.axvline(x=xc, color='blue')
# #plt.xlim([0,100])
# plt.plot(df['timestamps_bpod'],df['zdFF'], color = 'g') 
# plt.rcParams["figure.figsize"] = (20,10)
# #plt.xlim(1900, 1935)
# plt.show() 
#================================================


#%%
#===========================================================================
#           7.4 Restructure the behavior data & add trial column
#===========================================================================
# #* A. list of time variables table_x
# table_x = ["feedback_times",
#         "response_times", 
#         "goCueTrigger_times",
#         "goCue_times", 
#         "stimOnTrigger_times", 
#         "stimOn_times", 
#         "stimOff_times", 
#         "stimFreeze_times", 
#         "itiIn_times", 
#         "stimOffTrigger_times",
#         "stimFreezeTrigger_times",
#         "errorCueTrigger_times",
#         "intervals_0", 
#         "intervals_1", 
#         "firstMovement_times",
#         "wheel_moves_intervals_start" 
#         #"wheel_moves_intervals_stop", 
#         #"wheel_timestamps", 
#         #"peakVelocity_times",
#         ] 

# #* B. list of non-time variables table_y 
# table_y = pd.concat([df_events["ttl_nph"], 
#                             df_events["ttl_bpod"], 
#                             df_events["feedbackType"],
#                             df_events["contrastLeft"], 
#                             df_events["contrastRight"], 
#                             df_events["allContrasts"], #KB added 12-10-2022 
#                             df_events["probabilityLeft"], 
#                             df_events["choice"], 
#                             #df_events["repNum"], 
#                             df_events["rewardVolume"], 
#                             #df_events["stim_Right"], 
#                             df_events["reactionTime"],
#                             df_events["reactionTime_triggerTime"],
#                             df_events["feedback_correct"], 
#                             df_events["feedback_incorrect"], 
#                             df_events["wheel_moves_peak_amplitude"], #KB added 12-10-2022
#                             df_events["is_final_movement"], #KB added 12-10-2022
#                             df_events["phase"], #KB added 12-10-2022 
#                             df_events["position"], #KB added 12-10-2022 
#                             df_events["quiescence"]], #KB added 12-10-2022 
#                             axis=1) 

#%%
#================================================
#  *                    ALTERNATIVE
#    for older sessions
#================================================ 
#* A. list of time variables table_x
table_x = ["feedback_times",
        "response_times", 
        "goCueTrigger_times",
        "goCue_times", 
        "stimOnTrigger_times", 
        "stimOn_times", 
        #"stimOff_times", 
        #"stimFreeze_times", 
        #"itiIn_times", 
        #"stimOffTrigger_times",
        #"stimFreezeTrigger_times",
        #"errorCueTrigger_times",
        #"intervals_start", 
        #"intervals_stop", 
        "intervals_0", 
        "intervals_1", 
        "firstMovement_times",
        #"wheel_moves_intervals_start" 
        #"wheel_moves_intervals_stop", 
        #"wheel_timestamps", 
        #"peakVelocity_times",
        ] 

#* B. list of non-time variables table_y 
table_y = pd.concat([df_events["ttl_nph"], 
                            df_events["ttl_bpod"], 
                            df_events["feedbackType"],
                            df_events["contrastLeft"], 
                            df_events["contrastRight"], 
                            #df_events["cL"],
                            #df_events["cR"],    
                            df_events["allContrasts"], #KB added 12-10-2022 
                            df_events["probabilityLeft"], 
                            df_events["choice"], 
                            #df_events["repNum"], 
                            df_events["rewardVolume"], 
                            #df_events["stim_Right"], 
                            df_events["reactionTime"],
                            df_events["reactionTime_triggerTime"],
                            df_events["f_c_reactionTime"], 
                            df_events["f_inc_reactionTime"]], 
                            # df_events["wheel_moves_peak_amplitude"], #KB added 12-10-2022
                            # df_events["is_final_movement"], #KB added 12-10-2022
                            # df_events["phase"], #KB added 12-10-2022 
                            # df_events["position"], #KB added 12-10-2022 
                            # df_events["quiescence"]], #KB added 12-10-2022 
                            axis=1) 
#================================================



#%%
#================================================
#  *                    INFO
#    
""" works :D """
""" 
Goal: 
    a table with all the system/mouse outputs/choices (table_y) organized by the times of the events (column "name", for every table_x event) 

Description: 
    "{0}".format(x) substitutes {0} by x
    onetime_allnontime is data from one name from table_x which is the times-associated table into the entire non-associated to time table

    1st for loop: 
        1. I add the times of every table_x name to the whole table_y 
        2. create a column that has the table_x name that was joined to the table_y
        3. rename that column as "times" 

        4. create a data frame (onetime_allnontime_2) with the same names as one of the previously created (onetime_allnontime)

    2nd for loop:  
        5. append in loop the rest of the time created tables into that df created in 4. onetime_allnontime_2 
        6. reset the index
        7. sort the values by the "times" of the events 
        8. drop the nans, associated to the stimFreeze_times, stimFreezeTrigger_times, errorCueTrigger_times 
        9. reset again the index 
        10. add a "trial" column 
"""
# 
#================================================
onetime_allnontime={} 
for x in table_x: 
    for i in range(0, len(table_x)): 
        onetime_allnontime["{0}".format(x)] = pd.concat([df_events["{0}".format(x)], 
                                table_y], axis=1) #join df_events of each table_x to the entire table_y
        onetime_allnontime["{0}".format(x)]["name"] = "{0}".format(x) #names with "name" the column to which table_x time name it is associated to
        onetime_allnontime["{0}".format(x)] = onetime_allnontime["{0}".format(x)].rename(columns={"{0}".format(x): 'times'}) #renames the new created column with "times"

onetime_allnontime_2=pd.DataFrame(onetime_allnontime["feedback_times"]) #create a df with the data of the previous loop's first time event
for x in table_x[1:len(table_x)]: 
    onetime_allnontime_2 = onetime_allnontime_2.append((onetime_allnontime["{0}".format(x)])) #keep appending to the df the rest of the time events
onetime_allnontime_2 = onetime_allnontime_2.reset_index(drop=True) #reset the index
df_events_sorted = onetime_allnontime_2.sort_values(by=['times']) #sort all the rows by the time of the events
# to check what are the nans: 
#test = df_events_sorted[df_events_sorted['times'].isna()]
#test.name.unique()
df_events_sorted = df_events_sorted.dropna(subset=['times']) #drop the nan rows - may be associated to the stimFreeze_times, stimFreezeTrigger_times, errorCueTrigger_times
df_events_sorted = df_events_sorted.reset_index() 
#================================================


#%% 
#add column for the trials 
# df_events_sorted["trial"] = 0 
# n=0
# for i in range(0,len(df_events_sorted)): 
#     if df_events_sorted["name"][i] == "intervals_0": 
#         n = n+1
#         df_events_sorted["trial"][i] = n

#     else: 
#         df_events_sorted["trial"][i] = n 
# df_events_sorted 

## through a function KB 2022Jun02 
test_trial = []
def create_trial(name_column): 
    n=0
    for i in range(0,len(df_events_sorted)): 
        if name_column[i] == "intervals_0": 
            n = n+1
            test_trial.append(n)

        else: 
            test_trial.append(n)
    return test_trial 
def create_trial2(name_column): #KB 20221017
    n=0
    for i in range(0,len(df_events_sorted)): 
        if name_column[i] == "intervals_start": 
            n = n+1 
            test_trial.append(n)

        else: 
            test_trial.append(n)
    return test_trial 

if "intervals_0" in df_events_sorted: 
    test_trial = create_trial(df_events_sorted["name"]) 
else: 
    test_trial = create_trial2(df_events_sorted["name"]) 

df_events_sorted["trial"] = test_trial 
    

#%%
#===========================================================================
#                            8
#===========================================================================

"""
The code below creates a column where 1 is whenever a certain event occurs, in this case it is when a new trial starts
"""

""" 8.1 to check the photometry signal and two events during some trials of the session """
df_events_sorted["new_trial"] = 0 
for i in range(0, len(df_events_sorted["name"])): 
    if df_events_sorted["name"][i] == "intervals_0": 
        df_events_sorted["new_trial"][i] = 1
    elif df_events_sorted["name"][i] == "intervals_start": #KB 20221017
        df_events_sorted["new_trial"][i] = 1


xcoords = df_events_sorted[df_events_sorted['new_trial'] == 1]
xcoords2 = df_events_sorted[df_events_sorted['name'] == "stimOn_times"]
for xc in zip(xcoords["times"]):
    plt.axvline(x=xc, color='blue')
for xc in zip(xcoords2["times"]):
    plt.axvline(x=xc, color='orange')
plt.plot(df['timestamps_bpod'],df['zdFF'], color = 'g') 
plt.rcParams["figure.figsize"] = (20,10)
plt.xlim(0, 150)
plt.show() 
#================================================


#%%
""" JUST 1 EVENT """
""" 
join the df of one event to the photometry data
sort the entire merge of both dataframes
count the epoch based on the event 
switch the positive values of the epoch -1(?) 
remove the rows from the event 
"""
# df_events_stimOn_times = df_events_sorted[df_events_sorted['name'] == "stimOn_times"]

""" df_events_sorted and photometry """
df["times"] = df["timestamps_bpod"]
# df_all = df.append(onetime_allnontime["feedback_times"]) 
df_all = df.append(df_events_sorted) 
df_all = df_all.sort_values(by=['times']) 
df_all = df_all.reset_index(drop=True) 

b = df_all 
#================================================


#%%
""" 8.2 change "name" to the end - so it is possible to repeat the previous behavior rows throughout the photometry data rows """
cols = list(b.columns.values) #Make a list of all of the columns in the df
cols.pop(cols.index('name')) #Remove b from list
b = b[cols+['name']] #Create new dataframe with columns in the order you want
#================================================

#%%
#===========================================================================
#  *                                 INFO
"""
error found on Feb212022 - ffill will not work for the contrastL and contrastR variables
because it will fill everything, while one of them should be NaN while the other one has a value 

Solving this below: 
1. placing the columns b.contrastLeft and b.contrastRight to the end 
2. ffil the rest of the behav columns 
3. loop through the 2 contrast columns by checking if it's a new trial or not and 
repeating the values that appear in the first row of that trial b.trial 
*seems to work* 
"""
# 
#===========================================================================
#! b.trial doesnt work
#! b.new_trial gives 1.0 when we have "intervals_0" 
#! 0 is joined in allContrasts 
#! b.all_contrasts_separated does not work 
#? LEAVING LEFT AS NEGATIVE
""" COPY FROM 8.2 change cL and cR to the end - so it is possible to repeat the previous behavior rows throughout the photometry data rows """
cols = list(b.columns.values) #Make a list of all of the columns in the df
cols.pop(cols.index('contrastLeft')) #Remove b from list
b = b[cols+['contrastLeft']] #Create new dataframe with columns in the order you want
cols = list(b.columns.values) #Make a list of all of the columns in the df
cols.pop(cols.index('contrastRight')) #Remove b from list
b = b[cols+['contrastRight']] 
cols = list(b.columns.values) #Make a list of all of the columns in the df
cols.pop(cols.index('new_trial')) #Remove b from list #KB added 02112022
b = b[cols+['new_trial']] 
# cols = list(b.columns.values) #Make a list of all of the columns in the df #KB commented 02112022
# cols.pop(cols.index('allContrasts')) #Remove b from list
# b = b[cols+['allContrasts']]

b_1 = b.loc[:, 'feedbackType':'trial'] #new_trial doesnt work well when doing this... #KB changed to "contrastLeft" instead of "new_trial" 02112022
b_1 = b_1.fillna(method='ffill')
b.loc[:, 'feedbackType':'trial'] = b_1 
b = b.reset_index(drop=True) #added 15 December 2021 KB

#b["all_contrasts_separated"] = np.NaN #KB changed from new_column to all_contrasts 09102022
#b["all_contrasts_0joined"] = np.NaN #KB added 10102022 for joined -0 and 0 (it was like before)
b['new_trial'] = b['new_trial'].fillna(0) #KB added 02112022 1=change of trial 
b_trial = np.array(b.new_trial) #KB changed from "trial" to "new_trial" 02112022 
b_contrastL = np.array(b.contrastLeft) 
b_contrastR = np.array(b.contrastRight) 

# for i in range(1,len(b)): 
#     if b.trial[i] != b.trial[i-1]: 
#         number_1, number_2 = b["contrastLeft"][i], b["contrastRight"][i]
#     else: 
#         b["contrastLeft"][i],b["contrastRight"][i] = number_1,number_2 
number_1=np.nan #KB added 09082022 
number_2=np.nan
for i in range(1,len(b_trial)): 
    if b_trial[i] == 1.: #KB changed 02112022
        number_1, number_2 = b_contrastL[i], b_contrastR[i]
    else: 
        b_contrastL[i],b_contrastR[i] = number_1,number_2 
b.contrastLeft = b_contrastL
b.contrastRight = b_contrastR 
b.contrastLeft = (-1)*b.contrastLeft #KB changed from Right to Left, to be standardized with the rest of the columns 09102022
b.all_contrasts_0separated= b["allContrasts"].map(str) #KB added 10102022 for separated -0 and 0 in unique()
b.contrastLeft = (-1)*b.contrastLeft #KB changed from Right to Left, to be standardized with the rest of the columns 09102022 

#! not needed 
# testL = b["contrastLeft"].map(str)
# testR = b["contrastRight"].map(str)
# b.all_contrasts_separated.fillna(b["contrastRight"], inplace=True) 
# b.all_contrasts_separated.fillna(b["contrastLeft"], inplace=True) 
# # Import math Library
# import math 
# for i in range(0,len(b.contrastLeft)): 
#     if math.isnan(b.all_contrasts_separated[i]): 
#         b.all_contrasts_separated[i] = b.contrastLeft[i]
#================================================


#%%
""" 18Jan2021 """

b = b[(b["name"]=="feedback_times").values | b["name"].isna().values] 
b = b.reset_index(drop=True) #added 15 December 2021 KB

c = df_events_sorted[df_events_sorted["name"]=="feedback_times"] #feedback_times #goCue_times
b=b.reset_index(drop=True) 
c=c.reset_index(drop=True)
b_test = b 
b_test = b_test.reset_index(drop=True)
b_test["epoch"] = np.nan
b_time = np.array(b_test["times"])
b_something = np.array(b_test["epoch"]) 
e_something = np.array(b_test["epoch"]) 

c = df_events_sorted[df_events_sorted["name"]=="feedback_times"]
c_event = np.array(c["times"]) 

# # for i in range(0,len(b_time)): 
# for j in range(0,len(c_event)): 
#     b_ndx = np.nonzero(b_time==c_event[j])[0]
#     e_ndx = b_ndx + np.arange(-30,61)
#     e_ndx[30:91] = e_ndx[30:91] + 1
#     b_something[e_ndx] = np.arange(-30,61) #times framerate and itll be in seconds
#     e_something[e_ndx] = j


# for i in range(0,len(b_time)): 
if acq_FR == 30: 
    for j in range(0,len(c_event)): 
        b_ndx = np.nonzero(b_time==c_event[j])[0]
        e_ndx = b_ndx + np.arange(-30,61)
        e_ndx[30:91] = e_ndx[30:91] + 1
        b_something[e_ndx] = np.arange(-30,61) #times framerate and itll be in seconds
        e_something[e_ndx] = j
elif acq_FR == 15: 
    for j in range(0,len(c_event)): 
        b_ndx = np.nonzero(b_time==c_event[j])[0]
        e_ndx = b_ndx + np.arange(-15,31)
        e_ndx[15:46] = e_ndx[15:46] + 1
        b_something[e_ndx] = np.arange(-15,31) #times framerate and itll be in seconds
        e_something[e_ndx] = j
elif acq_FR == 60: 
    for j in range(0,len(c_event)): 
        b_ndx = np.nonzero(b_time==c_event[j])[0]
        e_ndx = b_ndx + np.arange(-60,121)
        e_ndx[60:181] = e_ndx[60:181] + 1
        b_something[e_ndx] = np.arange(-60,121) #times framerate and itll be in seconds
        e_something[e_ndx] = j
else: 
    print(">>>>> WEIRD FR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

b_test["epoch"] = b_something 
b_test["epoch_trial"] = e_something 
b_test["epoch_sec"] = b_test["epoch"]/acq_FR


teste = b_test.dropna(subset=['epoch']) 
teste = teste.reset_index(drop=True)
#================================================


#%%
#%% 
tc_dual_to_plot = teste
#%%
#===========================================================================
#                            FUNCTIONS TO PLOT
# KB EDITED 2022Jun02 from plotting_processed_data.py code
#===========================================================================
def plot_feedback_choice(arg1): 
    # Plot the responses of the 5-HT DRN neurons according to x event
    sns.set(rc={'figure.figsize':(21,17),'figure.dpi':150},style="ticks") 
    sns.despine(offset={'left':10,'right':10,'top':20,'bottom':13},trim="True") 


    """COLORS"""
    a = tc_dual_to_plot[arg1].unique() 
    cleanedList = [x for x in a if str(x) != 'nan']
    number_of_colors = len(cleanedList)
    if number_of_colors < 3: 
        colors = ["#ea6276","#32aac8"] 
        palette = sns.color_palette(colors,number_of_colors) 
    else: 
        if (arg1 == ("contrastLeft")) or (arg1 == ("contrastRight")):  
            palette = sns.color_palette("rocket_r",number_of_colors) 
        elif (arg1 == ("probabilityLeft")) or (arg1 == ("repNum")): 
            palette = sns.color_palette("coolwarm",number_of_colors) 
        elif (arg1 == ("allContrasts")): 
            colors = ["#247ba0","#70c1b3","#b2dbbf","#f3ffbd","#ead2ac","#f3ffbd","#b2dbbf","#70c1b3","#247ba0"]
            palette = sns.color_palette(colors,number_of_colors)
        else: 
            colors = ["#BA98CE", "#FF83B0", "#79BEFF"] 
            palette = sns.color_palette(colors,number_of_colors) 

    """"LINES"""
    #sns.lineplot(x="epoch_sec", y="zdFF",
    #            data=tc_dual_to_plot, color= 'lightslategrey', linewidth = 0.25, alpha = 0.2, units="epoch_trial", estimator=None, hue=arg1,palette=palette)#, hue="feedbackType",palette=palette)
    sns.lineplot(x="epoch_sec", y=("zdFF"),
                data=tc_dual_to_plot, color= 'mediumblue', linewidth = 3, alpha = 0.85, hue=arg1,palette=palette)
    plt.axvline(x=0, color = "black", alpha=0.9, linewidth = 3, linestyle="dashed", label = "feedback")
    # plt.axhline(y=0, color = "gray", alpha=0.75, linewidth = 1.5, linestyle="dashed")

    """LABELS"""
    plt.title("TCW - DA - D4 22Oct2022"+" split by "+arg1, y=1.1, fontsize=45) 
    plt.xlabel("time since feedback outcome (s)", labelpad=20, fontsize=60)
    plt.ylabel("zdFF", labelpad=20, fontsize=60) 
    plt.ylim(-1,2) 
    #sns.set(rc={'figure.figsize':(21,17)},style="ticks") #KB commented 09-10-2022 used to use this one
    sns.set(rc={'figure.figsize':(8,6)}, style="ticks") #KB added 09-10-2022 
    sns.despine(offset={'left':10,'right':10,'top':20,'bottom':13},trim="True")
    plt.xticks(fontsize=50)
    plt.yticks(fontsize=50)
    leg = plt.legend(loc="upper right", fontsize=35,frameon=False)
    # leg = plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", fontsize=25,frameon=False)

    for line in leg.get_lines():
        line.set_linewidth(5.5)


    """AXIS and DIMENSION"""
    #plt.xlim(-30,60)
    plt.ylim(-1,2) 
    #sns.set(rc={'figure.figsize':(15,10)}) 
    #plt.savefig(save_the_figs+arg1+'_feedback.png') #,dpi=1200) 
    #plt.savefig('/home/kcenia/Documents/Photometry_results/Plots/Results_RawSignalAnalyses/B_GCAMP_feedback_'+arg1+'_epoch2.png', dpi=300, bbox_inches = "tight") 
    plt.show()
#================================================


#%%
b_1 = tc_dual_to_plot.loc[:, 'feedbackType':'rewardVolume'] 
a=[]
for col in b_1.columns: 
    a.append(str(col))
b_2 = tc_dual_to_plot.loc[:, 'contrastLeft':'contrastRight']
for col in b_2.columns: 
    a.append(str(col))

for i in range(0,len(a)): 
    plot_feedback_choice(a[i]) 
#================================================

#%% 
print(session_path,io_path,session_path_behav) 
#================================================

#%%
#tc_dual_to_plot.to_csv('/home/kcenia/Documents/Photometry_results/SfN_02Nov2022/D5_2022-06-10_goCue.csv') 
#================================================

 # %% 

 #===========================================================================
 #  *                                 INFO
 #    A. raw signal, entire session         DONE
 #    B. raw signal, first 20000 removed    DONE 
 #      B_GCAMP_feedback                    DONE many plots 
 #      B_GCAMP_feedback_epoch              DONE many plots
 # 
 #===========================================================================
#%%
# tc_dual_to_plot = teste 
# tc_dual_to_plot_last = tc_dual_to_plot.tail(10000)
#===========================================================================
#                            FUNCTIONS TO PLOT
# KB EDITED 2022Jun02 from plotting_processed_data.py code
#===========================================================================
def plot_feedback_choice(dataframe,arg1,title): 
    # Plot the responses of the 5-HT DRN neurons according to x event
    sns.set(rc={'figure.figsize':(10,8),'figure.dpi':150},style="ticks") #KB added 09-10-2022 
    sns.despine(offset={'left':10,'right':10,'top':15,'bottom':13},trim="True") #KB added 09-10-2022 

    """COLORS"""
    a = dataframe[arg1].unique() 
    cleanedList = [x for x in a if str(x) != 'nan']
    number_of_colors = len(cleanedList)
    if number_of_colors < 3: 
        colors = ["#ea6276","#32aac8"] 
        palette = sns.color_palette(colors,number_of_colors) 
    else: 
        if (arg1 == ("contrastLeft")) or (arg1 == ("contrastRight")):  
            palette = sns.color_palette("rocket_r",number_of_colors) 
        elif (arg1 == ("probabilityLeft")): 
            #!  CHANGED
            colors = ["#FE4A49", "#FED766", "#009FB7"] 
            palette = sns.color_palette(colors,number_of_colors) 
        elif (arg1 == ("repNum")): 
            palette = sns.color_palette("coolwarm",number_of_colors) 
        elif (arg1 == ("all_contrasts")): 
            colors = ["#247ba0","#70c1b3","#b2dbbf","#f3ffbd","#ead2ac","#f3ffbd","#b2dbbf","#70c1b3","#247ba0"]
            palette = sns.color_palette(colors,number_of_colors) 
        elif (arg1 == ("all_contrasts_0joined")):  
            colors = ["#247ba0","#70c1b3","#b2dbbf","#f3ffbd","#ead2ac","#ead2ac","#f3ffbd","#b2dbbf","#70c1b3","#247ba0"]
            palette = sns.color_palette(colors,number_of_colors)
        else: 
            colors = ["#BA98CE", "#FF83B0", "#79BEFF"] 
            palette = sns.color_palette(colors,number_of_colors) 

    """"LINES"""
    #sns.lineplot(x="epoch_sec", y="zdFF",
    #           data=dataframe, color= 'lightslategrey', linewidth = 0.25, alpha = 0.2, units="epoch_trial", estimator=None, hue=arg1,palette=palette)#, hue="feedbackType",palette=palette)
    sns.lineplot(x="epoch_sec", y=("zdFF"),
                data=dataframe, color= 'mediumblue', linewidth = 2, alpha = 0.85, hue=arg1,palette=palette)
    plt.axvline(x=0, color = "black", alpha=0.9, linewidth = 1.5, linestyle="dashed", label = "feedback")
    # plt.axhline(y=0, color = "gray", alpha=0.75, linewidth = 1.5, linestyle="dashed")

    """LABELS"""
    plt.suptitle("N1 21-09-2022"+" split by "+arg1, y=1.03, fontsize=22) 
    plt.title(str(title),fontsize=18,y=1.025)
    plt.xlabel("time since feedback outcome (s)", labelpad=20, fontsize=20)
    plt.ylabel("zdFF", labelpad=20, fontsize=20) 
    plt.ylim(-1,2) 
    #sns.set(rc={'figure.figsize':(21,17)},style="ticks") #KB commented 09-10-2022 used to use this one
    sns.set(rc={'figure.figsize':(10,8)}, style="ticks") #KB added 09-10-2022 
    sns.despine(offset={'left':10,'right':10,'top':15,'bottom':13},trim="True")
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    leg = plt.legend(loc="upper right", fontsize=15,frameon=False)
    # leg = plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", fontsize=25,frameon=False)

    for line in leg.get_lines():
        line.set_linewidth(3)


    """AXIS and DIMENSION"""
    #plt.xlim(-30,60)
    plt.ylim(-1,2) 
    #sns.set(rc={'figure.figsize':(15,10)}) 
    #plt.savefig(save_the_figs+arg1+'_feedback.png') #,dpi=1200) 
    #plt.savefig('/home/kcenia/Documents/Photometry_results/Plots/Results_RawSignalAnalyses/B_isos_feedback_'+arg1+'_epoch_first10000ofthesession.png', dpi=300, bbox_inches = "tight") 
    plt.show()

#%%

#================================================
#                      PLOT
#================================================
b_1 = tc_dual_to_plot.loc[:, 'feedbackType':'rewardVolume'] #KB changed to rewardVolume 10102022 because stim_Right was not considering -0 and 0, creating a bias in the stim appearance distribution 
a=[]
for col in b_1.columns: 
    a.append(str(col))
b_2 = tc_dual_to_plot.loc[:, 'contrastLeft':'all_contrasts'] #KB to be added 10102022 all_contrasts_0joined and change colors 
for col in b_2.columns: 
    a.append(str(col))

for i in range(0,len(a)): 
    plot_feedback_choice(tc_dual_to_plot,a[i])
#================================================ 


#%%
#================================================b.contrastLeft = (-1)*b.contrastLeft #KB changed from Right to Left, to be standardized with the rest of the columns 09102022
b.all_contrasts_0separated= b["allContrasts"].map(str) #KB added 10102022 for separated -0 and 0 in unique()
b.contrastLeft = (-1)*b.contrastLeft #KB changed from Right to Left, to be standardized with the rest of the columns 09102022 

#                   Extra plots - feedbackType
#================================================
#===========================
#  todo      1. Only correct
#=========================== 
df_correct = tc_dual_to_plot[tc_dual_to_plot["feedbackType"] == 1.] 
# for i in range(0,len(a)): 
#     plot_feedback_choice(df_correct,a[i]) 

#===========================
#  todo      1. Correct and beginning 
#=========================== 
number = int((len(tc_dual_to_plot)/3)) 
df_initial = tc_dual_to_plot[0:number] 
df_initial_correct = df_initial[df_initial["feedbackType"] == 1.] 
# for i in range(0,len(a)): 
#     plot_feedback_choice(df_initial_correct,a[i],"beginning of the session") 

#%% 
#===========================
#  todo      2. Only incorrect
#=========================== 
df_incorrect = tc_dual_to_plot[tc_dual_to_plot["feedbackType"] == -1.] 
# for i in range(0,len(a)): 
#     plot_feedback_choice(df_incorrect,a[i]) 

#===========================
#  todo      2. Incorrect and beginning
#=========================== 
df_initial_incorrect = df_initial[df_initial["feedbackType"] == -1.] 

# for i in range(0,len(a)): 
#     plot_feedback_choice(df_initial_incorrect,a[i],"beginning of the session") 


#%% 
#===========================
#  todo      2. Only final
#=========================== 
number = int((len(tc_dual_to_plot)/3)) 
df_final = tc_dual_to_plot[(len(tc_dual_to_plot)-number):]

# for i in range(0,len(a)): 
#     plot_feedback_choice(df_final,a[i],"last 1/3 of the session") 



#%%
for i in range(0,len(a)): 
    sns.set(rc={'figure.figsize':(10,8),'figure.dpi':150},style="ticks") 
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharey=True)
    fig.suptitle('beginning of the session') 

    # Upper 
    # LeftPlot
    sns.lineplot(ax=axes[0,0], x="epoch_sec", y=("zdFF"),
                    data=df_correct, color= 'mediumblue', linewidth = 2, alpha = 0.85, hue=a[i]) 
    axes[0,0].set_title("correct " + a[i]) 

    #Upper 
    # RightPlot
    sns.lineplot(ax=axes[0,1], x="epoch_sec", y=("zdFF"),
                    data=df_incorrect, color= 'mediumblue', linewidth = 2, alpha = 0.85, hue=a[i]) 
    axes[0,1].set_title("inc " + a[i])
    
    # LeftPlot
    sns.lineplot(ax=axes[1,0], x="epoch_sec", y=("zdFF"),
                    data=df_initial_correct, color= 'mediumblue', linewidth = 2, alpha = 0.85, hue=a[i]) 
    axes[1,0].set_title("c and b " + a[i])
    
    # RightPlot
    sns.lineplot(ax=axes[1,1], x="epoch_sec", y=("zdFF"),
                    data=df_initial_incorrect, color= 'mediumblue', linewidth = 2, alpha = 0.85, hue=a[i])
    axes[1,1].set_title("inc and b " + a[i])
    
    leg = plt.legend(loc="upper right", fontsize=12,frameon=False) 

    plt.subplots_adjust(hspace = 0.5)
    plt.plot()
    plt.show() 


#%%
#%% 
#================================================
#                save the data
#================================================
#tc_dual_to_plot.to_csv('/home/kcenia/Documents/Photometry_results/Plots/Results_RawSignalAnalyses/ZFM-04533_20220921_feedback.csv') 
# tc_dual_to_plot.to_csv('/home/kcenia/Documents/Photometry_results/csvs/S7_20221124_feedback_notsure.csv') 
#%%
#tc_dual_to_plot_2 = pd.read_csv('/home/kcenia/Documents/Photometry_results/Plots/Results_RawSignalAnalyses/ZFM-04022_20220922_feedback_B_RawSignal20000on_GCAMP.csv')



























# %%
#===========================================================================
#                            Sanity check
#=========================================================================== 

#================================================
#                PLot the last values
#================================================ 
tc_dual_to_plot_last = tc_dual_to_plot.tail(10000)
#================================================
#%%
#================================================
#           count the number of unique
#================================================
column_list = ['feedbackType', 'probabilityLeft', 'choice', 'rewardVolume',
       'contrastLeft', 'contrastRight', 'all_contrasts']
def checking_the_data(test): 
    values, counts = np.unique(tc_dual_to_plot_last[test], return_counts=True)
    #performance = 
    print(i, "| values = ",values, "| counts = ", counts) 
    return i,values,counts

for i in column_list: 
    checking_the_data(i) 
#================================================
#================================================
#               Performance % calc
#================================================
values, counts = np.unique(tc_dual_to_plot_last["feedbackType"], return_counts=True) 
tc_dual_to_plot_correct = tc_dual_to_plot_last[tc_dual_to_plot_last["feedbackType"]==1.0]
values2, counts2 = np.unique(tc_dual_to_plot_correct["feedbackType"], return_counts=True)
if values2[0] == 1.: 
    performance = (counts2[0]*100)/(sum(counts))
    print("Performance = ", performance, "%") 
else: 
    print("READJUST THIS!") 
#================================================


#================================================
#             Plot the distribution - probability Left
#================================================ 
#! PROB WRONG WHEN 3(?) 
values, counts = np.unique(tc_dual_to_plot_last["probabilityLeft"], return_counts=True) 
tc_dual_to_plot_probabilityLeft = tc_dual_to_plot_last[tc_dual_to_plot_last["probabilityLeft"]==0.2]
values2, counts2 = np.unique(tc_dual_to_plot_probabilityLeft["probabilityLeft"], return_counts=True)
if values2[0] == 0.2: 
    count_probabilityRight = (counts2[0]*100)/(sum(counts)) 
    count_probabilityLeft = (counts[1]*100)/(sum(counts)) 
    print("Probability Right = ", count_probabilityRight, "%\n",
    "Probability Left = ", count_probabilityLeft, "%") 
else: 
    print("READJUST THIS! Probably also has 0.5") 

#%%
BAR_WIDTH = 0.9     # 0. < BAR_WIDTH <= 1.
def main():
    # the data you want to plot 
    sns.set(rc={'figure.figsize':(3,3)}, style="ticks") 
    categories  = values 
    values_plot = [count_probabilityRight,count_probabilityLeft]
    # x-values for the center of each bar
    xs = np.arange(1, len(categories) + 1)
    # plot each bar centered
    plt.bar(xs - BAR_WIDTH/20, values_plot, width=BAR_WIDTH, color=["#ea6276","#32aac8"])
    # make sure the chart is centered 
    plt.suptitle("Probability Left",y=1,fontsize=10) 
    plt.title("block left",fontsize=8)
    plt.xlim(0, len(categories) + 1)
    plt.ylim(0,100)
    sns.despine(offset={'left':5,'right':5,'top':5,'bottom':5},trim="True") 
    # add bar labels
    plt.xticks(xs, categories,fontsize=10) 
    plt.yticks(fontsize=10) 
    # show the results
    plt.show() 

main() 
#================================================
#%% 
#================================================
#                Plot the distribution - choice
# ! WRONG IF NO CHOICE IS NOT PRESENT(?) 
#================================================ 
#* -1 is leftturn 
values, counts = np.unique(tc_dual_to_plot_last["choice"], return_counts=True) 
tc_dual_to_plot_choiceLeft = tc_dual_to_plot_last[tc_dual_to_plot_last["choice"]==-1.]
values2, counts2 = np.unique(tc_dual_to_plot_choiceLeft["choice"], return_counts=True)
tc_dual_to_plot_choiceRight = tc_dual_to_plot_last[tc_dual_to_plot_last["choice"]==1.]
values3, counts3 = np.unique(tc_dual_to_plot_choiceRight["choice"], return_counts=True)
if values2[0] == -1.: 
    count_choiceLeft = (counts2[0]*100)/(sum(counts)) 
    count_nochoice = (counts[1]*100)/sum(counts)
    count_choiceRight = (counts3[0]*100)/(sum(counts)) 
    print("Choice Left = ", count_choiceLeft, "%\n", 
    "No choice = ", count_nochoice, "%\n",
    "Choice Right = ", count_choiceRight, "%") 
else: 
    print("READJUST THIS! Probably also has 0.5") 
#================================================
#%% 
import matplotlib.patches as mpatches
BAR_WIDTH = 0.9     # 0. < BAR_WIDTH <= 1.
def main():
    # the data you want to plot 
    sns.set(rc={'figure.figsize':(3,3)}, style="ticks") 
    categories  = values 
    values_plot = [count_choiceLeft,count_nochoice,count_choiceRight]
    # x-values for the center of each bar
    xs = np.arange(1, len(categories) + 1)
    # plot each bar centered 
    colors = ["#79BEFF", "#FF83B0", "#BA98CE"] 
    plt.bar(xs - BAR_WIDTH/20, values_plot, width=BAR_WIDTH, color=colors)
    # make sure the chart is centered 
    plt.suptitle("Choice",y=1,fontsize=10) 
    plt.title("check colors and labels",fontsize=8)
    plt.xlim(0, len(categories) + 1)
    plt.ylim(0,100)
    sns.despine(offset={'left':5,'right':5,'top':5,'bottom':5},trim="True") 
    # add bar labels
    plt.xticks(xs, categories,fontsize=10) 
    plt.yticks(fontsize=10) 
    #create legend manually 
    legend=('choiceL','no choice','choiceR') 
    patch0 = mpatches.Patch(color=colors[0], label=legend[0])
    patch1 = mpatches.Patch(color=colors[1], label=legend[1]) 
    patch2 = mpatches.Patch(color=colors[2], label=legend[2])
    plt.legend(handles=[patch0,patch1,patch2], fontsize=8, frameon=False)
    # show the results
    plt.show() 

main() 
#================================================
#%% 
# pie plot
y = counts
#mylabels = values
mylabels = ('choiceL','no choice','choiceR') 
colors = ["#79BEFF", "#FF83B0", "#BA98CE"] 
plt.pie(y, autopct='%1.1f%%',labels = mylabels, colors=colors,textprops={'fontsize': 8})
plt.suptitle("Choice",y=1,fontsize=10) 
plt.title("check colors and labels", fontsize=8)
plt.show() 
#================================================ 
# # %%
# #================================================
# #                Plot the distribution - stim_Right (stim appearance)
# #================================================ 
# values, counts = np.unique(tc_dual_to_plot_last["stim_Right"], return_counts=True) 
# tc_dual_to_plot_stim_Left = tc_dual_to_plot_last[tc_dual_to_plot_last["stim_Right"]==-1.]
# values2, counts2 = np.unique(tc_dual_to_plot_stim_Left["stim_Right"], return_counts=True)
# if values2[0] == -1.: 
#     count_stimL = (counts2[0]*100)/(sum(counts)) 
#     count_stimR = (counts[1]*100)/sum(counts)
#     print("Stim appears in the left side = ", count_stimL, "%\n", 
#     "Stim appears in the right side = ", count_stimR, "%") 
# else: 
#     print("READJUST THIS!") 
# #================================================
# #%% 
# # bar plot 
# import matplotlib.patches as mpatches
# BAR_WIDTH = 0.9     # 0. < BAR_WIDTH <= 1.
# def main():
#     # the data you want to plot 
#     sns.set(rc={'figure.figsize':(3,3)}, style="ticks") 
#     categories  = values 
#     values_plot = [count_stimL,count_stimR]
#     # x-values for the center of each bar
#     xs = np.arange(1, len(categories) + 1)
#     # plot each bar centered 
#     colors = ["#ea6276","#32aac8"]
#     plt.bar(xs - BAR_WIDTH/20, values_plot, width=BAR_WIDTH, color=colors)
#     # make sure the chart is centered 
#     plt.suptitle("stim appearance",y=1,fontsize=10) 
#     plt.title("appearing on the right or left side",fontsize=8)
#     plt.xlim(0, len(categories) + 1)
#     plt.ylim(0,100)
#     sns.despine(offset={'left':5,'right':5,'top':5,'bottom':5},trim="True") 
#     # add bar labels
#     plt.xticks(xs, categories,fontsize=10) 
#     plt.yticks(fontsize=10) 
#     #create legend manually 
#     legend=('left','right') 
#     patch0 = mpatches.Patch(color=colors[0], label=legend[0])
#     patch1 = mpatches.Patch(color=colors[1], label=legend[1]) 
#     plt.legend(handles=[patch0,patch1], fontsize=8, frameon=False)
#     # show the results
#     plt.show() 

# main() 
# #================================================
# #%% 
# # pie plot
# y = counts
# #mylabels = values
# mylabels = ('left','right') 
# colors = ["#ea6276","#32aac8"]
# plt.pie(y, autopct='%1.1f%%',labels = mylabels, colors=colors, textprops={'fontsize': 8}) 
# plt.suptitle("stim appearance",y=1,fontsize=10) 
# plt.title("test",fontsize=8)
# plt.show() 
#================================================
#%% 
#================================================
#                Plot the distribution - all_contrasts - by contrasts and by sides 
#================================================ 
values, counts = np.unique(tc_dual_to_plot_last["all_contrasts_0joined"], return_counts=True) 
#create dict for all the contrasts 
dicts = {}
keys = values
for i in range(len(keys)):
    dicts[keys[i]] = counts[i]
print(dicts) 
#calc percentage 
def percentage(part, whole):
  return 100 * float(part)/float(whole) 
whole = sum(dicts.values())
values_percentage=[] 
for i in dicts.values(): 
    values_percentage.append(percentage(i,whole)) 
dicts
#================================================
#%% 
# bar plot 
BAR_WIDTH = 0.9     # 0. < BAR_WIDTH <= 1.
def main():
    # the data you want to plot 
    sns.set(rc={'figure.figsize':(3,3)}, style="ticks") 
    categories  = list(dicts.keys())
    # values_plot = list(dicts.values()) 
    values_plot = values_percentage
    # x-values for the center of each bar
    xs = np.arange(1, len(categories) + 1)
    # plot each bar centered 
    colors = ["#ead2ac","#f3ffbd","#b2dbbf","#70c1b3","#247ba0", "#ead2ac", "#f3ffbd","#b2dbbf","#70c1b3","#247ba0"] 
    plt.bar(xs - BAR_WIDTH/20, values_plot, color = colors, width=BAR_WIDTH)
    # make sure the chart is centered 
    plt.suptitle("contrast distribution",y=1,fontsize=10) 
    plt.title("test",fontsize=8)
    plt.xlim(0, len(categories) + 1)
    plt.ylim(0,20)
    sns.despine(offset={'left':5,'right':5,'top':5,'bottom':5},trim="True") 
    # add bar labels
    plt.xticks(xs, categories,fontsize=10, rotation=45) 
    plt.yticks(fontsize=10) 
    #create legend manually 
    legend=categories
    # patch0 = mpatches.Patch(color=colors[0], label=legend[0])
    # patch1 = mpatches.Patch(color=colors[1], label=legend[1]) 
    # patch2 = mpatches.Patch(color=colors[2], label=legend[2])
    # patch3 = mpatches.Patch(color=colors[3], label=legend[3]) 
    # patch4 = mpatches.Patch(color=colors[4], label=legend[4])
    patch5 = mpatches.Patch(color=colors[5], label=legend[5]) 
    patch6 = mpatches.Patch(color=colors[6], label=legend[6]) 
    patch7 = mpatches.Patch(color=colors[7], label=legend[7])
    patch8 = mpatches.Patch(color=colors[8], label=legend[8]) 
    patch9 = mpatches.Patch(color=colors[9], label=legend[9]) 
    # plt.legend(handles=[patch0,patch1,patch2,patch3,patch4,patch5,patch6,patch7,patch8,patch9],fontsize=6,ncol=2,frameon=False) 
    plt.legend(handles=[patch5,patch6,patch7,patch8,patch9],fontsize=6,frameon=False)
    # show the results
    plt.show() 
main() 
#================================================
#%% 
# # pie plot
# y = list(dicts.values())
# #mylabels = values
# mylabels = list(dicts.keys())
# colors = ["#ead2ac","#f3ffbd","#b2dbbf","#70c1b3","#247ba0", "#ead2ac", "#f3ffbd","#b2dbbf","#70c1b3","#247ba0"] 
# #colors = ["#E1E67E","#FCF300","#FEDD00","#FFC600","#FFA417", "#A2E6FA", "#60B6FB","#1E96FC","#1360E2","#072AC8"] 
# plt.pie(y, autopct='%1.1f%%', labels = mylabels, colors = colors, textprops={'fontsize': 8})
# plt.suptitle("contrast distribution",y=1,fontsize=10) 
# plt.title("test",fontsize=8)
# plt.show() 
#================================================ 
#%% 
# bar plot 
categoriesL = dict(list(dicts.items())[:len(dicts)//2]) #pick negative values
categoriesR = dict(list(dicts.items())[len(dicts)//2:]) #pick positive values 
countsL = list(categoriesL.values()) 
countsR = list(categoriesR.values()) 
whole = (sum(countsL))+(sum(countsR)) 
countsL_percentage = percentage(sum(countsL),whole)
countsR_percentage = percentage(sum(countsR),whole)
print("stim appears in the right = ", countsR_percentage, "%\n",
    "stim appears in the left = ", countsL_percentage, "%") 

BAR_WIDTH = 0.9     # 0. < BAR_WIDTH <= 1.
def main():
    # the data you want to plot 
    sns.set(rc={'figure.figsize':(3,3)}, style="ticks") 
    categories  = ["right","left"] 
    values_plot = [countsR_percentage,countsL_percentage]
    # x-values for the center of each bar
    xs = np.arange(1, len(categories) + 1)
    # plot each bar centered
    plt.bar(xs - BAR_WIDTH/20, values_plot, width=BAR_WIDTH, color=["#A6C5F7","#A3E8B0"])
    # make sure the chart is centered 
    plt.suptitle("stim appearance side",y=1,fontsize=10) 
    plt.title("test",fontsize=8)
    plt.xlim(0, len(categories) + 1)
    plt.ylim(0,100)
    sns.despine(offset={'left':5,'right':5,'top':5,'bottom':5},trim="True") 
    # add bar labels
    plt.xticks(xs, categories,fontsize=10) 
    plt.yticks(fontsize=10) 
    # show the results
    plt.show() 
main() 
#================================================
#%%
# pie plot
y = list(dicts.values())
#mylabels = values
mylabels = list(dicts.keys())
# colors = ["#ead2ac","#f3ffbd","#b2dbbf","#70c1b3","#247ba0", "#ead2ac", "#f3ffbd","#b2dbbf","#70c1b3","#247ba0"] 
# colors = ["#E1E67E","#FCF300","#FEDD00","#FFC600","#FFA417", "#A2E6FA", "#60B6FB","#1E96FC","#1360E2","#072AC8"] 
colors = ["#DFFFCC","#C1F4BE","#A3E8B0","#85DDA1","#67D193", "#D7E3FC", "#C4DBFA","#A6C5F7","#88AFF4","#6A99F1"] 
plt.pie(y, labels = mylabels, colors = colors, textprops={'fontsize': 8})
plt.suptitle("contrast distribution",y=1,fontsize=10) 
plt.title("test",fontsize=8)
plt.show() 
#================================================ 
# %%

# %%
