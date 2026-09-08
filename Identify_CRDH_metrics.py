import numpy as np
import tifffile as tf
from joblib import Parallel, delayed
import os
import glob

def create_and_make_compound_paths(base_dir, model):
    """
    Batch create output paths for compound event analysis for a given model, and actually create the folders.

    Parameters:
        base_dir : str
            Base path, e.g., '.../Observed/'
        model : str
            Model name, e.g., 'MPI-ESM1-2-HR'

    Returns:
        dict : Dictionary containing all paths, keys named according to event type and variable
    """
    # Three event types
    event_types = {
        'compound': 'compound',
        'HF': 'compound_HF',
        'DF': 'compound_DF'
    }

    # Subdirectories under each event type
    subfolders = {
        'compound_path': 'Compound_event',
        'disturbance_path': 'Disturbance_period',
        'speed_path': 'Compound_speed',
        'ratio_path': 'Compound_ratio',
        'com_dry_path': 'Compound_dry',
        'com_heat_path': 'Compound_heat'
    }

    all_paths = {}

    for event_key, event_folder in event_types.items():
        for path_key, sub in subfolders.items():
            # Path key: compound_path, compound_path_HF, compound_path_DF, etc.
            full_key = path_key if event_key == 'compound' else f"{path_key}_{event_key}"
            # Construct full path
            full_path = os.path.join(base_dir, event_folder, sub, model) + '/'
            # Create folder
            os.makedirs(full_path, exist_ok=True)
            # Save to dictionary
            all_paths[full_key] = full_path

    return all_paths

# Read land mask
reference = tf.imread('.../reference_no_antarctica.tif')
reference = reference.astype(float)
reference[62:, :] = np.nan

# Read hottest day
hottest_d = np.load('.../hottest_day.npy')
temp_value = -9999
hottest_d_temp = np.nan_to_num(hottest_d, nan=temp_value)
# Convert to integer
hottest_day = hottest_d_temp.astype(int)

# import matplotlib.pyplot as plt
# plt.imshow(reference, cmap='jet')
# plt.colorbar(label='PET')  # Add colorbar

Model_name = ['ACCESS-CM2', 'ACCESS-ESM1-5', 'AWI-ESM-1-1-LR', 'CMCC-ESM2', 'EC-Earth3', 
              'EC-Earth3-CC', 'EC-Earth3-Veg', 'EC-Earth3-Veg-LR', 'INM-CM4-8',  'INM-CM5-0', 
              'IPSL-CM6A-LR', 'MIROC6', 'MPI-ESM-1-2-HAM','MPI-ESM1-2-HR', 'MPI-ESM1-2-LR',
              'MRI-ESM2-0']

for model in Model_name: 

    # model = 'ACCESS-CM2'  
    
    base_path = '.../Compound/180days/ALL/3days_SPEI128/'
    paths = create_and_make_compound_paths(base_path, model)
        
    # Input paths for drought and heatwave events    
    dry_path_in = '.../Droughts/180days/3days_SPEI128/ALL/' + model  + '/3days_events/'
    dry_path_all = '.../Droughts/180days/3days_SPEI128/ALL/' + model  + '/3days_events/' + 'NH_Dry_End_*.npy'
    dry_path = glob.glob(dry_path_all)
    dry_path.sort()
    start_year = int(dry_path[0][-8:-4])
    end_year = int(dry_path[-1][-8:-4])
    
    heat_path_in = '.../Heatwaves/ALL/3days_events/' + model  + '/'
    
    for m in range(start_year, end_year+1):

        # m = start_year + 10
        
        dryspell_start = np.load(f"{dry_path_in}NH_Dry_Start_{m}.npy") 
        heatwave_start = np.load(f"{heat_path_in}NH_Heat_Start_{m}.npy")
        dryspell_end = np.load(f"{dry_path_in}NH_Dry_End_{m}.npy")
        heatwave_end = np.load(f"{heat_path_in}NH_Heat_End_{m}.npy")
        dryspell_length = np.load(f"{dry_path_in}NH_Dry_Length_{m}.npy")
        heatwave_length = np.load(f"{heat_path_in}NH_Heat_Length_{m}.npy")
        
        # Overall compound event features
        compound_length = np.full((72, 144, 20), np.nan)
        compound_start = np.full((72, 144, 20), np.nan)
        compound_end = np.full((72, 144, 20), np.nan)
        disturbance_period_start = np.full((72, 144, 20), np.nan)
        disturbance_period_end = np.full((72, 144, 20), np.nan)
        disturbance_period_length = np.full((72, 144, 20), np.nan)
        compound_dryspell_start = np.full((72, 144, 20), np.nan)
        compound_dryspell_end = np.full((72, 144, 20), np.nan)
        compound_dryspell_length = np.full((72, 144, 20), np.nan)
        compound_heatwave_start = np.full((72, 144, 20), np.nan)
        compound_heatwave_end = np.full((72, 144, 20), np.nan)
        compound_heatwave_length = np.full((72, 144, 20), np.nan)        
        ratio = np.full((72, 144, 20), np.nan)
        speed = np.full((72, 144, 20), np.nan)
        compound_percent = np.full((72, 144), np.nan)
        compound_ratio = np.full((72, 144), np.nan)
        
        # Heatwave-first compound event features
        compound_length_HF = np.full((72, 144, 20), np.nan)
        compound_start_HF = np.full((72, 144, 20), np.nan)
        compound_end_HF = np.full((72, 144, 20), np.nan)
        disturbance_period_start_HF = np.full((72, 144, 20), np.nan)
        disturbance_period_end_HF = np.full((72, 144, 20), np.nan)
        disturbance_period_length_HF = np.full((72, 144, 20), np.nan)
        compound_dryspell_start_HF = np.full((72, 144, 20), np.nan)
        compound_dryspell_end_HF = np.full((72, 144, 20), np.nan)
        compound_dryspell_length_HF = np.full((72, 144, 20), np.nan)
        compound_heatwave_start_HF = np.full((72, 144, 20), np.nan)
        compound_heatwave_end_HF = np.full((72, 144, 20), np.nan)
        compound_heatwave_length_HF = np.full((72, 144, 20), np.nan)        
        ratio_HF = np.full((72, 144, 20), np.nan)
        speed_HF = np.full((72, 144, 20), np.nan)
        compound_percent_HF = np.full((72, 144), np.nan)
        compound_ratio_HF = np.full((72, 144), np.nan)
        
        # Drought-first compound event features
        compound_length_DF = np.full((72, 144, 20), np.nan)
        compound_start_DF = np.full((72, 144, 20), np.nan)
        compound_end_DF = np.full((72, 144, 20), np.nan)
        disturbance_period_start_DF = np.full((72, 144, 20), np.nan)
        disturbance_period_end_DF = np.full((72, 144, 20), np.nan)
        disturbance_period_length_DF = np.full((72, 144, 20), np.nan)
        compound_dryspell_start_DF = np.full((72, 144, 20), np.nan)
        compound_dryspell_end_DF = np.full((72, 144, 20), np.nan)
        compound_dryspell_length_DF = np.full((72, 144, 20), np.nan)
        compound_heatwave_start_DF = np.full((72, 144, 20), np.nan)
        compound_heatwave_end_DF = np.full((72, 144, 20), np.nan)
        compound_heatwave_length_DF = np.full((72, 144, 20), np.nan)        
        ratio_DF = np.full((72, 144, 20), np.nan)
        speed_DF = np.full((72, 144, 20), np.nan)     
        compound_percent_DF = np.full((72, 144), np.nan) 
        compound_ratio_DF = np.full((72, 144), np.nan)
        
        for i in range(72):
            for j in range(144):

                # i,j = 25,30 i,j = 40,50

                if np.isnan(reference[i, j]):
                    continue

                tem1 = dryspell_start[i, j, :].flatten()
                tem2 = dryspell_end[i, j, :].flatten()
                tem3 = heatwave_start[i, j, :].flatten()
                tem4 = heatwave_end[i, j, :].flatten()
                tem5 = dryspell_length[i, j, :].flatten()
                tem6 = heatwave_length[i, j, :].flatten()

                tem1 = tem1[~np.isnan(tem1)]
                tem2 = tem2[~np.isnan(tem2)]
                tem3 = tem3[~np.isnan(tem3)]
                tem4 = tem4[~np.isnan(tem4)]
                tem5 = tem5[~np.isnan(tem5)]
                tem6 = tem6[~np.isnan(tem6)]
                
                dry_count = len(tem1)
                heat_count = len(tem3)   
                compound_length_total = np.sum(tem5) +  np.sum(tem6) 

                if tem1.size == 0 or tem3.size == 0:
                    continue
        
            # --------------------- Filter and merge compound events ----------------------------        
                # Drought start, end, and duration arrays
                P_start_end3 = np.column_stack((tem1, tem2, tem2 - tem1 + 1))

                # Heatwave start, end, and duration arrays
                T_start_end3 = np.column_stack((tem3, tem4, tem4 - tem3 + 1))

                # Expand the days of heatwave and drought events to prepare for matching
                P_start_end4 = np.full((len(P_start_end3), int(max(P_start_end3[:, 2].max(), T_start_end3[:, 2].max()))), np.nan)
                for m1 in range(len(P_start_end3)):
                    P_start_end4[m1, :int(P_start_end3[m1, 2])] = np.arange(P_start_end3[m1, 0], P_start_end3[m1, 1] + 1)

                T_start_end4 = np.full((len(T_start_end3), int(max(P_start_end3[:, 2].max(), T_start_end3[:, 2].max()))), np.nan)
                for m1 in range(len(T_start_end3)):
                    T_start_end4[m1, :int(T_start_end3[m1, 2])] = np.arange(T_start_end3[m1, 0], T_start_end3[m1, 1] + 1)

                compound_dryspell_start1, compound_dryspell_end1, compound_dryspell_length1 = [], [], []
                compound_heatwave_start1, compound_heatwave_end1, compound_heatwave_length1 = [], [], []
                
                for o in range(len(T_start_end3)):

                    # o = 0

                    T_event = T_start_end4[o, :]
                    T_event = T_event[~np.isnan(T_event)]

                    for p in range(len(P_start_end3)):

                        # p = 0

                        P_event = P_start_end4[p, :]
                        P_event = P_event[~np.isnan(P_event)]

                        compound, T_C, P_C = np.intersect1d(T_event, P_event, return_indices=True)

                        if len(compound) == 0:
                            continue

                        compound_dryspell_start1.append(P_event[0])
                        compound_dryspell_end1.append(P_event[-1])
                        compound_dryspell_length1.append(P_event[-1] - P_event[0] + 1)

                        compound_heatwave_start1.append(T_event[0])
                        compound_heatwave_end1.append(T_event[-1])
                        compound_heatwave_length1.append(T_event[-1] - T_event[0] + 1)

                CDS = np.array(compound_dryspell_start1)
                CHS = np.array(compound_heatwave_start1)
                CDE = np.array(compound_dryspell_end1)
                CHE = np.array(compound_heatwave_end1)
                
                if CDS.size == 0 or CDE.size == 0:
                    ratio[i, j] = 0
                    compound_percent[i, j] = 0
                    compound_ratio[i,j] = 0
                    
                    ratio_HF[i, j] = 0
                    compound_percent_HF[i, j] = 0
                    compound_ratio_HF[i,j] = 0
                    
                    ratio_DF[i, j] = 0
                    compound_percent_DF[i, j] = 0
                    compound_ratio_DF[i,j] = 0
                    continue
        
                # Classify identified compound events, handle cases where one heatwave event contains multiple drought events, or vice versa
                # Get unique values of CHS and their frequencies
                x1, p = np.unique(CHS, return_counts=True) 

                # Find values with frequency greater than 1
                s1 = x1[p > 1]

                if len(s1) > 0:
                    
                    for dd in range(len(s1)):

                        # dd=0
                        # Find indices in CDS where value equals current one
                        s2 = np.where(CHS == s1[dd])[0]

                        for ee in range(len(s2)-1):

                            CHE[s2[ee]] = CDS[s2[ee]+1]-1
                            CHS[s2[ee]+1] = CDE[s2[ee]]+1
                
                # Get unique values of CDS and their frequencies
                x1, p = np.unique(CDS, return_counts=True) 

                # Find values with frequency greater than 1
                s1 = x1[p > 1]

                if len(s1) > 0:
                    
                    for dd in range(len(s1)):

                        # dd=0
                        # Find indices in CDS where value equals current one
                        s2 = np.where(CDS == s1[dd])[0]

                        for ee in range(len(s2)-1):

                            CDE[s2[ee]] = CHS[s2[ee]+1]-1
                            CDS[s2[ee]+1] = CHE[s2[ee]]+1        
                
                compound_length1, compound_start1, compound_end1 = [], [], []
                disturbance_period, disturbance_period_start1, disturbance_period_end1, disturbance_period_length1 = [], [], [], []
                ratio1, sequence_start1, sequence_end1 = [], [], []
            
                # ---------- Final calculation of compound event features ----------------------
                
                # Drought start, end, and duration arrays
                P_start_end1 = np.column_stack((CDS, CDE, CDE - CDS + 1))

                # Heatwave start, end, and duration arrays
                T_start_end1 = np.column_stack((CHS, CHE, CHE - CHS + 1))

                # Expand the days of heatwave and drought events to prepare for matching
                P_start_end2 = np.full((len(P_start_end1), int(max(P_start_end1[:, 2].max(), T_start_end1[:, 2].max()))), np.nan)
                for m1 in range(len(P_start_end1)):
                    P_start_end2[m1, :int(P_start_end1[m1, 2])] = np.arange(P_start_end1[m1, 0], P_start_end1[m1, 1] + 1)

                T_start_end2 = np.full((len(T_start_end1), int(max(P_start_end1[:, 2].max(), T_start_end1[:, 2].max()))), np.nan)
                for m1 in range(len(T_start_end1)):
                    T_start_end2[m1, :int(T_start_end1[m1, 2])] = np.arange(T_start_end1[m1, 0], T_start_end1[m1, 1] + 1)

                # Overall compound event features
                compound_dryspell_start2, compound_dryspell_end2, compound_dryspell_length2 = [], [], []
                compound_heatwave_start2, compound_heatwave_end2, compound_heatwave_length2 = [], [], []
                compound_length2, compound_start2, compound_end2 = [], [], []
                disturbance_period2, disturbance_period_start2, disturbance_period_end2, disturbance_period_length2 = [], [], [], []
                ratio2, speed2 = [], []
            
                # Drought-first features
                compound_dryspell_start_DF2, compound_dryspell_end_DF2, compound_dryspell_length_DF2 = [], [], []
                compound_heatwave_start_DF2, compound_heatwave_end_DF2, compound_heatwave_length_DF2 = [], [], []
                compound_length_DF2, compound_start_DF2, compound_end_DF2 = [], [], []
                disturbance_period_DF2, disturbance_period_start_DF2, disturbance_period_end_DF2, disturbance_period_length_DF2 = [], [], [], []
                ratio_DF2, speed_DF2 = [], []
                
                # Heatwave-first features
                compound_dryspell_start_HF2, compound_dryspell_end_HF2, compound_dryspell_length_HF2 = [], [], []
                compound_heatwave_start_HF2, compound_heatwave_end_HF2, compound_heatwave_length_HF2 = [], [], []
                compound_length_HF2, compound_start_HF2, compound_end_HF2 = [], [], []
                disturbance_period_HF2, disturbance_period_start_HF2, disturbance_period_end_HF2, disturbance_period_length_HF2 = [], [], [], []
                ratio_HF2, speed_HF2 = [], []
                
                for o in range(len(T_start_end2)):

                    # o = 0

                    T_event = T_start_end2[o, :]
                    T_event = T_event[~np.isnan(T_event)]

                    for p in range(len(P_start_end2)):

                        # p = 0

                        P_event = P_start_end2[p, :]
                        P_event = P_event[~np.isnan(P_event)]

                        compound, T_C, P_C = np.intersect1d(T_event, P_event, return_indices=True)

                        if len(compound) == 0:
                            continue

                        compound_length2.append(len(compound))
                        compound_start2.append(compound[0])
                        compound_end2.append(compound[-1])

                        disturbance_period2 = np.union1d(T_event, P_event)
                        disturbance_period_start2.append(disturbance_period2[0])
                        disturbance_period_end2.append(disturbance_period2[-1])
                        disturbance_period_length2.append(len(disturbance_period2))
                        
                        compound_dryspell_start2.append(P_event[0])
                        compound_dryspell_end2.append(P_event[-1])
                        compound_dryspell_length2.append(P_event[-1] - P_event[0] + 1)

                        compound_heatwave_start2.append(T_event[0])
                        compound_heatwave_end2.append(T_event[-1])
                        compound_heatwave_length2.append(T_event[-1] - T_event[0] + 1)
                        
                        ratio2.append(len(compound)/len(disturbance_period2)*100)
                        if np.abs(P_event[0]-T_event[0])==0:
                            speed2.append(2*100)
                        else:   
                            speed2.append(len(compound)/np.abs(P_event[0]-T_event[0])*100)
                                        
                        if T_event[0] < P_event[0]:  # Heatwave-first features
                            
                            compound_length_HF2.append(len(compound))
                            compound_start_HF2.append(compound[0])
                            compound_end_HF2.append(compound[-1])

                            disturbance_period_HF2 = np.union1d(T_event, P_event)
                            disturbance_period_start_HF2.append(disturbance_period_HF2[0])
                            disturbance_period_end_HF2.append(disturbance_period_HF2[-1])
                            disturbance_period_length_HF2.append(len(disturbance_period_HF2)) 
                            
                            compound_dryspell_start_HF2.append(P_event[0])
                            compound_dryspell_end_HF2.append(P_event[-1])
                            compound_dryspell_length_HF2.append(P_event[-1] - P_event[0] + 1)

                            compound_heatwave_start_HF2.append(T_event[0])
                            compound_heatwave_end_HF2.append(T_event[-1])
                            compound_heatwave_length_HF2.append(T_event[-1] - T_event[0] + 1)
                            
                            ratio_HF2.append(len(compound)/len(disturbance_period_HF2)*100)
                            if np.abs(P_event[0]-T_event[0])==0:
                                speed_HF2.append(2*100)
                            else:   
                                speed_HF2.append(len(compound)/np.abs(P_event[0]-T_event[0])*100)
                            
                                                    
                        if T_event[0] > P_event[0]:  # Drought-first features

                            compound_length_DF2.append(len(compound))
                            compound_start_DF2.append(compound[0])
                            compound_end_DF2.append(compound[-1])

                            disturbance_period_DF2 = np.union1d(T_event, P_event)
                            disturbance_period_start_DF2.append(disturbance_period_DF2[0])
                            disturbance_period_end_DF2.append(disturbance_period_DF2[-1])
                            disturbance_period_length_DF2.append(len(disturbance_period_DF2)) 
                            
                            compound_dryspell_start_DF2.append(P_event[0])
                            compound_dryspell_end_DF2.append(P_event[-1])
                            compound_dryspell_length_DF2.append(P_event[-1] - P_event[0] + 1)

                            compound_heatwave_start_DF2.append(T_event[0])
                            compound_heatwave_end_DF2.append(T_event[-1])
                            compound_heatwave_length_DF2.append(T_event[-1] - T_event[0] + 1)
                            
                            ratio_DF2.append(len(compound)/len(disturbance_period_DF2)*100)
                            if np.abs(P_event[0]-T_event[0])==0:
                                speed_DF2.append(2*100)
                            else:   
                                speed_DF2.append(len(compound)/np.abs(P_event[0]-T_event[0])*100)

                # Overall compound event features               
                compound_dryspell_start[i, j, :len(compound_dryspell_length2)] = np.array(compound_dryspell_start2)
                compound_dryspell_end[i, j, :len(compound_dryspell_length2)] = np.array(compound_dryspell_end2)
                compound_dryspell_length[i, j, :len(compound_dryspell_length2)] = np.array(compound_dryspell_length2)

                compound_heatwave_start[i, j, :len(compound_heatwave_length2)] = np.array(compound_heatwave_start2)
                compound_heatwave_end[i, j, :len(compound_heatwave_length2)] = np.array(compound_heatwave_end2)
                compound_heatwave_length[i, j, :len(compound_heatwave_length2)] = np.array(compound_heatwave_length2)

                compound_length[i, j, :len(compound_length2)] = np.array(compound_length2)
                compound_start[i, j, :len(compound_start2)] = np.array(compound_start2)
                compound_end[i, j, :len(compound_end2)] = np.array(compound_end2)

                disturbance_period_length[i, j, :len(disturbance_period_length2)] = np.array(disturbance_period_length2)
                disturbance_period_start[i, j, :len(disturbance_period_start2)] = np.array(disturbance_period_start2)
                disturbance_period_end[i, j, :len(disturbance_period_end2)] = np.array(disturbance_period_end2)

                ratio[i, j, :len(ratio2)] = np.array(ratio2) 
                speed[i, j, :len(speed2)] = np.array(speed2) 
                
                temp_per = len(ratio2)/(dry_count + heat_count) *100
                if temp_per > 100:
                    compound_percent[i,j] = 100
                else:
                    compound_percent[i,j] = temp_per
                    
                temp_com_ratio = np.sum(compound_length2)/compound_length_total *100
                if temp_com_ratio > 100:
                    compound_ratio[i,j] = 100
                else:
                    compound_ratio[i,j] = temp_com_ratio
                
                # Heatwave-first compound event features               
                compound_dryspell_start_HF[i, j, :len(compound_dryspell_length_HF2)] = np.array(compound_dryspell_start_HF2)
                compound_dryspell_end_HF[i, j, :len(compound_dryspell_length_HF2)] = np.array(compound_dryspell_end_HF2)
                compound_dryspell_length_HF[i, j, :len(compound_dryspell_length_HF2)] = np.array(compound_dryspell_length_HF2)

                compound_heatwave_start_HF[i, j, :len(compound_heatwave_length_HF2)] = np.array(compound_heatwave_start_HF2)
                compound_heatwave_end_HF[i, j, :len(compound_heatwave_length_HF2)] = np.array(compound_heatwave_end_HF2)
                compound_heatwave_length_HF[i, j, :len(compound_heatwave_length_HF2)] = np.array(compound_heatwave_length_HF2)

                compound_length_HF[i, j, :len(compound_length_HF2)] = np.array(compound_length_HF2)
                compound_start_HF[i, j, :len(compound_start_HF2)] = np.array(compound_start_HF2)
                compound_end_HF[i, j, :len(compound_end_HF2)] = np.array(compound_end_HF2)

                disturbance_period_length_HF[i, j, :len(disturbance_period_length_HF2)] = np.array(disturbance_period_length_HF2)
                disturbance_period_start_HF[i, j, :len(disturbance_period_start_HF2)] = np.array(disturbance_period_start_HF2)
                disturbance_period_end_HF[i, j, :len(disturbance_period_end_HF2)] = np.array(disturbance_period_end_HF2)

                ratio_HF[i, j, :len(ratio_HF2)] = np.array(ratio_HF2) 
                speed_HF[i, j, :len(speed_HF2)] = np.array(speed_HF2) 
                
                temp_per_HF = len(ratio_HF2)/(dry_count + heat_count) *100
                if temp_per_HF > 100:
                    compound_percent_HF[i,j] = 100
                else:
                    compound_percent_HF[i,j] = temp_per_HF
                    
                temp_com_ratio_HF = np.sum(compound_length_HF2)/compound_length_total *100
                if temp_com_ratio_HF > 100:
                    compound_ratio_HF[i,j] = 100
                else:
                    compound_ratio_HF[i,j] = temp_com_ratio_HF
                       
                # Drought-first compound event features               
                compound_dryspell_start_DF[i, j, :len(compound_dryspell_length_DF2)] = np.array(compound_dryspell_start_DF2)
                compound_dryspell_end_DF[i, j, :len(compound_dryspell_length_DF2)] = np.array(compound_dryspell_end_DF2)
                compound_dryspell_length_DF[i, j, :len(compound_dryspell_length_DF2)] = np.array(compound_dryspell_length_DF2)

                compound_heatwave_start_DF[i, j, :len(compound_heatwave_length_DF2)] = np.array(compound_heatwave_start_DF2)
                compound_heatwave_end_DF[i, j, :len(compound_heatwave_length_DF2)] = np.array(compound_heatwave_end_DF2)
                compound_heatwave_length_DF[i, j, :len(compound_heatwave_length_DF2)] = np.array(compound_heatwave_length_DF2)

                compound_length_DF[i, j, :len(compound_length_DF2)] = np.array(compound_length_DF2)
                compound_start_DF[i, j, :len(compound_start_DF2)] = np.array(compound_start_DF2)
                compound_end_DF[i, j, :len(compound_end_DF2)] = np.array(compound_end_DF2)

                disturbance_period_length_DF[i, j, :len(disturbance_period_length_DF2)] = np.array(disturbance_period_length_DF2)
                disturbance_period_start_DF[i, j, :len(disturbance_period_start_DF2)] = np.array(disturbance_period_start_DF2)
                disturbance_period_end_DF[i, j, :len(disturbance_period_end_DF2)] = np.array(disturbance_period_end_DF2)

                ratio_DF[i, j, :len(ratio_DF2)] = np.array(ratio_DF2) 
                speed_DF[i, j, :len(speed_DF2)] = np.array(speed_DF2) 
                
                temp_per_DF = len(ratio_DF2)/(dry_count + heat_count) *100
                if temp_per_DF > 100:
                    compound_percent_DF[i,j] = 100
                else:
                    compound_percent_DF[i,j] = temp_per_DF
                    
                temp_com_ratio_DF = np.sum(compound_length_DF2)/compound_length_total *100
                if temp_com_ratio_DF > 100:
                    compound_ratio_DF[i,j] = 100
                else:
                    compound_ratio_DF[i,j] = temp_com_ratio_DF 

                    
        # Output compound event features             
        np.save(f"{paths['com_dry_path']}Compound_Dryspell_Start_{m}.npy", compound_dryspell_start)
        np.save(f"{paths['com_dry_path']}Compound_Dryspell_End_{m}.npy", compound_dryspell_end)
        np.save(f"{paths['com_dry_path']}Compound_Dryspell_Length_{m}.npy", compound_dryspell_length)

        np.save(f"{paths['com_heat_path']}Compound_Heatwave_Start_{m}.npy", compound_heatwave_start)
        np.save(f"{paths['com_heat_path']}Compound_Heatwave_End_{m}.npy", compound_heatwave_end)
        np.save(f"{paths['com_heat_path']}Compound_Heatwave_Length_{m}.npy", compound_heatwave_length)

        np.save(f"{paths['compound_path']}Compound_Length_{m}.npy", compound_length)
        np.save(f"{paths['compound_path']}Compound_Start_{m}.npy", compound_start)
        np.save(f"{paths['compound_path']}Compound_End_{m}.npy", compound_end)

        np.save(f"{paths['disturbance_path']}Disturbance_Period_Length_{m}.npy", disturbance_period_length)
        np.save(f"{paths['disturbance_path']}Disturbance_Period_Start_{m}.npy", disturbance_period_start)
        np.save(f"{paths['disturbance_path']}Disturbance_Period_End_{m}.npy", disturbance_period_end)

        np.save(f"{paths['speed_path']}Speed_{m}.npy", speed)
        np.save(f"{paths['ratio_path']}Ratio_{m}.npy", ratio)
        
        np.save(f"{paths['compound_path']}Compound_percent_{m}.npy", compound_percent)
        np.save(f"{paths['compound_path']}Compound_ratio_{m}.npy", compound_ratio)
        
        # Output heatwave-first event features             
        np.save(f"{paths['com_dry_path_HF']}Compound_Dryspell_Start_{m}.npy", compound_dryspell_start_HF)
        np.save(f"{paths['com_dry_path_HF']}Compound_Dryspell_End_{m}.npy", compound_dryspell_end_HF)
        np.save(f"{paths['com_dry_path_HF']}Compound_Dryspell_Length_{m}.npy", compound_dryspell_length_HF)

        np.save(f"{paths['com_heat_path_HF']}Compound_Heatwave_Start_{m}.npy", compound_heatwave_start_HF)
        np.save(f"{paths['com_heat_path_HF']}Compound_Heatwave_End_{m}.npy", compound_heatwave_end_HF)
        np.save(f"{paths['com_heat_path_HF']}Compound_Heatwave_Length_{m}.npy", compound_heatwave_length_HF)

        np.save(f"{paths['compound_path_HF']}Compound_Length_{m}.npy", compound_length_HF)
        np.save(f"{paths['compound_path_HF']}Compound_Start_{m}.npy", compound_start_HF)
        np.save(f"{paths['compound_path_HF']}Compound_End_{m}.npy", compound_end_HF)

        np.save(f"{paths['disturbance_path_HF']}Disturbance_Period_Length_{m}.npy", disturbance_period_length_HF)
        np.save(f"{paths['disturbance_path_HF']}Disturbance_Period_Start_{m}.npy", disturbance_period_start_HF)
        np.save(f"{paths['disturbance_path_HF']}Disturbance_Period_End_{m}.npy", disturbance_period_end_HF)

        np.save(f"{paths['speed_path_HF']}Speed_{m}.npy", speed_HF)
        np.save(f"{paths['ratio_path_HF']}Ratio_{m}.npy", ratio_HF)
        
        np.save(f"{paths['compound_path_HF']}Compound_percent_{m}.npy", compound_percent_HF)
        np.save(f"{paths['compound_path_HF']}Compound_ratio_{m}.npy", compound_ratio_HF)
        
        # Output drought-first event features             
        np.save(f"{paths['com_dry_path_DF']}Compound_Dryspell_Start_{m}.npy", compound_dryspell_start_DF)
        np.save(f"{paths['com_dry_path_DF']}Compound_Dryspell_End_{m}.npy", compound_dryspell_end_DF)
        np.save(f"{paths['com_dry_path_DF']}Compound_Dryspell_Length_{m}.npy", compound_dryspell_length_DF)

        np.save(f"{paths['com_heat_path_DF']}Compound_Heatwave_Start_{m}.npy", compound_heatwave_start_DF)
        np.save(f"{paths['com_heat_path_DF']}Compound_Heatwave_End_{m}.npy", compound_heatwave_end_DF)
        np.save(f"{paths['com_heat_path_DF']}Compound_Heatwave_Length_{m}.npy", compound_heatwave_length_DF)

        np.save(f"{paths['compound_path_DF']}Compound_Length_{m}.npy", compound_length_DF)
        np.save(f"{paths['compound_path_DF']}Compound_Start_{m}.npy", compound_start_DF)
        np.save(f"{paths['compound_path_DF']}Compound_End_{m}.npy", compound_end_DF)

        np.save(f"{paths['disturbance_path_DF']}Disturbance_Period_Length_{m}.npy", disturbance_period_length_DF)
        np.save(f"{paths['disturbance_path_DF']}Disturbance_Period_Start_{m}.npy", disturbance_period_start_DF)
        np.save(f"{paths['disturbance_path_DF']}Disturbance_Period_End_{m}.npy", disturbance_period_end_DF)

        np.save(f"{paths['speed_path_DF']}Speed_{m}.npy", speed_DF)
        np.save(f"{paths['ratio_path_DF']}Ratio_{m}.npy", ratio_DF)
        
        np.save(f"{paths['compound_path_DF']}Compound_percent_{m}.npy", compound_percent_DF)
        np.save(f"{paths['compound_path_DF']}Compound_ratio_{m}.npy", compound_ratio_DF)
        
    print(f"{model} is done!")