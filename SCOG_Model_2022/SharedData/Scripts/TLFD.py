"""
Trip Length Frequency Distribution for Visum

michael.mccarthy@rsginc.com 2025-07-29

"""

# Libraries
import sys
import numpy as np
import VisumPy.helpers
import VisumPy.matrices
import pandas as pd
from datetime import datetime
import os.path

def tlfd (trip_mtx, trip_label, dist_mtx, mtx_label):
    # get trip PA and distance matrix
    # group into bins by distance
    
    trips = trip_mtx.flatten()
    dist = dist_mtx.flatten()
    
    # Create bins
    max_distance = 75
    bin_size = 1
    bins = np.arange(0, max_distance + bin_size, bin_size)
    bin_labels = [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(bins)-1)]
    bin_labels2 = [f"{int(bins[i])}" for i in range(len(bins)-1)]
    

    # Assign each trip to a bin based on its distance
    bin_indices = np.digitize(dist, bins) - 1  # -1 to get correct index

    # Aggregate total trips per bin
    trip_totals = np.zeros(len(bin_labels))
    for i in range(len(trips)):
        if 0 <= bin_indices[i] < len(trip_totals):
            trip_totals[bin_indices[i]] += trips[i]
            
     # Create dataframe
    tlfd_df = pd.DataFrame({
        mtx_label+' Bin': bin_labels2,
        trip_label+' Trips': trip_totals.astype(int)
    }).set_index(mtx_label+' Bin')
    
    return(tlfd_df)

# set paths 
out_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..', 'Outputs'))
timestamp = datetime.now().strftime("%m%d%Y_%H%M%S")

# skims: 100 = MSA, 104 = distance
# trips: 21 = daily HBW OD, 22 = HBO, 25 = External, 26 = NHB

dist_mat = VisumPy.helpers.GetMatrix(Visum, 104)
time_mat = VisumPy.helpers.GetMatrix(Visum, 100)
hbw_mat = VisumPy.helpers.GetMatrix(Visum, 1)
hbo_mat = VisumPy.helpers.GetMatrix(Visum, 2)
nhb_mat = VisumPy.helpers.GetMatrix(Visum, 7)
xiw_mat = VisumPy.helpers.GetMatrix(Visum, 71)
ixw_mat = VisumPy.helpers.GetMatrix(Visum, 72)
xi_nw_mat = VisumPy.helpers.GetMatrix(Visum, 73)
xx_mat = VisumPy.helpers.GetMatrix(Visum, 76)


skim_label = 'Distance'
hbw_dist_df = tlfd(hbw_mat, 'HBW', dist_mat, skim_label)
hbo_dist_df = tlfd(hbo_mat, 'HBO', dist_mat, skim_label)
nhb_dist_df = tlfd(nhb_mat, 'NHB', dist_mat, skim_label)
xiw_dist_df = tlfd(xiw_mat, 'XI Work', dist_mat, skim_label)
ixw_dist_df = tlfd(ixw_mat, 'IX Work', dist_mat, skim_label)
ixxinw_dist_df = tlfd(xi_nw_mat, 'IX/XI Non-work', dist_mat, skim_label)
xx_dist_df = tlfd(xx_mat, 'XX', dist_mat, skim_label)

distance_df = pd.concat([hbw_dist_df, hbo_dist_df, nhb_dist_df, xiw_dist_df, ixw_dist_df, ixxinw_dist_df, xx_dist_df],axis=1,sort=False).reset_index()
distance_df.to_csv(os.path.join(out_path, "TLFD_miles_"+timestamp+".csv"))

#hbw_dist_df.merge(hbo_dist_df,on=skim_label+' Bin').merge(nhb_dist_df,on=skim_label+' Bin')
# .merge(xiw_dist_df,on=skim_label+' Bin').merge(ixw_dist_df,on=skim_label+' Bin').merge(ixxinw_dist_df,on=skim_label+' Bin').merge(xx_dist_df,on=skim_label+' Bin')

skim_label = 'Time'
hbw_time_df = tlfd(hbw_mat, 'HBW', time_mat, skim_label)
hbo_time_df = tlfd(hbo_mat, 'HBO', time_mat, skim_label)
nhb_time_df = tlfd(nhb_mat, 'NHB', time_mat, skim_label)
xiw_time_df = tlfd(xiw_mat, 'XI Work', time_mat, skim_label)
ixw_time_df = tlfd(ixw_mat, 'IX Work', time_mat, skim_label)
ixxinw_time_df = tlfd(xi_nw_mat, 'IX/XI Non-work', time_mat, skim_label)
xx_time_df = tlfd(xx_mat, 'XX', time_mat, skim_label)

time_df = pd.concat([hbw_time_df, hbo_time_df, nhb_time_df, xiw_time_df, ixw_time_df, ixxinw_time_df, xx_time_df],axis=1,sort=False).reset_index()
time_df.to_csv(os.path.join(out_path, "TLFD_minutes_"+timestamp+".csv"))


# hbw_df.to_csv(os.path.join(out_path, "TLFD_HBW_"+timestamp+".csv"))
# 
# hbo_df = tlfd(hbo_mat, dist_mat)
# hbo_df.to_csv(os.path.join(out_path, "TLFD_HBO_"+timestamp+".csv"))
# 
# nhb_df = tlfd(nhb_mat, dist_mat)
# nhb_df.to_csv(os.path.join(out_path, "TLFD_NHB_"+timestamp+".csv"))
# 
# ext_df = tlfd(xiw_mat, dist_mat)
# ext_df.to_csv(os.path.join(out_path, "TLFD_XIW_"+timestamp+".csv"))
# 
# ixw_df = tlfd(ixw_mat, dist_mat)
# ixw_df.to_csv(os.path.join(out_path, "TLFD_IXW_"+timestamp+".csv"))
# 
# ixxinw_df = tlfd(xi_nw_mat, dist_mat)
# ixxinw_df.to_csv(os.path.join(out_path, "TLFD_IXXI_NW_"+timestamp+".csv"))
# 
# xx_df = tlfd(xx_mat, dist_mat)
# xx_df.to_csv(os.path.join(out_path, "TLFD_XX_"+timestamp+".csv"))


