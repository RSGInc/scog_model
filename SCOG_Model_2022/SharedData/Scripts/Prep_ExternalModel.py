"""
External Growth

Luke Gordon, luke.gordon@rsginc.com 09/26/2023
Michael McCarthy, michael.mccarthy@rsginc.com 12/17/2024
"""


# Libraries
import VisumPy.helpers
import VisumPy.csvHelpers
import VisumPy.matrices
import traceback
import pandas as pd
import numpy as np
import csv, sys, os

# set paths 
external_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'External'))

# Pull EXT_GROWTH from Visum table
no               = VisumPy.helpers.GetMulti(Visum.Net.Zones, "NO", activeOnly = True)
ext_auto_growth  = VisumPy.helpers.GetMulti(Visum.Net.Zones, "EXT_AUTO_GROWTH", activeOnly = True)
#ext_truck_growth = VisumPy.helpers.GetMulti(Visum.Net.Zones, "EXT_TRUCK_GROWTH", activeOnly = True)

growth_df = pd.DataFrame({'NO':no, 'Auto_Ext_Growth':ext_auto_growth}) #, 'Trk_Ext_Growth':ext_truck_growth})

# External counts and percentages lookup table
externalLookupTableFileName = external_path + "/Ext_Sta_Summary.csv" # TODO dummy file
external_lookup_table = VisumPy.csvHelpers.readCSV(externalLookupTableFileName)
df = pd.DataFrame(external_lookup_table[1:], columns=external_lookup_table[0])
# Sort increasing order of NO
df.sort_values(by='NO', ascending=True)

# Convert NO to integer for growth_df and df
growth_df[['NO']] = growth_df[['NO']].astype(int)
df[['NO']]        = df[['NO']].astype(int)

# Join growth_df to df by NO to include growth values
df = pd.merge(df, growth_df, on='NO', how='left')

# Grow auto and truck XX marginals by growth rate by zone
scen_year = Visum.Net.AttValue("SCEN_YEAR")
base_year = Visum.Net.AttValue("BASE_YEAR")
grow_years = int(scen_year)-int(base_year)

# Convert all count and growth fields to float to make multiplication and other operations run smoothly
df[['AADT','Auto_AADT','Trk_AADT','Auto_XX_O','Auto_XX_D','Trk_XX_O','Trk_XX_D','Auto_XIIX_OD','Trk_XIIX_OD','XI_W%','IX_W%','X_NW%','Auto_XX_O_Out',
	'Auto_XX_D_Out','Trk_XX_O_Out','Trk_XX_D_Out','Auto_XIIX_OD_Out','Trk_XIIX_OD_Out','Auto_IX%','Auto_XX%','AutoXXO%','AutoXXD%','Trk_IX%','Trk_XX%','Trk_XXO%',
	'Trk_XXD%','Truck XXO %','Truck XXD %','Auto_Ext_Growth']] = df[['AADT','Auto_AADT','Trk_AADT','Auto_XX_O','Auto_XX_D','Trk_XX_O','Trk_XX_D',
	'Auto_XIIX_OD','Trk_XIIX_OD','XI_W%','IX_W%','X_NW%','Auto_XX_O_Out','Auto_XX_D_Out','Trk_XX_O_Out','Trk_XX_D_Out','Auto_XIIX_OD_Out','Trk_XIIX_OD_Out',
	'Auto_IX%','Auto_XX%','AutoXXO%','AutoXXD%','Trk_IX%','Trk_XX%','Trk_XXO%','Trk_XXD%','Truck XXO %','Truck XXD %','Auto_Ext_Growth']].astype(float)
	


# Multiply XX and XIIX by growth rates
for i in range(len(df)):
	df.at[i,'Auto_XX_O_Out']    = df.at[i,'Auto_XX_O']   *((1+df.at[i,'Auto_Ext_Growth'])**grow_years)
	df.at[i,'Auto_XX_D_Out']    = df.at[i,'Auto_XX_D']   *((1+df.at[i,'Auto_Ext_Growth'])**grow_years)
	df.at[i,'Auto_XIIX_OD_Out'] = df.at[i,'Auto_XIIX_OD']*((1+df.at[i,'Auto_Ext_Growth'])**grow_years)
	# df.at[i,'Trk_XX_O_Out']     = df.at[i,'Trk_XX_O']    *((1+df.at[i,'Trk_Ext_Growth']) **grow_years)
	# df.at[i,'Trk_XX_D_Out']     = df.at[i,'Trk_XX_D']    *((1+df.at[i,'Trk_Ext_Growth']) **grow_years)
	# df.at[i,'Trk_XIIX_OD_Out']  = df.at[i,'Trk_XIIX_OD'] *((1+df.at[i,'Trk_Ext_Growth']) **grow_years)


# Zero Visum fields to ensure internals are 0
df['Zero'] = 0
VisumPy.helpers.SetMulti(Visum.Net.Zones, "XX_AUTO_P", df['Zero'])
VisumPy.helpers.SetMulti(Visum.Net.Zones, "XX_AUTO_A", df['Zero'])
VisumPy.helpers.SetMulti(Visum.Net.Zones, "XIIX_AUTO_OD", df['Zero'])
# VisumPy.helpers.SetMulti(Visum.Net.Zones, "XX_TRUCK_P", df['Zero'])
# VisumPy.helpers.SetMulti(Visum.Net.Zones, "XX_TRUCK_A", df['Zero'])
# VisumPy.helpers.SetMulti(Visum.Net.Zones, "XIIX_TRUCK_OD", df['Zero'])
VisumPy.helpers.SetMulti(Visum.Net.Zones, "XI_W_PCT", df['Zero'])
VisumPy.helpers.SetMulti(Visum.Net.Zones, "IX_W_PCT", df['Zero'])
VisumPy.helpers.SetMulti(Visum.Net.Zones, "X_NW_PCT", df['Zero'])

# Paste XX and XIIX into Visum for balancing and frataring of matrices only in external zones
VisumPy.helpers.SetMulti(Visum.Net.Zones, "XX_AUTO_P", df['Auto_XX_O_Out'],activeOnly = True)
VisumPy.helpers.SetMulti(Visum.Net.Zones, "XX_AUTO_A", df['Auto_XX_D_Out'],activeOnly = True)
VisumPy.helpers.SetMulti(Visum.Net.Zones, "XIIX_AUTO_OD", df['Auto_XIIX_OD_Out'],activeOnly = True)
# VisumPy.helpers.SetMulti(Visum.Net.Zones, "XX_TRUCK_P", df['Trk_XX_O_Out'],activeOnly = True)
# VisumPy.helpers.SetMulti(Visum.Net.Zones, "XX_TRUCK_A", df['Trk_XX_D_Out'],activeOnly = True)
# VisumPy.helpers.SetMulti(Visum.Net.Zones, "XIIX_TRUCK_OD", df['Trk_XIIX_OD_Out'],activeOnly = True)

# Paste other Ext_Sta_Summary.csv fields into Visum
VisumPy.helpers.SetMulti(Visum.Net.Zones, "XI_W_PCT", df['XI_W%'],activeOnly = True)
VisumPy.helpers.SetMulti(Visum.Net.Zones, "IX_W_PCT", df['IX_W%'],activeOnly = True)
VisumPy.helpers.SetMulti(Visum.Net.Zones, "X_NW_PCT", df['X_NW%'],activeOnly = True)


#if (int(Visum.Net.AttValue("YEAR"))) > 2022: # Fratar matrix for future years
if scen_year > base_year: # Fratar matrix for future years
	# Pull full fields (not just externals) for matrix balancing
	XX_auto_P  = VisumPy.helpers.GetMulti(Visum.Net.Zones, "XX_AUTO_P")
	XX_auto_A  = VisumPy.helpers.GetMulti(Visum.Net.Zones, "XX_AUTO_A")
	# XX_truck_P = VisumPy.helpers.GetMulti(Visum.Net.Zones, "XX_TRUCK_P")
	# XX_truck_A = VisumPy.helpers.GetMulti(Visum.Net.Zones, "XX_TRUCK_A")
	
	# Balance XX Matrices to Grown P's and A's for Autos
	mat = VisumPy.helpers.GetMatrix(Visum, 6) # Auto XX Matrix # bug? initially zeros
	r = np.array(XX_auto_P)
	c = np.array(XX_auto_A)
	#  Run Visum balanceMatrix function
	balanced_mat = VisumPy.matrices.balanceMatrix(mat,r,c)
	# Set matrix in Visum
	VisumPy.helpers.SetMatrix(Visum, 76, balanced_mat)
	
	
	# Balance XX Matrices to Grown P's and A's for Trucks
	# mat = VisumPy.helpers.GetMatrix(Visum, 567) # Truck XX Matrix
	# r = np.array(XX_truck_P)
	# c = np.array(XX_truck_A)
	# #  Run Visum balanceMatrix function
	# balanced_mat = VisumPy.matrices.balanceMatrix(mat,r,c)
	# # Set matrix in Visum
	# VisumPy.helpers.SetMatrix(Visum, 569, balanced_mat)

else: # Base year just copy Base Matrices to working matrices
	# Auto
	mat = VisumPy.helpers.GetMatrix(Visum, 6) # Auto XX Matrix
	VisumPy.helpers.SetMatrix(Visum, 76, mat)
	# Truck
	# mat = VisumPy.helpers.GetMatrix(Visum, 567) # Truck XX Matrix
	# VisumPy.helpers.SetMatrix(Visum, 569, mat)
	
	


