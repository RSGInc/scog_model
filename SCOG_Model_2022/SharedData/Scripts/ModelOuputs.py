# Full script to handle Assignment Summary, Model Reporting, and Results export. Uses same timestamped folder for all outputs

"""
created 1/16/2024

@author: luke.gordon
@author: michael.mccarthy
mm modified 1/16/2025

"""

# Libraries
import VisumPy.helpers
import VisumPy.excel
import pandas as pd
import numpy as np
import csv
from datetime import datetime
import math
import os.path

# set paths 
outputs_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..', 'Outputs'))
	
# Calculate timestamp for folder name
date = datetime.now().strftime("(%Y-%m-%d)-%H_%M_%S")

# Create timestamped folder for all results
os.mkdir(outputs_path + "/ModelRun_"+date)


# Create folder for storing this runs assignment results
os.mkdir(outputs_path + "/ModelRun_"+date+"/Assignment Results")

# Create folder for storing this runs reporting results
#os.mkdir(outputs_path + "/ModelRun_"+date+"/Model Reporting")

# Create folder for storing unformatted (float) versions of Model Reporting results for use in comparison
#os.mkdir(outputs_path + "/ModelRun_"+date+"/Model Reporting/unformatted")

# Create folder for storing this runs matrices
#os.mkdir(outputs_path + "/ModelRun_"+date+"/Matrices")

# Create folder for storing this runs link table
#os.mkdir(outputs_path + "/ModelRun_"+date+"/Network")



# Formatting functions
# Formatting for columns
# Create formatting function for large numbers (no decimals and thousand commas)
def format_commas(column):
	return column.apply(lambda x: '{:,.0f}'.format(x) if pd.notna(x) else None)
# Create formatting function for percentages (2 decimals and percent symbol, also multiplies by 100)
def format_percent(column):
	return column.apply(lambda x: '{:.2%}'.format(x) if pd.notna(x) else None)
# Create formatting function for small numbers (2 decimals)
def format_twoplaces(column):
	return column.apply(lambda x: '{:.2f}'.format(x) if pd.notna(x) else None)
# Create formatting function for small numbers (1 decimal)
def format_oneplace(column):
	return column.apply(lambda x: '{:.1f}'.format(x) if pd.notna(x) else None)
# Create formatting function for small numbers (0 decimals)
def format_zeroplaces(column):
	return column.apply(lambda x: '{:.0f}'.format(x) if pd.notna(x) else None)
	
	
# Formatting for single cells
# Create formatting function for large numbers (no decimals and thousand commas)
def format_commas_cell(cell_value):
	return '{:,.0f}'.format(cell_value) if pd.notna(cell_value) else None
# Create formatting function for percentages (2 decimals and percent symbol, also multiplies by 100)
def format_percent_cell(cell_value):
	return '{:.2%}'.format(cell_value) if pd.notna(cell_value) else None
# Create formatting function for small numbers (2 decimals)
def format_twoplaces_cell(cell_value):
	return '{:.2f}'.format(cell_value) if pd.notna(cell_value) else None
# Create formatting function for small numbers (0 decimals)
def format_zeroplaces_cell(cell_value):
	return '{:.0f}'.format(cell_value) if pd.notna(cell_value) else None

	
# Round to nearest even number function. Used in Volume corridor reporting
def round_to_nearest_even(number):
    rounded = round(number)
    return rounded + (rounded % 2 == 1)

# Create Percent Error function
def pct_error(count , flow):
	error = ((sum(flow)/len(flow)) - (sum(count)/len(count))) / (sum(count)/len(count))

	return error
	
# Create Percent RMSE function
def pct_rmse(count , sqerror):
	rmse = math.sqrt(sum(sqerror)/len(sqerror))/(sum(count)/len(count))

	return rmse
	
# Create VMT function (needs to pull data from all links, not just ones with counts)	
def vmt(flow , length):
	vmt = sum(flow*length)

	return vmt


# Create VHT function (needs to pull data from all links and periods, not just ones with counts) 
		# Need to have logic to handle daily vs. a single period
def vht_dly(am_flow, am_time, pm_flow, pm_time, op_flow, op_time):
		vht = sum(am_flow*am_time) + sum(md_flow*md_time) + sum(pm_flow*pm_time) + sum(nt_flow*nt_time)
		return vht
def vht_per(flow, time):
		vht = sum(flow*time)
		return vht



# Assignment summary function
def assignment_summary(auto_count, auto_flow, cong_auto_time, period):
	
	# DAILY
	# Percent Error and Percent RMSE
	
	# Import ID fields and fields with Counts and Flows
	# Link ID fields
	NO          = VisumPy.helpers.GetMulti(Visum.Net.Links,"No", activeOnly = True)
	FCLASS      = VisumPy.helpers.GetMulti(Visum.Net.Links,"TYPENO", activeOnly = True)
	LENGTH       = VisumPy.helpers.GetMulti(Visum.Net.Links,"Length", activeOnly = True)
	EXT         = VisumPy.helpers.GetMulti(Visum.Net.Links,"EXT_COUNT", activeOnly = True)
	SCRNLINE    = VisumPy.helpers.GetMulti(Visum.Net.Links,r"CONCATENATE:SCREENLINES\CODE", activeOnly = True)
	# Counts
	Auto_Count  = VisumPy.helpers.GetMulti(Visum.Net.Links,auto_count, activeOnly = True)
	#Truck_Count = VisumPy.helpers.GetMulti(Visum.Net.Links,truck_count, activeOnly = True)
	#Tot_Count   = VisumPy.helpers.GetMulti(Visum.Net.Links,all_count, activeOnly = True)
	#AADT        = VisumPy.helpers.GetMulti(Visum.Net.Links,'AADT', activeOnly = True)
	# Link Daily Flows
	Auto_Flow  = VisumPy.helpers.GetMulti(Visum.Net.Links,auto_flow, activeOnly = True)
	#Truck_Flow = VisumPy.helpers.GetMulti(Visum.Net.Links,truck_flow, activeOnly = True)
	#Tot_Flow   = VisumPy.helpers.GetMulti(Visum.Net.Links,all_flow, activeOnly = True)

	
	# Make Visum list with link data
	summary_list = [NO, FCLASS, LENGTH, EXT, SCRNLINE, Auto_Flow, Auto_Count]    
			
	# Put Visum link list into dataframe
	df = pd.DataFrame(np.column_stack(summary_list), columns = ['NO','FCLASS', 'LENGTH', 'EXTERNAL', 'SCRNLINE',
																'Auto_Flow', 'Auto_Count'])
																															

	# # Break out SCRNLINE field to separate by commas into individual columns																														
	# df = pd.concat([df,df['SCRNLINE'].str.split(',', expand = True)], axis = 1)
	# # Change Screenline field names
	# df = df.rename(columns = {0:'SCRNLINE1',1:'SCRNLINE2'})
	# # Replace null values with 0 in the screenline fields
	# df['SCRNLINE1'] = df['SCRNLINE1'].replace('',np.nan).fillna(0)
	# df['SCRNLINE2'] = df['SCRNLINE2'].replace('',np.nan).fillna(0)
	

	# Define custom_sum function to maintain null values when aggregating Counts and Flows by LinkNO
	def custom_sum(series):
    # If all values are null, return null; otherwise, return the sum of the values
		return series.sum() if series.notna().any() else None																												
	# GROUP EACH DATAFRAME BY 'NO' COLUMN TO COMBINE COUNTS ON EACH LINK INTO BOTH DIRECTIONS
	df = df.groupby('NO').agg(
		FCLASS      =('FCLASS', 'min'),
		LENGTH      =('LENGTH', 'max'),
		EXTERNAL    =('EXTERNAL', 'max'),
		# SCRNLINE1   =('SCRNLINE1', 'first'),
		# SCRNLINE2   =('SCRNLINE2', 'first'),
		Auto_Flow   =('Auto_Flow', custom_sum),
		Auto_Count  =('Auto_Count', custom_sum)
	).reset_index()

	df['AADT'] = df['Auto_Count']
	
	# Convert SCRNLINE1 and SCRNLINE2 to Integer
	# df[['SCRNLINE1','SCRNLINE2']] = df[['SCRNLINE1','SCRNLINE2']].astype(float)
	# df[['SCRNLINE1','SCRNLINE2']] = df[['SCRNLINE1','SCRNLINE2']].astype(int)
	
	
	# Build results dictionary to use as results dataframe to save summary stats for group stats
	results = {"Segment":['Auto','Auto: Internal','Auto: External',  # Auto
						'Auto: AADT <5k','Auto: AADT 5-10k','Auto: AADT 10-15k','Auto: AADT 15-20k','Auto: AADT 20-30k','Auto: AADT 30-40k','Auto: AADT 40-50k','Auto: AADT >50k',
						'Auto: Rural Interstate', 'Auto: Rural Highways',    'Auto: Rural Principal Arterial', 'Auto: Rural Minor Arterial', 'Auto: Rural Major Collector', 'Auto: Rural Minor Collector', 'Auto: Rural Local Street',
						'Auto: Urban Interstate', 'Auto: Urban Expressways', 'Auto: Urban Principal Arterial', 'Auto: Urban Minor Arterial', 'Auto: Urban Major Collector', 'Auto: Urban Minor Collector', 'Auto: Urban Local Street','Auto: Ramps']}
	"""
						'Auto: Screenline #1', 'Auto: Screenline #2', 'Auto: Screenline #3', 'Auto: Screenline #4', 'Auto: Screenline #5', 'Auto: Screenline #6', 'Auto: Screenline #8', 'Auto: Screenline #9', 'Auto: Screenline #10', 
						'Auto: Screenline #11', 'Auto: Screenline #12', 'Auto: Screenline #13', 'Auto: Screenline #14', 'Auto: Screenline #20', 'Auto: Screenline #21', 'Auto: Screenline #22', 'Auto: Screenline #23', 
						'Auto: Screenline #24', 'Auto: Screenline #28', 'Auto: Screenline #29', 'Auto: Screenline #30', 'Auto: Screenline #31', 'Auto: Screenline #32', 'Auto: Screenline #33', 'Auto: Screenline #34', 
						'Auto: Screenline #35', 'Auto: Screenline #36',
						'Truck','Truck: Internal','Truck: External', # Truck
						'Truck: AADT <5k','Truck: AADT 5-10k','Truck: AADT 10-15k','Truck: AADT 15-20k','Truck: AADT 20-30k','Truck: AADT 30-40k','Truck: AADT 40-50k','Truck: AADT >50k',
						'Truck: Rural Interstate', 'Truck: Rural Highways',    'Truck: Rural Principal Arterial', 'Truck: Rural Minor Arterial', 'Truck: Rural Major Collector', 'Truck: Rural Minor Collector', 'Truck: Rural Local Street',
						'Truck: Urban Interstate', 'Truck: Urban Expressways', 'Truck: Urban Principal Arterial', 'Truck: Urban Minor Arterial', 'Truck: Urban Major Collector', 'Truck: Urban Minor Collector', 'Truck: Urban Local Street','Truck: Ramps',
						'Truck: Screenline #1', 'Truck: Screenline #2', 'Truck: Screenline #3', 'Truck: Screenline #4', 'Truck: Screenline #5', 'Truck: Screenline #6', 'Truck: Screenline #8', 'Truck: Screenline #9', 'Truck: Screenline #10', 
						'Truck: Screenline #11', 'Truck: Screenline #12', 'Truck: Screenline #13', 'Truck: Screenline #14', 'Truck: Screenline #20', 'Truck: Screenline #21', 'Truck: Screenline #22', 'Truck: Screenline #23', 
						'Truck: Screenline #24', 'Truck: Screenline #28', 'Truck: Screenline #29', 'Truck: Screenline #30', 'Truck: Screenline #31', 'Truck: Screenline #32', 'Truck: Screenline #33', 'Truck: Screenline #34', 
						'Truck: Screenline #35', 'Truck: Screenline #36',
						'All Modes','All Modes: Internal','All Modes: External', # All Modes
						'All Modes: AADT <5k','All Modes: AADT 5-10k','All Modes: AADT 10-15k','All Modes: AADT 15-20k','All Modes: AADT 20-30k','All Modes: AADT 30-40k','All Modes: AADT 40-50k','All Modes: AADT >50k',
						'All Modes: Rural Interstate', 'All Modes: Rural Highways',    'All Modes: Rural Principal Arterial', 'All Modes: Rural Minor Arterial', 'All Modes: Rural Major Collector', 'All Modes: Rural Minor Collector', 'All Modes: Rural Local Street',
						'All Modes: Urban Interstate', 'All Modes: Urban Expressways', 'All Modes: Urban Principal Arterial', 'All Modes: Urban Minor Arterial', 'All Modes: Urban Major Collector', 'All Modes: Urban Minor Collector', 'All Modes: Urban Local Street','All Modes: Ramps',
						'All Modes: Screenline #1', 'All Modes: Screenline #2', 'All Modes: Screenline #3', 'All Modes: Screenline #4', 'All Modes: Screenline #5', 'All Modes: Screenline #6', 'All Modes: Screenline #8', 'All Modes: Screenline #9', 'All Modes: Screenline #10', 
						'All Modes: Screenline #11', 'All Modes: Screenline #12', 'All Modes: Screenline #13', 'All Modes: Screenline #14', 'All Modes: Screenline #20', 'All Modes: Screenline #21', 'All Modes: Screenline #22', 'All Modes: Screenline #23', 
						'All Modes: Screenline #24', 'All Modes: Screenline #28', 'All Modes: Screenline #29', 'All Modes: Screenline #30', 'All Modes: Screenline #31', 'All Modes: Screenline #32', 'All Modes: Screenline #33', 'All Modes: Screenline #34', 
						'All Modes: Screenline #35', 'All Modes: Screenline #36',
						'LCV'
					]}
	"""
						
						
						
	# Plug results dictionary into results_df dataframe                           
	results_df = pd.DataFrame(data = results)
	
	# Add stats columns
	results_df['Percent Error'] = None
	results_df['Percent RMSE'] = None
	results_df['Total VMT'] = None
	results_df['Total VHT'] = None
	results_df['Number of Observations'] = None
	results_df['Sum of Counts'] = None
	results_df['Mean of Counts'] = None
	results_df['Median of Counts'] = None
	results_df['Count VMT, Links with Counts'] = None
	results_df['Modeled VMT, Links with Counts'] = None
	

	
	
	# For links with counts only, used for Pct. Error and Pct. RMSE
	# Filter out links where count is null or 0 and by each condition
	# All Links with Auto Counts
	count_df     = df[df['Auto_Count'].notna()]
	# Internal/External
	internal_df  = count_df[(count_df['EXTERNAL'] == 0)]
	external_df  = count_df[(count_df['EXTERNAL'] == 1)]

	# By AADT Volume
	under_5k_df     = count_df[(count_df['AADT'] < 5000)]
	btwn_5_10k_df   = count_df[(count_df['AADT'] >= 5000) & (count_df['AADT'] < 10000)]
	btwn_10_15k_df  = count_df[(count_df['AADT'] >= 10000) & (count_df['AADT'] < 15000)]
	btwn_15_20k_df  = count_df[(count_df['AADT'] >= 15000) & (count_df['AADT'] < 20000)]
	btwn_20_30k_df  = count_df[(count_df['AADT'] >= 20000) & (count_df['AADT'] < 30000)]
	btwn_30_40k_df  = count_df[(count_df['AADT'] >= 30000) & (count_df['AADT'] < 40000)]
	btwn_40_50k_df  = count_df[(count_df['AADT'] >= 40000) & (count_df['AADT'] < 50000)]
	over_50k_df     = count_df[(count_df['AADT'] >= 50000)]
	# By Functional Class
	fc1_df    = count_df[(count_df['FCLASS'] == 1)]
	fc2_df    = count_df[(count_df['FCLASS'] == 2)]
	fc4_df    = count_df[(count_df['FCLASS'] == 4)]
	fc6_df    = count_df[(count_df['FCLASS'] == 6)]
	fc7_df    = count_df[(count_df['FCLASS'] == 7)]
	fc8_df    = count_df[(count_df['FCLASS'] == 8)]
	fc9_df    = count_df[(count_df['FCLASS'] == 9)]
	fc11_df   = count_df[(count_df['FCLASS'] == 11)]
	fc12_df   = count_df[(count_df['FCLASS'] == 12)]
	fc14_df   = count_df[(count_df['FCLASS'] == 14)]
	fc16_df   = count_df[(count_df['FCLASS'] == 16)]
	fc17_df   = count_df[(count_df['FCLASS'] == 17)]
	fc18_df   = count_df[(count_df['FCLASS'] == 18)]
	fc19_df   = count_df[(count_df['FCLASS'] == 19)]
	fcramp_df = count_df[(count_df['FCLASS'] == 50) | (count_df['FCLASS'] == 53) | (count_df['FCLASS'] == 54)]
	# By Screenline
	# sl_1_df   = count_df[(count_df['SCRNLINE1'] == 1) | (count_df['SCRNLINE2'] == 1)]
	# sl_2_df   = count_df[(count_df['SCRNLINE1'] == 2) | (count_df['SCRNLINE2'] == 2)]
	# sl_3_df   = count_df[(count_df['SCRNLINE1'] == 3) | (count_df['SCRNLINE2'] == 3)]
	# sl_4_df   = count_df[(count_df['SCRNLINE1'] == 4) | (count_df['SCRNLINE2'] == 4)]
	# sl_5_df   = count_df[(count_df['SCRNLINE1'] == 5) | (count_df['SCRNLINE2'] == 5)]
	# sl_6_df   = count_df[(count_df['SCRNLINE1'] == 6) | (count_df['SCRNLINE2'] == 6)]
	# sl_8_df   = count_df[(count_df['SCRNLINE1'] == 8) | (count_df['SCRNLINE2'] == 8)]
	# sl_9_df   = count_df[(count_df['SCRNLINE1'] == 9) | (count_df['SCRNLINE2'] == 9)]
	# sl_10_df   = count_df[(count_df['SCRNLINE1'] == 10) | (count_df['SCRNLINE2'] == 10)]
	# sl_11_df   = count_df[(count_df['SCRNLINE1'] == 11) | (count_df['SCRNLINE2'] == 11)]
	# sl_12_df   = count_df[(count_df['SCRNLINE1'] == 12) | (count_df['SCRNLINE2'] == 12)]
	# sl_13_df   = count_df[(count_df['SCRNLINE1'] == 13) | (count_df['SCRNLINE2'] == 13)]
	# sl_14_df   = count_df[(count_df['SCRNLINE1'] == 14) | (count_df['SCRNLINE2'] == 14)]
	# sl_20_df   = count_df[(count_df['SCRNLINE1'] == 20) | (count_df['SCRNLINE2'] == 20)]
	# sl_21_df   = count_df[(count_df['SCRNLINE1'] == 21) | (count_df['SCRNLINE2'] == 21)]
	# sl_22_df   = count_df[(count_df['SCRNLINE1'] == 22) | (count_df['SCRNLINE2'] == 22)]
	# sl_23_df   = count_df[(count_df['SCRNLINE1'] == 23) | (count_df['SCRNLINE2'] == 23)]
	# sl_24_df   = count_df[(count_df['SCRNLINE1'] == 24) | (count_df['SCRNLINE2'] == 24)]
	# sl_28_df   = count_df[(count_df['SCRNLINE1'] == 28) | (count_df['SCRNLINE2'] == 28)]
	# sl_29_df   = count_df[(count_df['SCRNLINE1'] == 29) | (count_df['SCRNLINE2'] == 29)]
	# sl_30_df   = count_df[(count_df['SCRNLINE1'] == 30) | (count_df['SCRNLINE2'] == 30)]
	# sl_31_df   = count_df[(count_df['SCRNLINE1'] == 31) | (count_df['SCRNLINE2'] == 31)]
	# sl_32_df   = count_df[(count_df['SCRNLINE1'] == 32) | (count_df['SCRNLINE2'] == 32)]
	# sl_33_df   = count_df[(count_df['SCRNLINE1'] == 33) | (count_df['SCRNLINE2'] == 33)]
	# sl_34_df   = count_df[(count_df['SCRNLINE1'] == 34) | (count_df['SCRNLINE2'] == 34)]
	# sl_35_df   = count_df[(count_df['SCRNLINE1'] == 35) | (count_df['SCRNLINE2'] == 35)]
	# sl_36_df   = count_df[(count_df['SCRNLINE1'] == 36) | (count_df['SCRNLINE2'] == 36)]
	
	
	# Build list of dataframes to loop thru
	auto_df_list = [count_df,internal_df,external_df,
					under_5k_df,btwn_5_10k_df,btwn_10_15k_df,btwn_15_20k_df,btwn_20_30k_df,btwn_30_40k_df,btwn_40_50k_df,over_50k_df,
					fc1_df,fc2_df,fc4_df,fc6_df,fc7_df,fc8_df,fc9_df,fc11_df,fc12_df,fc14_df,fc16_df,fc17_df,fc18_df,fc19_df,fcramp_df]
				#	sl_1_df,sl_2_df,sl_3_df,sl_4_df,sl_5_df,sl_6_df,sl_8_df,sl_9_df,sl_10_df,sl_11_df,sl_12_df,sl_13_df,sl_14_df,sl_20_df,
				#	sl_21_df,sl_22_df,sl_23_df,sl_24_df,sl_28_df,sl_29_df,sl_30_df,sl_31_df,sl_32_df,sl_33_df,sl_34_df,sl_35_df,sl_36_df]


	"""
	# All Links with All Modes Counts
	count_df     = df[df['Tot_Count'].notna()]
	# Internal/External
	internal_df  = count_df[(count_df['EXTERNAL'] == 0)]
	external_df  = count_df[(count_df['EXTERNAL'] == 1)]
	# By AADT Volume
	under_5k_df     = count_df[(count_df['AADT'] < 5000)]
	btwn_5_10k_df   = count_df[(count_df['AADT'] >= 5000) & (count_df['AADT'] < 10000)]
	btwn_10_15k_df  = count_df[(count_df['AADT'] >= 10000) & (count_df['AADT'] < 15000)]
	btwn_15_20k_df  = count_df[(count_df['AADT'] >= 15000) & (count_df['AADT'] < 20000)]
	btwn_20_30k_df  = count_df[(count_df['AADT'] >= 20000) & (count_df['AADT'] < 30000)]
	btwn_30_40k_df  = count_df[(count_df['AADT'] >= 30000) & (count_df['AADT'] < 40000)]
	btwn_40_50k_df  = count_df[(count_df['AADT'] >= 40000) & (count_df['AADT'] < 50000)]
	over_50k_df     = count_df[(count_df['AADT'] >= 50000)]
	# By Functional Class
	fc1_df    = count_df[(count_df['FCLASS'] == 1)]
	fc2_df    = count_df[(count_df['FCLASS'] == 2)]
	fc4_df    = count_df[(count_df['FCLASS'] == 4)]
	fc6_df    = count_df[(count_df['FCLASS'] == 6)]
	fc7_df    = count_df[(count_df['FCLASS'] == 7)]
	fc8_df    = count_df[(count_df['FCLASS'] == 8)]
	fc9_df    = count_df[(count_df['FCLASS'] == 9)]
	fc11_df   = count_df[(count_df['FCLASS'] == 11)]
	fc12_df   = count_df[(count_df['FCLASS'] == 12)]
	fc14_df   = count_df[(count_df['FCLASS'] == 14)]
	fc16_df   = count_df[(count_df['FCLASS'] == 16)]
	fc17_df   = count_df[(count_df['FCLASS'] == 17)]
	fc18_df   = count_df[(count_df['FCLASS'] == 18)]
	fc19_df   = count_df[(count_df['FCLASS'] == 19)]
	fcramp_df = count_df[(count_df['FCLASS'] == 50) | (count_df['FCLASS'] == 53) | (count_df['FCLASS'] == 54)]
	# By Screenline
	sl_1_df   = count_df[(count_df['SCRNLINE1'] == 1) | (count_df['SCRNLINE2'] == 1)]
	sl_2_df   = count_df[(count_df['SCRNLINE1'] == 2) | (count_df['SCRNLINE2'] == 2)]
	sl_3_df   = count_df[(count_df['SCRNLINE1'] == 3) | (count_df['SCRNLINE2'] == 3)]
	sl_4_df   = count_df[(count_df['SCRNLINE1'] == 4) | (count_df['SCRNLINE2'] == 4)]
	sl_5_df   = count_df[(count_df['SCRNLINE1'] == 5) | (count_df['SCRNLINE2'] == 5)]
	sl_6_df   = count_df[(count_df['SCRNLINE1'] == 6) | (count_df['SCRNLINE2'] == 6)]
	sl_8_df   = count_df[(count_df['SCRNLINE1'] == 8) | (count_df['SCRNLINE2'] == 8)]
	sl_9_df   = count_df[(count_df['SCRNLINE1'] == 9) | (count_df['SCRNLINE2'] == 9)]
	sl_10_df   = count_df[(count_df['SCRNLINE1'] == 10) | (count_df['SCRNLINE2'] == 10)]
	sl_11_df   = count_df[(count_df['SCRNLINE1'] == 11) | (count_df['SCRNLINE2'] == 11)]
	sl_12_df   = count_df[(count_df['SCRNLINE1'] == 12) | (count_df['SCRNLINE2'] == 12)]
	sl_13_df   = count_df[(count_df['SCRNLINE1'] == 13) | (count_df['SCRNLINE2'] == 13)]
	sl_14_df   = count_df[(count_df['SCRNLINE1'] == 14) | (count_df['SCRNLINE2'] == 14)]
	sl_20_df   = count_df[(count_df['SCRNLINE1'] == 20) | (count_df['SCRNLINE2'] == 20)]
	sl_21_df   = count_df[(count_df['SCRNLINE1'] == 21) | (count_df['SCRNLINE2'] == 21)]
	sl_22_df   = count_df[(count_df['SCRNLINE1'] == 22) | (count_df['SCRNLINE2'] == 22)]
	sl_23_df   = count_df[(count_df['SCRNLINE1'] == 23) | (count_df['SCRNLINE2'] == 23)]
	sl_24_df   = count_df[(count_df['SCRNLINE1'] == 24) | (count_df['SCRNLINE2'] == 24)]
	sl_28_df   = count_df[(count_df['SCRNLINE1'] == 28) | (count_df['SCRNLINE2'] == 28)]
	sl_29_df   = count_df[(count_df['SCRNLINE1'] == 29) | (count_df['SCRNLINE2'] == 29)]
	sl_30_df   = count_df[(count_df['SCRNLINE1'] == 30) | (count_df['SCRNLINE2'] == 30)]
	sl_31_df   = count_df[(count_df['SCRNLINE1'] == 31) | (count_df['SCRNLINE2'] == 31)]
	sl_32_df   = count_df[(count_df['SCRNLINE1'] == 32) | (count_df['SCRNLINE2'] == 32)]
	sl_33_df   = count_df[(count_df['SCRNLINE1'] == 33) | (count_df['SCRNLINE2'] == 33)]
	sl_34_df   = count_df[(count_df['SCRNLINE1'] == 34) | (count_df['SCRNLINE2'] == 34)]
	sl_35_df   = count_df[(count_df['SCRNLINE1'] == 35) | (count_df['SCRNLINE2'] == 35)]
	sl_36_df   = count_df[(count_df['SCRNLINE1'] == 36) | (count_df['SCRNLINE2'] == 36)]
	
	
	# Build list of dataframes to loop thru
	allmodes_df_list = [count_df,internal_df,external_df,
				under_5k_df,btwn_5_10k_df,btwn_10_15k_df,btwn_15_20k_df,btwn_20_30k_df,btwn_30_40k_df,btwn_40_50k_df,over_50k_df,
				fc1_df,fc2_df,fc4_df,fc6_df,fc7_df,fc8_df,fc9_df,fc11_df,fc12_df,fc14_df,fc16_df,fc17_df,fc18_df,fc19_df,fcramp_df,
				sl_1_df,sl_2_df,sl_3_df,sl_4_df,sl_5_df,sl_6_df,sl_8_df,sl_9_df,sl_10_df,sl_11_df,sl_12_df,sl_13_df,sl_14_df,sl_20_df,
				sl_21_df,sl_22_df,sl_23_df,sl_24_df,sl_28_df,sl_29_df,sl_30_df,sl_31_df,sl_32_df,sl_33_df,sl_34_df,sl_35_df,sl_36_df]
	"""
	
	
	

	# Add squared error column to each df 
	# Auto
	for i in auto_df_list:
		i['Auto_SqError']  = (i.Auto_Flow - i.Auto_Count)**2	
	# Truck          
	# for i in truck_df_list:
	# 	i['Truck_SqError'] = (i.Truck_Flow - i.Truck_Count)**2
	# # All Modes          
	# for i in allmodes_df_list:
	# 	i['Tot_SqError']   = (i.Tot_Flow - i.Tot_Count)**2
			
			
	# Calculate Auto pct error and pct rmse from each dataframe and save in results dataframe
	y = 0
	for i in auto_df_list:
		if len(i) == 0: #(sum(i.Auto_Flow) == 0) | (sum(i.Auto_Count) == 0): #len(i) == 0:
			results_df.at[y,"Number of Observations"] = 0
			y = y + 1
			continue
		else:
			results_df.at[y,"Percent Error"]                  = pct_error(i.Auto_Count,i.Auto_Flow)
			results_df.at[y,"Percent RMSE"]                   = pct_rmse(i.Auto_Count,i.Auto_SqError)
			results_df.at[y,"Number of Observations"]         = len(i)
			results_df.at[y,"Sum of Counts"]                  = np.sum(i.Auto_Count)
			results_df.at[y,"Mean of Counts"]                 = np.mean(i.Auto_Count)
			results_df.at[y,"Median of Counts"]               = np.median(i.Auto_Count)
			results_df.at[y,"Count VMT, Links with Counts"]   = vmt(i.Auto_Count,i.LENGTH)
			results_df.at[y,'Modeled VMT, Links with Counts'] = vmt(i.Auto_Flow,i.LENGTH)
			y = y + 1
	
	# Calculate Truck pct error and pct rmse from each dataframe and save in results dataframe
	# y = 53
	# for i in truck_df_list:
	# 	if len(i) == 0: #(sum(i.Truck_Flow) == 0) | (sum(i.Truck_Count) == 0):
	# 		results_df.at[y,"Number of Observations"] = 0
	# 		y = y + 1
	# 		continue
	# 	else:
	# 		results_df.at[y,"Percent Error"]                  = pct_error(i.Truck_Count,i.Truck_Flow)
	# 		results_df.at[y,"Percent RMSE"]                   = pct_rmse(i.Truck_Count, i.Truck_SqError)
	# 		results_df.at[y,"Number of Observations"]         = len(i)
	# 		results_df.at[y,"Sum of Counts"]                  = np.sum(i.Truck_Count)
	# 		results_df.at[y,"Mean of Counts"]                 = np.mean(i.Truck_Count)
	# 		results_df.at[y,"Median of Counts"]               = np.median(i.Truck_Count)
	# 		results_df.at[y,"Count VMT, Links with Counts"]   = vmt(i.Truck_Count,i.LENGTH)
	# 		results_df.at[y,'Modeled VMT, Links with Counts'] = vmt(i.Truck_Flow,i.LENGTH)
	# 		y = y + 1     

	# Calculate All Modes pct error and pct rmse from each dataframe and save in results dataframe
	# y = 106
	# for i in allmodes_df_list:
	# 	if len(i) == 0: #(sum(i.Tot_Flow) == 0) | (sum(i.Tot_Count) == 0):
	# 		results_df.at[y,"Number of Observations"] = 0
	# 		y = y + 1
	# 		continue
	# 	else:
	# 		results_df.at[y,"Percent Error"]                  = pct_error(i.Tot_Count,i.Tot_Flow)
	# 		results_df.at[y,"Percent RMSE"]                   = pct_rmse(i.Tot_Count,i.Tot_SqError)
	# 		results_df.at[y,"Number of Observations"]         = len(i)
	# 		results_df.at[y,"Sum of Counts"]                  = np.sum(i.Tot_Count)
	# 		results_df.at[y,"Mean of Counts"]                 = np.mean(i.Tot_Count)
	# 		results_df.at[y,"Median of Counts"]               = np.median(i.Tot_Count)
	# 		results_df.at[y,"Count VMT, Links with Counts"]   = vmt(i.Tot_Count,i.LENGTH)
	# 		results_df.at[y,'Modeled VMT, Links with Counts'] = vmt(i.Tot_Flow,i.LENGTH)
	# 		y = y + 1     
	
	
	# Total VMT and Total VHT
	
	# Import ID fields and fields with Counts and Flows
	# Link ID fields
	NO          = VisumPy.helpers.GetMulti(Visum.Net.Links,"No", activeOnly = True)
	FCLASS      = VisumPy.helpers.GetMulti(Visum.Net.Links,"TYPENO", activeOnly = True)
	EXT         = VisumPy.helpers.GetMulti(Visum.Net.Links,"EXT_COUNT", activeOnly = True)
	SCRNLINE    = VisumPy.helpers.GetMulti(Visum.Net.Links,r"CONCATENATE:SCREENLINES\CODE", activeOnly = True)
	LENGTH       = VisumPy.helpers.GetMulti(Visum.Net.Links,"Length", activeOnly = True)
	# Pull CONGTIME Auto by period, Length, and Flows by Period for Total VMT/Total VHT Calculations
	CONGTIME_AM_C   = VisumPy.helpers.GetMulti(Visum.Net.Links,"AM_CTIME", activeOnly = True)
	# CONGTIME_MD_C   = VisumPy.helpers.GetMulti(Visum.Net.Links,"MD_CTIME_C", activeOnly = True)
	CONGTIME_PM_C   = VisumPy.helpers.GetMulti(Visum.Net.Links,"PM_CTIME", activeOnly = True)
	CONGTIME_PMPK_C   = VisumPy.helpers.GetMulti(Visum.Net.Links,"PMPK_CTIME", activeOnly = True)
	CONGTIME_OP_C   = VisumPy.helpers.GetMulti(Visum.Net.Links,"OP_CTIME", activeOnly = True)
	CONGTIME_PER_C  = VisumPy.helpers.GetMulti(Visum.Net.Links,cong_auto_time, activeOnly = True)
	# # Pull CONGTIME Truck by period, Length, and Flows by Period for Total VMT/Total VHT Calculations 	
	# CONGTIME_AM_T   = VisumPy.helpers.GetMulti(Visum.Net.Links,"AM3_CTIME_T", activeOnly = True)
	# CONGTIME_MD_T   = VisumPy.helpers.GetMulti(Visum.Net.Links,"MD_CTIME_T", activeOnly = True)
	# CONGTIME_PM_T   = VisumPy.helpers.GetMulti(Visum.Net.Links,"PM3_CTIME_T", activeOnly = True)
	# CONGTIME_NI_T   = VisumPy.helpers.GetMulti(Visum.Net.Links,"NI_CTIME_T", activeOnly = True)
	# CONGTIME_PER_T  = VisumPy.helpers.GetMulti(Visum.Net.Links,cong_trk_time, activeOnly = True)
	
	# Link Flows by Period
	# AM
	AM_Auto_Flow  = VisumPy.helpers.GetMulti(Visum.Net.Links,"AM_AUTO_VOLUME", activeOnly = True)
	
	# PM
	PM_Auto_Flow  = VisumPy.helpers.GetMulti(Visum.Net.Links,"PM_AUTO_VOLUME", activeOnly = True)
	PMPeak_Auto_Flow  = VisumPy.helpers.GetMulti(Visum.Net.Links,"PMPK_AUTO_VOLUME", activeOnly = True)
	
	# OP
	OP_Auto_Flow  = VisumPy.helpers.GetMulti(Visum.Net.Links,"OP_AUTO_VOLUME", activeOnly = True)

	# Period for functions
	Auto_Flow  = VisumPy.helpers.GetMulti(Visum.Net.Links,auto_flow, activeOnly = True)
	#Truck_Flow = VisumPy.helpers.GetMulti(Visum.Net.Links,truck_flow, activeOnly = True)
	#Tot_Flow   = VisumPy.helpers.GetMulti(Visum.Net.Links,all_flow, activeOnly = True)
	#LCV_Flow   = VisumPy.helpers.GetMulti(Visum.Net.Links,lcv_flow, activeOnly = True)
	
	
	# Make Visum list with link data
	summary_list = [NO, FCLASS, EXT, SCRNLINE, LENGTH, 
					CONGTIME_AM_C, CONGTIME_PM_C, CONGTIME_PMPK_C, CONGTIME_OP_C, CONGTIME_PER_C,
					AM_Auto_Flow, PM_Auto_Flow, PMPeak_Auto_Flow, OP_Auto_Flow, Auto_Flow]
			
	# Put Visum link list into dataframe  
	df_all = pd.DataFrame(np.column_stack(summary_list), columns = ['NO', 'FCLASS', 'EXTERNAL', 'SCRNLINE', 'LENGTH', 
																'CONGTIME_AM_C', 'CONGTIME_PM_C', 'CONGTIME_PMPK_C', 'CONGTIME_OP_C', 'CONGTIME_PER_C',
																'AM_Auto_Flow', 'PM_Auto_Flow', 'PMPeak_Auto_Flow', 'OP_Auto_Flow', 'Auto_Flow'])
																
	# Break out SCRNLINE field to separate by commas into individual columns
	df_all = pd.concat([df_all,df_all['SCRNLINE'].str.split(',', expand = True)], axis = 1)
	# Change Screenline field names
	df_all = df_all.rename(columns = {0:'SCRNLINE1',1:'SCRNLINE2'})
	
	# Replace null values with 0 in the screenline fields
	df_all['SCRNLINE1'] = df_all['SCRNLINE1'].replace('',np.nan).fillna(0)
	# df_all['SCRNLINE2'] = df_all['SCRNLINE2'].replace('',np.nan).fillna(0)


	# Convert all flow and time fields to float to make multiplication and other operations run smoothly. Read in as strings
	df_all[['NO','EXTERNAL', 'FCLASS','SCRNLINE1','LENGTH',
		'CONGTIME_AM_C', 'CONGTIME_PM_C', 'CONGTIME_PMPK_C', 'CONGTIME_OP_C', 'CONGTIME_PER_C',
		'AM_Auto_Flow', 'PM_Auto_Flow', 'PMPeak_Auto_Flow', 'OP_Auto_Flow', 'Auto_Flow']] = df_all[['NO','EXTERNAL', 'FCLASS','SCRNLINE1','LENGTH',
		'CONGTIME_AM_C', 'CONGTIME_PM_C', 'CONGTIME_PMPK_C', 'CONGTIME_OP_C', 'CONGTIME_PER_C',
		'AM_Auto_Flow', 'PM_Auto_Flow', 'PMPeak_Auto_Flow', 'OP_Auto_Flow', 'Auto_Flow']].astype(float) # ,'SCRNLINE2'
	# Convert ID fields to integer
	df_all[['NO','EXTERNAL','FCLASS','SCRNLINE1']] = df_all[['NO','EXTERNAL','FCLASS','SCRNLINE1']].astype(int) # ,'SCRNLINE2'
	

	
	# For links with counts only, used for Pct. Error and Pct. RMSE
	# Filter out links where count is null and by each condition
	# All Links with Counts
	count_df_all     = df_all
	# Internal/External
	internal_df_all  = df_all[(df_all['EXTERNAL'] == 0)]
	external_df_all  = df_all[(df_all['EXTERNAL'] == 1)]
	# By AADT Volume
	under_5k_df_all     = df_all[df_all['NO'].isin(under_5k_df['NO'])]
	btwn_5_10k_df_all   = df_all[df_all['NO'].isin(btwn_5_10k_df['NO'])]
	btwn_10_15k_df_all  = df_all[df_all['NO'].isin(btwn_10_15k_df['NO'])]
	btwn_15_20k_df_all  = df_all[df_all['NO'].isin(btwn_15_20k_df['NO'])]
	btwn_20_30k_df_all  = df_all[df_all['NO'].isin(btwn_20_30k_df['NO'])]
	btwn_30_40k_df_all  = df_all[df_all['NO'].isin(btwn_30_40k_df['NO'])]
	btwn_40_50k_df_all  = df_all[df_all['NO'].isin(btwn_40_50k_df['NO'])]
	over_50k_df_all     = df_all[df_all['NO'].isin(over_50k_df['NO'])]	
	# By Functional Class
	fc1_df_all    = df_all[(df_all['FCLASS'] == 1)]
	fc2_df_all    = df_all[(df_all['FCLASS'] == 2)]
	fc4_df_all    = df_all[(df_all['FCLASS'] == 4)]
	fc6_df_all    = df_all[(df_all['FCLASS'] == 6)]
	fc7_df_all    = df_all[(df_all['FCLASS'] == 7)]
	fc8_df_all    = df_all[(df_all['FCLASS'] == 8)]
	fc9_df_all    = df_all[(df_all['FCLASS'] == 9)]
	fc11_df_all   = df_all[(df_all['FCLASS'] == 11)]
	fc12_df_all   = df_all[(df_all['FCLASS'] == 12)]
	fc14_df_all   = df_all[(df_all['FCLASS'] == 14)]
	fc16_df_all   = df_all[(df_all['FCLASS'] == 16)]
	fc17_df_all   = df_all[(df_all['FCLASS'] == 17)]
	fc18_df_all   = df_all[(df_all['FCLASS'] == 18)]
	fc19_df_all   = df_all[(df_all['FCLASS'] == 19)]
	fcramp_df_all = df_all[(df_all['FCLASS'] == 50) | (df_all['FCLASS'] == 53) | (df_all['FCLASS'] == 54)]
	# By Screenline
	sl_1_df_all   = df_all[(df_all['SCRNLINE1'] == 1)] #| (df_all['SCRNLINE2'] == 1)]
	sl_2_df_all   = df_all[(df_all['SCRNLINE1'] == 2)] #| (df_all['SCRNLINE2'] == 2)]
	sl_3_df_all   = df_all[(df_all['SCRNLINE1'] == 3)] #| (df_all['SCRNLINE2'] == 3)]
	sl_4_df_all   = df_all[(df_all['SCRNLINE1'] == 4)] #| (df_all['SCRNLINE2'] == 4)]
	sl_5_df_all   = df_all[(df_all['SCRNLINE1'] == 5)] #| (df_all['SCRNLINE2'] == 5)]
	sl_6_df_all   = df_all[(df_all['SCRNLINE1'] == 6)] #| (df_all['SCRNLINE2'] == 6)]
	sl_8_df_all   = df_all[(df_all['SCRNLINE1'] == 8)] #| (df_all['SCRNLINE2'] == 8)]
	sl_9_df_all   = df_all[(df_all['SCRNLINE1'] == 9)] #| (df_all['SCRNLINE2'] == 9)]
	sl_10_df_all   = df_all[(df_all['SCRNLINE1'] == 10)] # | (df_all['SCRNLINE2'] == 10)]
	sl_11_df_all   = df_all[(df_all['SCRNLINE1'] == 11)] # | (df_all['SCRNLINE2'] == 11)]
	sl_12_df_all   = df_all[(df_all['SCRNLINE1'] == 12)] # | (df_all['SCRNLINE2'] == 12)]
	sl_13_df_all   = df_all[(df_all['SCRNLINE1'] == 13)] # | (df_all['SCRNLINE2'] == 13)]
	sl_14_df_all   = df_all[(df_all['SCRNLINE1'] == 14)] # | (df_all['SCRNLINE2'] == 14)]
	sl_20_df_all   = df_all[(df_all['SCRNLINE1'] == 20)] # | (df_all['SCRNLINE2'] == 20)]
	sl_21_df_all   = df_all[(df_all['SCRNLINE1'] == 21)] # | (df_all['SCRNLINE2'] == 21)]
	sl_22_df_all   = df_all[(df_all['SCRNLINE1'] == 22)] # | (df_all['SCRNLINE2'] == 22)]
	sl_23_df_all   = df_all[(df_all['SCRNLINE1'] == 23)] # | (df_all['SCRNLINE2'] == 23)]
	sl_24_df_all   = df_all[(df_all['SCRNLINE1'] == 24)] # | (df_all['SCRNLINE2'] == 24)]
	sl_28_df_all   = df_all[(df_all['SCRNLINE1'] == 28)] # | (df_all['SCRNLINE2'] == 28)]
	sl_29_df_all   = df_all[(df_all['SCRNLINE1'] == 29)] # | (df_all['SCRNLINE2'] == 29)]
	sl_30_df_all   = df_all[(df_all['SCRNLINE1'] == 30)] # | (df_all['SCRNLINE2'] == 30)]
	sl_31_df_all   = df_all[(df_all['SCRNLINE1'] == 31)] # | (df_all['SCRNLINE2'] == 31)]
	sl_32_df_all   = df_all[(df_all['SCRNLINE1'] == 32)] # | (df_all['SCRNLINE2'] == 32)]
	sl_33_df_all   = df_all[(df_all['SCRNLINE1'] == 33)] # | (df_all['SCRNLINE2'] == 33)]
	sl_34_df_all   = df_all[(df_all['SCRNLINE1'] == 34)] # | (df_all['SCRNLINE2'] == 34)]
	sl_35_df_all   = df_all[(df_all['SCRNLINE1'] == 35)] # | (df_all['SCRNLINE2'] == 35)]
	sl_36_df_all   = df_all[(df_all['SCRNLINE1'] == 36)] # | (df_all['SCRNLINE2'] == 36)]
	
	# Build list of dataframes to loop thru
	df_list_all = [count_df_all,internal_df_all,external_df_all,
					under_5k_df_all,btwn_5_10k_df_all,btwn_10_15k_df_all,btwn_15_20k_df_all,btwn_20_30k_df_all,btwn_30_40k_df_all,btwn_40_50k_df_all,over_50k_df_all,
					fc1_df_all,fc2_df_all,fc4_df_all,fc6_df_all,fc7_df_all,fc8_df_all,fc9_df_all,fc11_df_all,fc12_df_all,fc14_df_all,fc16_df_all,fc17_df_all,fc18_df_all,fc19_df_all,fcramp_df_all]
					# sl_1_df_all,sl_2_df_all,sl_3_df_all,sl_4_df_all,sl_5_df_all,sl_6_df_all,sl_8_df_all,sl_9_df_all,sl_10_df_all,sl_11_df_all,sl_12_df_all,sl_13_df_all,sl_14_df_all,
					# sl_20_df_all,sl_21_df_all,sl_22_df_all,sl_23_df_all,sl_24_df_all,sl_28_df_all,sl_29_df_all,sl_30_df_all,sl_31_df_all,sl_32_df_all,sl_33_df_all,sl_34_df_all,
					# sl_35_df_all,sl_36_df_all]
	
	
	# Calculate Auto Total VMT and Total VHT from each dataframe and save in results dataframe
	y = 0
	for i in df_list_all:
		if len(i) == 0: #sum(i.Auto_Flow) == 0.0:
			y = y + 1
			continue
		else:
			results_df.at[y,"Total VMT"]           = vmt(i.Auto_Flow,i.LENGTH)
			if period == 'Daily':
				results_df.at[y,"Total VHT"]       = vht_dly(i.AM_Auto_Flow,i.CONGTIME_AM_C,i.PM_Auto_Flow,i.CONGTIME_PM_C,i.OP_Auto_Flow,i.CONGTIME_OP_C) # sum of 3 periods
			else:
				results_df.at[y,"Total VHT"]       = vht_per(i.Auto_Flow,i.CONGTIME_PER_C)
			y = y + 1

	# Calculate Truck Total VMT and Total VHT from each dataframe and save in results dataframe
	# y = 53
	# for i in df_list_all:
	# 	if len(i) == 0: #sum(i.Truck_Flow) == 0.0:
	# 		y = y + 1
	# 		continue
	# 	else:
	# 		results_df.at[y,"Total VMT"]           = vmt(i.Truck_Flow,i.LENGTH)
	# 		if period == 'Daily':
	# 			results_df.at[y,"Total VHT"]       = vht_dly(i.AM_Truck_Flow,i.CONGTIME_AM_T,i.MD_Truck_Flow,i.CONGTIME_MD_T,i.PM_Truck_Flow,i.CONGTIME_PM_T,i.NI_Truck_Flow,i.CONGTIME_NI_T)
	# 		else:
	# 			results_df.at[y,"Total VHT"]       = vht_per(i.Truck_Flow,i.CONGTIME_PER_T)
	# 		y = y + 1     
	
	# Calculate All Modes Total VMT and Total VHT from each dataframe and save in results dataframe
	# y = 106
	# a = 0
	# t = 53
	# for i in df_list_all:
	# 	if len(i) == 0: #sum(i.Tot_Flow) == 0.0:
	# 		y = y + 1
	# 		continue
	# 	else:
	# 		results_df.at[y,"Total VMT"]           = results_df.at[a,"Total VMT"] + results_df.at[t,"Total VMT"]       # vmt(i.Tot_Flow,i.LENGTH)
	# 		results_df.at[y,"Total VHT"]           = results_df.at[a,"Total VHT"] + results_df.at[t,"Total VHT"]       # vht(i.AM_Tot_Flow,i.CONGTIME_AM,i.MD_Tot_Flow,i.CONGTIME_MD,i.PM_Tot_Flow,i.CONGTIME_PM,i.NI_Tot_Flow,i.CONGTIME_NT)
	# 		y = y + 1
	# 		a = a + 1
	# 		t = t + 1
	# 
	# 
	# # Calculate LCV Total VMT and save in results dataframe
	# y = 159
	# results_df.at[y,"Total VMT"]           = vmt(count_df_all.LCV_Flow,count_df_all.LENGTH)  
	
	
	# Format fields in results_df before export
	# Create formatting function for large numbers (no decimals and thousand commas)
	#def format_commas(column):
	#	return column.apply(lambda x: '{:,.0f}'.format(x) if pd.notna(x) else None)
	# Create formatting function for percentages (2 decimals and percent symbol, also multiplies by 100)
	#def format_percent(column):
	#	return column.apply(lambda x: '{:.2%}'.format(x) if pd.notna(x) else None)

	# Apply the formatting function to specific columns
	results_df['Percent Error'] = format_percent(results_df['Percent Error'])
	results_df['Percent RMSE'] = format_percent(results_df['Percent RMSE'])	
	results_df['Total VMT'] = format_commas(results_df['Total VMT'])
	results_df['Total VHT'] = format_commas(results_df['Total VHT'])
	results_df['Number of Observations'] = format_commas(results_df['Number of Observations'])
	results_df['Sum of Counts'] = format_commas(results_df['Sum of Counts'])
	results_df['Mean of Counts'] = format_commas(results_df['Mean of Counts'])
	results_df['Median of Counts'] = format_commas(results_df['Median of Counts'])
	results_df['Count VMT, Links with Counts'] = format_commas(results_df['Count VMT, Links with Counts'])
	results_df['Modeled VMT, Links with Counts'] = format_commas(results_df['Modeled VMT, Links with Counts'])

	
	# Save Daily Summary file to new timestamped folder
	results_df.to_csv(outputs_path + "/ModelRun_"+date+"/Assignment Results/" + period + "_Summary.csv")


# Count station summary (goes in Model Reporting folder)
def count_station_summary():
	
	# Pull template
	count_summary_df = pd.read_csv(settings_dict['SharedData_Path']+'/template/reporting/count_station_summary_template.csv')
	
	# Set indices for count_summary_df source column
	count_summary_df.set_index('Source', inplace=True)
	
	# Pull count station fields
	master_id = VisumPy.helpers.GetMulti(Visum.Net.CountLocations,"MASTER_LOCAL_ID")
	source    = VisumPy.helpers.GetMulti(Visum.Net.CountLocations,"Code")
	auto_dly  = VisumPy.helpers.GetMulti(Visum.Net.CountLocations,"AUTO_DLY_COUNT") # For class/non-class check
	
	# Make Visum list with link data
	summary_list = [master_id, source, auto_dly]    
			
	# Put Visum link list into dataframe
	df = pd.DataFrame(np.column_stack(summary_list), columns = ['MASTER_ID','SOURCE', 'AUTO_DLY'])
	
	# Define custom_sum function to maintain null values when aggregating Counts and Flows by LinkNO
	def custom_sum(series):
    # If all values are null, return null; otherwise, return the sum of the values
		return series.sum() if series.notna().any() else None																												
	# GROUP EACH DATAFRAME BY 'MASTER_ID' COLUMN TO GET 1 CODE/AUTO_DLY FOR EACH STATION
	df = df.groupby('MASTER_ID').agg(
		SOURCE      =('SOURCE', 'first'),
		AUTO_DLY    =('AUTO_DLY', custom_sum)
	).reset_index()
	
	# Take counts by row: source and class/non-class to fill summary table
	# MS2
	count_summary_df.at['MS2','Class']     = df[df['AUTO_DLY'].notnull() & (df['SOURCE'] == 'MS2')].shape[0]
	count_summary_df.at['MS2','Non-Class'] = df[df['AUTO_DLY'].isnull()  & (df['SOURCE'] == 'MS2')].shape[0]
	count_summary_df.at['MS2','Total']     = df[df['SOURCE'] == 'MS2'].shape[0]
	# DKS
	count_summary_df.at['DKS','Class']     = df[df['AUTO_DLY'].notnull() & ((df['SOURCE'] == 'DKS') | (df['SOURCE'] == 'DKS_2'))].shape[0]
	count_summary_df.at['DKS','Non-Class'] = df[df['AUTO_DLY'].isnull()  & ((df['SOURCE'] == 'DKS') | (df['SOURCE'] == 'DKS_2'))].shape[0]
	count_summary_df.at['DKS','Total']     = df[((df['SOURCE'] == 'DKS') | (df['SOURCE'] == 'DKS_2'))].shape[0]
	# COS
	count_summary_df.at['COS','Class'] = df[df['AUTO_DLY'].notnull() & (df['SOURCE'] == 'COS')].shape[0]
	count_summary_df.at['COS','Non-Class'] = df[df['AUTO_DLY'].isnull()  & (df['SOURCE'] == 'COS')].shape[0]
	count_summary_df.at['COS','Total'] = df[df['SOURCE'] == 'COS'].shape[0]
	# Total
	count_summary_df.at['Total','Class'] = df[df['AUTO_DLY'].notnull()].shape[0]
	count_summary_df.at['Total','Non-Class'] = df[df['AUTO_DLY'].isnull()].shape[0]
	count_summary_df.at['Total','Total'] = len(df)

	# Format sum fields to have commas and rounding
	count_summary_df['Class'] = format_commas(count_summary_df['Class'])
	count_summary_df['Non-Class'] = format_commas(count_summary_df['Non-Class'])
	count_summary_df['Total'] = format_commas(count_summary_df['Total'])
	
	# Export count_summary_df to csv file in timestamped folder
	count_summary_df.to_csv(outputs_path + "/ModelRun_"+date+"/Model Reporting/Count_Station_Summary.csv")

# Model reporting function
def model_reporting():

	
	# 1: MATRIX SUMMARY
	
	# Pull matrix list from Visum
	NO   = VisumPy.helpers.GetMulti(Visum.Net.Matrices,"No")
	Code = VisumPy.helpers.GetMulti(Visum.Net.Matrices,"Code")
	Name = VisumPy.helpers.GetMulti(Visum.Net.Matrices,"Name")
	Type = VisumPy.helpers.GetMulti(Visum.Net.Matrices,"MatrixType")
	Sum  = VisumPy.helpers.GetMulti(Visum.Net.Matrices,"Sum")
	
	# Make Visum list with link data
	mtx_list = [NO,Code,Name,Type,Sum]    
			
	# Put Visum link list into dataframe  
	mtx_df = pd.DataFrame(np.column_stack(mtx_list), columns = ['Num','Code','Name','Type','Sum'])
	
	# Convert Sum to float for later functionality
	mtx_df[['Sum']] = mtx_df[['Sum']].astype(float)
	
	# Remove skim matrices and drop Type field
	mtx_df = mtx_df[(mtx_df['Type'] == 'MATRIXTYPE_DEMAND')]
	mtx_df = mtx_df.drop('Type', axis=1)
	
	# Export mtx_df to csv file in timestamped folder
	mtx_df.to_csv(outputs_path + "/ModelRun_"+date+"/Model Reporting/unformatted/matrixsummary.csv")
	
	# Copy mtx_df to mtx_df_export to maintain formatting for future operations in mtx_df
	mtx_df_export = mtx_df.copy()
	
	# Format sum field to have commas and rounding
	mtx_df_export['Sum'] = format_commas(mtx_df_export['Sum'])
	
	# Export mtx_df_export to csv file in timestamped folder
	mtx_df_export.to_csv(outputs_path + "/ModelRun_"+date+"/Model Reporting/Matrix_Summary.csv")
	
	
	
	# 2: TOTAL PERSON TRIPS
	
	# Import blank dataframe with purposes and totals columns from template folder
	tot_person_trips_df = pd.read_csv(settings_dict['SharedData_Path']+'/template/reporting/total_internal_person_trips_template.csv')

	# Set indices for tot_person_trips_df and mtx_df to use the purpose/code columns
	tot_person_trips_df.set_index('Purpose', inplace=True)
	mtx_df.set_index('Code', inplace=True)
	
	# Create purpose list
	purpose = ['HBW','HBSch','HBR','HBO','HBC','NHB']
	# Fill Sum of Income Matrices in tot_person_trips_df from mtx_df sum column
	for i in purpose:
		if i+'1' in mtx_df.index:
			tot_person_trips_df.at[i,'Sum of Income Matrices'] = mtx_df.at[i+'1','Sum'] + mtx_df.at[i+'2','Sum'] + mtx_df.at[i+'3','Sum'] + mtx_df.at[i+'4','Sum']
		else:
			tot_person_trips_df.at[i,'Sum of Income Matrices'] = 0
			continue
	# Fill Sum of Total Matrices in tot_person_trips_df from mtx_df sum column
	for i in purpose:
		if i in mtx_df.index:
			tot_person_trips_df.at[i,'Sum of Total Matrix'] = mtx_df.at[i,'Sum']
		else:
			tot_person_trips_df.at[i,'Sum of Total Matrix'] = 0
			continue
	
	# Export tot_person_trips_df to csv file in timestamped folder
	tot_person_trips_df.to_csv(outputs_path + "/ModelRun_"+date+"/Model Reporting/unformatted/totalinternalpersontrips.csv")
	
	# Copy tot_person_trips_df to tot_person_trips_df_export to maintain formatting for future operations in tot_person_trips_df
	tot_person_trips_df_export = tot_person_trips_df.copy()
	
	# Format sum fields to have commas and rounding
	tot_person_trips_df_export['Sum of Income Matrices'] = format_commas(tot_person_trips_df_export['Sum of Income Matrices'])
	tot_person_trips_df_export['Sum of Total Matrix']    = format_commas(tot_person_trips_df_export['Sum of Total Matrix'])
			
	# Export tot_person_trips_df_export to csv file in timestamped folder
	tot_person_trips_df_export.to_csv(outputs_path + "/ModelRun_"+date+"/Model Reporting/Total_Internal_Person_Trips_Summary.csv")
	
	
	
	# 3: TRIPS BY MODE
	
	# Import blank dataframe with purposes and totals columns from template folder
	trips_by_mode_df = pd.read_csv(settings_dict['SharedData_Path']+'/template/reporting/trips_by_mode_template.csv')
	
	# Set indices for trips_by_mode_df code column
	trips_by_mode_df.set_index('Mode', inplace=True)
	
	# Create purpose and mode lists
	purpose = ['HBW','HBSch','HBR','HBO','HBC','NHB']
	mode    = ['DA','SR','WLK_Bus','DRV_Bus','WLK_Rail','DRV_Rail','Walk','Bike']
	# Fill Sum of Matrices in trips_by_mode_df from mtx_df sum column
	for p in purpose:
		for m in mode:
			if p+'_'+m+'_Inc1' in mtx_df.index:
				trips_by_mode_df.at[m,p] = mtx_df.at[p+'_'+m+'_Inc1','Sum'] + mtx_df.at[p+'_'+m+'_Inc2','Sum'] + mtx_df.at[p+'_'+m+'_Inc3','Sum'] + mtx_df.at[p+'_'+m+'_Inc4','Sum']
			else:
				trips_by_mode_df.at[m,p] = mtx_df.at[p+'_'+m,'Sum']
			continue
	# Fill Sum of Matrices in trips_by_mode_df from mtx_df sum column for School Bus (HBSch only)
	for p in purpose:
		if p != 'HBSch':
			trips_by_mode_df.at['SchoolBus',p] = 0
		else:
			trips_by_mode_df.at['SchoolBus',p] = mtx_df.at['HBSch_SchoolBus','Sum']
	
	# External purposes (all trips DA)
	all_modes = ['DA','SR','WLK_Bus','DRV_Bus','WLK_Rail','DRV_Rail','SchoolBus','Walk','Bike']
	ext_purps = ['Auto_XI_W','Auto_IX_W','Auto_XIIX_NW','Auto_XX']
	for p in ext_purps:
		for m in all_modes:
			if m == 'DA':
				trips_by_mode_df.at[m,p] = mtx_df.at[p,'Sum']
			else:
				trips_by_mode_df.at[m,p] = 0
	
	# Internal and External columns
	temp = 0.0
	# Internal
	for m in all_modes:
		for p in purpose:
			temp = temp + trips_by_mode_df.at[m,p]
		trips_by_mode_df.at[m,'All Internal Purposes'] = temp
		temp = 0.0
	# External Purposes
	for m in all_modes:
		for p in ext_purps:
			temp = temp + trips_by_mode_df.at[m,p]
		trips_by_mode_df.at[m,'All External Purposes'] = temp
		temp = 0.0

	# Fill Person Trips Totals by Mode column
	all_modes    = ['DA','SR','WLK_Bus','DRV_Bus','WLK_Rail','DRV_Rail','SchoolBus','Walk','Bike']
	all_purps   = ['HBW','HBSch','HBR','HBO','HBC','NHB','Auto_XI_W','Auto_IX_W','Auto_XIIX_NW','Auto_XX']
	all_trips = 0.0
	temp = 0.0
	for m in all_modes:
		for p in all_purps:
			temp = temp + trips_by_mode_df.at[m,p]
		trips_by_mode_df.at[m,'All Purposes Person Trips'] = temp
		all_trips = all_trips + temp
		temp = 0.0
	# Fill person trip totals by Purpose row
	temp = 0.0
	for p in all_purps:
		for m in all_modes:
			temp =  temp + trips_by_mode_df.at[m,p]
		trips_by_mode_df.at['All Modes Person Trips',p] = temp
		temp = 0.0
	
	# Fill sum of All Internal Purposes
	temp = 0.0
	for m in all_modes:
		temp =  temp + trips_by_mode_df.at[m,'All Internal Purposes']
	trips_by_mode_df.at['All Modes Person Trips','All Internal Purposes'] = temp
	temp = 0.0
	# Fill sum of All External Purposes
	temp = 0.0
	for m in all_modes:
		temp =  temp + trips_by_mode_df.at[m,'All External Purposes']
	trips_by_mode_df.at['All Modes Person Trips','All External Purposes'] = temp
	temp = 0.0
	
	
	# Fill total person trips for all modes and purposes cell
	trips_by_mode_df.at['All Modes Person Trips','All Purposes Person Trips'] = all_trips
	
	"""
	# Copy DA person trips to Auto trips
	trips_by_mode_df.at['DA','Auto Trips'] = trips_by_mode_df.at['DA','All Purposes Person Trips']
	# Convert Shared Ride person trips to Auto trips
	#HOV occupancies    HBW	    HBSch	HBR	    HBO	    HBC	    NHB
	occ     =          [2.17,   2.38,   2.26,   2.38,   2.67,   2.34]
	trips_by_mode_df.at['SR','Auto Trips'] = trips_by_mode_df.at['SR','HBW']/occ[0] + trips_by_mode_df.at['SR','HBSch']/occ[1] + trips_by_mode_df.at['SR','HBR']/occ[2] + trips_by_mode_df.at['SR','HBO']/occ[3] + trips_by_mode_df.at['SR','HBC']/occ[4] + trips_by_mode_df.at['SR','NHB']/occ[5]
	"""
	
	# Export trips_by_mode_df to csv file in timestamped folder
	trips_by_mode_df.to_csv(outputs_path + "/ModelRun_"+date+"/Model Reporting/unformatted/tripsbymode.csv")

	# Copy trips_by_mode_df to trips_by_mode_df_export to maintain formatting for future operations in trips_by_mode_df
	trips_by_mode_df_export = trips_by_mode_df.copy()
	
	# Format sum fields to have commas and rounding
	all_cols   = ['HBW','HBSch','HBR','HBO','HBC','NHB','Auto_XI_W','Auto_IX_W','Auto_XIIX_NW','All Internal Purposes','All External Purposes','All Purposes Person Trips']#,'Auto Trips']

	for p in all_cols:
		trips_by_mode_df_export[p] = format_commas(trips_by_mode_df_export[p])
			
	# Export trips_by_mode_df_export to csv file in timestamped folder
	trips_by_mode_df_export.to_csv(outputs_path + "/ModelRun_"+date+"/Model Reporting/Trips_By_Mode_Summary.csv")



	# 4: MODE SHARES
	
	# Import blank dataframe with purposes and totals columns from template folder
	mode_shares_df = pd.read_csv(settings_dict['SharedData_Path']+'/template/reporting/mode_shares_template.csv')
	
	# Set indices for mode_shares_df and mtx_df to use the purpose/code columns
	mode_shares_df.set_index('Mode', inplace=True)
	
	# Create purpose and mode lists
	mode    = ['DA','SR','WLK_Bus','DRV_Bus','WLK_Rail','DRV_Rail','SchoolBus','Walk','Bike'] 
	purpose = ['HBW','HBSch','HBR','HBO','HBC','NHB','Auto_XI_W','Auto_IX_W','Auto_XIIX_NW','Auto_XX','All Internal Purposes','All External Purposes']
	
	# Fill Percents in mode_shares_df from trips_by_mode_df table
	for m in mode:
		for p in purpose:
			mode_shares_df.at[m,p] = float(trips_by_mode_df.at[m,p]) / float(trips_by_mode_df.at['All Modes Person Trips',p])
		mode_shares_df.at[m,'All Purposes Person Trips'] = float(trips_by_mode_df.at[m,'All Purposes Person Trips']) / float(trips_by_mode_df.at['All Modes Person Trips','All Purposes Person Trips'])
	
	# Export mode_shares_df to csv file in timestamped folder
	mode_shares_df.to_csv(outputs_path + "/ModelRun_"+date+"/Model Reporting/unformatted/modeshares.csv")
	
	# Copy mode_shares_df to mode_shares_df_export to maintain formatting for future operations in mode_shares_df
	mode_shares_df_export = mode_shares_df.copy()
	
	# Format percents fields to have percent notation and 2 decimals
	for p in purpose:
		mode_shares_df_export[p] = format_percent(mode_shares_df_export[p])
	mode_shares_df_export['All Purposes Person Trips'] = format_percent(mode_shares_df_export['All Purposes Person Trips'])
			
	# Export mode_shares_df_export to csv file in timestamped folder
	mode_shares_df_export.to_csv(outputs_path + "/ModelRun_"+date+"/Model Reporting/Mode_Shares_Summary.csv")
	
	
	
	# 5: VEHICLE TRIPS
	
	# Import blank dataframe with purposes and totals columns from template folder
	vehicle_trips_df = pd.read_csv(settings_dict['SharedData_Path']+'/template/reporting/vehicle_trips_template.csv')
	
	# Set indices for vehicle_trips_df to use the purpose/code columns
	vehicle_trips_df.set_index('Indicator', inplace=True)
	
	# Internal Vehicle Trips by Period
	period = ['AM_PR','MD_PR','PM_PR','NI_PR','AM_PK_HR','PM_PK_HR','Daily']
	veh_types = ['LOV','HOV','LCV','Truck']
	# Loop through periods and vehicle types
	for t in period:
		for v in veh_types:
			vehicle_trips_df.at['Internal ' + t + ' Trips',v]   = mtx_df.at[t + '_' + v,'Sum']
		vehicle_trips_df.at['Internal ' + t + ' Trips','Total Auto']            = vehicle_trips_df.at['Internal ' + t + ' Trips','LOV']        + vehicle_trips_df.at['Internal ' + t + ' Trips','HOV'] + vehicle_trips_df.at['Internal ' + t + ' Trips','LCV']
		vehicle_trips_df.at['Internal ' + t + ' Trips','Total Vehicle Trips']   = vehicle_trips_df.at['Internal ' + t + ' Trips','Total Auto'] + vehicle_trips_df.at['Internal ' + t + ' Trips','Truck'] 
		
	# External Vehicle Trips by Period
	# AM
	vehicle_trips_df.at['External AM_PR Trips','LOV']     = mtx_df.at['Auto_Ext_AM','Sum']
	vehicle_trips_df.at['External AM_PR Trips','HOV']     = 0
	vehicle_trips_df.at['External AM_PR Trips','LCV']     = 0
	vehicle_trips_df.at['External AM_PR Trips','Truck']   = mtx_df.at['Truck_Ext_AM','Sum']
	vehicle_trips_df.at['External AM_PR Trips','Total Auto']            = vehicle_trips_df.at['External AM_PR Trips','LOV']
	vehicle_trips_df.at['External AM_PR Trips','Total Vehicle Trips']   = vehicle_trips_df.at['External AM_PR Trips','Total Auto'] + vehicle_trips_df.at['External AM_PR Trips','Truck']
	# MD
	vehicle_trips_df.at['External MD_PR Trips','LOV']     = mtx_df.at['Auto_Ext_MD','Sum']
	vehicle_trips_df.at['External MD_PR Trips','HOV']     = 0
	vehicle_trips_df.at['External MD_PR Trips','LCV']     = 0
	vehicle_trips_df.at['External MD_PR Trips','Truck']   = mtx_df.at['Truck_Ext_MD','Sum']
	vehicle_trips_df.at['External MD_PR Trips','Total Auto']            = vehicle_trips_df.at['External MD_PR Trips','LOV']
	vehicle_trips_df.at['External MD_PR Trips','Total Vehicle Trips']   = vehicle_trips_df.at['External MD_PR Trips','Total Auto'] + vehicle_trips_df.at['External MD_PR Trips','Truck']
	# PM
	vehicle_trips_df.at['External PM_PR Trips','LOV']     = mtx_df.at['Auto_Ext_PM','Sum']
	vehicle_trips_df.at['External PM_PR Trips','HOV']     = 0
	vehicle_trips_df.at['External PM_PR Trips','LCV']     = 0
	vehicle_trips_df.at['External PM_PR Trips','Truck']   = mtx_df.at['Truck_Ext_PM','Sum']
	vehicle_trips_df.at['External PM_PR Trips','Total Auto']            = vehicle_trips_df.at['External PM_PR Trips','LOV']
	vehicle_trips_df.at['External PM_PR Trips','Total Vehicle Trips']   = vehicle_trips_df.at['External PM_PR Trips','Total Auto'] + vehicle_trips_df.at['External PM_PR Trips','Truck']
	# NI
	vehicle_trips_df.at['External NI_PR Trips','LOV']     = mtx_df.at['Auto_Ext_NI','Sum']
	vehicle_trips_df.at['External NI_PR Trips','HOV']     = 0
	vehicle_trips_df.at['External NI_PR Trips','LCV']     = 0
	vehicle_trips_df.at['External NI_PR Trips','Truck']   = mtx_df.at['Truck_Ext_NI','Sum']
	vehicle_trips_df.at['External NI_PR Trips','Total Auto']            = vehicle_trips_df.at['External NI_PR Trips','LOV']
	vehicle_trips_df.at['External NI_PR Trips','Total Vehicle Trips']   = vehicle_trips_df.at['External NI_PR Trips','Total Auto'] + vehicle_trips_df.at['External NI_PR Trips','Truck']
	# AM PK
	vehicle_trips_df.at['External AM_PK_HR Trips','LOV']     = mtx_df.at['Auto_Ext_AMPK','Sum']
	vehicle_trips_df.at['External AM_PK_HR Trips','HOV']     = 0
	vehicle_trips_df.at['External AM_PK_HR Trips','LCV']     = 0
	vehicle_trips_df.at['External AM_PK_HR Trips','Truck']   = mtx_df.at['Truck_Ext_AMPK','Sum']
	vehicle_trips_df.at['External AM_PK_HR Trips','Total Auto']            = vehicle_trips_df.at['External AM_PK_HR Trips','LOV']
	vehicle_trips_df.at['External AM_PK_HR Trips','Total Vehicle Trips']   = vehicle_trips_df.at['External AM_PK_HR Trips','Total Auto'] + vehicle_trips_df.at['External AM_PK_HR Trips','Truck']
	# PM PK
	vehicle_trips_df.at['External PM_PK_HR Trips','LOV']     = mtx_df.at['Auto_Ext_PMPK','Sum']
	vehicle_trips_df.at['External PM_PK_HR Trips','HOV']     = 0
	vehicle_trips_df.at['External PM_PK_HR Trips','LCV']     = 0
	vehicle_trips_df.at['External PM_PK_HR Trips','Truck']   = mtx_df.at['Truck_Ext_PMPK','Sum']
	vehicle_trips_df.at['External PM_PK_HR Trips','Total Auto']            = vehicle_trips_df.at['External PM_PK_HR Trips','LOV']
	vehicle_trips_df.at['External PM_PK_HR Trips','Total Vehicle Trips']   = vehicle_trips_df.at['External PM_PK_HR Trips','Total Auto'] + vehicle_trips_df.at['External PM_PK_HR Trips','Truck']
	# Daily
	vehicle_trips_df.at['External Daily Trips','LOV']     = mtx_df.at['Auto_Ext_Dly','Sum']
	vehicle_trips_df.at['External Daily Trips','HOV']     = 0
	vehicle_trips_df.at['External Daily Trips','LCV']     = 0
	vehicle_trips_df.at['External Daily Trips','Truck']   = mtx_df.at['Truck_Ext_Dly','Sum']
	vehicle_trips_df.at['External Daily Trips','Total Auto']            = vehicle_trips_df.at['External Daily Trips','LOV']
	vehicle_trips_df.at['External Daily Trips','Total Vehicle Trips']   = vehicle_trips_df.at['External Daily Trips','Total Auto'] + vehicle_trips_df.at['External Daily Trips','Truck']
	
	
	# Multiply AM and PM rows by 3 and MD and NI periods by 6 (hours in each period)
	# AM and PM
	vehicle_trips_df.loc['Internal AM_PR Trips'] = vehicle_trips_df.loc['Internal AM_PR Trips'] * 3
	vehicle_trips_df.loc['External AM_PR Trips'] = vehicle_trips_df.loc['External AM_PR Trips'] * 3
	vehicle_trips_df.loc['Internal PM_PR Trips'] = vehicle_trips_df.loc['Internal PM_PR Trips'] * 3
	vehicle_trips_df.loc['External PM_PR Trips'] = vehicle_trips_df.loc['External PM_PR Trips'] * 3
	# MD and NI
	vehicle_trips_df.loc['Internal MD_PR Trips'] = vehicle_trips_df.loc['Internal MD_PR Trips'] * 6
	vehicle_trips_df.loc['External MD_PR Trips'] = vehicle_trips_df.loc['External MD_PR Trips'] * 6
	vehicle_trips_df.loc['Internal NI_PR Trips'] = vehicle_trips_df.loc['Internal NI_PR Trips'] * 6
	vehicle_trips_df.loc['External NI_PR Trips'] = vehicle_trips_df.loc['External NI_PR Trips'] * 6
	
	
	
	# Export vehicle_trips_df to csv file in timestamped folder
	vehicle_trips_df.to_csv(outputs_path + "/ModelRun_"+date+"/Model Reporting/unformatted/vehicletrips.csv")
	
	# Copy vehicle_trips_df to vehicle_trips_df_export to maintain formatting for future operations in vehicle_trips_df
	vehicle_trips_df_export = vehicle_trips_df.copy()

	# Format trip fields to have commas and rounding
	vehicle_fields = ['LOV','HOV','LCV','Total Auto','Truck','Total Vehicle Trips']
	for v in vehicle_fields:
		vehicle_trips_df_export[v] = format_commas(vehicle_trips_df_export[v])
			
	# Export vehicle_trips_df_export to csv file in timestamped folder
	vehicle_trips_df_export.to_csv(outputs_path + "/ModelRun_"+date+"/Model Reporting/Vehicle_Trips_Summary.csv")
	
	
	
	# 6: VEHICLE TRIP MATRICES
	
	# Import blank dataframe with from template folder
	vehicle_trip_mtxs_df = pd.read_csv(settings_dict['SharedData_Path']+'/template/reporting/vehicle_trip_matrices_template.csv')
	
	# Set indices for vehicle_trip_mtxs_df to use the Mode columns
	vehicle_trip_mtxs_df.set_index('Mode', inplace=True)
	
	# Internals
	# Create purpose and mode lists
	mode    = ['LOV','HOV','LCV','Truck']
	period  = ['AM_PR','MD_PR','PM_PR','NI_PR','AM_PK_HR','PM_PK_HR']
	
	# Fill trips in vehicle_trip_mtxs_df from mtx_df table
	for m in mode:
		for per in period:
			vehicle_trip_mtxs_df.at[m,per] = mtx_df.at[per+'_'+m,'Sum']
		vehicle_trip_mtxs_df.at[m,'Period Total'] = mtx_df.at['AM_PR_'+m,'Sum'] + mtx_df.at['MD_PR_'+m,'Sum'] + mtx_df.at['PM_PR_'+m,'Sum'] + mtx_df.at['NI_PR_'+m,'Sum']
		vehicle_trip_mtxs_df.at[m,'Daily Matrix'] = mtx_df.at['Daily_'+m,'Sum']
	
	# Externals
	# Create purpose and mode lists
	mode    = ['Auto_Ext','Truck_Ext']
	
	# Fill trips in vehicle_trip_mtxs_df from mtx_df table
	for m in mode:
		vehicle_trip_mtxs_df.at[m,'AM_PR']        = mtx_df.at[m+'_AM','Sum']
		vehicle_trip_mtxs_df.at[m,'MD_PR']        = mtx_df.at[m+'_MD','Sum']
		vehicle_trip_mtxs_df.at[m,'PM_PR']        = mtx_df.at[m+'_PM','Sum']
		vehicle_trip_mtxs_df.at[m,'NI_PR']        = mtx_df.at[m+'_NI','Sum']
		vehicle_trip_mtxs_df.at[m,'AM_PK_HR']     = mtx_df.at[m+'_AMPK','Sum']
		vehicle_trip_mtxs_df.at[m,'PM_PK_HR']     = mtx_df.at[m+'_PMPK','Sum']
		vehicle_trip_mtxs_df.at[m,'Period Total'] = mtx_df.at[m+'_AM','Sum'] + mtx_df.at[m+'_MD','Sum'] + mtx_df.at[m+'_PM','Sum'] + mtx_df.at[m+'_NI','Sum']
		vehicle_trip_mtxs_df.at[m,'Daily Matrix'] = mtx_df.at[m+'_Dly','Sum']
		
	# Export vehicle_trip_mtxs_df to csv file in timestamped folder
	vehicle_trip_mtxs_df.to_csv(outputs_path + "/ModelRun_"+date+"/Model Reporting/unformatted/vehicletripmatrices.csv")

	# Copy vehicle_trip_mtxs_df to vehicle_trip_mtxs_df_export to maintain formatting for future operations in vehicle_trip_mtxs_df
	vehicle_trip_mtxs_df_export = vehicle_trip_mtxs_df.copy()	

	# Format trip fields to have commas and rounding
	for per in period:
		vehicle_trip_mtxs_df_export[per] = format_commas(vehicle_trip_mtxs_df_export[per])
	vehicle_trip_mtxs_df_export['Period Total'] = format_commas(vehicle_trip_mtxs_df_export['Period Total'])
	vehicle_trip_mtxs_df_export['Daily Matrix'] = format_commas(vehicle_trip_mtxs_df_export['Daily Matrix'])
			
	# Export vehicle_trip_mtxs_df_export to csv file in timestamped folder
	vehicle_trip_mtxs_df_export.to_csv(outputs_path + "/ModelRun_"+date+"/Model Reporting/Vehicle_Trips_Matrices_Summary.csv")
	
	
	
	# 7: TRANSIT TRIPS
	
	# Import blank dataframe with from template folder
	transit_trip_df = pd.read_csv(settings_dict['SharedData_Path']+'/template/reporting/transit_trips_template.csv')
	
	# Set indices for vehicle_trip_mtxs_df to use the Mode columns
	transit_trip_df.set_index('Period', inplace=True)
	
	# Create purpose and mode lists
	income  = ['Inc1','Inc2','Inc3','Inc4']
	pk      = ['HBW','HBSch','HBC']
	op		= ['HBR','HBO',  'NHB']
	
	# Fill trips in transit_trip_df from mtx_df table
	# Peak
	temp_wlk = 0.0
	temp_drv = 0.0
	for purp in pk:
		if purp+'1' in mtx_df.index:
			for i in income:
				temp_wlk = temp_wlk + mtx_df.at[purp+'_WLK_Bus_'+i,'Sum'] + mtx_df.at[purp+'_WLK_Rail_'+i,'Sum']
				temp_drv = temp_drv + mtx_df.at[purp+'_DRV_Bus_'+i,'Sum'] + mtx_df.at[purp+'_DRV_Rail_'+i,'Sum']
		else:
			temp_wlk = temp_wlk + mtx_df.at[purp+'_WLK_Bus','Sum'] + mtx_df.at[purp+'_WLK_Rail','Sum']
			temp_drv = temp_drv + mtx_df.at[purp+'_DRV_Bus','Sum'] + mtx_df.at[purp+'_DRV_Rail','Sum']
	transit_trip_df.at['Peak','Walk Access']  = temp_wlk
	transit_trip_df.at['Peak','Drive Access'] = temp_drv
	# Off-Peak
	temp_wlk = 0.0
	temp_drv = 0.0
	for purp in op:
		if purp+'1' in mtx_df.index:
			for i in income:
				temp_wlk = temp_wlk + mtx_df.at[purp+'_WLK_Bus_'+i,'Sum'] + mtx_df.at[purp+'_WLK_Rail_'+i,'Sum']
				temp_drv = temp_drv + mtx_df.at[purp+'_DRV_Bus_'+i,'Sum'] + mtx_df.at[purp+'_DRV_Rail_'+i,'Sum']
		else:
			temp_wlk = temp_wlk + mtx_df.at[purp+'_WLK_Bus','Sum'] + mtx_df.at[purp+'_WLK_Rail','Sum']
			temp_drv = temp_drv + mtx_df.at[purp+'_DRV_Bus','Sum'] + mtx_df.at[purp+'_DRV_Rail','Sum']
	transit_trip_df.at['Off-Peak','Walk Access']  = temp_wlk
	transit_trip_df.at['Off-Peak','Drive Access'] = temp_drv
	
	# Daily
	transit_trip_df.at['Daily','Walk Access']  = transit_trip_df.at['Peak','Walk Access']  + transit_trip_df.at['Off-Peak','Walk Access']
	transit_trip_df.at['Daily','Drive Access'] = transit_trip_df.at['Peak','Drive Access'] + transit_trip_df.at['Off-Peak','Drive Access']	
	# Total Linked
	transit_trip_df.at['Peak','Linked']     = transit_trip_df.at['Peak','Walk Access']     + transit_trip_df.at['Peak','Drive Access']
	transit_trip_df.at['Off-Peak','Linked'] = transit_trip_df.at['Off-Peak','Walk Access'] + transit_trip_df.at['Off-Peak','Drive Access']	
	transit_trip_df.at['Daily','Linked']    = transit_trip_df.at['Daily','Walk Access']    + transit_trip_df.at['Daily','Drive Access']	
	
	# Import Unlinked trips value from Visum Line Routes list and add the sum to transit_trip_df
	pk_unlinked_transit_trips    = VisumPy.helpers.GetMulti(Visum.Net.LineRoutes,r"PEAK_UNLINKED_TRIPS")
	offpk_unlinked_transit_trips = VisumPy.helpers.GetMulti(Visum.Net.LineRoutes,r"OFFPEAK_UNLINKED_TRIPS")
	dly_unlinked_transit_trips   = VisumPy.helpers.GetMulti(Visum.Net.LineRoutes,r"PTRIPSUNLINKED(AP)")
	
	# Paste unlinkeds trips into transit_trip_df
	transit_trip_df.at['Peak','Unlinked']  = sum(pk_unlinked_transit_trips)
	transit_trip_df.at['Off-Peak','Unlinked']  = sum(offpk_unlinked_transit_trips)
	transit_trip_df.at['Daily','Unlinked']  = sum(dly_unlinked_transit_trips)
	
	
	# Export transit_trip_df to csv file in timestamped folder
	transit_trip_df.to_csv(outputs_path + "/ModelRun_"+date+"/Model Reporting/unformatted/transittrips.csv")
	
	# Copy transit_trip_df to transit_trip_df_export to maintain formatting for future operations in transit_trip_df
	transit_trip_df_export = transit_trip_df.copy()	
	
	# Format trip fields to have commas and rounding
	transit_trip_df_export['Walk Access']  = format_commas(transit_trip_df_export['Walk Access'])
	transit_trip_df_export['Drive Access'] = format_commas(transit_trip_df_export['Drive Access'])
	transit_trip_df_export['Linked']       = format_commas(transit_trip_df_export['Linked'])
	transit_trip_df_export['Unlinked']     = format_commas(transit_trip_df_export['Unlinked'])
			
	# Export transit_trip_df_export to csv file in timestamped folder
	transit_trip_df_export.to_csv(outputs_path + "/ModelRun_"+date+"/Model Reporting/Transit_Trips_Summary.csv")
	
	
	
	
	# 8: MODEL SUMMARY FILE

	# Import blank dataframe with from template folder
	model_summary_df = pd.read_csv(settings_dict['SharedData_Path']+'/template/reporting/model_summary_template.csv')
	
	# Set indices for model_summary_df to use the Mode columns
	model_summary_df.set_index('Primary Indicators', inplace=True)
	
	# Fill in rows of Model Summary that come from matrix_df and other matrix aggregated df's from above outputs
	model_summary_df.at['Person trips (PrT)','Statistics'] = trips_by_mode_df.at['All Modes Person Trips','All Purposes Person Trips']
	
	model_summary_df.at['Vehicle trips (PrT)','Statistics'] = vehicle_trips_df.at['Internal Daily Trips','Total Vehicle Trips'] + vehicle_trips_df.at['External Daily Trips','Total Vehicle Trips']
	
	model_summary_df.at['Linked transit passenger trips (PuT)','Statistics'] = transit_trip_df.at['Daily','Linked']
	
	model_summary_df.at['Unlinked transit passenger trips (PuT)','Statistics'] = transit_trip_df.at['Daily','Unlinked']
	
	model_summary_df.at['Park & Ride Trips (drive access) (PuT)','Statistics'] = transit_trip_df.at['Daily','Drive Access']
	
	model_summary_df.at['Combined walk and bike trips','Statistics'] = trips_by_mode_df.at['Walk','All Purposes Person Trips'] + trips_by_mode_df.at['Bike','All Purposes Person Trips']
	
	model_summary_df.at['Single Occupancy Vehicle %','Statistics'] = mode_shares_df.at['DA','All Purposes Person Trips']
	
	model_summary_df.at['High Occupancy Vehicle %','Statistics'] = mode_shares_df.at['SR','All Purposes Person Trips']
	
	model_summary_df.at['Walk to bus %','Statistics'] = mode_shares_df.at['WLK_Bus','All Purposes Person Trips']
	
	model_summary_df.at['Drive to bus (park & ride) %','Statistics'] = mode_shares_df.at['DRV_Bus','All Purposes Person Trips']
	
	model_summary_df.at['Walk %','Statistics'] = mode_shares_df.at['Walk','All Purposes Person Trips']
	
	model_summary_df.at['Bike %','Statistics'] = mode_shares_df.at['Bike','All Purposes Person Trips']
	

	# Import Zone fields for Population, Employment, and Housing Units in model_summary_df
	pop  = VisumPy.helpers.GetMulti(Visum.Net.Zones,"HHPOP")
	emp  = VisumPy.helpers.GetMulti(Visum.Net.Zones,"TOTEMP")
	hu   = VisumPy.helpers.GetMulti(Visum.Net.Zones,"HH_SRTC")
	
	# Fill zone summary fields in model_summary_df
	#model_summary_df.at['Population','Statistics']        = sum(pop)
	model_summary_df.at['Total Employment','Statistics']   = sum(emp)
	model_summary_df.at['Housing Units (HU)','Statistics'] = sum(hu)

	
	# Import Link fields for VMT and other flow calculations
	# Link ID fields
	NO          = VisumPy.helpers.GetMulti(Visum.Net.Links,"No", activeOnly = True)
	FCLASS      = VisumPy.helpers.GetMulti(Visum.Net.Links,"TYPENO", activeOnly = True)
	EXT         = VisumPy.helpers.GetMulti(Visum.Net.Links,"EXT_COUNT", activeOnly = True)
	LENGTH      = VisumPy.helpers.GetMulti(Visum.Net.Links,"Length", activeOnly = True)
	NUMLANES    = VisumPy.helpers.GetMulti(Visum.Net.Links,"NUMLANES", activeOnly = True)
	VOL_CORR    = VisumPy.helpers.GetMulti(Visum.Net.Links,"VOL_CORRIDOR", activeOnly = True)
	VOL_CORR_2  = VisumPy.helpers.GetMulti(Visum.Net.Links,"VOL_CORRIDOR_2", activeOnly = True)
	TT_CORR     = VisumPy.helpers.GetMulti(Visum.Net.Links,"TT_CORRIDOR", activeOnly = True)
	# Free-Flow Time
	FFTIME_C    = VisumPy.helpers.GetMulti(Visum.Net.Links,"FFTIME_C", activeOnly = True)
	FFTIME_T    = VisumPy.helpers.GetMulti(Visum.Net.Links,"FFTIME_T", activeOnly = True)
	# Pull CONGTIME Auto by period, Length, and Flows by Period for VMT/VHT Calculations 	
	CONGTIME_AM_C   = VisumPy.helpers.GetMulti(Visum.Net.Links,"AM3_CTIME_C", activeOnly = True)
	CONGTIME_MD_C   = VisumPy.helpers.GetMulti(Visum.Net.Links,"MD_CTIME_C", activeOnly = True)
	CONGTIME_PM_C   = VisumPy.helpers.GetMulti(Visum.Net.Links,"PM3_CTIME_C", activeOnly = True)
	CONGTIME_NI_C   = VisumPy.helpers.GetMulti(Visum.Net.Links,"NI_CTIME_C", activeOnly = True)
	CONGTIME_PMPK_C = VisumPy.helpers.GetMulti(Visum.Net.Links,"PM1_CTIME_C", activeOnly = True)
	# Pull CONGTIME Truck by period, Length, and Flows by Period for VMT/VHT Calculations 	
	CONGTIME_AM_T   = VisumPy.helpers.GetMulti(Visum.Net.Links,"AM3_CTIME_T", activeOnly = True)
	CONGTIME_MD_T   = VisumPy.helpers.GetMulti(Visum.Net.Links,"MD_CTIME_T", activeOnly = True)
	CONGTIME_PM_T   = VisumPy.helpers.GetMulti(Visum.Net.Links,"PM3_CTIME_T", activeOnly = True)
	CONGTIME_NI_T   = VisumPy.helpers.GetMulti(Visum.Net.Links,"NI_CTIME_T", activeOnly = True)
	CONGTIME_PMPK_T = VisumPy.helpers.GetMulti(Visum.Net.Links,"PM1_CTIME_T", activeOnly = True)
	# Link Flows by Period
	# AM
	AM_Auto_Flow  = VisumPy.helpers.GetMulti(Visum.Net.Links,"AM_PERIOD_AUTO_VOLUME", activeOnly = True)
	AM_Truck_Flow = VisumPy.helpers.GetMulti(Visum.Net.Links,"AM_PERIOD_TRUCK_VOLUME", activeOnly = True)
	AM_Tot_Flow   = VisumPy.helpers.GetMulti(Visum.Net.Links,"AM_PERIOD_MODEL_VOLUME", activeOnly = True)
	# MD
	MD_Auto_Flow  = VisumPy.helpers.GetMulti(Visum.Net.Links,"MIDDAY_PERIOD_AUTO_VOLUME", activeOnly = True)
	MD_Truck_Flow = VisumPy.helpers.GetMulti(Visum.Net.Links,"MIDDAY_PERIOD_TRUCK_VOLUME", activeOnly = True)
	MD_Tot_Flow   = VisumPy.helpers.GetMulti(Visum.Net.Links,"MIDDAY_PERIOD_MODEL_VOLUME", activeOnly = True)
	# PM
	PM_Auto_Flow  = VisumPy.helpers.GetMulti(Visum.Net.Links,"PM_PERIOD_AUTO_VOLUME", activeOnly = True)
	PM_Truck_Flow = VisumPy.helpers.GetMulti(Visum.Net.Links,"PM_PERIOD_TRUCK_VOLUME", activeOnly = True)
	PM_Tot_Flow   = VisumPy.helpers.GetMulti(Visum.Net.Links,"PM_PERIOD_MODEL_VOLUME", activeOnly = True)
	# NT
	NI_Auto_Flow  = VisumPy.helpers.GetMulti(Visum.Net.Links,"NIGHT_PERIOD_AUTO_VOLUME", activeOnly = True)
	NI_Truck_Flow = VisumPy.helpers.GetMulti(Visum.Net.Links,"NIGHT_PERIOD_TRUCK_VOLUME", activeOnly = True)
	NI_Tot_Flow   = VisumPy.helpers.GetMulti(Visum.Net.Links,"NIGHT_PERIOD_MODEL_VOLUME", activeOnly = True)
	# PM Peak Hour
	PMPKHR_Auto_Flow  = VisumPy.helpers.GetMulti(Visum.Net.Links,"PMPKHR_AUTO_VOL", activeOnly = True)
	PMPKHR_Truck_Flow = VisumPy.helpers.GetMulti(Visum.Net.Links,"PMPKHR_TRUCK_VOL", activeOnly = True)
	PMPKHR_Tot_Flow   = VisumPy.helpers.GetMulti(Visum.Net.Links,"PMPKHR_VOL", activeOnly = True)
	# Daily
	Auto_Flow  = VisumPy.helpers.GetMulti(Visum.Net.Links,"DAILY_AUTO_VOLUME", activeOnly = True)
	Truck_Flow = VisumPy.helpers.GetMulti(Visum.Net.Links,"DAILY_TRUCK_VOLUME", activeOnly = True)
	Tot_Flow   = VisumPy.helpers.GetMulti(Visum.Net.Links,"DAILY_MODEL_VOLUME", activeOnly = True)
	
	
	# Make Visum list with link data
	summary_list = [NO, FCLASS, EXT, LENGTH, NUMLANES, VOL_CORR, VOL_CORR_2, TT_CORR, FFTIME_C, FFTIME_T,
					CONGTIME_AM_C, CONGTIME_MD_C, CONGTIME_PM_C, CONGTIME_NI_C, CONGTIME_PMPK_C,
					CONGTIME_AM_T, CONGTIME_MD_T, CONGTIME_PM_T, CONGTIME_NI_T, CONGTIME_PMPK_T,
					AM_Auto_Flow, AM_Truck_Flow, AM_Tot_Flow, MD_Auto_Flow, MD_Truck_Flow, MD_Tot_Flow,
					PM_Auto_Flow, PM_Truck_Flow, PM_Tot_Flow, NI_Auto_Flow, NI_Truck_Flow, NI_Tot_Flow,
					PMPKHR_Auto_Flow, PMPKHR_Truck_Flow, PMPKHR_Tot_Flow,
					Auto_Flow, Truck_Flow, Tot_Flow]
			
	# Put Visum link list into dataframe  
	link_df = pd.DataFrame(np.column_stack(summary_list), columns = ['NO', 'FCLASS', 'EXTERNAL', 'LENGTH', 'NUMLANES', 'VOL_CORR', 'VOL_CORR_2', 'TT_CORR', 'FFTIME_C', 'FFTIME_T',
																'CONGTIME_AM_C', 'CONGTIME_MD_C', 'CONGTIME_PM_C', 'CONGTIME_NI_C', 'CONGTIME_PMPK_C',
																'CONGTIME_AM_T', 'CONGTIME_MD_T', 'CONGTIME_PM_T', 'CONGTIME_NI_T', 'CONGTIME_PMPK_T',
																'AM_Auto_Flow', 'AM_Truck_Flow', 'AM_Tot_Flow', 'MD_Auto_Flow', 'MD_Truck_Flow', 'MD_Tot_Flow',
																'PM_Auto_Flow', 'PM_Truck_Flow', 'PM_Tot_Flow', 'NI_Auto_Flow', 'NI_Truck_Flow', 'NI_Tot_Flow',
																'PMPKHR_Auto_Flow', 'PMPKHR_Truck_Flow', 'PMPKHR_Tot_Flow',
																'Auto_Flow', 'Truck_Flow', 'Tot_Flow'])#, 'LCV_Flow'])
	
	# Fill in rows of Model Summary that come from link_df
	# VMT
	model_summary_df.at['Daily VMT','Statistics']             = np.dot(link_df['LENGTH'], link_df['Tot_Flow'])
	model_summary_df.at['Daily Per Capita VMT','Statistics']  = model_summary_df.at['Daily VMT','Statistics']/sum(pop)
	model_summary_df.at['Daily VMT Per HU','Statistics']      = model_summary_df.at['Daily VMT','Statistics']/model_summary_df.at['Housing Units (HU)','Statistics']
	model_summary_df.at['PM Peak Hr VMT','Statistics']        = np.dot(link_df['LENGTH'], link_df['PMPKHR_Tot_Flow'])
	model_summary_df.at['PM Peak Hr VMT Per HU','Statistics'] = model_summary_df.at['PM Peak Hr VMT','Statistics']/model_summary_df.at['Housing Units (HU)','Statistics']
	# VHT
	model_summary_df.at['Daily VHT','Statistics']             = (np.dot(link_df['CONGTIME_AM_C'], link_df['AM_Auto_Flow']) +
																 np.dot(link_df['CONGTIME_MD_C'], link_df['MD_Auto_Flow']) +
																 np.dot(link_df['CONGTIME_PM_C'], link_df['PM_Auto_Flow']) +
																 np.dot(link_df['CONGTIME_NI_C'], link_df['NI_Auto_Flow']) +
																 np.dot(link_df['CONGTIME_AM_T'], link_df['AM_Truck_Flow']) +
																 np.dot(link_df['CONGTIME_MD_T'], link_df['MD_Truck_Flow']) +
																 np.dot(link_df['CONGTIME_PM_T'], link_df['PM_Truck_Flow']) +
																 np.dot(link_df['CONGTIME_NI_T'], link_df['NI_Truck_Flow']))
	model_summary_df.at['Daily Per Capita VHT','Statistics']  = model_summary_df.at['Daily VHT','Statistics']/sum(pop)
	model_summary_df.at['Daily VHT Per HU','Statistics']      = model_summary_df.at['Daily VHT','Statistics']/model_summary_df.at['Housing Units (HU)','Statistics']
	model_summary_df.at['PM Peak Hr VHT','Statistics']        = (np.dot(link_df['CONGTIME_PMPK_C'], link_df['PMPKHR_Auto_Flow']) +
															     np.dot(link_df['CONGTIME_PMPK_T'], link_df['PMPKHR_Truck_Flow']))
	model_summary_df.at['PM Peak Hr VHT Per HU','Statistics'] = model_summary_df.at['PM Peak Hr VHT','Statistics']/model_summary_df.at['Housing Units (HU)','Statistics']
	# VHD
	model_summary_df.at['Daily VHD','Statistics']             = (model_summary_df.at['Daily VHT','Statistics'] - 
																(np.dot(link_df['FFTIME_C'], link_df['Auto_Flow']) + 
																 np.dot(link_df['FFTIME_T'], link_df['Truck_Flow'])))
	model_summary_df.at['Daily Per Capita VHD','Statistics']  = model_summary_df.at['Daily VHD','Statistics']/sum(pop)
	model_summary_df.at['Daily VHD Per HU','Statistics']      = model_summary_df.at['Daily VHD','Statistics']/model_summary_df.at['Housing Units (HU)','Statistics']
	model_summary_df.at['PM Peak Hr VHD','Statistics']        = (model_summary_df.at['PM Peak Hr VHT','Statistics'] - 
																(np.dot(link_df['FFTIME_C'], link_df['PMPKHR_Auto_Flow']) + 
																 np.dot(link_df['FFTIME_T'], link_df['PMPKHR_Truck_Flow'])))
	model_summary_df.at['PM Peak Hr VHD Per HU','Statistics'] = model_summary_df.at['PM Peak Hr VHD','Statistics']/model_summary_df.at['Housing Units (HU)','Statistics']

	# Export model_summary_df to csv file in timestamped folder
	model_summary_df.to_csv(outputs_path + "/ModelRun_"+date+"/Model Reporting/unformatted/modelsummary.csv")
	
	# Copy model_summary_df to model_summary_df_export to maintain formatting for future operations in model_summary_df
	model_summary_df_export = model_summary_df.copy()
	# Convert Statistics column in model_summary_df_export to string so it can accept the formatted statistics values
	model_summary_df_export['Statistics'] = model_summary_df_export['Statistics'].astype(str)
	
	
	# Format model summary table output cells to make them look better. Each line needs one of three formats
	model_summary_df_export.at['Person trips (PrT)','Statistics']                     = format_commas_cell(model_summary_df.at['Person trips (PrT)','Statistics'])
	model_summary_df_export.at['Vehicle trips (PrT)','Statistics']                    = format_commas_cell(model_summary_df.at['Vehicle trips (PrT)','Statistics'])
	model_summary_df_export.at['Linked transit passenger trips (PuT)','Statistics']   = format_commas_cell(model_summary_df.at['Linked transit passenger trips (PuT)','Statistics'])
	model_summary_df_export.at['Unlinked transit passenger trips (PuT)','Statistics'] = format_commas_cell(model_summary_df.at['Unlinked transit passenger trips (PuT)','Statistics'])
	model_summary_df_export.at['Park & Ride Trips (drive access) (PuT)','Statistics'] = format_commas_cell(model_summary_df.at['Park & Ride Trips (drive access) (PuT)','Statistics'])
	model_summary_df_export.at['Combined walk and bike trips','Statistics']           = format_commas_cell(model_summary_df.at['Combined walk and bike trips','Statistics'])
	model_summary_df_export.at['Single Occupancy Vehicle %','Statistics']             = format_percent_cell(model_summary_df.at['Single Occupancy Vehicle %','Statistics'])
	model_summary_df_export.at['High Occupancy Vehicle %','Statistics']               = format_percent_cell(model_summary_df.at['High Occupancy Vehicle %','Statistics'])
	model_summary_df_export.at['Walk to bus %','Statistics']                          = format_percent_cell(model_summary_df.at['Walk to bus %','Statistics'])
	model_summary_df_export.at['Drive to bus (park & ride) %','Statistics']           = format_percent_cell(model_summary_df.at['Drive to bus (park & ride) %','Statistics'])
	model_summary_df_export.at['Walk %','Statistics']                                 = format_percent_cell(model_summary_df.at['Walk %','Statistics'])
	model_summary_df_export.at['Bike %','Statistics']                                 = format_percent_cell(model_summary_df.at['Bike %','Statistics'])
	model_summary_df_export.at['Daily VMT','Statistics']                              = format_commas_cell(model_summary_df.at['Daily VMT','Statistics'])
	model_summary_df_export.at['Daily Per Capita VMT','Statistics']                   = format_twoplaces_cell(model_summary_df.at['Daily Per Capita VMT','Statistics'])
	model_summary_df_export.at['Daily VMT Per HU','Statistics']                       = format_twoplaces_cell(model_summary_df.at['Daily VMT Per HU','Statistics'])
	model_summary_df_export.at['PM Peak Hr VMT','Statistics']                         = format_commas_cell(model_summary_df.at['PM Peak Hr VMT','Statistics'])
	model_summary_df_export.at['PM Peak Hr VMT Per HU','Statistics']                  = format_twoplaces_cell(model_summary_df.at['PM Peak Hr VMT Per HU','Statistics'])
	model_summary_df_export.at['Daily VHT','Statistics']                              = format_commas_cell(model_summary_df.at['Daily VHT','Statistics'])
	model_summary_df_export.at['Daily Per Capita VHT','Statistics']                   = format_twoplaces_cell(model_summary_df.at['Daily Per Capita VHT','Statistics'])
	model_summary_df_export.at['Daily VHT Per HU','Statistics']                       = format_twoplaces_cell(model_summary_df.at['Daily VHT Per HU','Statistics'])
	model_summary_df_export.at['PM Peak Hr VHT','Statistics']                         = format_commas_cell(model_summary_df.at['PM Peak Hr VHT','Statistics'])
	model_summary_df_export.at['PM Peak Hr VHT Per HU','Statistics']                  = format_twoplaces_cell(model_summary_df.at['PM Peak Hr VHT Per HU','Statistics'])
	model_summary_df_export.at['Daily VHD','Statistics']                              = format_commas_cell(model_summary_df.at['Daily VHD','Statistics'])
	model_summary_df_export.at['Daily Per Capita VHD','Statistics']                   = format_twoplaces_cell(model_summary_df.at['Daily Per Capita VHD','Statistics'])
	model_summary_df_export.at['Daily VHD Per HU','Statistics']                       = format_twoplaces_cell(model_summary_df.at['Daily VHD Per HU','Statistics'])
	model_summary_df_export.at['PM Peak Hr VHD','Statistics']                         = format_commas_cell(model_summary_df.at['PM Peak Hr VHD','Statistics'])
	model_summary_df_export.at['PM Peak Hr VHD Per HU','Statistics']                  = format_twoplaces_cell(model_summary_df.at['PM Peak Hr VHD Per HU','Statistics'])
	#model_summary_df_export.at['Population','Statistics']                             = format_commas_cell(model_summary_df.at['Population','Statistics'])
	model_summary_df_export.at['Total Employment','Statistics']                       = format_commas_cell(model_summary_df.at['Total Employment','Statistics'])
	model_summary_df_export.at['Housing Units (HU)','Statistics']                     = format_commas_cell(model_summary_df.at['Housing Units (HU)','Statistics'])
	
	
	# Export model_summary_df_export to csv file in timestamped folder
	model_summary_df_export.to_csv(outputs_path + "/ModelRun_"+date+"/Model Reporting/Model_Summary.csv")
	
	
	
	
	
	# 9: VOLUME CORRIDORS SUMMARY FILE

	# Import blank dataframe with from template folder
	vol_corridor_summary_df = pd.read_csv(settings_dict['SharedData_Path']+'/template/reporting/volume_corridors_template.csv')

	# Set indices for vol_corridor_summary_df to use the Volume Corridor column
	vol_corridor_summary_df.set_index('Volume Corridor', inplace=True)
	
	# Drop rows with empty 'VOL_CORR' and 'VOL_CORR_2'
	#vol_link_df = link_df.dropna(subset=['VOL_CORR','VOL_CORR_2'], how = 'all')
	
	# Pull unique Volume Corridor values
	array1 = link_df['VOL_CORR'].dropna().to_numpy()
	array2 = link_df['VOL_CORR_2'].dropna().to_numpy()
	combined_array = np.concatenate((array1, array2), axis=0)
	unique_vol_corr = np.unique(combined_array)

	
	# Calculate average volume by corridor for PM Peak and Daily
	for i in unique_vol_corr:
		temp_df = link_df[(link_df['VOL_CORR'] == i) | (link_df['VOL_CORR_2'] == i)]
		# Multiply Number of lanes by 2 on divided links (only 1 instance of LinkNo). For later when calculating average number of lanes for the corridor
		# Identify rows with unique LinkNo values
		unique_linkno_rows = temp_df.drop_duplicates(subset='NO', keep=False)
		# For rows with unique LinkNo values, multiply NUMLANES by 2 and divide Length by 2
		temp_df.loc[temp_df['NO'].isin(unique_linkno_rows['NO']), 'NUMLANES'] *= 2
		temp_df.loc[temp_df['NO'].isin(unique_linkno_rows['NO']), 'LENGTH'] /= 2

		# Groupby LinkNo to get total volume and lanes for both directions
		temp_df = temp_df.groupby('NO').agg(
			LENGTH           =('LENGTH', 'max'),
			NUMLANES         =('NUMLANES', 'sum'),
			PMPKHR_Tot_Flow  =('PMPKHR_Tot_Flow', 'sum'),
			Tot_Flow         =('Tot_Flow', 'sum')
		).reset_index()

		# Use grouped dataframe for summary stats
		vol_corridor_summary_df.at[i,'Length (miles)'] = temp_df['LENGTH'].sum()
		vol_corridor_summary_df.at[i,'Avg PM Peak Hour Volume (both directions)'] = temp_df['PMPKHR_Tot_Flow'].mean()
		vol_corridor_summary_df.at[i,'Avg Daily Model Volume (both directions)']  = temp_df['Tot_Flow'].mean()
		
		
		# Calculate geometric average number of lanes and round to nearest even number, for LOS calculations
		temp_df['Pct_of_Length'] = temp_df['LENGTH'] / temp_df['LENGTH'].sum()
		vol_corridor_summary_df.at[i,'Number of Lanes'] = round_to_nearest_even(np.dot(temp_df['Pct_of_Length'], temp_df['NUMLANES']))
		numlanes = int(vol_corridor_summary_df.at[i,'Number of Lanes'])


		# Calculate Level of Service for each corridor based on volume and type of road from lookup tables
		# PM Peak LOS
		pm_los_num = vol_corridor_summary_df.at[i,'PM Peak Hour LOS Table']
		if pd.notna(pm_los_num):
			pm_los_tbl = pd.read_csv(settings_dict['SharedData_Path']+'/template/reporting/Volume_LOS_Tables/Individual_Tables/' + pm_los_num + '.csv')
			pm_los_tbl.set_index(pm_los_tbl['Lanes'].astype(int), inplace=True)
			# Multiply threshold values by adjustment factor specific to each corridor (USER INPUT)
			los_values = ['B','C','D','E','F']
			for los in los_values:
				if los in pm_los_tbl:
					pm_los_tbl[los] = pm_los_tbl[los] * (1 + vol_corridor_summary_df.at[i,'LOS Pct Adjustment']) + vol_corridor_summary_df.at[i,'LOS PM Peak Additive Adjustment']
			# Loop through to check which LOS bucket the specific corridor falls in based on Volume, NumLanes, and classification by LOS table
					if  vol_corridor_summary_df.at[i,'Avg PM Peak Hour Volume (both directions)'] <  pm_los_tbl.at[numlanes,los]:
						vol_corridor_summary_df.at[i,'PM Peak Hour LOS'] = los
						break
					elif vol_corridor_summary_df.at[i,'Avg PM Peak Hour Volume (both directions)'] > pm_los_tbl.at[numlanes,'F']:
						vol_corridor_summary_df.at[i,'PM Peak Hour LOS'] = 'F'
					else:
						continue
		else:
			continue
		# Daily LOS
		dly_los_num = vol_corridor_summary_df.at[i,'Daily LOS Table']
		if pd.notna(dly_los_num):
			dly_los_tbl = pd.read_csv(settings_dict['SharedData_Path']+'/template/reporting/Volume_LOS_Tables/Individual_Tables/' + dly_los_num + '.csv')
			dly_los_tbl.set_index(dly_los_tbl['Lanes'].astype(int), inplace=True)
			# Multiply threshold values by adjustment factor specific to each corridor (USER INPUT)
			los_values = ['B','C','D','E','F']
			for los in los_values:
				if los in dly_los_tbl:
					dly_los_tbl[los] = dly_los_tbl[los] * (1 + vol_corridor_summary_df.at[i,'LOS Pct Adjustment']) + vol_corridor_summary_df.at[i,'LOS Daily Additive Adjustment']
			# Loop through to check which LOS bucket the specific corridor falls in based on Volume, NumLanes, and classification by LOS table
					if  vol_corridor_summary_df.at[i,'Avg Daily Model Volume (both directions)'] <  dly_los_tbl.at[numlanes,los]:
						vol_corridor_summary_df.at[i,'Daily LOS'] = los
						break
					elif vol_corridor_summary_df.at[i,'Avg Daily Model Volume (both directions)'] >  dly_los_tbl.at[numlanes,'F']:
						vol_corridor_summary_df.at[i,'Daily LOS'] = 'F'
					else:
						continue

		else:
			continue
	
	# Export vol_corridor_summary_df to csv file in timestamped folder
	vol_corridor_summary_df.to_csv(outputs_path + "/ModelRun_"+date+"/Model Reporting/unformatted/volumecorridorsummary.csv")
	
	# Copy vol_corridor_summary_df to vol_corridor_summary_df_export to maintain formatting for future operations in vol_corridor_summary_df
	vol_corridor_summary_df_export = vol_corridor_summary_df.copy()
	
	# Format volume corridor summary table output cells to make them look better
	vol_corridor_summary_df_export['Length (miles)']                            = format_oneplace(vol_corridor_summary_df_export['Length (miles)'])
	vol_corridor_summary_df_export['Avg PM Peak Hour Volume (both directions)'] = format_commas(vol_corridor_summary_df_export['Avg PM Peak Hour Volume (both directions)'])
	vol_corridor_summary_df_export['Avg Daily Model Volume (both directions)']  = format_commas(vol_corridor_summary_df_export['Avg Daily Model Volume (both directions)'])
	
	# Drop LOS ID fields. They remain in original vol_corridor_summary_df
	vol_corridor_summary_df_export.drop(columns=['PM Peak Hour LOS Table'], inplace=True)
	vol_corridor_summary_df_export.drop(columns=['Daily LOS Table'], inplace=True)
	vol_corridor_summary_df_export.drop(columns=['LOS Pct Adjustment'], inplace=True)
	vol_corridor_summary_df_export.drop(columns=['LOS PM Peak Additive Adjustment'], inplace=True)
	vol_corridor_summary_df_export.drop(columns=['LOS Daily Additive Adjustment'], inplace=True)
	
	# Export vol_corridor_summary_df_export to csv file in timestamped folder
	vol_corridor_summary_df_export.to_csv(outputs_path + "/ModelRun_"+date+"/Model Reporting//Volume_Corridor_Summary.csv")
	
	
	
	
	
	
	# 10: TRAVEL TIME CORRIDORS SUMMARY FILE

	# Import blank dataframe with from template folder
	tt_corridor_summary_df = pd.read_csv(settings_dict['SharedData_Path']+'/template/reporting/traveltime_corridors_template.csv')
	
	# Set indices for vol_corridor_summary_df to use the Volume Corridor column
	tt_corridor_summary_df.set_index('Travel Time Corridor', inplace=True)
	
	# Drop rows with empty 'VOL_CORR'
	tt_link_df = link_df.dropna(subset=['TT_CORR'])
	
	# Pull unique Volume Corridor values
	unique_tt_corr = tt_link_df['TT_CORR'].unique()
	
	# Calculate average volume by corridor for PM Peak and Daily
	for i in unique_tt_corr:
		temp_df = tt_link_df[(tt_link_df['TT_CORR'] == i)]
		tt_corridor_summary_df.at[i,'Length (miles)']                        = temp_df['LENGTH'].sum()
		tt_corridor_summary_df.at[i,'Free Flow Travel Time (seconds)']       = temp_df['FFTIME_C'].sum()*3600
		tt_corridor_summary_df.at[i,'PM Peak Loaded Travel Time (seconds)']  = temp_df['CONGTIME_PMPK_C'].sum()*3600
		tt_corridor_summary_df.at[i,'Travel Time Ratio']                     = (temp_df['CONGTIME_PMPK_C'].sum()*3600) / (temp_df['FFTIME_C'].sum()*3600)
		if tt_corridor_summary_df.at[i,'Travel Time Ratio'] <= 1.17:
			tt_corridor_summary_df.at[i,'LOS'] = 'A'
		elif tt_corridor_summary_df.at[i,'Travel Time Ratio'] <= 1.50:
			tt_corridor_summary_df.at[i,'LOS'] = 'B'
		elif tt_corridor_summary_df.at[i,'Travel Time Ratio'] <= 2.00:
			tt_corridor_summary_df.at[i,'LOS'] = 'C'
		elif tt_corridor_summary_df.at[i,'Travel Time Ratio'] <= 2.50:
			tt_corridor_summary_df.at[i,'LOS'] = 'D'
		elif tt_corridor_summary_df.at[i,'Travel Time Ratio'] <= 3.32:
			tt_corridor_summary_df.at[i,'LOS'] = 'E'
		elif tt_corridor_summary_df.at[i,'Travel Time Ratio'] >  3.32:
			tt_corridor_summary_df.at[i,'LOS'] = 'F'
	
	
	# Export tt_corridor_summary_df to csv file in timestamped folder
	tt_corridor_summary_df.to_csv(outputs_path + "/ModelRun_"+date+"/Model Reporting/unformatted/ttcorridorsummary.csv")
	
	# Copy tt_corridor_summary_df to tt_corridor_summary_df_export to maintain formatting for future operations in tt_corridor_summary_df
	tt_corridor_summary_df_export = tt_corridor_summary_df.copy()
	
	# Format travel time corridor summary table output cells to make them look better
	tt_corridor_summary_df_export['Length (miles)']                        = format_oneplace(tt_corridor_summary_df_export['Length (miles)'])
	tt_corridor_summary_df_export['Free Flow Travel Time (seconds)']       = format_commas(tt_corridor_summary_df_export['Free Flow Travel Time (seconds)'])
	tt_corridor_summary_df_export['PM Peak Loaded Travel Time (seconds)']  = format_commas(tt_corridor_summary_df_export['PM Peak Loaded Travel Time (seconds)'])
	tt_corridor_summary_df_export['Travel Time Ratio']                     = format_twoplaces(tt_corridor_summary_df_export['Travel Time Ratio'])
	
	
	# Export tt_corridor_summary_df_export to csv file in timestamped folder
	tt_corridor_summary_df_export.to_csv(outputs_path + "/ModelRun_"+date+"/Model Reporting/Travel_Time_Corridor_Summary.csv")
	


# Export Matrices in list view to csv file
def matrix_export():
	
	# EXPORT MATRICES TO CSV FILE IN LIST VIEW
	
	# Import matrices in OD list view as a dataframe
	# From/To Zone
	FromZoneNo   = VisumPy.helpers.GetMulti(Visum.Net.ODPairs,"FROMZONENO")
	ToZoneNo   = VisumPy.helpers.GetMulti(Visum.Net.ODPairs,"TOZONENO")
	# LOV
	DLY_LOV      = VisumPy.helpers.GetMulti(Visum.Net.ODPairs,"MATVALUE(241)")
	AM_PK_HR_LOV = VisumPy.helpers.GetMulti(Visum.Net.ODPairs,"MATVALUE(255)")
	PM_PK_HR_LOV = VisumPy.helpers.GetMulti(Visum.Net.ODPairs,"MATVALUE(261)")
	AM_PER_LOV   = VisumPy.helpers.GetMulti(Visum.Net.ODPairs,"MATVALUE(267)")
	MD_PER_LOV   = VisumPy.helpers.GetMulti(Visum.Net.ODPairs,"MATVALUE(273)")	
	PM_PER_LOV   = VisumPy.helpers.GetMulti(Visum.Net.ODPairs,"MATVALUE(279)")
	NI_PER_LOV   = VisumPy.helpers.GetMulti(Visum.Net.ODPairs,"MATVALUE(285)")	
	# HOV
	DLY_HOV      = VisumPy.helpers.GetMulti(Visum.Net.ODPairs,"MATVALUE(243)")
	AM_PK_HR_HOV = VisumPy.helpers.GetMulti(Visum.Net.ODPairs,"MATVALUE(257)")
	PM_PK_HR_HOV = VisumPy.helpers.GetMulti(Visum.Net.ODPairs,"MATVALUE(263)")
	AM_PER_HOV   = VisumPy.helpers.GetMulti(Visum.Net.ODPairs,"MATVALUE(269)")
	MD_PER_HOV   = VisumPy.helpers.GetMulti(Visum.Net.ODPairs,"MATVALUE(275)")	
	PM_PER_HOV   = VisumPy.helpers.GetMulti(Visum.Net.ODPairs,"MATVALUE(281)")
	NI_PER_HOV   = VisumPy.helpers.GetMulti(Visum.Net.ODPairs,"MATVALUE(287)")	
	# Truck
	DLY_Truck      = VisumPy.helpers.GetMulti(Visum.Net.ODPairs,"MATVALUE(245)")
	AM_PK_HR_Truck = VisumPy.helpers.GetMulti(Visum.Net.ODPairs,"MATVALUE(259)")
	PM_PK_HR_Truck = VisumPy.helpers.GetMulti(Visum.Net.ODPairs,"MATVALUE(265)")
	AM_PER_Truck   = VisumPy.helpers.GetMulti(Visum.Net.ODPairs,"MATVALUE(271)")
	MD_PER_Truck   = VisumPy.helpers.GetMulti(Visum.Net.ODPairs,"MATVALUE(277)")	
	PM_PER_Truck   = VisumPy.helpers.GetMulti(Visum.Net.ODPairs,"MATVALUE(283)")
	NI_PER_Truck   = VisumPy.helpers.GetMulti(Visum.Net.ODPairs,"MATVALUE(289)")	
	# LCV
	DLY_LCV      = VisumPy.helpers.GetMulti(Visum.Net.ODPairs,"MATVALUE(246)")
	AM_PK_HR_LCV = VisumPy.helpers.GetMulti(Visum.Net.ODPairs,"MATVALUE(538)")
	PM_PK_HR_LCV = VisumPy.helpers.GetMulti(Visum.Net.ODPairs,"MATVALUE(539)")
	AM_PER_LCV   = VisumPy.helpers.GetMulti(Visum.Net.ODPairs,"MATVALUE(534)")
	MD_PER_LCV   = VisumPy.helpers.GetMulti(Visum.Net.ODPairs,"MATVALUE(535)")	
	PM_PER_LCV   = VisumPy.helpers.GetMulti(Visum.Net.ODPairs,"MATVALUE(536)")
	NI_PER_LCV   = VisumPy.helpers.GetMulti(Visum.Net.ODPairs,"MATVALUE(537)")	
	# Ext_Auto
	DLY_Ext_Auto      = VisumPy.helpers.GetMulti(Visum.Net.ODPairs,"MATVALUE(576)")
	AM_PK_HR_Ext_Auto = VisumPy.helpers.GetMulti(Visum.Net.ODPairs,"MATVALUE(582)")
	PM_PK_HR_Ext_Auto = VisumPy.helpers.GetMulti(Visum.Net.ODPairs,"MATVALUE(583)")
	AM_PER_Ext_Auto   = VisumPy.helpers.GetMulti(Visum.Net.ODPairs,"MATVALUE(578)")
	MD_PER_Ext_Auto   = VisumPy.helpers.GetMulti(Visum.Net.ODPairs,"MATVALUE(579)")	
	PM_PER_Ext_Auto   = VisumPy.helpers.GetMulti(Visum.Net.ODPairs,"MATVALUE(580)")
	NI_PER_Ext_Auto   = VisumPy.helpers.GetMulti(Visum.Net.ODPairs,"MATVALUE(581)")	
	# Ext_Truck
	DLY_Ext_Truck      = VisumPy.helpers.GetMulti(Visum.Net.ODPairs,"MATVALUE(577)")
	AM_PK_HR_Ext_Truck = VisumPy.helpers.GetMulti(Visum.Net.ODPairs,"MATVALUE(589)")
	PM_PK_HR_Ext_Truck = VisumPy.helpers.GetMulti(Visum.Net.ODPairs,"MATVALUE(590)")
	AM_PER_Ext_Truck   = VisumPy.helpers.GetMulti(Visum.Net.ODPairs,"MATVALUE(585)")
	MD_PER_Ext_Truck   = VisumPy.helpers.GetMulti(Visum.Net.ODPairs,"MATVALUE(586)")	
	PM_PER_Ext_Truck   = VisumPy.helpers.GetMulti(Visum.Net.ODPairs,"MATVALUE(587)")
	NI_PER_Ext_Truck   = VisumPy.helpers.GetMulti(Visum.Net.ODPairs,"MATVALUE(588)")
	
	
	# Make Visum list with matrix (OD Pair) data
	summary_list = [FromZoneNo, ToZoneNo,
					DLY_LOV,   AM_PK_HR_LOV,  PM_PK_HR_LOV,  AM_PER_LOV,  MD_PER_LOV,  PM_PER_LOV,  NI_PER_LOV,
					DLY_HOV,   AM_PK_HR_HOV,  PM_PK_HR_HOV,  AM_PER_HOV,  MD_PER_HOV,  PM_PER_HOV,  NI_PER_HOV,
					DLY_Truck, AM_PK_HR_Truck,PM_PK_HR_Truck,AM_PER_Truck,MD_PER_Truck,PM_PER_Truck,NI_PER_Truck,
					DLY_LCV,   AM_PK_HR_LCV,  PM_PK_HR_LCV,  AM_PER_LCV,  MD_PER_LCV,  PM_PER_LCV,  NI_PER_LCV ,
					DLY_Ext_Auto,  AM_PK_HR_Ext_Auto ,PM_PK_HR_Ext_Auto ,AM_PER_Ext_Auto ,MD_PER_Ext_Auto ,PM_PER_Ext_Auto ,NI_PER_Ext_Auto,
					DLY_Ext_Truck, AM_PK_HR_Ext_Truck,PM_PK_HR_Ext_Truck,AM_PER_Ext_Truck,MD_PER_Ext_Truck,PM_PER_Ext_Truck,NI_PER_Ext_Truck]
			
	# Put Visum link list into dataframe  
	output_mtxs_df = pd.DataFrame(np.column_stack(summary_list), columns = ['FromZoneNo', 'ToZoneNo', 
								'DLY_LOV',  'AM_PK_HR_LOV',  'PM_PK_HR_LOV',  'AM_PER_LOV',  'MD_PER_LOV',  'PM_PER_LOV',  'NI_PER_LOV',
								'DLY_HOV',  'AM_PK_HR_HOV',  'PM_PK_HR_HOV',  'AM_PER_HOV',  'MD_PER_HOV',  'PM_PER_HOV',  'NI_PER_HOV',
								'DLY_Truck','AM_PK_HR_Truck','PM_PK_HR_Truck','AM_PER_Truck','MD_PER_Truck','PM_PER_Truck','NI_PER_Truck',
								'DLY_LCV',  'AM_PK_HR_LCV',  'PM_PK_HR_LCV',  'AM_PER_LCV',  'MD_PER_LCV',  'PM_PER_LCV',  'NI_PER_LCV',
								'DLY_Ext_Auto', 'AM_PK_HR_Ext_Auto' ,'PM_PK_HR_Ext_Auto' ,'AM_PER_Ext_Auto' ,'MD_PER_Ext_Auto' ,'PM_PER_Ext_Auto' ,'NI_PER_Ext_Auto',
								'DLY_Ext_Truck','AM_PK_HR_Ext_Truck','PM_PK_HR_Ext_Truck','AM_PER_Ext_Truck','MD_PER_Ext_Truck','PM_PER_Ext_Truck','NI_PER_Ext_Truck'])
								
								
	# Format matrix value to have 2 decimal places
	matrices = ['FromZoneNo', 'ToZoneNo', 
				'DLY_LOV',  'AM_PK_HR_LOV',  'PM_PK_HR_LOV',  'AM_PER_LOV',  'MD_PER_LOV',  'PM_PER_LOV',  'NI_PER_LOV',
				'DLY_HOV',  'AM_PK_HR_HOV',  'PM_PK_HR_HOV',  'AM_PER_HOV',  'MD_PER_HOV',  'PM_PER_HOV',  'NI_PER_HOV',
				'DLY_Truck','AM_PK_HR_Truck','PM_PK_HR_Truck','AM_PER_Truck','MD_PER_Truck','PM_PER_Truck','NI_PER_Truck',
				'DLY_LCV',  'AM_PK_HR_LCV',  'PM_PK_HR_LCV',  'AM_PER_LCV',  'MD_PER_LCV',  'PM_PER_LCV',  'NI_PER_LCV',
				'DLY_Ext_Auto', 'AM_PK_HR_Ext_Auto' ,'PM_PK_HR_Ext_Auto' ,'AM_PER_Ext_Auto' ,'MD_PER_Ext_Auto' ,'PM_PER_Ext_Auto' ,'NI_PER_Ext_Auto',
				'DLY_Ext_Truck','AM_PK_HR_Ext_Truck','PM_PK_HR_Ext_Truck','AM_PER_Ext_Truck','MD_PER_Ext_Truck','PM_PER_Ext_Truck','NI_PER_Ext_Truck']
	
	for mtx in matrices:
		output_mtxs_df[mtx] = format_twoplaces(output_mtxs_df[mtx])
	
	
	
	# Export matrices as csv
	output_mtxs_df.to_csv(outputs_path + "/ModelRun_"+date+"/Matrices/Model_Matrices.csv")
	
	


# Export Link Table to csv
def linktbl_export():
	
	# EXPORT LINK TABLE TO CSV FILE
	
	# Import Link fields 
	# Link ID fields
	NO          = VisumPy.helpers.GetMulti(Visum.Net.Links,"No", activeOnly = True)
	FromNodeNO  = VisumPy.helpers.GetMulti(Visum.Net.Links,"FROMNODENO", activeOnly = True)
	ToNodeNO    = VisumPy.helpers.GetMulti(Visum.Net.Links,"TONODENO", activeOnly = True)
	LINKSERIAL  = VisumPy.helpers.GetMulti(Visum.Net.Links,"LINKSERIAL", activeOnly = True)
	FCLASS      = VisumPy.helpers.GetMulti(Visum.Net.Links,"TYPENO", activeOnly = True)
	EXT         = VisumPy.helpers.GetMulti(Visum.Net.Links,"EXT_COUNT", activeOnly = True)
	LENGTH      = VisumPy.helpers.GetMulti(Visum.Net.Links,"Length", activeOnly = True)
	NUMLANES    = VisumPy.helpers.GetMulti(Visum.Net.Links,"NUMLANES", activeOnly = True)
	# Free-Flow Time
	FFTIME_C    = VisumPy.helpers.GetMulti(Visum.Net.Links,"FFTIME_C", activeOnly = True)
	FFTIME_T    = VisumPy.helpers.GetMulti(Visum.Net.Links,"FFTIME_T", activeOnly = True)
	# Pull CONGTIME Auto by period, Length, and Flows by Period for VMT/VHT Calculations 	
	CONGTIME_AM_C   = VisumPy.helpers.GetMulti(Visum.Net.Links,"AM3_CTIME_C", activeOnly = True)
	CONGTIME_MD_C   = VisumPy.helpers.GetMulti(Visum.Net.Links,"MD_CTIME_C", activeOnly = True)
	CONGTIME_PM_C   = VisumPy.helpers.GetMulti(Visum.Net.Links,"PM3_CTIME_C", activeOnly = True)
	CONGTIME_NI_C   = VisumPy.helpers.GetMulti(Visum.Net.Links,"NI_CTIME_C", activeOnly = True)
	CONGTIME_PMPK_C = VisumPy.helpers.GetMulti(Visum.Net.Links,"PM1_CTIME_C", activeOnly = True)
	# Pull CONGTIME Truck by period, Length, and Flows by Period for VMT/VHT Calculations 	
	CONGTIME_AM_T   = VisumPy.helpers.GetMulti(Visum.Net.Links,"AM3_CTIME_T", activeOnly = True)
	CONGTIME_MD_T   = VisumPy.helpers.GetMulti(Visum.Net.Links,"MD_CTIME_T", activeOnly = True)
	CONGTIME_PM_T   = VisumPy.helpers.GetMulti(Visum.Net.Links,"PM3_CTIME_T", activeOnly = True)
	CONGTIME_NI_T   = VisumPy.helpers.GetMulti(Visum.Net.Links,"NI_CTIME_T", activeOnly = True)
	CONGTIME_PMPK_T = VisumPy.helpers.GetMulti(Visum.Net.Links,"PM1_CTIME_T", activeOnly = True)
	
	# Link Counts by Period
	# AM
	AM_Auto_Count  = VisumPy.helpers.GetMulti(Visum.Net.Links,"AUTO_AM_COUNT", activeOnly = True)
	AM_Truck_Count = VisumPy.helpers.GetMulti(Visum.Net.Links,"TRUCK_AM_COUNT", activeOnly = True)
	AM_Tot_Count   = VisumPy.helpers.GetMulti(Visum.Net.Links,"ALL_AM_COUNT", activeOnly = True)
	# MD
	MD_Auto_Count  = VisumPy.helpers.GetMulti(Visum.Net.Links,"AUTO_MD_COUNT", activeOnly = True)
	MD_Truck_Count = VisumPy.helpers.GetMulti(Visum.Net.Links,"TRUCK_MD_COUNT", activeOnly = True)
	MD_Tot_Count   = VisumPy.helpers.GetMulti(Visum.Net.Links,"ALL_MD_COUNT", activeOnly = True)
	# PM
	PM_Auto_Count  = VisumPy.helpers.GetMulti(Visum.Net.Links,"AUTO_PM_COUNT", activeOnly = True)
	PM_Truck_Count = VisumPy.helpers.GetMulti(Visum.Net.Links,"TRUCK_PM_COUNT", activeOnly = True)
	PM_Tot_Count   = VisumPy.helpers.GetMulti(Visum.Net.Links,"ALL_PM_COUNT", activeOnly = True)
	# NT
	NI_Auto_Count  = VisumPy.helpers.GetMulti(Visum.Net.Links,"AUTO_NI_COUNT", activeOnly = True)
	NI_Truck_Count = VisumPy.helpers.GetMulti(Visum.Net.Links,"TRUCK_NI_COUNT", activeOnly = True)
	NI_Tot_Count   = VisumPy.helpers.GetMulti(Visum.Net.Links,"ALL_NI_COUNT", activeOnly = True)
	# PM Peak Hour
	PMPKHR_Auto_Count  = VisumPy.helpers.GetMulti(Visum.Net.Links,"AUTO_PMPK_COUNT", activeOnly = True)
	PMPKHR_Truck_Count = VisumPy.helpers.GetMulti(Visum.Net.Links,"TRUCK_PMPK_COUNT", activeOnly = True)
	PMPKHR_Tot_Count   = VisumPy.helpers.GetMulti(Visum.Net.Links,"ALL_PMPK_COUNT", activeOnly = True)
	# AM Peak Hour
	AMPKHR_Auto_Count  = VisumPy.helpers.GetMulti(Visum.Net.Links,"AUTO_AMPK_COUNT", activeOnly = True)
	AMPKHR_Truck_Count = VisumPy.helpers.GetMulti(Visum.Net.Links,"TRUCK_AMPK_COUNT", activeOnly = True)
	AMPKHR_Tot_Count   = VisumPy.helpers.GetMulti(Visum.Net.Links,"ALL_AMPK_COUNT", activeOnly = True)
	# Daily
	Auto_Count  = VisumPy.helpers.GetMulti(Visum.Net.Links,"AADT_AUTO", activeOnly = True)
	Truck_Count = VisumPy.helpers.GetMulti(Visum.Net.Links,"AADT_TRUCK", activeOnly = True)
	Tot_Count   = VisumPy.helpers.GetMulti(Visum.Net.Links,"AADT", activeOnly = True)
	
	# Link Flows by Period
	# AM
	AM_Auto_Flow  = VisumPy.helpers.GetMulti(Visum.Net.Links,"AM_PERIOD_AUTO_VOLUME", activeOnly = True)
	AM_Truck_Flow = VisumPy.helpers.GetMulti(Visum.Net.Links,"AM_PERIOD_TRUCK_VOLUME", activeOnly = True)
	AM_Tot_Flow   = VisumPy.helpers.GetMulti(Visum.Net.Links,"AM_PERIOD_MODEL_VOLUME", activeOnly = True)
	# MD
	MD_Auto_Flow  = VisumPy.helpers.GetMulti(Visum.Net.Links,"MIDDAY_PERIOD_AUTO_VOLUME", activeOnly = True)
	MD_Truck_Flow = VisumPy.helpers.GetMulti(Visum.Net.Links,"MIDDAY_PERIOD_TRUCK_VOLUME", activeOnly = True)
	MD_Tot_Flow   = VisumPy.helpers.GetMulti(Visum.Net.Links,"MIDDAY_PERIOD_MODEL_VOLUME", activeOnly = True)
	# PM
	PM_Auto_Flow  = VisumPy.helpers.GetMulti(Visum.Net.Links,"PM_PERIOD_AUTO_VOLUME", activeOnly = True)
	PM_Truck_Flow = VisumPy.helpers.GetMulti(Visum.Net.Links,"PM_PERIOD_TRUCK_VOLUME", activeOnly = True)
	PM_Tot_Flow   = VisumPy.helpers.GetMulti(Visum.Net.Links,"PM_PERIOD_MODEL_VOLUME", activeOnly = True)
	# NT
	NI_Auto_Flow  = VisumPy.helpers.GetMulti(Visum.Net.Links,"NIGHT_PERIOD_AUTO_VOLUME", activeOnly = True)
	NI_Truck_Flow = VisumPy.helpers.GetMulti(Visum.Net.Links,"NIGHT_PERIOD_TRUCK_VOLUME", activeOnly = True)
	NI_Tot_Flow   = VisumPy.helpers.GetMulti(Visum.Net.Links,"NIGHT_PERIOD_MODEL_VOLUME", activeOnly = True)
	# PM Peak Hour
	PMPKHR_Auto_Flow  = VisumPy.helpers.GetMulti(Visum.Net.Links,"PMPKHR_AUTO_VOL", activeOnly = True)
	PMPKHR_Truck_Flow = VisumPy.helpers.GetMulti(Visum.Net.Links,"PMPKHR_TRUCK_VOL", activeOnly = True)
	PMPKHR_Tot_Flow   = VisumPy.helpers.GetMulti(Visum.Net.Links,"PMPKHR_VOL", activeOnly = True)
	# AM Peak Hour
	AMPKHR_Auto_Flow  = VisumPy.helpers.GetMulti(Visum.Net.Links,"AMPKHR_AUTO_VOL", activeOnly = True)
	AMPKHR_Truck_Flow = VisumPy.helpers.GetMulti(Visum.Net.Links,"AMPKHR_TRUCK_VOL", activeOnly = True)
	AMPKHR_Tot_Flow   = VisumPy.helpers.GetMulti(Visum.Net.Links,"AMPKHR_VOL", activeOnly = True)
	# Daily
	Auto_Flow  = VisumPy.helpers.GetMulti(Visum.Net.Links,"DAILY_AUTO_VOLUME", activeOnly = True)
	Truck_Flow = VisumPy.helpers.GetMulti(Visum.Net.Links,"DAILY_TRUCK_VOLUME", activeOnly = True)
	Tot_Flow   = VisumPy.helpers.GetMulti(Visum.Net.Links,"DAILY_MODEL_VOLUME", activeOnly = True)
	LCV_Flow   = VisumPy.helpers.GetMulti(Visum.Net.Links,"DAILY_LCV_VOL", activeOnly = True)
	
	
	# Make Visum list with link data
	summary_list = [NO, FromNodeNO, ToNodeNO, LINKSERIAL, FCLASS, EXT, LENGTH, NUMLANES, FFTIME_C, FFTIME_T,
					CONGTIME_AM_C, CONGTIME_MD_C, CONGTIME_PM_C, CONGTIME_NI_C, CONGTIME_PMPK_C,
					CONGTIME_AM_T, CONGTIME_MD_T, CONGTIME_PM_T, CONGTIME_NI_T, CONGTIME_PMPK_T,
					AM_Auto_Count, AM_Truck_Count, AM_Tot_Count, MD_Auto_Count, MD_Truck_Count, MD_Tot_Count,
					PM_Auto_Count, PM_Truck_Count, PM_Tot_Count, NI_Auto_Count, NI_Truck_Count, NI_Tot_Count,
					PMPKHR_Auto_Count, PMPKHR_Truck_Count, PMPKHR_Tot_Count, AMPKHR_Auto_Count, AMPKHR_Truck_Count, AMPKHR_Tot_Count,
					Auto_Count, Truck_Count, Tot_Count,
					AM_Auto_Flow, AM_Truck_Flow, AM_Tot_Flow, MD_Auto_Flow, MD_Truck_Flow, MD_Tot_Flow,
					PM_Auto_Flow, PM_Truck_Flow, PM_Tot_Flow, NI_Auto_Flow, NI_Truck_Flow, NI_Tot_Flow,
					PMPKHR_Auto_Flow, PMPKHR_Truck_Flow, PMPKHR_Tot_Flow, AMPKHR_Auto_Flow, AMPKHR_Truck_Flow, AMPKHR_Tot_Flow,
					Auto_Flow, Truck_Flow, Tot_Flow, LCV_Flow]
			
	# Put Visum link list into dataframe  
	link_df = pd.DataFrame(np.column_stack(summary_list), columns = ['NO', 'FromNodeNO', 'ToNodeNO', 'LINKSERIAL', 'FCLASS', 'EXTERNAL', 'LENGTH', 'NUMLANES','FFTIME_C', 'FFTIME_T',
																'CONGTIME_AM_C', 'CONGTIME_MD_C', 'CONGTIME_PM_C', 'CONGTIME_NI_C', 'CONGTIME_PMPK_C',
																'CONGTIME_AM_T', 'CONGTIME_MD_T', 'CONGTIME_PM_T', 'CONGTIME_NI_T', 'CONGTIME_PMPK_T',
																'AM_Auto_Count', 'AM_Truck_Count', 'AM_Tot_Count', 'MD_Auto_Count', 'MD_Truck_Count', 'MD_Tot_Count',
																'PM_Auto_Count', 'PM_Truck_Count', 'PM_Tot_Count', 'NI_Auto_Count', 'NI_Truck_Count', 'NI_Tot_Count',
																'PMPKHR_Auto_Count', 'PMPKHR_Truck_Count', 'PMPKHR_Tot_Count', 'AMPKHR_Auto_Count', 'AMPKHR_Truck_Count', 'AMPKHR_Tot_Count',
																'Auto_Count', 'Truck_Count', 'Tot_Count',
																'AM_Auto_Flow', 'AM_Truck_Flow', 'AM_Tot_Flow', 'MD_Auto_Flow', 'MD_Truck_Flow', 'MD_Tot_Flow',
																'PM_Auto_Flow', 'PM_Truck_Flow', 'PM_Tot_Flow', 'NI_Auto_Flow', 'NI_Truck_Flow', 'NI_Tot_Flow',
																'PMPKHR_Auto_Flow', 'PMPKHR_Truck_Flow', 'PMPKHR_Tot_Flow', 'AMPKHR_Auto_Flow', 'AMPKHR_Truck_Flow', 'AMPKHR_Tot_Flow',
																'Auto_Flow', 'Truck_Flow', 'Tot_Flow', 'LCV_Flow'])

	
	# Export link table as csv
	link_df.to_csv(outputs_path + "/ModelRun_"+date+"/Network/LinkTable.csv")
	
	
	
	

# Export Shapefile of network
def network_shp_export():

	# Create export shapefile parameters object
	shp_export_params = Visum.IO.CreateExportShapeFilePara()
	
	# Set options for export in parameters object
	shp_export_params.SetAttValue("OBJECTTYPE", 0)
	shp_export_params.SetAttValue("DIRECTED", 1)
	shp_export_params.SetAttValue("ONLYACTIVE", 1)
	
	# Set columns for export in parameters object
	shp_export_params.ClearLayout()
	shp_export_params.AddColumn("NO")
	shp_export_params.AddColumn("FROMNODENO")
	shp_export_params.AddColumn("TONODENO")
	shp_export_params.AddColumn("LINKSERIAL")
	shp_export_params.AddColumn("Length")
	
	Visum.IO.ExportShapefile(outputs_path + "/ModelRun_"+date+"/Network/Network.shp", shp_export_params)
	
	
	

# Run all of the above functions
# assignment_summary(auto_count, auto_flow, cong_auto_time, period_label):
# Daily

assignment_summary('COUNT_2018', 'PMPK_AUTO_VOLUME',  'PMPK_CTIME', 'PMPeak2018')  # Daily VHT Calculated from full day, so AM3_CTIME here is just a placeholder for the function

# AM Period
#assignment_summary('AUTO_AM_COUNT', 'TRUCK_AM_COUNT', 'ALL_AM_COUNT', 'AM_PERIOD_AUTO_VOLUME', 'AM_PERIOD_TRUCK_VOLUME', 'AM_PERIOD_MODEL_VOLUME', 'AM_PERIOD_LCV_VOL', 'AM3_CTIME_C', 'AM3_CTIME_T', 'AM_Period')

# MD Period
#assignment_summary('AUTO_MD_COUNT', 'TRUCK_MD_COUNT', 'ALL_MD_COUNT', 'MIDDAY_PERIOD_AUTO_VOLUME', 'MIDDAY_PERIOD_TRUCK_VOLUME', 'MIDDAY_PERIOD_MODEL_VOLUME', 'MIDDAY_PERIOD_LCV_VOL', 'MD_CTIME_C', 'MD_CTIME_T', 'MD_Period')

# PM Period
#assignment_summary('AUTO_PM_COUNT', 'TRUCK_PM_COUNT', 'ALL_PM_COUNT', 'PM_PERIOD_AUTO_VOLUME', 'PM_PERIOD_TRUCK_VOLUME', 'PM_PERIOD_MODEL_VOLUME', 'PM_PERIOD_LCV_VOL', 'PM3_CTIME_C', 'PM3_CTIME_T', 'PM_Period')

# NT Period
# assignment_summary('AUTO_NI_COUNT', 'TRUCK_NI_COUNT', 'ALL_NI_COUNT', 'NIGHT_PERIOD_AUTO_VOLUME', 'NIGHT_PERIOD_TRUCK_VOLUME', 'NIGHT_PERIOD_MODEL_VOLUME', 'NIGHT_PERIOD_LCV_VOL', 'NI_CTIME_C', 'NI_CTIME_T', 'NI_Period')

# AMPK Period
#assignment_summary('AUTO_AMPK_COUNT', 'TRUCK_AMPK_COUNT', 'ALL_AMPK_COUNT', 'AMPKHR_AUTO_VOL', 'AMPKHR_TRUCK_VOL', 'AMPKHR_VOL', 'AMPKHR_LCV_VOL', 'AM1_CTIME_C', 'AM1_CTIME_T', 'AM_Peak_Hour')

# PMPK Period
#assignment_summary('AUTO_PMPK_COUNT', 'TRUCK_PMPK_COUNT', 'ALL_PMPK_COUNT', 'PMPKHR_AUTO_VOL', 'PMPKHR_TRUCK_VOL', 'PMPKHR_VOL', 'PMPKHR_LCV_VOL', 'PM1_CTIME_C', 'PM1_CTIME_T', 'PM_Peak_Hour')


# Count station summary
#count_station_summary()


# Model Reporting
#model_reporting()



# Export Matrices
#matrix_export()


# Export Link table
#linktbl_export()


# Export network shapefile
#network_shp_export()







