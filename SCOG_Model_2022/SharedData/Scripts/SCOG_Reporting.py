"""
Model Trip, VMT, VHT, VHD, LOS reporting
michael.mccarthy@rsginc.com 2025-07-28
revised 2025-11-13

"""

# Libraries
import sys
import numpy as np
import VisumPy.helpers
import VisumPy.matrices
import pandas as pd
from datetime import datetime
import os.path

# set paths 
out_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..', 'Outputs'))
shared_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..', 'SharedData'))
reports_path = shared_path + "/Reports/"
timestamp = datetime.now().strftime("%Y%m%d") # _%H%M
scenname = Visum.Net.AttValue("SCENARIO")

# Create timestamped folder for all results
scen_out = out_path + "/ModelRun_"+scenname+"_"+timestamp

if not os.path.exists(scen_out):
	os.mkdir(scen_out)

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

# Pull Link attributes from Visum and create dataframe
no          = VisumPy.helpers.GetMulti(Visum.Net.Links,"No")
length      = VisumPy.helpers.GetMulti(Visum.Net.Links,"Length")
typeno      = VisumPy.helpers.GetMulti(Visum.Net.Links,"TypeNo")
tsys        = VisumPy.helpers.GetMulti(Visum.Net.Links,"TSysSet")
nfc         = VisumPy.helpers.GetMulti(Visum.Net.Links,"NFCLASS")
lanes         = VisumPy.helpers.GetMulti(Visum.Net.Links,"NUMLANES")
amvol       = VisumPy.helpers.GetMulti(Visum.Net.Links,"AM_AUTO_VOLUME")
pmvol       = VisumPy.helpers.GetMulti(Visum.Net.Links,"PM_AUTO_VOLUME")
pmpkvol     = VisumPy.helpers.GetMulti(Visum.Net.Links,"PMPK_AUTO_VOLUME")
opvol       = VisumPy.helpers.GetMulti(Visum.Net.Links,"OP_AUTO_VOLUME")
dlyvol      = VisumPy.helpers.GetMulti(Visum.Net.Links,"DLY_AUTO_VOLUME")
fftime      = VisumPy.helpers.GetMulti(Visum.Net.Links,"T0_PRTSYS(C)")
amctime     = VisumPy.helpers.GetMulti(Visum.Net.Links,"AM_CTIME")
pmctime     = VisumPy.helpers.GetMulti(Visum.Net.Links,"PM_CTIME")
opctime     = VisumPy.helpers.GetMulti(Visum.Net.Links,"OP_CTIME")
pmpkctime   = VisumPy.helpers.GetMulti(Visum.Net.Links,"PMPK_CTIME")
VOL_CORR    = VisumPy.helpers.GetMulti(Visum.Net.Links,"VOL_CORRIDOR")
VOL_CORR_2  = VisumPy.helpers.GetMulti(Visum.Net.Links,"VOL_CORRIDOR_2")
TT_CORR     = VisumPy.helpers.GetMulti(Visum.Net.Links,"TT_CORRIDOR")


# Set up dataframe to use for all needed zone attributes. Update as needed during coding
links_df = pd.DataFrame({'NO':no, 'LENGTH': length, 'TYPENO': typeno, 'TSYSSET': tsys, 'NFCLASS': nfc, 'AM_AUTO_VOLUME': amvol, 'PM_AUTO_VOLUME': pmvol, 'PMPK_AUTO_VOLUME': pmpkvol, 'OP_AUTO_VOLUME': opvol, 'DLY_AUTO_VOLUME': dlyvol,  'FFTIME': fftime, 
    'AMTIME': amctime, 'PMTIME': pmctime, 'OPTIME': opctime, 'PMPKTIME': pmpkctime, 'NUMLANES': lanes, 'VOL_CORR': VOL_CORR, 'VOL_CORR_2': VOL_CORR_2, 'TT_CORR': TT_CORR})

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

# Vehicle Trips
HBW_vehicle = mtx_df.loc[mtx_df['Name'] == 'HBWpa','Sum'].sum()
HBO_vehicle = mtx_df.loc[mtx_df['Name'] == 'HBOpa','Sum'].sum()
NHB_vehicle = mtx_df.loc[mtx_df['Name'] == 'NHBpa','Sum'].sum()
XIW_vehicle = mtx_df.loc[mtx_df['Name'] == ',XI_NonRes_Work','Sum'].sum()
IXW_vehicle = mtx_df.loc[mtx_df['Name'] == ',IX_Res_Work','Sum'].sum()
XIIXNW_vehicle = mtx_df.loc[mtx_df['Name'] == 'XIIX_NonWork','Sum'].sum()
XX_vehicle = mtx_df.loc[mtx_df['Name'] == 'AutoXX','Sum'].sum()
Int_vehicle = HBW_vehicle + HBO_vehicle + NHB_vehicle
Ext_vehicle = IXW_vehicle + IXW_vehicle + XIIXNW_vehicle + XX_vehicle
Tot_vehicle = Int_vehicle + Ext_vehicle

# Person Trips
HBW_person = HBW_vehicle * 1.031
HBO_person = HBO_vehicle * 1.49
NHB_person = NHB_vehicle * 1.2259
XIW_person = mtx_df.loc[mtx_df['Name'] == 'XI_NonRes_Work_person','Sum'].sum()
IXW_person = mtx_df.loc[mtx_df['Name'] == 'IX_Res_Work_person','Sum'].sum()
XIIXNW_person = mtx_df.loc[mtx_df['Name'] == 'XIIX_NonWork_person','Sum'].sum()
XX_person = mtx_df.loc[mtx_df['Name'] == 'XX_person','Sum'].sum()
Int_person = HBW_person + HBO_person + NHB_person
Ext_person = IXW_person + IXW_person + XIIXNW_person + XX_person
Tot_person = Int_person + Ext_person

# Vehicle Trips
# Regional VMT, VHT, VHD (times are in seconds)
links_df['VMT_DLY'] = links_df['LENGTH'] * links_df['DLY_AUTO_VOLUME']
links_df['VMT_AM'] = links_df['LENGTH'] * links_df['AM_AUTO_VOLUME']
links_df['VMT_PM'] = links_df['LENGTH'] * links_df['PM_AUTO_VOLUME']
links_df['VMT_OP'] = links_df['LENGTH'] * links_df['OP_AUTO_VOLUME']
links_df['VMT_PMPK'] = links_df['LENGTH'] * links_df['PMPK_AUTO_VOLUME']
links_df['VHT_DLY_ff']    = (links_df['FFTIME'] / (60*60)) * links_df['DLY_AUTO_VOLUME']
links_df['VHT_AM_ff']     = (links_df['FFTIME'] / (60*60)) * links_df['AM_AUTO_VOLUME']
links_df['VHT_PM_ff']     = (links_df['FFTIME'] / (60*60)) * links_df['PM_AUTO_VOLUME']
links_df['VHT_OP_ff']     = (links_df['FFTIME'] / (60*60)) * links_df['OP_AUTO_VOLUME']
links_df['VHT_PMPK_ff']   = (links_df['FFTIME'] / (60*60)) * links_df['PMPK_AUTO_VOLUME']
links_df['VHT_AM_cong']   = (links_df['AMTIME'] / (60*60)) * links_df['AM_AUTO_VOLUME']
links_df['VHT_PM_cong']   = (links_df['PMTIME'] / (60*60)) * links_df['PM_AUTO_VOLUME']
links_df['VHT_OP_cong']   = (links_df['OPTIME'] / (60*60)) * links_df['OP_AUTO_VOLUME']
links_df['VHT_PMPK_cong'] = (links_df['PMPKTIME'] / (60*60)) * links_df['PMPK_AUTO_VOLUME']
links_df['VHT_DLY_cong'] = links_df['VHT_AM_cong'] + links_df['VHT_PM_cong'] + links_df['VHT_OP_cong']
links_df['VHD_DLY'] = links_df['VHT_DLY_cong'] - links_df['VHT_DLY_ff']
links_df['VHD_AM'] = links_df['VHT_AM_cong'] - links_df['VHT_AM_ff']
links_df['VHD_PM'] = links_df['VHT_PM_cong'] - links_df['VHT_PM_ff']
links_df['VHD_OP'] = links_df['VHT_OP_cong'] - links_df['VHT_OP_ff']
links_df['VHD_PMPK'] = links_df['VHT_PMPK_cong'] - links_df['VHT_PMPK_ff']

# Summarize link statistics
summary_df = links_df.sum().to_frame().T

# Import Zone fields for Population, Employment, and Housing Units
pop  = VisumPy.helpers.GetMulti(Visum.Net.Zones,"TotPopulation")
emp  = VisumPy.helpers.GetMulti(Visum.Net.Zones,"TotEmployment")
hu   = VisumPy.helpers.GetMulti(Visum.Net.Zones,"HOUSINGUNITS")
	
# Fill summary fields
summary_df['Population']         = sum(pop)
summary_df['Total Employment']   = sum(emp)
summary_df['Housing Units (HU)'] = sum(hu)

summary_df['Person Trips (PrT)'] = Tot_person
summary_df['Vehicle Trips (PrT)'] = Tot_vehicle

summary_df['Daily VMT'] = summary_df['VMT_DLY']
summary_df['Daily Per Capita VMT'] = summary_df['VMT_DLY'] / summary_df['Population']
summary_df['Daily VMT Per HU'] = summary_df['VMT_DLY'] / summary_df['Housing Units (HU)']
summary_df['AM VMT'] = summary_df['VMT_AM']
summary_df['AM VMT Per HU'] = summary_df['VMT_AM'] / summary_df['Housing Units (HU)']
summary_df['PM VMT'] = summary_df['VMT_PM']
summary_df['PM VMT Per HU'] = summary_df['VMT_PM'] / summary_df['Housing Units (HU)']
summary_df['OP VMT'] = summary_df['VMT_OP']
summary_df['OP VMT Per HU'] = summary_df['VMT_OP'] / summary_df['Housing Units (HU)']
summary_df['PM Peak Hr VMT'] = summary_df['VMT_PMPK']
summary_df['PM Peak Hr VMT Per HU'] = summary_df['VMT_PMPK'] / summary_df['Housing Units (HU)']

summary_df['Daily VHT'] = summary_df['VHT_DLY_cong']
summary_df['Daily Per Capita VHT'] = summary_df['VHT_DLY_cong'] / summary_df['Population']
summary_df['Daily VHT Per HU'] = summary_df['VHT_DLY_cong'] / summary_df['Housing Units (HU)']
summary_df['AM VHT'] = summary_df['VHT_AM_cong']
summary_df['AM VHT Per HU'] = summary_df['VHT_AM_cong'] / summary_df['Housing Units (HU)']
summary_df['PM VHT'] = summary_df['VHT_PM_cong']
summary_df['PM VHT Per HU'] = summary_df['VHT_PM_cong'] / summary_df['Housing Units (HU)']
summary_df['OP VHT'] = summary_df['VHT_OP_cong']
summary_df['OP VHT Per HU'] = summary_df['VHT_OP_cong'] / summary_df['Housing Units (HU)']
summary_df['PM Peak Hr VHT'] = summary_df['VHT_PMPK_cong']
summary_df['PM Peak Hr VHT Per HU'] = summary_df['VHT_PMPK_cong'] / summary_df['Housing Units (HU)']

summary_df['Daily VHD'] = summary_df['VHD_DLY']
summary_df['Daily Per Capita VHD'] = summary_df['VHD_DLY'] / summary_df['Population']
summary_df['Daily VHD Per HU'] = summary_df['VHD_DLY'] / summary_df['Housing Units (HU)']
summary_df['AM VHD'] = summary_df['VHD_AM']
summary_df['AM VHD Per HU'] = summary_df['VHD_AM'] / summary_df['Housing Units (HU)']
summary_df['PM VHD'] = summary_df['VHD_PM']
summary_df['PM VHD Per HU'] = summary_df['VHD_PM'] / summary_df['Housing Units (HU)']
summary_df['OP VHD'] = summary_df['VHD_OP']
summary_df['OP VHD Per HU'] = summary_df['VHD_OP'] / summary_df['Housing Units (HU)']
summary_df['PM Peak Hr VHD'] = summary_df['VHD_PMPK']
summary_df['PM Peak Hr VHD Per HU'] = summary_df['VHD_PMPK'] / summary_df['Housing Units (HU)']

selection = ['Population','Total Employment','Housing Units (HU)','Person Trips (PrT)','Vehicle Trips (PrT)',
	'Daily VMT','Daily Per Capita VMT','Daily VMT Per HU',
	'AM VMT','AM VMT Per HU','PM VMT','PM VMT Per HU','OP VMT','OP VMT Per HU',
	'PM Peak Hr VMT','PM Peak Hr VMT Per HU',
	'Daily VHT','Daily Per Capita VHT','Daily VHT Per HU',
	'AM VHT','AM VHT Per HU','PM VHT','PM VHT Per HU','OP VHT','OP VHT Per HU',
	'PM Peak Hr VHT','PM Peak Hr VHT Per HU',
	'Daily VHD','Daily Per Capita VHD' ,'Daily VHD Per HU',
	'AM VHD','AM VHD Per HU','PM VHD','PM VHD Per HU','OP VHD','OP VHD Per HU',
	'PM Peak Hr VHD','PM Peak Hr VHD Per HU']

# Copy model_summary_df to model_summary_df_export to maintain formatting for future operations in model_summary_df
summary_df = summary_df[selection].rename(index={0: 'Statistics'}).T
model_summary_df_export = summary_df.copy()
# Convert Statistics column in model_summary_df_export to string so it can accept the formatted statistics values
model_summary_df_export['Statistics'] = model_summary_df_export['Statistics'].astype(str)
	
	
# Format model summary table output cells to make them look better. Each line needs one of three formats
model_summary_df_export.at['Person Trips (PrT)','Statistics']                     = format_commas_cell(summary_df.at['Person Trips (PrT)','Statistics'])
model_summary_df_export.at['Vehicle Trips (PrT)','Statistics']                    = format_commas_cell(summary_df.at['Vehicle Trips (PrT)','Statistics'])
model_summary_df_export.at['Daily VMT','Statistics']                              = format_commas_cell(summary_df.at['Daily VMT','Statistics'])
model_summary_df_export.at['Daily Per Capita VMT','Statistics']                   = format_twoplaces_cell(summary_df.at['Daily Per Capita VMT','Statistics'])
model_summary_df_export.at['Daily VMT Per HU','Statistics']                       = format_twoplaces_cell(summary_df.at['Daily VMT Per HU','Statistics'])
model_summary_df_export.at['AM VMT','Statistics']                                 = format_commas_cell(summary_df.at['AM VMT','Statistics'])
model_summary_df_export.at['AM VMT Per HU','Statistics']                          = format_twoplaces_cell(summary_df.at['AM VMT Per HU','Statistics'])
model_summary_df_export.at['PM VMT','Statistics']                                 = format_commas_cell(summary_df.at['PM VMT','Statistics'])
model_summary_df_export.at['PM VMT Per HU','Statistics']                          = format_twoplaces_cell(summary_df.at['PM VMT Per HU','Statistics'])
model_summary_df_export.at['OP VMT','Statistics']                                 = format_commas_cell(summary_df.at['OP VMT','Statistics'])
model_summary_df_export.at['OP VMT Per HU','Statistics']                          = format_twoplaces_cell(summary_df.at['OP VMT Per HU','Statistics'])
model_summary_df_export.at['PM Peak Hr VMT','Statistics']                         = format_commas_cell(summary_df.at['PM Peak Hr VMT','Statistics'])
model_summary_df_export.at['PM Peak Hr VMT Per HU','Statistics']                  = format_twoplaces_cell(summary_df.at['PM Peak Hr VMT Per HU','Statistics'])
model_summary_df_export.at['Daily VHT','Statistics']                              = format_commas_cell(summary_df.at['Daily VHT','Statistics'])
model_summary_df_export.at['Daily Per Capita VHT','Statistics']                   = format_twoplaces_cell(summary_df.at['Daily Per Capita VHT','Statistics'])
model_summary_df_export.at['Daily VHT Per HU','Statistics']                       = format_twoplaces_cell(summary_df.at['Daily VHT Per HU','Statistics'])
model_summary_df_export.at['AM VHT','Statistics']                                 = format_commas_cell(summary_df.at['AM VHT','Statistics'])
model_summary_df_export.at['AM VHT Per HU','Statistics']                          = format_twoplaces_cell(summary_df.at['AM VHT Per HU','Statistics'])
model_summary_df_export.at['PM VHT','Statistics']                                 = format_commas_cell(summary_df.at['PM VHT','Statistics'])
model_summary_df_export.at['PM VHT Per HU','Statistics']                          = format_twoplaces_cell(summary_df.at['PM VHT Per HU','Statistics'])
model_summary_df_export.at['OP VHT','Statistics']                                 = format_commas_cell(summary_df.at['OP VHT','Statistics'])
model_summary_df_export.at['OP VHT Per HU','Statistics']                          = format_twoplaces_cell(summary_df.at['OP VHT Per HU','Statistics'])
model_summary_df_export.at['PM Peak Hr VHT','Statistics']                         = format_commas_cell(summary_df.at['PM Peak Hr VHT','Statistics'])
model_summary_df_export.at['PM Peak Hr VHT Per HU','Statistics']                  = format_twoplaces_cell(summary_df.at['PM Peak Hr VHT Per HU','Statistics'])
model_summary_df_export.at['Daily VHD','Statistics']                              = format_commas_cell(summary_df.at['Daily VHD','Statistics'])
model_summary_df_export.at['Daily Per Capita VHD','Statistics']                   = format_twoplaces_cell(summary_df.at['Daily Per Capita VHD','Statistics'])
model_summary_df_export.at['Daily VHD Per HU','Statistics']                       = format_twoplaces_cell(summary_df.at['Daily VHD Per HU','Statistics'])
model_summary_df_export.at['AM VHD','Statistics']                                 = format_commas_cell(summary_df.at['AM VHD','Statistics'])
model_summary_df_export.at['AM VHD Per HU','Statistics']                          = format_twoplaces_cell(summary_df.at['AM VHD Per HU','Statistics'])
model_summary_df_export.at['PM VHD','Statistics']                                 = format_commas_cell(summary_df.at['PM VHD','Statistics'])
model_summary_df_export.at['PM VHD Per HU','Statistics']                          = format_twoplaces_cell(summary_df.at['PM VHD Per HU','Statistics'])
model_summary_df_export.at['OP VHD','Statistics']                                 = format_commas_cell(summary_df.at['OP VHD','Statistics'])
model_summary_df_export.at['OP VHD Per HU','Statistics']                          = format_twoplaces_cell(summary_df.at['OP VHD Per HU','Statistics'])
model_summary_df_export.at['PM Peak Hr VHD','Statistics']                         = format_commas_cell(summary_df.at['PM Peak Hr VHD','Statistics'])
model_summary_df_export.at['PM Peak Hr VHD Per HU','Statistics']                  = format_twoplaces_cell(summary_df.at['PM Peak Hr VHD Per HU','Statistics'])
model_summary_df_export.at['Population','Statistics']                             = format_commas_cell(summary_df.at['Population','Statistics'])
model_summary_df_export.at['Total Employment','Statistics']                       = format_commas_cell(summary_df.at['Total Employment','Statistics'])
model_summary_df_export.at['Housing Units (HU)','Statistics']                     = format_commas_cell(summary_df.at['Housing Units (HU)','Statistics'])
	
model_summary_df_export.to_csv(os.path.join(scen_out, "Model_Summary.csv"))


# 9: VOLUME CORRIDORS SUMMARY FILE

# Import blank dataframe with from template folder
vol_corridor_summary_df = pd.read_csv(os.path.join(shared_path,'Reports/Template/volume_corridors_template.csv'))
# Set indices for vol_corridor_summary_df to use the Volume Corridor column
vol_corridor_summary_df.set_index('Volume Corridor', inplace=True)

# Drop rows with empty 'VOL_CORR' and 'VOL_CORR_2'
#vol_links_df = links_df.dropna(subset=['VOL_CORR','VOL_CORR_2'], how = 'all')

# Pull unique Volume Corridor values
array1 = links_df['VOL_CORR'].dropna().to_numpy()
array2 = links_df['VOL_CORR_2'].dropna().to_numpy()
combined_array = np.concatenate((array1, array2), axis=0)
unique_vol_corr = np.unique(combined_array)

# Calculate average volume by corridor for PM Peak and Daily
for i in unique_vol_corr:
	temp_df = links_df[(links_df['VOL_CORR'] == i) | (links_df['VOL_CORR_2'] == i)]
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
		PMPKHR_Tot_Flow  =('PMPK_AUTO_VOLUME', 'sum'),
		Tot_Flow         =('DLY_AUTO_VOLUME', 'sum')
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
		pm_los_tbl = pd.read_csv(os.path.join(shared_path,'Reports/Template/Volume_LOS_Tables/Individual_Tables/' + pm_los_num + '.csv'))
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
		dly_los_tbl = pd.read_csv(os.path.join(shared_path,'Reports/Template/Volume_LOS_Tables/Individual_Tables/' + dly_los_num + '.csv'))
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
#vol_corridor_summary_df.to_csv(outputs_path + "/ModelRun_"+date+"/Model Reporting/unformatted/volumecorridorsummary.csv")

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
vol_corridor_summary_df_export.to_csv(os.path.join(scen_out, "Volume_Corridor_Summary.csv"))

# 10: TRAVEL TIME CORRIDORS SUMMARY FILE
# Import blank dataframe with from template folder
tt_corridor_summary_df = pd.read_csv(os.path.join(shared_path,'Reports/Template/traveltime_corridors_template.csv'))

# Set indices for vol_corridor_summary_df to use the Volume Corridor column
tt_corridor_summary_df.set_index('Travel Time Corridor', inplace=True)

# Drop rows with empty 'VOL_CORR'
tt_links_df = links_df.dropna(subset=['TT_CORR'])

# Pull unique Volume Corridor values
unique_tt_corr = tt_links_df['TT_CORR'].unique()

# Calculate average volume by corridor for PM Peak and Daily
for i in unique_tt_corr:
	temp_df = tt_links_df[(tt_links_df['TT_CORR'] == i)]
	tt_corridor_summary_df.at[i,'Length (miles)']                        = temp_df['LENGTH'].sum()
	tt_corridor_summary_df.at[i,'Free Flow Travel Time (seconds)']       = temp_df['FFTIME'].sum()*3600
	tt_corridor_summary_df.at[i,'PM Peak Loaded Travel Time (seconds)']  = temp_df['PMPKTIME'].sum()*3600
	tt_corridor_summary_df.at[i,'Travel Time Ratio']                     = (temp_df['PMPKTIME'].sum()*3600) / (temp_df['FFTIME'].sum()*3600)
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
#tt_corridor_summary_df.to_csv(outputs_path + "/ModelRun_"+date+"/Model Reporting/unformatted/ttcorridorsummary.csv")

# Copy tt_corridor_summary_df to tt_corridor_summary_df_export to maintain formatting for future operations in tt_corridor_summary_df
tt_corridor_summary_df_export = tt_corridor_summary_df.copy()

# Format travel time corridor summary table output cells to make them look better
tt_corridor_summary_df_export['Length (miles)']                        = format_oneplace(tt_corridor_summary_df_export['Length (miles)'])
tt_corridor_summary_df_export['Free Flow Travel Time (seconds)']       = format_commas(tt_corridor_summary_df_export['Free Flow Travel Time (seconds)'])
tt_corridor_summary_df_export['PM Peak Loaded Travel Time (seconds)']  = format_commas(tt_corridor_summary_df_export['PM Peak Loaded Travel Time (seconds)'])
tt_corridor_summary_df_export['Travel Time Ratio']                     = format_twoplaces(tt_corridor_summary_df_export['Travel Time Ratio'])


# Export tt_corridor_summary_df_export to csv file in timestamped folder
tt_corridor_summary_df_export.to_csv(os.path.join(scen_out, "Travel_Time_Corridor_Summary.csv"))