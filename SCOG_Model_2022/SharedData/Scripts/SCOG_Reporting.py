"""
Model Trip, VMT, VHT, VHD reporting
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
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
scenname = Visum.Net.AttValue("SCENARIO")

# Create timestamped folder for all results
scen_out = out_path + "/ModelRun_"+scenname+"_"+timestamp
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

# Pull Link attributes from Visum and create dataframe
no          = VisumPy.helpers.GetMulti(Visum.Net.Links,"No")
length      = VisumPy.helpers.GetMulti(Visum.Net.Links,"Length")
typeno      = VisumPy.helpers.GetMulti(Visum.Net.Links,"TypeNo")
tsys        = VisumPy.helpers.GetMulti(Visum.Net.Links,"TSysSet")
nfc         = VisumPy.helpers.GetMulti(Visum.Net.Links,"NFCLASS")
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


# Set up dataframe to use for all needed zone attributes. Update as needed during coding
links_df = pd.DataFrame({'NO':no, 'LENGTH': length, 'TYPENO': typeno, 'TSYSSET': tsys, 'NFCLASS': nfc, 'AM_AUTO_VOLUME': amvol, 'PM_AUTO_VOLUME': pmvol, 'PMPK_AUTO_VOLUME': pmpkvol, 'OP_AUTO_VOLUME': opvol, 'DLY_AUTO_VOLUME': dlyvol,  'FFTIME': fftime, 
    'AMTIME': amctime, 'PMTIME': pmctime, 'OPTIME': opctime, 'PMPKTIME': pmpkctime})

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
summary_df['PM Peak Hr VMT'] = summary_df['VMT_PMPK']
summary_df['PM Peak Hr VMT Per HU'] = summary_df['VMT_PMPK'] / summary_df['Housing Units (HU)']

summary_df['Daily VHT'] = summary_df['VHT_DLY_cong']
summary_df['Daily Per Capita VHT'] = summary_df['VHT_DLY_cong'] / summary_df['Population']
summary_df['Daily VHT Per HU'] = summary_df['VHT_DLY_cong'] / summary_df['Housing Units (HU)']
summary_df['PM Peak Hr VHT'] = summary_df['VHT_PMPK_cong']
summary_df['PM Peak Hr VHT Per HU'] = summary_df['VHT_PMPK_cong'] / summary_df['Housing Units (HU)']

summary_df['Daily VHD'] = summary_df['VHD_DLY']
summary_df['Daily Per Capita VHD'] = summary_df['VHD_DLY'] / summary_df['Population']
summary_df['Daily VHD Per HU'] = summary_df['VHD_DLY'] / summary_df['Housing Units (HU)']
summary_df['PM Peak Hr VHD'] = summary_df['VHD_PMPK']
summary_df['PM Peak Hr VHD Per HU'] = summary_df['VHD_PMPK'] / summary_df['Housing Units (HU)']

selection = ['Population','Total Employment','Housing Units (HU)','Person Trips (PrT)','Vehicle Trips (PrT)','Daily VMT','Daily Per Capita VMT','Daily VMT Per HU','PM Peak Hr VMT','PM Peak Hr VMT Per HU','Daily VHT','Daily Per Capita VHT','Daily VHT Per HU','PM Peak Hr VHT','PM Peak Hr VHT Per HU','Daily VHD','Daily Per Capita VHD' ,'Daily VHD Per HU','PM Peak Hr VHD','PM Peak Hr VHD Per HU']

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
model_summary_df_export.at['PM Peak Hr VMT','Statistics']                         = format_commas_cell(summary_df.at['PM Peak Hr VMT','Statistics'])
model_summary_df_export.at['PM Peak Hr VMT Per HU','Statistics']                  = format_twoplaces_cell(summary_df.at['PM Peak Hr VMT Per HU','Statistics'])
model_summary_df_export.at['Daily VHT','Statistics']                              = format_commas_cell(summary_df.at['Daily VHT','Statistics'])
model_summary_df_export.at['Daily Per Capita VHT','Statistics']                   = format_twoplaces_cell(summary_df.at['Daily Per Capita VHT','Statistics'])
model_summary_df_export.at['Daily VHT Per HU','Statistics']                       = format_twoplaces_cell(summary_df.at['Daily VHT Per HU','Statistics'])
model_summary_df_export.at['PM Peak Hr VHT','Statistics']                         = format_commas_cell(summary_df.at['PM Peak Hr VHT','Statistics'])
model_summary_df_export.at['PM Peak Hr VHT Per HU','Statistics']                  = format_twoplaces_cell(summary_df.at['PM Peak Hr VHT Per HU','Statistics'])
model_summary_df_export.at['Daily VHD','Statistics']                              = format_commas_cell(summary_df.at['Daily VHD','Statistics'])
model_summary_df_export.at['Daily Per Capita VHD','Statistics']                   = format_twoplaces_cell(summary_df.at['Daily Per Capita VHD','Statistics'])
model_summary_df_export.at['Daily VHD Per HU','Statistics']                       = format_twoplaces_cell(summary_df.at['Daily VHD Per HU','Statistics'])
model_summary_df_export.at['PM Peak Hr VHD','Statistics']                         = format_commas_cell(summary_df.at['PM Peak Hr VHD','Statistics'])
model_summary_df_export.at['PM Peak Hr VHD Per HU','Statistics']                  = format_twoplaces_cell(summary_df.at['PM Peak Hr VHD Per HU','Statistics'])
model_summary_df_export.at['Population','Statistics']                             = format_commas_cell(summary_df.at['Population','Statistics'])
model_summary_df_export.at['Total Employment','Statistics']                       = format_commas_cell(summary_df.at['Total Employment','Statistics'])
model_summary_df_export.at['Housing Units (HU)','Statistics']                     = format_commas_cell(summary_df.at['Housing Units (HU)','Statistics'])
	
model_summary_df_export.to_csv(os.path.join(scen_out, "Model_Summary.csv"))