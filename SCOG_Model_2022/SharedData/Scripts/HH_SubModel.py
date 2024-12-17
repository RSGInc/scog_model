# Python Script to generate disaggregated households distributions by TAZ
# Created by RSG: August 2023
# luke.gordon@rsginc.com for SRTC, michael.mccarthy@rsginc.com adapted for SCOG

# Libraries
import numpy as np
import VisumPy.helpers
import VisumPy.matrices
import pandas as pd
import csv
from itertools import product


# def CheckAttr(attrID,dtype):
	# check if attribute does not exist


# mm for SCOG avoid settings outside of visum database
# # Run settings script
# from SRTC_Settings import settings_dict
# 
# # Need to run settings reader again here for some reason
# 		# Only need to change values in SRTC_Setting.py and SRTC_Setting.csv, handled by Batch file
# settings_csv = csv.reader(open(settings_dict['SharedData_Path']+'SRTC_Settings.csv'))
# settings = list(settings_csv)
# settings_dict = {}
# for row in settings:
#     key = row[0]
#     value = row[1]
#     settings_dict[key] = value

# TODO hardcoded
hhsubmodel_path = "U:/Projects/Clients/SCOG/Model/scog_model/SCOG_Model_2022/SharedData/HH_Submodel/"
hh_parameters_path = hhsubmodel_path + "Parameters/"
hh_out_path = hhsubmodel_path + "Outputs/"

""" Define Input files and Constants """
HHSizeModel = pd.read_csv(hh_parameters_path+'HHSizeModel.csv')
IncomeModel = pd.read_csv(hh_parameters_path+'IncomeModel.csv')
HHSeedMtx   = pd.read_csv(hh_parameters_path+'HHsize_income_2d_table.csv')
NumWorkersModel = pd.read_csv(hh_parameters_path+'NumberOfWorkersModel_MNL.csv')
#NumChildrenModel = pd.read_csv(hh_parameters_path+'NumberOfChildrenModel.csv')
#NumAutosModel = pd.read_csv(hh_parameters_path+'NumberOfVehiclesModel_NIRCC.csv')

# Pull Zone attributes from Visum and create dataframe
no          = VisumPy.helpers.GetMulti(Visum.Net.Zones,"No")
tothh       = VisumPy.helpers.GetMulti(Visum.Net.Zones,"HousingUnits")
#tothh_h12   = VisumPy.helpers.GetMulti(Visum.Net.Zones,"HH_CENSUS")
hhinc       = VisumPy.helpers.GetMulti(Visum.Net.Zones,"HHINC")
#hhpop       = VisumPy.helpers.GetMulti(Visum.Net.Zones,"HHPOP")
#hhveh	    = VisumPy.helpers.GetMulti(Visum.Net.Zones,"HHVEH")
#hhwrk	    = VisumPy.helpers.GetMulti(Visum.Net.Zones,"HHWRK")
hhsize      = VisumPy.helpers.GetMulti(Visum.Net.Zones,"AvgHHSize")

# Set up dataframe to use for all needed zone attributes. Update as needed during coding
zone_df = pd.DataFrame({'NO':no, 'TOTHH':tothh, 'HHINC':hhinc, 'HHSIZE':hhsize})

""" Generate Output csvs """
hhsize = [1,2,3,4] 
hhinc = [1,2,3,4]
hhwrk = [0,1,2,3]
tothh = [0]
combinations1 = list(product(no,hhsize,hhinc,tothh))
combinations2 = list(product(no,hhsize,hhinc,hhwrk,tothh))
HIOut = pd.DataFrame(combinations1, columns=['ZONE','HHSIZE','INCOME','TOTHH'])
HIWOut = pd.DataFrame(combinations2, columns=['ZONE','HHSIZE','INCOME','WORKERS','TOTHH'])

HIOut.to_csv(hh_out_path+"HHSize_Inc.csv", index = False)
HIWOut.to_csv(hh_out_path+"HHSize_Inc_Workers.csv", index = False)

HIOutputFile   = pd.read_csv(hh_out_path+"HHSize_Inc.csv")
HIWOutputFile  = pd.read_csv(hh_out_path+"HHSize_Inc_Workers.csv")
#HIWCOutputFile = pd.read_csv(hh_out_path+"HHSize_Inc_Workers_Childrn.csv")

# Instead, filter to relevant zones/set active prior to running
# Drop P&R and External TAZs
# zone_df = zone_df[(zone_df['NO'] < 8000)]


""" Step 1: HH Size and HH Income models """

# HH Size Model

# Compute average HH size by TAZ
#for x in range(len(zone_df)):
#	if zone_df.loc[x,'TOTHH'] > 0:
#		zone_df.loc[x,'HHSIZE'] = zone_df.loc[x,'HHPOP']/zone_df.loc[x,'TOTHH_H12']
		
# Round HHSIZE to 1 decimal place
zone_df = zone_df.round({'HHSIZE':1})

# Replace HHSIZE < 1 or null with 1, replace HHSIZE > 4.2 with 4.2
for x in range(len(zone_df)):
	if zone_df.loc[x,'HHSIZE'] < 1 or zone_df.loc[x,'HHSIZE'] == None:
		zone_df.loc[x,'HHSIZE'] = 1
	elif zone_df.loc[x,'HHSIZE'] > 4.2:
		zone_df.loc[x,'HHSIZE'] = 4.2

# Join in HHSizeModel (lookup table) to zone_df on HHSIZE for multiplication
taz_records = len(zone_df)
zone_df = pd.merge(zone_df, HHSizeModel, on='HHSIZE', how='left')
if len(zone_df) != taz_records:
    raise Exception("Merge process in HHSizeModel is not correct")

# Multiply TOTHH by HH1/2/3/4 values from lookup table
for x in range(len(zone_df)):
	zone_df.loc[x,'HHS1'] = zone_df.loc[x,'HH1'] * zone_df.loc[x,'TOTHH']
	zone_df.loc[x,'HHS2'] = zone_df.loc[x,'HH2'] * zone_df.loc[x,'TOTHH']
	zone_df.loc[x,'HHS3'] = zone_df.loc[x,'HH3'] * zone_df.loc[x,'TOTHH']
	zone_df.loc[x,'HHS4'] = zone_df.loc[x,'HH4'] * zone_df.loc[x,'TOTHH']

# Replace empty cells with 0
zone_df.fillna(0, inplace=True) # Replace blank cells with 0	
	
# Set Visum fields with HH size totals
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HHS1",zone_df['HHS1'])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HHS2",zone_df['HHS2'])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HHS3",zone_df['HHS3'])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HHS4",zone_df['HHS4'])

# Set Visum field with average HH size by TAZ
# already formula? VisumPy.helpers.SetMulti(Visum.Net.Zones,"AvgHHSize",zone_df['HHSIZE'])
	

## HH Income Model

# Compute weighted average for average regional income
avg_regionalINC = np.dot(zone_df['TOTHH'],zone_df['HHINC'])/zone_df['TOTHH'].sum()
	
# Compute average HH income by TAZ
for x in range(len(zone_df)):
	if zone_df.loc[x,'TOTHH'] > 0:
		zone_df.loc[x,'INC_INDEX'] = zone_df.loc[x,'HHINC']/avg_regionalINC

# Round INC_INDEX to 1 decimal place
zone_df = zone_df.round({'INC_INDEX':1})

# Replace INC_INDEX < 0.2 with 0.2, replace INC_INDEX > 2 with 2
for x in range(len(zone_df)):
	if zone_df.loc[x,'INC_INDEX'] < 0.2:
		zone_df.loc[x,'INC_INDEX'] = 0.2
	elif zone_df.loc[x,'INC_INDEX'] > 2:
		zone_df.loc[x,'INC_INDEX'] = 2

# Replace empty cells with 0
zone_df.fillna(0, inplace=True) # Replace blank cells with 0		
		
# Paste in INC_INDEX to Visum zone table
VisumPy.helpers.SetMulti(Visum.Net.Zones,"INC_INDEX",zone_df['INC_INDEX'])

# Join in IncomeModel (lookup table) to zone_df on INC_INDEX for multiplication
zone_df = pd.merge(zone_df, IncomeModel, on='INC_INDEX', how='left')
if len(zone_df) != taz_records:
    raise Exception("Merge process in IncomeModel is not correct")

# Multiply TOTHH by HHINC1/2/3/4 values from lookup table
for x in range(len(zone_df)):
	zone_df.loc[x,'INC1'] = zone_df.loc[x,'HHINC1'] * zone_df.loc[x,'TOTHH']
	zone_df.loc[x,'INC2'] = zone_df.loc[x,'HHINC2'] * zone_df.loc[x,'TOTHH']
	zone_df.loc[x,'INC3'] = zone_df.loc[x,'HHINC3'] * zone_df.loc[x,'TOTHH']
	zone_df.loc[x,'INC4'] = zone_df.loc[x,'HHINC4'] * zone_df.loc[x,'TOTHH']

# Replace empty cells with 0
zone_df.fillna(0, inplace=True) # Replace blank cells with 0

# Set Visum fields with HH income totals
VisumPy.helpers.SetMulti(Visum.Net.Zones,"INC1",zone_df['INC1'])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"INC2",zone_df['INC2'])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"INC3",zone_df['INC3'])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"INC4",zone_df['INC4'])




""" Step 2: Iterative Proportional Fitting/Balancing """
# Loop thru each zone to run matrix balancing with seed matrix, 
		# Using the outputs of Step 1 HHSize as rowsums and HHInc as colsums
y = 0
for x in range(len(zone_df)):
	if zone_df.loc[x,'TOTHH'] > 0:
		mat = np.array([[HHSeedMtx['inc1'].values[0],HHSeedMtx['inc2'].values[0],HHSeedMtx['inc3'].values[0],HHSeedMtx['inc4'].values[0]],  # HH1
                        [HHSeedMtx['inc1'].values[1],HHSeedMtx['inc2'].values[1],HHSeedMtx['inc3'].values[1],HHSeedMtx['inc4'].values[1]],  # HH2
                        [HHSeedMtx['inc1'].values[2],HHSeedMtx['inc2'].values[2],HHSeedMtx['inc3'].values[2],HHSeedMtx['inc4'].values[2]],  # HH3
                        [HHSeedMtx['inc1'].values[3],HHSeedMtx['inc2'].values[3],HHSeedMtx['inc3'].values[3],HHSeedMtx['inc4'].values[3]]]) # HH4
		r = np.array([zone_df.loc[x,'HHS1'],zone_df.loc[x,'HHS2'],zone_df.loc[x,'HHS3'],zone_df.loc[x,'HHS4']]) # HHSize from the HH size submodel by zone
		c = np.array([zone_df.loc[x,'INC1'],zone_df.loc[x,'INC2'],zone_df.loc[x,'INC3'],zone_df.loc[x,'INC4']]) # HHIncome from the HH Income submodel by zone
		""" Run Visum balanceMatrix function """
		balanced_mat = VisumPy.matrices.balanceMatrix(mat,r,c,closePctDiff=0.001)
		# Paste in balanced values to new df fields
		zone_df.loc[x,'HH1INC1'] = balanced_mat[0,0]
		zone_df.loc[x,'HH1INC2'] = balanced_mat[0,1]
		zone_df.loc[x,'HH1INC3'] = balanced_mat[0,2]
		zone_df.loc[x,'HH1INC4'] = balanced_mat[0,3]
		zone_df.loc[x,'HH2INC1'] = balanced_mat[1,0]
		zone_df.loc[x,'HH2INC2'] = balanced_mat[1,1]
		zone_df.loc[x,'HH2INC3'] = balanced_mat[1,2]
		zone_df.loc[x,'HH2INC4'] = balanced_mat[1,3]
		zone_df.loc[x,'HH3INC1'] = balanced_mat[2,0]
		zone_df.loc[x,'HH3INC2'] = balanced_mat[2,1]
		zone_df.loc[x,'HH3INC3'] = balanced_mat[2,2]
		zone_df.loc[x,'HH3INC4'] = balanced_mat[2,3]
		zone_df.loc[x,'HH4INC1'] = balanced_mat[3,0]
		zone_df.loc[x,'HH4INC2'] = balanced_mat[3,1]
		zone_df.loc[x,'HH4INC3'] = balanced_mat[3,2]
		zone_df.loc[x,'HH4INC4'] = balanced_mat[3,3]
		
		# Also Paste values into csv file with 1 row per zone/HH size/income group
		HIOutputFile.loc[y,'TOTHH'] = balanced_mat[0,0]
		y = y + 1
		HIOutputFile.loc[y,'TOTHH'] = balanced_mat[0,1]
		y = y + 1
		HIOutputFile.loc[y,'TOTHH'] = balanced_mat[0,2]
		y = y + 1
		HIOutputFile.loc[y,'TOTHH'] = balanced_mat[0,3]
		y = y + 1
		HIOutputFile.loc[y,'TOTHH'] = balanced_mat[1,0]
		y = y + 1
		HIOutputFile.loc[y,'TOTHH'] = balanced_mat[1,1]
		y = y + 1
		HIOutputFile.loc[y,'TOTHH'] = balanced_mat[1,2]
		y = y + 1
		HIOutputFile.loc[y,'TOTHH'] = balanced_mat[1,3]
		y = y + 1
		HIOutputFile.loc[y,'TOTHH'] = balanced_mat[2,0]
		y = y + 1
		HIOutputFile.loc[y,'TOTHH'] = balanced_mat[2,1]
		y = y + 1
		HIOutputFile.loc[y,'TOTHH'] = balanced_mat[2,2]
		y = y + 1
		HIOutputFile.loc[y,'TOTHH'] = balanced_mat[2,3]
		y = y + 1
		HIOutputFile.loc[y,'TOTHH'] = balanced_mat[3,0]
		y = y + 1
		HIOutputFile.loc[y,'TOTHH'] = balanced_mat[3,1]
		y = y + 1
		HIOutputFile.loc[y,'TOTHH'] = balanced_mat[3,2]
		y = y + 1
		HIOutputFile.loc[y,'TOTHH'] = balanced_mat[3,3]
		y = y + 1
		
	else:
		y = y + 16
		continue
	
# Replace empty cells with 0
zone_df.fillna(0, inplace=True) # Replace blank cells with 0

# Paste Joint distribution fields from zone_df fields into Visum zone table
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH1INC1",zone_df['HH1INC1'])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH1INC2",zone_df['HH1INC2'])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH1INC3",zone_df['HH1INC3'])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH1INC4",zone_df['HH1INC4'])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH2INC1",zone_df['HH2INC1'])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH2INC2",zone_df['HH2INC2'])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH2INC3",zone_df['HH2INC3'])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH2INC4",zone_df['HH2INC4'])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH3INC1",zone_df['HH3INC1'])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH3INC2",zone_df['HH3INC2'])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH3INC3",zone_df['HH3INC3'])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH3INC4",zone_df['HH3INC4'])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH4INC1",zone_df['HH4INC1'])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH4INC2",zone_df['HH4INC2'])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH4INC3",zone_df['HH4INC3'])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH4INC4",zone_df['HH4INC4'])

# Paste Joint distribution output table into csv file TOTHH field
HIOutputFile['TOTHH'].fillna(0, inplace=True) # Replace blank cells with 0
HIOutputFile.to_csv(hh_out_path+"HHSize_Inc.csv", index = False)


""" Step 3: Number of Workers per HH (BASED ON AMATS) """
# Create array to match up with coefficient array for worker model
XArray = np.zeros(len(NumWorkersModel))

y = 0
for x in range(len(HIOutputFile)):
	if HIOutputFile.loc[x,'TOTHH'] > 0:
		XArray[0] = 1 # ASC
		XArray[1] = 1 if HIOutputFile.loc[x,'HHSIZE'] == 1 else 0
		XArray[2] = 1 if HIOutputFile.loc[x,'HHSIZE'] == 2 else 0
		XArray[3] = 1 if HIOutputFile.loc[x,'HHSIZE'] == 3 else 0
		XArray[4] = 1 if HIOutputFile.loc[x,'HHSIZE'] == 4 else 0
		XArray[5] = 1 if HIOutputFile.loc[x,'INCOME'] == 1 else 0
		XArray[6] = 1 if HIOutputFile.loc[x,'INCOME'] == 2 else 0
		XArray[7] = 1 if HIOutputFile.loc[x,'INCOME'] == 3 else 0
		XArray[8] = 1 if HIOutputFile.loc[x,'INCOME'] == 4 else 0
		
		# Initialize Utilities
		v0 = 0
		v1 = 0
		v2 = 0
		v3 = 0
		# Calculate utilities
		for i in range(len(XArray)):
			# Compute utility for each number of workers by TAZ/HH Size/Income 
			v0 =  v0 + NumWorkersModel.loc[i,'Work0']  * XArray[i]
			v1 =  v1 + NumWorkersModel.loc[i,'Work1']  * XArray[i]
			v2 =  v2 + NumWorkersModel.loc[i,'Work2']  * XArray[i]
			v3 =  v3 + NumWorkersModel.loc[i,'Work3p'] * XArray[i]
		
		# Constrain number of workers by HHSize [# Workers <= HH Size)
		v2 = v2 - 500 if HIOutputFile.loc[x,'HHSIZE'] == 1 else v2
		v3 = v3 - 500 if HIOutputFile.loc[x,'HHSIZE'] <= 2 else v3
		
		# Compute probabilities for each number of workers 
		p0 = np.exp(v0)/(np.exp(v0)+np.exp(v1)+np.exp(v2)+np.exp(v3))
		p1 = np.exp(v1)/(np.exp(v0)+np.exp(v1)+np.exp(v2)+np.exp(v3))
		p2 = np.exp(v2)/(np.exp(v0)+np.exp(v1)+np.exp(v2)+np.exp(v3))
		p3 = np.exp(v3)/(np.exp(v0)+np.exp(v1)+np.exp(v2)+np.exp(v3))
		
		# Compute number workers distribution for given TAZ/HH Size/Income
		w0 = p0 * HIOutputFile.loc[x,'TOTHH']
		w1 = p1 * HIOutputFile.loc[x,'TOTHH']
		w2 = p2 * HIOutputFile.loc[x,'TOTHH']
		w3 = p3 * HIOutputFile.loc[x,'TOTHH']
		
		# Replace values from instances of # Workers > HH Size with 0
		if HIOutputFile.loc[x,'HHSIZE'] == 1:
			w2 = 0
			w3 = 0
		elif HIOutputFile.loc[x,'HHSIZE'] == 2:
			w3 = 0
		
		# Paste in number of workers to HIWOutputFile by TAZ/HH Size/Income/Workers
		HIWOutputFile.loc[y,'TOTHH'] = w0
		y = y + 1
		HIWOutputFile.loc[y,'TOTHH'] = w1
		y = y + 1
		HIWOutputFile.loc[y,'TOTHH'] = w2
		y = y + 1
		HIWOutputFile.loc[y,'TOTHH'] = w3
		y = y + 1
	else:
		HIWOutputFile.loc[y,'TOTHH'] = 0
		y = y + 1
		HIWOutputFile.loc[y,'TOTHH'] = 0
		y = y + 1
		HIWOutputFile.loc[y,'TOTHH'] = 0
		y = y + 1
		HIWOutputFile.loc[y,'TOTHH'] = 0
		y = y + 1
		continue
	
# Paste 3-Way distribution output table into csv file TOTHH field
HIWOutputFile['TOTHH'].fillna(0, inplace=True) # Replace blank cells with 0
HIWOutputFile.to_csv(hh_out_path+"HHSize_Inc_Workers.csv", index = False)


# Collapse Number of Workers by TAZ to get HHWRK(0-3)
# Filter by # of workers
worker0 = HIWOutputFile[HIWOutputFile['WORKERS'] == 0].reset_index(drop=True)
worker1 = HIWOutputFile[HIWOutputFile['WORKERS'] == 1].reset_index(drop=True)
worker2 = HIWOutputFile[HIWOutputFile['WORKERS'] == 2].reset_index(drop=True)
worker3 = HIWOutputFile[HIWOutputFile['WORKERS'] == 3].reset_index(drop=True)

zone_df['HHWRK0'] = 0
zone_df['HHWRK1'] = 0
zone_df['HHWRK2'] = 0
zone_df['HHWRK3'] = 0
# Groupby ZONE 
worker0 = worker0.groupby(['ZONE'])
worker1 = worker1.groupby(['ZONE'])
worker2 = worker2.groupby(['ZONE'])
worker3 = worker3.groupby(['ZONE'])
# Sum TOTHH by ZONE
zone_df['HHWRK0'] = worker0['TOTHH'].sum().reset_index(drop=True)
zone_df['HHWRK1'] = worker1['TOTHH'].sum().reset_index(drop=True)
zone_df['HHWRK2'] = worker2['TOTHH'].sum().reset_index(drop=True)
zone_df['HHWRK3'] = worker3['TOTHH'].sum().reset_index(drop=True)
# Replace empty cells with 0
zone_df.fillna(0, inplace=True) # Replace blank cells with 0	
# Set Visum zone fields with HHWRK(0-3) values
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HHWRK0",zone_df['HHWRK0'])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HHWRK1",zone_df['HHWRK1'])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HHWRK2",zone_df['HHWRK2'])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HHWRK3",zone_df['HHWRK3'])



# Reshape 3-way distribution to be pasted into Visum zone layer
triple_dist = pd.DataFrame({'NO':zone_df['NO']})
i = 0
for x in range(len(triple_dist)):
	for y in range(1,65):
		triple_dist.loc[x,y] = HIWOutputFile.loc[i,'TOTHH']
		i = i + 1

# Replace empty cells with 0
triple_dist.fillna(0, inplace=True) # Replace blank cells with 0		
		
# Paste 3-way distribution into Visum zone layer
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH1INC1W0",triple_dist[1])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH1INC1W1",triple_dist[2])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH1INC2W0",triple_dist[5])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH1INC2W1",triple_dist[6])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH1INC3W0",triple_dist[9])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH1INC3W1",triple_dist[10])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH1INC4W0",triple_dist[13])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH1INC4W1",triple_dist[14])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH2INC1W0",triple_dist[17])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH2INC1W1",triple_dist[18])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH2INC1W2",triple_dist[19])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH2INC2W0",triple_dist[21])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH2INC2W1",triple_dist[22])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH2INC2W2",triple_dist[23])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH2INC3W0",triple_dist[25])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH2INC3W1",triple_dist[26])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH2INC3W2",triple_dist[27])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH2INC4W0",triple_dist[29])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH2INC4W1",triple_dist[30])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH2INC4W2",triple_dist[31])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH3INC1W0",triple_dist[33])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH3INC1W1",triple_dist[34])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH3INC1W2",triple_dist[35])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH3INC1W3",triple_dist[36])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH3INC2W0",triple_dist[37])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH3INC2W1",triple_dist[38])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH3INC2W2",triple_dist[39])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH3INC2W3",triple_dist[40])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH3INC3W0",triple_dist[41])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH3INC3W1",triple_dist[42])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH3INC3W2",triple_dist[43])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH3INC3W3",triple_dist[44])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH3INC4W0",triple_dist[45])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH3INC4W1",triple_dist[46])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH3INC4W2",triple_dist[47])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH3INC4W3",triple_dist[48])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH4INC1W0",triple_dist[49])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH4INC1W1",triple_dist[50])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH4INC1W2",triple_dist[51])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH4INC1W3",triple_dist[52])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH4INC2W0",triple_dist[53])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH4INC2W1",triple_dist[54])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH4INC2W2",triple_dist[55])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH4INC2W3",triple_dist[56])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH4INC3W0",triple_dist[57])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH4INC3W1",triple_dist[58])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH4INC3W2",triple_dist[59])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH4INC3W3",triple_dist[60])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH4INC4W0",triple_dist[61])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH4INC4W1",triple_dist[62])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH4INC4W2",triple_dist[63])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH4INC4W3",triple_dist[64])

