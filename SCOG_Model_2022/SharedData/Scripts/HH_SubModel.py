# Python Script to generate disaggregated households distributions by TAZ
# Created by RSG: August 2023
# luke.gordon@rsginc.com for SRTC, michael.mccarthy@rsginc.com adapted for SCOG

# Libraries
import sys
import numpy as np
import VisumPy.helpers
import VisumPy.matrices
import pandas as pd
import csv
from itertools import product
import os.path
# import logging
from VisumPy.AddIn import AddIn, AddInState, MessageType, AddInParameter


if len(sys.argv) > 1:
    # Set translation functionality, because body script is in debug mode and the dialog script got not called before
    # The VISUM object will be created by functionality of class 'AddIn'  ( the debug sys.argv[2] - parameter is used for this)
    addIn = AddIn()
else:
    # In else condition (the AddIn was called out of Visum) the AddIn class object is needed only to get logging functionality. The passed VISUM object
    # is needed for logging functionality. Translation functionality must not be set because it already got set in dialog script
    addIn = AddIn(Visum)

# set paths 
hhsubmodel_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'HH_Submodel'))
hh_parameters_path = hhsubmodel_path + "/Parameters/"
hh_out_path = hhsubmodel_path + "/Outputs/"

""" Define Input files and Constants """
HHSizeModel = pd.read_csv(hh_parameters_path+'HHSizeModel.csv')
IncomeModel = pd.read_csv(hh_parameters_path+'IncomeModel.csv')
HHIncSeedMtx   = pd.read_csv(hh_parameters_path+'HHsize_income_2d_table.csv')
HHWrkSeedMtx   = pd.read_csv(hh_parameters_path+'HHsize_workers_2d_table.csv')
HHVehSeedMtx   = pd.read_csv(hh_parameters_path+'HHsize_vehicles_2d_table_PUMS.csv')
NumWorkersModel = pd.read_csv(hh_parameters_path+'NumberOfWorkersModel_MNL.csv')
NumVehiclesModel = pd.read_csv(hh_parameters_path+'NumberOfVehiclesModel_MNL.csv')


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
hhveh = [0,1,2,3]
tothh = [0]
combinations1 = list(product(no,hhsize,hhinc,tothh))
combinations2 = list(product(no,hhsize,hhinc,hhwrk,tothh))
combinations3 = list(product(no,hhsize,hhinc,hhwrk,hhveh,tothh))
HIOut = pd.DataFrame(combinations1, columns=['ZONE','HHSIZE','INCOME','TOTHH'])
HIWOut = pd.DataFrame(combinations2, columns=['ZONE','HHSIZE','INCOME','WORKERS','TOTHH'])
HIWVOut = pd.DataFrame(combinations3, columns=['ZONE','HHSIZE','INCOME','WORKERS','VEHICLES','TOTHH'])

HIOut.to_csv(hh_out_path+"HHSize_Inc.csv", index = False)
HIWOut.to_csv(hh_out_path+"HHSize_Inc_Workers.csv", index = False)
HIWVOut.to_csv(hh_out_path+"HHSize_Inc_Workers_Veh.csv", index = False)

HIOutputFile   = pd.read_csv(hh_out_path+"HHSize_Inc.csv")
HIWOutputFile  = pd.read_csv(hh_out_path+"HHSize_Inc_Workers.csv")
HIWVOutputFile  = pd.read_csv(hh_out_path+"HHSize_Inc_Workers_Veh.csv")
#HIWCOutputFile = pd.read_csv(hh_out_path+"HHSize_Inc_Workers_Childrn.csv")

# Drop P&R and External TAZs
zone_df = zone_df[(zone_df['NO'] < 1000)] # hardcoded, still produces warnings for 15 zones


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
# and rescale to ensure same TOTHH

for x in range(len(zone_df)):
	zone_df.loc[x,'frac_tot'] = zone_df.loc[x,'HH1'] + zone_df.loc[x,'HH2'] + zone_df.loc[x,'HH3'] + zone_df.loc[x,'HH4']
	zone_df.loc[x,'HHS1'] = (zone_df.loc[x,'HH1'] / zone_df.loc[x,'frac_tot']) * zone_df.loc[x,'TOTHH']
	zone_df.loc[x,'HHS2'] = (zone_df.loc[x,'HH2'] / zone_df.loc[x,'frac_tot']) * zone_df.loc[x,'TOTHH']
	zone_df.loc[x,'HHS3'] = (zone_df.loc[x,'HH3'] / zone_df.loc[x,'frac_tot']) * zone_df.loc[x,'TOTHH']
	zone_df.loc[x,'HHS4'] = (zone_df.loc[x,'HH4'] / zone_df.loc[x,'frac_tot']) * zone_df.loc[x,'TOTHH']


# refactor
'''
zone_df['HHS1'] = zone_df['HH1'] * zone_df['TOTHH']
zone_df['HHS2'] = zone_df['HH2'] * zone_df['TOTHH']
zone_df['HHS3'] = zone_df['HH3'] * zone_df['TOTHH']
zone_df['HHS4'] = zone_df['HH4'] * zone_df['TOTHH']
'''

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
	zone_df.loc[x,'frac_tot'] = zone_df.loc[x,'HHINC1'] + zone_df.loc[x,'HHINC2'] + zone_df.loc[x,'HHINC3'] + zone_df.loc[x,'HHINC4']
	zone_df.loc[x,'INC1'] = (zone_df.loc[x,'HHINC1']  / zone_df.loc[x,'frac_tot']) * zone_df.loc[x,'TOTHH']
	zone_df.loc[x,'INC2'] = (zone_df.loc[x,'HHINC2']  / zone_df.loc[x,'frac_tot']) * zone_df.loc[x,'TOTHH']
	zone_df.loc[x,'INC3'] = (zone_df.loc[x,'HHINC3']  / zone_df.loc[x,'frac_tot']) * zone_df.loc[x,'TOTHH']
	zone_df.loc[x,'INC4'] = (zone_df.loc[x,'HHINC4']  / zone_df.loc[x,'frac_tot']) * zone_df.loc[x,'TOTHH']
'''
# refactor
zone_df['INC1'] = zone_df['HHINC1'] * zone_df['TOTHH']
zone_df['INC2'] = zone_df['HHINC2'] * zone_df['TOTHH']
zone_df['INC3'] = zone_df['HHINC3'] * zone_df['TOTHH']
zone_df['INC4'] = zone_df['HHINC4'] * zone_df['TOTHH']
'''
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
		## HHSize x Income
		mat = np.array([[HHIncSeedMtx['inc1'].values[0],HHIncSeedMtx['inc2'].values[0],HHIncSeedMtx['inc3'].values[0],HHIncSeedMtx['inc4'].values[0]],  # HH1
                        [HHIncSeedMtx['inc1'].values[1],HHIncSeedMtx['inc2'].values[1],HHIncSeedMtx['inc3'].values[1],HHIncSeedMtx['inc4'].values[1]],  # HH2
                        [HHIncSeedMtx['inc1'].values[2],HHIncSeedMtx['inc2'].values[2],HHIncSeedMtx['inc3'].values[2],HHIncSeedMtx['inc4'].values[2]],  # HH3
                        [HHIncSeedMtx['inc1'].values[3],HHIncSeedMtx['inc2'].values[3],HHIncSeedMtx['inc3'].values[3],HHIncSeedMtx['inc4'].values[3]]]) # HH4
		r = np.array([zone_df.loc[x,'HHS1'],zone_df.loc[x,'HHS2'],zone_df.loc[x,'HHS3'],zone_df.loc[x,'HHS4']]) # HHSize from the HH size submodel by zone
		c = np.array([zone_df.loc[x,'INC1'],zone_df.loc[x,'INC2'],zone_df.loc[x,'INC3'],zone_df.loc[x,'INC4']]) # HHIncome from the HH Income submodel by zone
		""" Run Visum balanceMatrix function """
		
		try:
			balanced_mat = VisumPy.matrices.balanceMatrix(mat,r,c,closePctDiff=0.001) # TODO matrix balancing issue default 0.0001
			
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
		except:
			errstring = f"Could not balance NO {zone_df.loc[x,'NO']}	HH size: {np.array2string(r)} sum: {np.sum(r)}	INC: {np.array2string(c)} sum: {np.sum(c)}\n"
			addIn.ReportMessage(errstring, MessageType.Error)
			continue
		
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

for x in range(len(zone_df)):

	if zone_df.loc[x,'TOTHH'] > 0:
		## HHSize x Workers
		mat = np.array([[HHWrkSeedMtx['wrk0'].values[0],HHWrkSeedMtx['wrk1'].values[0],HHWrkSeedMtx['wrk2'].values[0],HHWrkSeedMtx['wrk3'].values[0]],  # HH1
                        [HHWrkSeedMtx['wrk0'].values[1],HHWrkSeedMtx['wrk1'].values[1],HHWrkSeedMtx['wrk2'].values[1],HHWrkSeedMtx['wrk3'].values[1]],  # HH2
                        [HHWrkSeedMtx['wrk0'].values[2],HHWrkSeedMtx['wrk1'].values[2],HHWrkSeedMtx['wrk2'].values[2],HHWrkSeedMtx['wrk3'].values[2]],  # HH3
                        [HHWrkSeedMtx['wrk0'].values[3],HHWrkSeedMtx['wrk1'].values[3],HHWrkSeedMtx['wrk2'].values[3],HHWrkSeedMtx['wrk3'].values[3]]]) # HH4
		r = np.array([zone_df.loc[x,'HHS1'],zone_df.loc[x,'HHS2'],zone_df.loc[x,'HHS3'],zone_df.loc[x,'HHS4']]) # HHSize from the HH size submodel by zone
		c = np.array([zone_df.loc[x,'HHWRK0'],zone_df.loc[x,'HHWRK1'],zone_df.loc[x,'HHWRK2'],zone_df.loc[x,'HHWRK3']]) # HHWorkers from the HH workers submodel by zone
		""" Run Visum balanceMatrix function """
		try:
			balanced_mat = VisumPy.matrices.balanceMatrix(mat,r,c,closePctDiff=0.001) # matrix balancing default 0.0001
			# Paste in balanced values to new df fields
			zone_df.loc[x,'HH1W0'] = balanced_mat[0,0]
			zone_df.loc[x,'HH1W1'] = balanced_mat[0,1]
			zone_df.loc[x,'HH2W0'] = balanced_mat[1,0]
			zone_df.loc[x,'HH2W1'] = balanced_mat[1,1]
			zone_df.loc[x,'HH2W2'] = balanced_mat[1,2]
			zone_df.loc[x,'HH3W0'] = balanced_mat[2,0]
			zone_df.loc[x,'HH3W1'] = balanced_mat[2,1]
			zone_df.loc[x,'HH3W2'] = balanced_mat[2,2]
			zone_df.loc[x,'HH3W3'] = balanced_mat[2,3]
			zone_df.loc[x,'HH4W0'] = balanced_mat[3,0]
			zone_df.loc[x,'HH4W1'] = balanced_mat[3,1]
			zone_df.loc[x,'HH4W2'] = balanced_mat[3,2]
			zone_df.loc[x,'HH4W3'] = balanced_mat[3,3]
		except:
			errstring = f"Could not balance NO {zone_df.loc[x,'NO']}	HH size: {np.array2string(r)} sum: {np.sum(r)}	Workers: {np.array2string(c)} sum: {np.sum(c)}\n"
			addIn.ReportMessage(errstring, MessageType.Error)
			continue

# Set Visum zone fields with HHWRK(0-3) values
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH1W0",zone_df['HH1W0'])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH1W1",zone_df['HH1W1'])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH2W0",zone_df['HH2W0'])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH2W1",zone_df['HH2W1'])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH2W2",zone_df['HH2W2'])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH3W0",zone_df['HH3W0'])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH3W1",zone_df['HH3W1'])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH3W2",zone_df['HH3W2'])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH3W3",zone_df['HH3W3'])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH4W0",zone_df['HH4W0'])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH4W1",zone_df['HH4W1'])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH4W2",zone_df['HH4W2'])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH4W3",zone_df['HH4W3'])

""" Step 4: number of vehicles available """
# Create array to match up with coefficient array for worker model
XArray = np.zeros(len(NumVehiclesModel))

y = 0
for x in range(len(HIWOutputFile)):
	if HIWOutputFile.loc[x,'TOTHH'] > 0:
		XArray[0] =  1 # ASC
		XArray[1] =  1 if HIWOutputFile.loc[x,'HHSIZE'] == 1 else 0
		XArray[2] =  1 if HIWOutputFile.loc[x,'HHSIZE'] == 2 else 0
		XArray[3] =  1 if HIWOutputFile.loc[x,'HHSIZE'] == 3 else 0
		XArray[4] =  1 if HIWOutputFile.loc[x,'HHSIZE'] == 4 else 0
		XArray[5] =  1 if HIWOutputFile.loc[x,'INCOME'] == 1 else 0
		XArray[6] =  1 if HIWOutputFile.loc[x,'INCOME'] == 2 else 0
		XArray[7] =  1 if HIWOutputFile.loc[x,'INCOME'] == 3 else 0
		XArray[8] =  1 if HIWOutputFile.loc[x,'INCOME'] == 4 else 0
		XArray[9] =  1 if HIWOutputFile.loc[x,'WORKERS'] == 0 else 0 # veh 1, 2, 3 gt workers 
		XArray[10] = 1 if HIWOutputFile.loc[x,'WORKERS'] == 1 else 0 # veh 2, 3 gt workers
		XArray[11] = 1 if HIWOutputFile.loc[x,'WORKERS'] == 2 else 0 # veh 3 gt workers
		XArray[12] = 1 if HIWOutputFile.loc[x,'WORKERS'] == 3 else 0 # veh 0, 1, 2, 3 lt workers
		
		# Initialize Utilities
		v0 = 0
		v1 = 0
		v2 = 0
		v3 = 0
		# Calculate utilities
		for i in range(len(XArray)):
			# Compute utility for each number of workers by TAZ/HH Size/Income 
			v0 =  v0 + NumVehiclesModel.loc[i,'Veh0']  * XArray[i]
			v1 =  v1 + NumVehiclesModel.loc[i,'Veh1']  * XArray[i]
			v2 =  v2 + NumVehiclesModel.loc[i,'Veh2']  * XArray[i]
			v3 =  v3 + NumVehiclesModel.loc[i,'Veh3'] * XArray[i]
		
		# disable and check
		# Constrain number of workers by HHSize [# Workers <= HH Size)
		v2 = v2 - 500 if HIWOutputFile.loc[x,'HHSIZE'] == 1 else v2
		v3 = v3 - 500 if HIWOutputFile.loc[x,'HHSIZE'] <= 2 else v3
		
		# Compute probabilities for each number of vehicles
		p0 = np.exp(v0)/(np.exp(v0)+np.exp(v1)+np.exp(v2)+np.exp(v3))
		p1 = np.exp(v1)/(np.exp(v0)+np.exp(v1)+np.exp(v2)+np.exp(v3))
		p2 = np.exp(v2)/(np.exp(v0)+np.exp(v1)+np.exp(v2)+np.exp(v3))
		p3 = np.exp(v3)/(np.exp(v0)+np.exp(v1)+np.exp(v2)+np.exp(v3))
		
		# Compute number workers distribution for given TAZ/HH Size/Income
		v0 = p0 * HIWOutputFile.loc[x,'TOTHH']
		v1 = p1 * HIWOutputFile.loc[x,'TOTHH']
		v2 = p2 * HIWOutputFile.loc[x,'TOTHH']
		v3 = p3 * HIWOutputFile.loc[x,'TOTHH']
		
		# Replace values from instances of # Workers > HH Size with 0
		if HIWOutputFile.loc[x,'HHSIZE'] == 1:
			w2 = 0
			w3 = 0
		elif HIWOutputFile.loc[x,'HHSIZE'] == 2:
			w3 = 0
		
		# Paste in number of vehicles to HIWVOutputFile by TAZ/HH Size/Income/Workers
		HIWVOutputFile.loc[y,'TOTHH'] = v0
		y = y + 1
		HIWVOutputFile.loc[y,'TOTHH'] = v1
		y = y + 1
		HIWVOutputFile.loc[y,'TOTHH'] = v2
		y = y + 1
		HIWVOutputFile.loc[y,'TOTHH'] = v3
		y = y + 1
	else:
		HIWVOutputFile.loc[y,'TOTHH'] = 0
		y = y + 1
		HIWVOutputFile.loc[y,'TOTHH'] = 0
		y = y + 1
		HIWVOutputFile.loc[y,'TOTHH'] = 0
		y = y + 1
		HIWVOutputFile.loc[y,'TOTHH'] = 0
		y = y + 1
		continue


# Paste 3-Way distribution output table into csv file TOTHH field
HIWVOutputFile['TOTHH'].fillna(0, inplace=True) # Replace blank cells with 0
HIWVOutputFile.to_csv(hh_out_path+"HHSize_Inc_Workers_Veh.csv", index = False)

# Collapse Number of Vehicles by TAZ to get HHVEH(0-3)
# Filter by # of vehicles
veh0 = HIWVOutputFile[HIWVOutputFile['VEHICLES'] == 0].reset_index(drop=True)
veh1 = HIWVOutputFile[HIWVOutputFile['VEHICLES'] == 1].reset_index(drop=True)
veh2 = HIWVOutputFile[HIWVOutputFile['VEHICLES'] == 2].reset_index(drop=True)
veh3 = HIWVOutputFile[HIWVOutputFile['VEHICLES'] == 3].reset_index(drop=True)

zone_df['HHVEH0'] = 0
zone_df['HHVEH1'] = 0
zone_df['HHVEH2'] = 0
zone_df['HHVEH3'] = 0
# Groupby ZONE 
veh0 = veh0.groupby(['ZONE'])
veh1 = veh1.groupby(['ZONE'])
veh2 = veh2.groupby(['ZONE'])
veh3 = veh3.groupby(['ZONE'])
# Sum TOTHH by ZONE
zone_df['HHVEH0'] = veh0['TOTHH'].sum().reset_index(drop=True)
zone_df['HHVEH1'] = veh1['TOTHH'].sum().reset_index(drop=True)
zone_df['HHVEH2'] = veh2['TOTHH'].sum().reset_index(drop=True)
zone_df['HHVEH3'] = veh3['TOTHH'].sum().reset_index(drop=True)
# Replace empty cells with 0
zone_df.fillna(0, inplace=True) # Replace blank cells with 0	
# Set Visum zone fields with HHWRK(0-3) values
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HHVEH0",zone_df['HHVEH0'])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HHVEH1",zone_df['HHVEH1'])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HHVEH2",zone_df['HHVEH2'])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HHVEH3",zone_df['HHVEH3'])

for x in range(len(zone_df)):

	if zone_df.loc[x,'TOTHH'] > 0:
		## HHSize x Workers
		mat = np.array([[HHVehSeedMtx ['VEH0'].values[0],HHVehSeedMtx['VEH1'].values[0],HHVehSeedMtx['VEH2'].values[0],HHVehSeedMtx['VEH3'].values[0]],  # HH1
                        [HHVehSeedMtx ['VEH0'].values[1],HHVehSeedMtx['VEH1'].values[1],HHVehSeedMtx['VEH2'].values[1],HHVehSeedMtx['VEH3'].values[1]],  # HH2
                        [HHVehSeedMtx ['VEH0'].values[2],HHVehSeedMtx['VEH1'].values[2],HHVehSeedMtx['VEH2'].values[2],HHVehSeedMtx['VEH3'].values[2]],  # HH3
                        [HHVehSeedMtx ['VEH0'].values[3],HHVehSeedMtx['VEH1'].values[3],HHVehSeedMtx['VEH2'].values[3],HHVehSeedMtx['VEH3'].values[3]]]) # HH4
		r = np.array([zone_df.loc[x,'HHS1'],zone_df.loc[x,'HHS2'],zone_df.loc[x,'HHS3'],zone_df.loc[x,'HHS4']]) # HHSize from the HH size submodel by zone
		c = np.array([zone_df.loc[x,'HHVEH0'],zone_df.loc[x,'HHVEH1'],zone_df.loc[x,'HHVEH2'],zone_df.loc[x,'HHVEH3']]) # HHVehicles from the HH Vehicles submodel by zone
		""" Run Visum balanceMatrix function """
		try:
			balanced_mat = VisumPy.matrices.balanceMatrix(mat,r,c,closePctDiff=0.001) # matrix balancing default 0.0001
			# Paste in balanced values to new df fields
			zone_df.loc[x,'HH1VEH0'] = balanced_mat[0,0]
			zone_df.loc[x,'HH1VEH1'] = balanced_mat[0,1]
			zone_df.loc[x,'HH2VEH0'] = balanced_mat[1,0]
			zone_df.loc[x,'HH2VEH1'] = balanced_mat[1,1]
			zone_df.loc[x,'HH2VEH2'] = balanced_mat[1,2]
			zone_df.loc[x,'HH3VEH0'] = balanced_mat[2,0]
			zone_df.loc[x,'HH3VEH1'] = balanced_mat[2,1]
			zone_df.loc[x,'HH3VEH2'] = balanced_mat[2,2]
			zone_df.loc[x,'HH3VEH3'] = balanced_mat[2,3]
			zone_df.loc[x,'HH4VEH0'] = balanced_mat[3,0]
			zone_df.loc[x,'HH4VEH1'] = balanced_mat[3,1]
			zone_df.loc[x,'HH4VEH2'] = balanced_mat[3,2]
			zone_df.loc[x,'HH4VEH3'] = balanced_mat[3,3]
		except:
			errstring = f"Could not balance NO {zone_df.loc[x,'NO']}	HH size: {np.array2string(r)} sum: {np.sum(r)}	Vehicles: {np.array2string(c)} sum: {np.sum(c)}\n"
			addIn.ReportMessage(errstring, MessageType.Error)
			continue

# Set Visum zone fields with HHSize x Vehicle values
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH1VEH0",zone_df['HH1VEH0'])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH1VEH1",zone_df['HH1VEH1'])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH2VEH0",zone_df['HH2VEH0'])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH2VEH1",zone_df['HH2VEH1'])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH2VEH2",zone_df['HH2VEH2'])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH3VEH0",zone_df['HH3VEH0'])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH3VEH1",zone_df['HH3VEH1'])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH3VEH2",zone_df['HH3VEH2'])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH3VEH3",zone_df['HH3VEH3'])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH4VEH0",zone_df['HH4VEH0'])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH4VEH1",zone_df['HH4VEH1'])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH4VEH2",zone_df['HH4VEH2'])
VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH4VEH3",zone_df['HH4VEH3'])

# TODO dummy values as placeholder
# zone_df['HH1VEH0'] = zone_df['HHS1'] * 0.1
# zone_df['HH1VEH1'] = zone_df['HHS1'] * 0.9
# zone_df['HH2VEH0'] = zone_df['HHS2'] * 0.1 
# zone_df['HH2VEH1'] = zone_df['HHS2'] * 0.3 
# zone_df['HH2VEH2'] = zone_df['HHS2'] * 0.6 
# zone_df['HH3VEH0'] = zone_df['HHS3'] * 0.1
# zone_df['HH3VEH1'] = zone_df['HHS3'] * 0.2 
# zone_df['HH3VEH2'] = zone_df['HHS3'] * 0.7 
# zone_df['HH4VEH0'] = zone_df['HHS4'] * 0.1
# zone_df['HH4VEH1'] = zone_df['HHS4'] * 0.1
# zone_df['HH4VEH2'] = zone_df['HHS4'] * 0.5
# zone_df['HH4VEH3'] = zone_df['HHS4'] * 0.2
# 
# # Set Visum zone fields with HHWRK(0-3) values
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH1VEH0",zone_df['HH1VEH0'])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH1VEH1",zone_df['HH1VEH1'])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH2VEH0",zone_df['HH2VEH0'])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH2VEH1",zone_df['HH2VEH1'])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH2VEH2",zone_df['HH2VEH2'])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH3VEH0",zone_df['HH3VEH0'])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH3VEH1",zone_df['HH3VEH1'])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH3VEH2",zone_df['HH3VEH2'])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH4VEH0",zone_df['HH4VEH0'])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH4VEH1",zone_df['HH4VEH1'])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH4VEH2",zone_df['HH4VEH2'])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH4VEH3",zone_df['HH4VEH3'])


# Reshape 3-way distribution to be pasted into Visum zone layer
# triple_dist = pd.DataFrame({'NO':zone_df['NO']})
# i = 0
# for x in range(len(triple_dist)):
# 	for y in range(1,65):
# 		triple_dist.loc[x,y] = HIWOutputFile.loc[i,'TOTHH']
# 		i = i + 1
# 
# # Replace empty cells with 0
# triple_dist.fillna(0, inplace=True) # Replace blank cells with 0		
# 		
# # Paste 3-way distribution into Visum zone layer
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH1INC1W0",triple_dist[1])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH1INC1W1",triple_dist[2])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH1INC2W0",triple_dist[5])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH1INC2W1",triple_dist[6])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH1INC3W0",triple_dist[9])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH1INC3W1",triple_dist[10])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH1INC4W0",triple_dist[13])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH1INC4W1",triple_dist[14])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH2INC1W0",triple_dist[17])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH2INC1W1",triple_dist[18])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH2INC1W2",triple_dist[19])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH2INC2W0",triple_dist[21])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH2INC2W1",triple_dist[22])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH2INC2W2",triple_dist[23])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH2INC3W0",triple_dist[25])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH2INC3W1",triple_dist[26])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH2INC3W2",triple_dist[27])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH2INC4W0",triple_dist[29])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH2INC4W1",triple_dist[30])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH2INC4W2",triple_dist[31])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH3INC1W0",triple_dist[33])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH3INC1W1",triple_dist[34])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH3INC1W2",triple_dist[35])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH3INC1W3",triple_dist[36])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH3INC2W0",triple_dist[37])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH3INC2W1",triple_dist[38])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH3INC2W2",triple_dist[39])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH3INC2W3",triple_dist[40])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH3INC3W0",triple_dist[41])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH3INC3W1",triple_dist[42])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH3INC3W2",triple_dist[43])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH3INC3W3",triple_dist[44])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH3INC4W0",triple_dist[45])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH3INC4W1",triple_dist[46])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH3INC4W2",triple_dist[47])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH3INC4W3",triple_dist[48])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH4INC1W0",triple_dist[49])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH4INC1W1",triple_dist[50])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH4INC1W2",triple_dist[51])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH4INC1W3",triple_dist[52])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH4INC2W0",triple_dist[53])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH4INC2W1",triple_dist[54])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH4INC2W2",triple_dist[55])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH4INC2W3",triple_dist[56])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH4INC3W0",triple_dist[57])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH4INC3W1",triple_dist[58])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH4INC3W2",triple_dist[59])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH4INC3W3",triple_dist[60])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH4INC4W0",triple_dist[61])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH4INC4W1",triple_dist[62])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH4INC4W2",triple_dist[63])
# VisumPy.helpers.SetMulti(Visum.Net.Zones,"HH4INC4W3",triple_dist[64])

 