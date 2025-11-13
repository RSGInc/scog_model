"""
VMT reporting by Functional Class
michael.mccarthy@rsginc.com 2025-07-28

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
timestamp = datetime.now().strftime("%m%d%Y_%H%M%S")

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

# Person Trips
# Vehicle Trips
# Regional VMT, VHT, VHD
links_df['VMT'] = links_df['LENGTH'] * links_df['DLY_AUTO_VOLUME']
links_df['VHT_DLY_ff'] = links_df['FFTIME'] * links_df['DLY_AUTO_VOLUME']
links_df['VHT_AM_ff'] = links_df['FFTIME'] * links_df['AM_AUTO_VOLUME']
links_df['VHT_PM_ff'] = links_df['FFTIME'] * links_df['PM_AUTO_VOLUME']
links_df['VHT_OP_ff'] = links_df['FFTIME'] * links_df['OP_AUTO_VOLUME']
links_df['VHT_PMPK_ff'] = links_df['FFTIME'] * links_df['PMPK_AUTO_VOLUME']
links_df['VHT_AM_cong'] = links_df['AMTIME'] * links_df['AM_AUTO_VOLUME']
links_df['VHT_PM_cong'] = links_df['PMTIME'] * links_df['PM_AUTO_VOLUME']
links_df['VHT_OP_cong'] = links_df['OPTIME'] * links_df['OP_AUTO_VOLUME']
links_df['VHT_PMPK_cong'] = links_df['PMPKTIME'] * links_df['PMPK_AUTO_VOLUME']
links_df['VHT_DLY_cong'] = links_df['VHT_AM_cong'] + links_df['VHT_PM_cong'] + links_df['VHT_OP_cong']
links_df['VHD_DLY'] = links_df['VHT_DLY_cong'] - links_df['VHT_DLY_ff']
links_df['VHD_AM'] = links_df['VHT_AM_cong'] - links_df['VHT_AM_ff']
links_df['VHD_PM'] = links_df['VHT_PM_cong'] - links_df['VHT_PM_ff']
links_df['VHD_OP'] = links_df['VHT_OP_cong'] - links_df['VHT_OP_ff']
links_df['VHD_PMPK'] = links_df['VHT_PMPK_cong'] - links_df['VHT_PMPK_ff']

sum_df = links_df.sum().reset_index()
sum_df.to_csv(os.path.join(out_path, "ModelReport_"+timestamp+".csv"), index = False)

# VMT by FC
# vmt_out = links_df.groupby('NFCLASS')['vmt'].sum().reset_index()
# filename = "VMTRep_"+timestamp+".csv"
# vmt_out.to_csv(os.path.join(out_path, filename), index = False)

    