"""
Transfer Calrep from TransCAD GISDK to Python/Pandas

    - Developed for Visum but agnostic
    - Given dataframe of link data and CSV of queries, calculate summaries and write out CSV

michael.mccarthy@rsginc.com 2025-03-07

"""

# Libraries
import sys
import numpy as np
import VisumPy.helpers
import VisumPy.matrices
import pandas as pd
from datetime import datetime
import os.path

def calrep (name, links_df, flowfld, joinfld, count_df, cntfld, queryfile, outdir):
    
    # set defaults/handle null
    fftime = 'FFTIME'
    congtime = 'CTIME'

    # output path
    timestamp = datetime.now().strftime("%m%d%Y_%H%M%S")
    filename = "CalRep_"+name+"_"+timestamp+".csv"

    # create output
    cols = ["Type","Item","NumObs","TotCnt","TotMod","AvgCnt","AvgMod","Tstat","AvgErr","PctErr","PctRMSE","MAPE","CorrCoef","SumSqErr","MeanSqErr","Miles","kVMT","kVHT_FF","kVHT_CTime"]
    calrep_df = pd.DataFrame(columns = cols)

    # calc directional flow * time
    links_df['volfftime'] = links_df[flowfld] * links_df[fftime]
    links_df['volctime'] = links_df[flowfld] * links_df[congtime]

    # group links by No and sum each direction
    field_map = {
        'TYPENO': 'first',
        'TSYSSET': 'first',
        'LENGTH': 'first',
        flowfld: 'sum',
        'volfftime': 'sum',
        'volctime': 'sum'
    }

    links_agg = links_df.groupby('NO').agg(field_map).reset_index()

    # join counts to network
    join_df = links_agg.merge(count_df, how="left", on=joinfld)

    # loop through queries
    query_df = pd.read_csv(queryfile)

    # ensure links are either 1 per direction or grouped into 2
    for i in range(len(query_df)):
        qtype = query_df.loc[i,['Type', 'Item', 'Query']].to_list()
        q = qtype[2]
        if q.split(" ")[0] not in join_df.columns:
            print('Missing field in query: '+ str(q))
            continue
        group_df = join_df.query(q)
        numobs = len(group_df)

        if numobs > 0:
            cnt = group_df[cntfld]
            vol = group_df[flowfld]
            totcnt = cnt.sum()
            avgcnt = cnt.mean() 
            stdcnt = cnt.std()
            totvol = vol.sum()
            avgvol = vol.mean() 
            stdvol = vol.std()
            avgerr = avgvol - avgcnt
            tstat = avgerr / np.sqrt(((stdcnt ** 2)/numobs) + ((stdvol ** 2)/numobs))
            pcterr = 100*avgerr/avgcnt
            errvec = vol - cnt
            sqerr = errvec ** 2
            mse = sqerr.mean()
            pctrmse = 100*np.sqrt(mse)/avgcnt
            pctabserr = errvec.abs() / cnt
            mape = 100*pctabserr.mean()
            tmp = (cnt - avgcnt)*(vol - avgvol)
            corrcoef = tmp.sum()/max(1,(numobs - 1)*stdcnt*stdvol)
            sumsqerr = sqerr.sum()
        else:
            cnt = 0
            vol = 0
            totcnt = 0
            avgcnt = 0 
            stdcnt = 0
            totvol = 0
            avgvol = 0 
            stdvol = 0
            avgerr = 0
            tstat = 0
            pcterr = 0
            errvec = 0
            sqerr = 0
            mse = 0
            pctrmse = 0
            pctabserr = 0
            mape = 0
            tmp = 0
            corrcoef = 0

        # query whole network
        if numobs > 0:
            summiles = group_df['LENGTH'].sum()
            vmt_vect = group_df[flowfld] * group_df['LENGTH']
            vht_ff_vect = group_df['volfftime'] / 60
            vht_ct_vect = group_df['volctime'] / 60
            vmt = vmt_vect.sum()/1000
            vht_ff = vht_ff_vect.sum()/1000
            vht_ct = vht_ct_vect.sum()/1000
        else:
            summiles = 0
            vmt = 0
            vht_ff = 0
            vht_ct = 0

        if numobs == 0:
             totcnt, totvol, avgcnt, avgvol, tstat, avgerr, pcterr, pctrmse, mape, corrcoef, sumsqerr, mse, summiles, vmt, vht_ff, vht_ct = (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
		
        # add record/row
        calrep_df.loc[i] = list((qtype[0], qtype[1], numobs, totcnt, totvol, avgcnt, avgvol, tstat, avgerr, pcterr, pctrmse, mape, corrcoef, sumsqerr, mse, summiles, vmt, vht_ff, vht_ct))
    
    # Export count_summary_df to csv file in timestamped folder
    calrep_df.to_csv(os.path.join(outdir, filename))

# set paths 
out_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..', 'Outputs'))
shared_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..', 'SharedData'))
reports_path = shared_path + "/Reports/"

# Pull Link attributes from Visum and create dataframe
no          = VisumPy.helpers.GetMulti(Visum.Net.Links,"No")
length      = VisumPy.helpers.GetMulti(Visum.Net.Links,"Length")
typeno      = VisumPy.helpers.GetMulti(Visum.Net.Links,"TypeNo")
tsys        = VisumPy.helpers.GetMulti(Visum.Net.Links,"TSysSet")
amvol       = VisumPy.helpers.GetMulti(Visum.Net.Links,"AM_AUTO_VOLUME")
pmvol       = VisumPy.helpers.GetMulti(Visum.Net.Links,"PM_AUTO_VOLUME")
pmpkvol     = VisumPy.helpers.GetMulti(Visum.Net.Links,"PMPK_AUTO_VOLUME")
opvol       = VisumPy.helpers.GetMulti(Visum.Net.Links,"OP_AUTO_VOLUME")
fftime      = VisumPy.helpers.GetMulti(Visum.Net.Links,"T0_PRTSYS(C)")
ctime      = VisumPy.helpers.GetMulti(Visum.Net.Links,"PM_CTIME")

# Set up dataframe to use for all needed zone attributes. Update as needed during coding
links_df = pd.DataFrame({'NO':no, 'LENGTH': length, 'TYPENO': typeno, 'TSYSSET': tsys, 'AM_AUTO_VOLUME': amvol, 'PM_AUTO_VOLUME': pmvol, 'PMPK_AUTO_VOLUME': pmpkvol, 'OP_AUTO_VOLUME': opvol,  'FFTIME': fftime, 'CTIME': ctime})

# count_test = pd.read_csv(reports_path + 'counts_2way.csv') # testing file
count_file = pd.read_csv(reports_path + 'merged_Auto_counts_5_13.csv')
query_file = reports_path + 'calrepinfo.csv' # TODO testing

calrep("AMAuto", links_df, 'AM_AUTO_VOLUME', 'NO', count_file, 'Count_AM', query_file, out_path)
calrep("PMAuto", links_df, 'PM_AUTO_VOLUME', 'NO', count_file, 'Count_PM', query_file, out_path)
calrep("OPAuto", links_df, 'OP_AUTO_VOLUME', 'NO', count_file, 'Count_OP', query_file, out_path)
calrep("PMPeakAuto", links_df, 'PMPK_AUTO_VOLUME', 'NO', count_file, 'Count_PMPK', query_file, out_path)
#calrep("DLYAuto", links_df, 'DLY_AUTO_VOLUME', 'NO', count_file, 'Count_DLY', query_file, out_path) # TODO daily volume vs count


