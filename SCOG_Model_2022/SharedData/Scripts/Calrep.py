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

# split multiple screenlines (not neccessary)
# def screenlines (df):
#   # Break out SCRNLINE field to separate by commas into individual columns
#   df['SCREENLINES'] = df['SCREENLINES'].astype(str)
#   df = pd.concat([df,df['SCREENLINES'].str.split(',', expand = True)], axis = 1)
#   # Change Screenline field names
#   df = df.rename(columns = {0:'SCRNLINE1',1:'SCRNLINE2'})
#   # Replace null values with 0 in the screenline fields
#   df['SL1'] = df['SL1'].replace('',np.nan).fillna(0)
#   df['SL2'] = df['SL2'].replace('',np.nan).fillna(0)
    

def calrep (name, links_df, flowfld, joinfld, count_df, cntfld, queryfile, outdir):
    
    # set defaults/handle null
    fftime = 'FFTIME'
    congtime = 'CTIME'
    
    write_links = True if cntfld == "DLY" else False

    # output path
    timestamp = datetime.now().strftime("%m%d%Y_%H%M%S")
    filename = "CalRep_"+name+"_"+timestamp+".csv"
    joinexport = "CountJoin_"+name+"_"+timestamp+".csv"
    joinexport2 = "CountJoin2_"+name+"_"+timestamp+".csv"

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
        'NFCLASS': 'first',
        'AreaType': 'first',
        'SCREENLINE': 'max', # ','.join, # appears in one direction
        'LENGTH': 'first',
        flowfld: 'sum',
        'volfftime': 'sum',
        'volctime': 'sum'
    }

    # ensure links are either 1 per direction or grouped into 2
    links_agg = links_df.groupby('NO').agg(field_map).reset_index()

    # also group counts by link NO, exclude empty counts, and manually exclude some counts
    counts_agg = count_df[(count_df['USECOUNT'] == 1) & (count_df[cntfld] > 0)].groupby('NO').agg({'StationName': 'first', 'EXT_COUNT':'max','USECOUNT':'min','DLY':'sum','AM':'sum','PM':'sum','PM_PKHR':'sum','OP':'sum'}).reset_index()

    # join counts to network
    join_df = links_agg.merge(counts_agg, how="right", on=joinfld)

    # loop through queries
    query_df = pd.read_csv(queryfile)

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
    join_df.to_csv(os.path.join(outdir, joinexport))
    
    if write_links == True:
        # join totflow and totcount to links df
        join_df2 = links_df.merge(counts_agg, how="left", on=joinfld)
        join_df2 = join_df2[['NO','DLY']]
        join_df3 = join_df2.merge(links_agg, how="left", on=joinfld)
        
        join_df3.to_csv(os.path.join(outdir, joinexport2)) # this join has all links and duplicates the counts on the right
        
        # calc individual link errors for calibration
        join_df3['count_tot'] = join_df3[cntfld]
        join_df3['count_err'] = join_df3[flowfld] - join_df3[cntfld]
        join_df3['pct_err'] = 100*(join_df3['count_err'] / join_df3[cntfld])
        
        # write back err to visum network
        counts_list = join_df3['count_tot'].to_list()
        errs_list = join_df3['count_err'].to_list()
        pcterr_list = join_df3['pct_err'].to_list()
        VisumPy.helpers.SetMulti(Visum.Net.Links,"calrep_2way_count",counts_list)
        VisumPy.helpers.SetMulti(Visum.Net.Links,"calrep_2way_err",errs_list)
        VisumPy.helpers.SetMulti(Visum.Net.Links,"calrep_2way_pcterr",pcterr_list)
        
    

# set paths 
out_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..', 'Outputs'))
shared_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..', 'SharedData'))
reports_path = shared_path + "/Reports/"

# Pull Link attributes from Visum and create dataframe
no          = VisumPy.helpers.GetMulti(Visum.Net.Links,"No")
length      = VisumPy.helpers.GetMulti(Visum.Net.Links,"Length")
typeno      = VisumPy.helpers.GetMulti(Visum.Net.Links,"TypeNo")
tsys        = VisumPy.helpers.GetMulti(Visum.Net.Links,"TSysSet")
nfc         = VisumPy.helpers.GetMulti(Visum.Net.Links,"NFCLASS")
area         = VisumPy.helpers.GetMulti(Visum.Net.Links,"AREATYPE")
scrnlna    = VisumPy.helpers.GetMulti(Visum.Net.Links,r"CONCATENATE:SCREENLINES\NO")
scrnlnb    = VisumPy.helpers.GetMulti(Visum.Net.Links,r"ReverseLink\CONCATENATE:SCREENLINES\NO")
amvol       = VisumPy.helpers.GetMulti(Visum.Net.Links,"AM_AUTO_VOLUME")
pmvol       = VisumPy.helpers.GetMulti(Visum.Net.Links,"PM_AUTO_VOLUME")
pmpkvol     = VisumPy.helpers.GetMulti(Visum.Net.Links,"PMPK_AUTO_VOLUME")
opvol       = VisumPy.helpers.GetMulti(Visum.Net.Links,"OP_AUTO_VOLUME")
dlyvol      = VisumPy.helpers.GetMulti(Visum.Net.Links,"DLY_AUTO_VOLUME")
fftime      = VisumPy.helpers.GetMulti(Visum.Net.Links,"T0_PRTSYS(C)")
ctime       = VisumPy.helpers.GetMulti(Visum.Net.Links,"PM_CTIME")

# Set up dataframe to use for all needed zone attributes. Update as needed during coding
links_df = pd.DataFrame({'NO':no, 'LENGTH': length, 'TYPENO': typeno, 'TSYSSET': tsys, 'NFCLASS': nfc, 'AreaType': area, 'AM_AUTO_VOLUME': amvol, 'PM_AUTO_VOLUME': pmvol, 'PMPK_AUTO_VOLUME': pmpkvol, 'OP_AUTO_VOLUME': opvol, 'DLY_AUTO_VOLUME': dlyvol,  'FFTIME': fftime, 'CTIME': ctime, 
'SC_with':scrnlna, 'SC_against':scrnlnb})

links_df['SC_with'] = links_df['SC_with'].replace('', np.nan)
links_df['SCREENLINE'] = np.where(links_df['SC_with'].isna(),links_df['SC_against'],links_df['SC_with'])

# links_df.to_csv(reports_path + "calrep_df.csv")

# count_test = pd.read_csv(reports_path + 'counts_2way.csv') # testing file
# count_file = pd.read_csv(reports_path + 'merged_Auto_counts_5_13.csv')
# count_file = pd.read_csv(reports_path + '/SCOG_Counts_07142025/Auto_Counts.csv')
count_file = pd.read_csv(reports_path + '/SCOG_Counts_07142025/Auto_Counts_additionaljoins7-23_ext.csv')
query_file = reports_path + 'calrepinfo.csv'

calrep("AMAuto", links_df, 'AM_AUTO_VOLUME', 'NO', count_file, 'AM', query_file, out_path)
calrep("PMAuto", links_df, 'PM_AUTO_VOLUME', 'NO', count_file, 'PM', query_file, out_path)
calrep("OPAuto", links_df, 'OP_AUTO_VOLUME', 'NO', count_file, 'OP', query_file, out_path)
calrep("PMPeakAuto", links_df, 'PMPK_AUTO_VOLUME', 'NO', count_file, 'PM_PKHR', query_file, out_path)
calrep("DLYAuto", links_df, 'DLY_AUTO_VOLUME', 'NO', count_file, 'DLY', query_file, out_path)

