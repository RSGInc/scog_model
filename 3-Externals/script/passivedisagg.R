# SCOG
# Passive Data Agg/Disagg to External Stations

#----------------------------------
# Requirements

# Dplyr: to install run:
#   install.packages(c("dplyr"))

# rhdf5: to install from Bioconductor run:
#   if (!requireNamespace("BiocManager", quietly = TRUE))
#     install.packages("BiocManager")
#   BiocManager::install("rhdf5")

library(dplyr)
library(rhdf5)

options(scipen=999)

# Set working directory to source location

#OMX API
# source("https://raw.githubusercontent.com/osPlanning/omx-r/master/r/omx.R")
source("./input/r/omxapi.r") # use local copy

#-------------------------------
# Input Data

# OD Matrices
# path to rMerge OMX files (not included)
# demx = Demographic expansion
# netx = Network expansion
res_pm ="./input/20201119_sept2019_skagit_omx/resident_PM_Trips_purp_demx.omx"
vis_pm ="./input/20201119_sept2019_skagit_omx/visitor_PM_Trips_purp_demx.omx"

res_cores = listOMX(res_pm)$Matrices$name
vis_cores = listOMX(vis_pm)$Matrices$name
  
    # Res Matrix Stats
    # Rows 35
    # Columns 35
    # Matrices
    # name dclass     dim   type
    # 1    hbo  FLOAT 35 x 35 matrix
    # 2    hbw  FLOAT 35 x 35 matrix
    # 3   nhbo  FLOAT 35 x 35 matrix
    # 4   nhbw  FLOAT 35 x 35 matrix
    # 5 res_ld  FLOAT 35 x 35 matrix
    #-------------------------------
    # Vis Matrices
    # name dclass     dim   type
    # 1 vis_ld  FLOAT 35 x 35 matrix
    # 2 vis_sd  FLOAT 35 x 35 matrix


# Districts - defined as internal/external
districts = read.csv("./input/districts.csv", stringsAsFactors = FALSE)

districts_lookup = districts %>% arrange(District) %>% pull(Model)

# Districts to External Stations
ext_crosswalk = read.csv("./input/external_stations_crosswalk.csv",stringsAsFactors = FALSE)

# External Splits: Counts at External Stations
ext_counts = read.csv("./input/2018_External_Counts_alt2.csv", stringsAsFactors = FALSE)

# TAZ to District Crosswalk
int_crosswalk = read.csv("./input/internal_zone_crosswalk.csv",stringsAsFactors = FALSE)

# Internal Splits:  TAZ Productions and Attractions
PAtable = read.csv("./input/CalculatedTrips_PA_by_TAZ_0104.csv", stringsAsFactors = FALSE) %>%
  filter(!is.na(TAZ))

# List ext stations/internal TAZs
model_exts = ext_counts  %>%
  filter(!Ext.Station %in% c(905,909)) %>%
  pull(Ext.Station)

model_tazs = int_crosswalk %>%
  filter(NO < 900) %>% # internal TAZs
  pull(NO)
  

#-------------------------------
# External Station Splits
 ext_splits_all = ext_crosswalk %>%
   left_join(ext_counts, by = c("ExternalStation1" = "Ext.Station")) %>%
   mutate(ext1_ei = EI, ext1_ie = IE) %>%
   select(-c("Route.Name","EI","IE")) %>% # drop ext1 join
   left_join(ext_counts, by = c("ExternalStation2" = "Ext.Station")) %>%
   mutate(ext2_ei = EI, ext2_ie = IE) %>%
   select(-c("Route.Name","EI","IE")) # drop ext1 join
 
 ext_splits1 = ext_splits_all %>% # 1-1 District to External Station
     filter(is.na(ExternalStation2)) %>%
     mutate(dest = ExternalStation1, factor_ei = 1, factor_ie = 1)
 
 ext_splits2 = ext_splits_all %>% # District to 1st of 2 External Stations
   filter(!is.na(ExternalStation2)) %>%
   mutate(dest = ExternalStation1, factor_ei = ext1_ei / (ext1_ei + ext2_ei), factor_ie = ext1_ie / (ext1_ie + ext2_ie))
     
 ext_splits3 = ext_splits_all %>% # District to 2nd of 2 External Stations
   filter(!is.na(ExternalStation2)) %>%
   mutate(dest = ExternalStation2, factor_ei = ext2_ei / (ext1_ei + ext2_ei), factor_ie = ext2_ie / (ext1_ie + ext2_ie))
 
 disagg_table = rbind(ext_splits1, ext_splits2, ext_splits3) %>%
   select(rxyRow, dest, factor_ei, factor_ie)
 
 write.csv(disagg_table, "./output/external_disagg.csv", row.names = FALSE)

#-------------------------------
# Internal Zone Splits

int_splits = int_crosswalk %>%
 left_join(PAtable, by = c("NO" = "TAZ")) %>%
 group_by(rxy_dist) %>%
 mutate(P_split = P/sum(P), A_split = A/sum(A)) %>%
 ungroup() %>%
 select(NO, rxy_dist,P,A,P_split,A_split) %>%
 filter(NO < 900) # internal TAZs

write.csv(int_splits, "./output/internal_disagg.csv", row.names = FALSE)


#-------------------------------
# Read Trips

taz_lookup = 1:35 #readLookupOMX(res_pm, "taz")
# EILookup <- readLookupOMX( res_pm, "EI" )

matrix = list()
for(core in res_cores){
  matrix[[core]] = readSelectedOMX(res_pm, core, RowLabels = "taz", ColLabels = "taz")
}

for(core in vis_cores){
  matrix[[core]] = readSelectedOMX(vis_pm, core, RowLabels = "taz", ColLabels = "taz")
}

passive_od = matrix$hbo + matrix$hbw + matrix$nhbo + matrix$nhbw + matrix$res_ld + matrix$vis_ld + matrix$vis_sd

#-----------------------
# E-I
# Matrix Setup
agg_EI = matrix(nrow = length(model_exts), ncol = ncol(passive_od))
rownames(agg_EI) = model_exts
colnames(agg_EI) = taz_lookup$Lookup # rXY Districts

disagg_EI = matrix(nrow = length(model_exts), ncol = length(model_tazs))
rownames(disagg_EI) = model_exts
colnames(disagg_EI) = model_tazs

# split external stations first
for(i in seq_along(disagg_table$rxyRow)){ 
  O = disagg_table[i,]$rxyRow
  Otaz = disagg_table[i,]$dest
  rxytrips = passive_od[O,]
  Ofactor = disagg_table[i,]$factor_ei
  
  matrix_i = match(Otaz,model_exts)
  agg_EI[is.na(agg_EI)] = 0
  agg_EI[matrix_i,] = agg_EI[matrix_i,] + (rxytrips * Ofactor)
}

# Disagg internal TAZs  
for(i in seq_along(disagg_table$rxyRow)){ # external stations x 
  for(j in seq_along(int_splits$rxy_dist)) { # Internal districts -> internal TAZs
    O = disagg_table[i,]$rxyRow
    Otaz = disagg_table[i,]$dest
    D = int_splits[j,]$rxy_dist
    Dtaz = int_splits[j,]$NO
    
    matrix_i = match(Otaz,model_exts) # index of Otaz (SCOG Model External TAZ No.) in model_exts
    rxytrips = agg_EI[matrix_i,D]
    Dfactor = int_splits[j,]$P_split
    
   # matrix_i = match(Otaz,model_exts)
    matrix_j = match(Dtaz,model_tazs) # index of Dtaz (SCOG Model Internal TAZ No.) in model_tazs
    disagg_EI[matrix_i,matrix_j] = rxytrips * Dfactor
  }
}

# write output matrix
#write.csv(disagg_EI, paste0("./output/EI_sum.csv"))

#-----------------------
# I-E
# Matrix Setup
agg_IE = matrix(nrow = nrow(passive_od), ncol = length(model_exts))
rownames(agg_IE) = taz_lookup$Lookup # rXY Districts
colnames(agg_IE) = model_exts

disagg_IE = matrix(nrow = length(model_tazs), ncol = length(model_exts))
rownames(disagg_IE) = model_tazs
colnames(disagg_IE) = model_exts

# split external stations first
for(i in seq_along(disagg_table$rxyRow)){ 
  D = disagg_table[i,]$rxyRow
  Dtaz = disagg_table[i,]$dest
  rxytrips = passive_od[,D]
  Dfactor = disagg_table[i,]$factor_ie
  
  matrix_i = match(Dtaz,model_exts)
  agg_IE[is.na(agg_IE)] = 0
  agg_IE[,matrix_i] = agg_IE[,matrix_i] + (rxytrips * Dfactor)
} # Agg IE is ok (1-35 (all) x 1-12;21-35 (ext))

# Disagg internal TAZs  (Int TAZs x External Stations)
for(i in seq_along(int_splits$rxy_dist)) {
  for(j in seq_along(disagg_table$rxyRow)){
    O = int_splits[i,]$rxy_dist
    Otaz = int_splits[i,]$NO
    D = disagg_table[j,]$rxyRow
    Dtaz = disagg_table[j,]$dest
    
    matrix_i = match(Otaz,model_tazs) # index of Otaz (SCOG Model Internal TAZ No.) in model_tazs
    matrix_j = match(Dtaz,model_exts) # index of Dtaz (SCOG Model External TAZ No.) in model_exts

    rxytrips = agg_IE[O,matrix_j]
    Ofactor = int_splits[i,]$A_split
    
    disagg_IE[matrix_i,matrix_j] = rxytrips * Ofactor
  }
}

# write output matrix
#write.csv(disagg_IE, paste0("./output/IE_sum.csv"))


#----------------------
# E-E
# Matrix Setup
agg_EE = matrix(nrow = length(model_exts), ncol = length(model_exts))
rownames(agg_EE) = model_exts
colnames(agg_EE) = model_exts

# Clear intrazonals - non pass-thru
ee_passive_od = passive_od
diag(ee_passive_od) = 0

# Clear other non-pass-thru i,j pairs (by rXY district)
# North of Skagit: 31,32,33,34,35
# South of Skagit: 3,5,6,7,8,9,10,11,21,22,23,24,25,26,27,28,29,30
# Whidbey ferry trips: 1, 2, 4 to zones south of Tulalip

north_pairs = c(31,32,33,34,35)
south_pairs = c(3,5,6,7,8,9,10,11,21,22,23,24,25,26,27,28,29,30)

for(i in north_pairs){
  for(j in north_pairs){
    ee_passive_od[i,j] = 0
  }
}

for(i in south_pairs){
  for(j in south_pairs){
    ee_passive_od[i,j] = 0
  }
}

# Whidbey Island Trips routed via Ferry
whidbey_dist = c(1,2,4)
# whidbey_scog_pct = c(0.30,0.10,NA,0.05)
whidbey_scog_pct = c(0.30*0.5,0.05*0.5,NA,0.03*0.5)
southern_dist = c(25,26,27,28,29,30,5,6,7,8,9,10,11)

for(i in whidbey_dist){
  for(j in southern_dist){
    ee_passive_od[i,j] = ee_passive_od[i,j] * whidbey_scog_pct[i]
  }
}

for(i in southern_dist){
  for(j in whidbey_dist){
    ee_passive_od[i,j] = ee_passive_od[i,j] * whidbey_scog_pct[j]
  }
}

# Whidbey Island Trips routed via SCOG - keep
# whidbey_dist = c(1,2,4)
# whidbey_scog_pct = c(0.30,0.10,0.05)
# southern_dist = c(21,22,23,24)
# 
# for(i in whidbey_dist){
#   for(j in southern_dist){
#     ee_passive_od[i,j] = ee_passive_od[i,j] * whidbey_scog_pct[i]
#   }
# }
# 
# for(i in southern_dist){
#   for(j in whidbey_dist){
#     ee_passive_od[i,j] = ee_passive_od[i,j] * whidbey_scog_pct[j]
#   }
# }

# Aggregate to Ext stations and apply traffic count factors
for(i in seq_along(disagg_table$rxyRow)){
  for(j in seq_along(disagg_table$rxyRow)){
  O = disagg_table[i,]$rxyRow
  Otaz = disagg_table[i,]$dest
  D = disagg_table[j,]$rxyRow
  Dtaz = disagg_table[j,]$dest
  rxytrips = ee_passive_od[O,D]
  Ofactor = disagg_table[i,]$factor_ei
  Dfactor = disagg_table[j,]$factor_ie
  
  matrix_i = match(Otaz,model_exts)
  matrix_j = match(Dtaz,model_exts)
  agg_EE[is.na(agg_EE)] = 0
 # cat(O,Otaz,matrix_i,D,Dtaz,matrix_j,rxytrips,"\n")
  agg_EE[matrix_i,matrix_j] = agg_EE[matrix_i,matrix_j] + (rxytrips * Ofactor * Dfactor) # CHECK
  }
}


# write output matrix
#write.csv(agg_EE, paste0("./output/EE_sum.csv"))

#-------------------------------
# Special Zones: 905 Guemes (rxy 20) and 909 Eastern Skagit SR 20 (rxy 13)

# Assume traffic count equals external volume; keep distribution from passive OD
spec_counts = ext_counts %>% filter(Ext.Station %in% c(905,909))
spec_rows = passive_od[c(20,13),]
spec_cols = passive_od[,c(20,13)]

# Get Distribution
spec_rows_prop = spec_rows
spec_rows_prop[1,] = spec_rows[1,]/sum(spec_rows[1,]) # proportions by column
spec_rows_prop[2,] = spec_rows[2,]/sum(spec_rows[2,])

spec_cols_prop = spec_cols
spec_cols_prop[,1] = spec_cols[,1]/sum(spec_cols[,1]) # proportions by row
spec_cols_prop[,2] = spec_cols[,2]/sum(spec_cols[,2])

# Distribute Traffic Counts
spec_EI = matrix(nrow = 35, ncol = 35)
spec_EI[20,] = spec_counts[1,]$EI * spec_rows_prop[1,]
spec_EI[13,] = spec_counts[2,]$EI * spec_rows_prop[2,]
spec_EI[,20] = spec_counts[1,]$EI * spec_cols_prop[,1]
spec_EI[,13] = spec_counts[2,]$EI * spec_cols_prop[,2]

spec_IE = matrix(nrow = 35, ncol = 35)
spec_IE[20,] = spec_counts[1,]$IE * spec_rows_prop[1,]
spec_IE[13,] = spec_counts[2,]$IE * spec_rows_prop[2,]
spec_IE[,20] = spec_counts[1,]$IE * spec_cols_prop[,1]
spec_IE[,13] = spec_counts[2,]$IE * spec_cols_prop[,2]

spec_mtx = spec_EI + spec_IE

# EI Split into TAZs
spec_disagg_EI = matrix(nrow = 2, ncol = length(model_tazs))
for(i in 1:2) { # Special Externals 20, 13
  for(j in seq_along(int_splits$rxy_dist)) { # Internal districts -> internal TAZs
    O = i
    Otaz = spec_counts[i,]$Ext.Station
    D = int_splits[j,]$rxy_dist
    Dtaz = int_splits[j,]$NO
    
    spec = c(20,13)
    spec_mtx_i = spec[i] # row index in spec_EI
    rxytrips = spec_EI[spec_mtx_i,D]
    Dfactor = int_splits[j,]$P_split
    
    matrix_i = i
    matrix_j = match(Dtaz,model_tazs) # index of Dtaz (SCOG Model Internal TAZ No.) in model_tazs
    spec_disagg_EI[matrix_i,matrix_j] = rxytrips * Dfactor
  }
}

# IE Split into TAZs
spec_disagg_IE = matrix(ncol = 2, nrow = length(model_tazs))
for(j in 1:2) { # Special Externals 20, 13
  for(i in seq_along(int_splits$rxy_dist)) { # Internal districts -> internal TAZs
    O = j
    Otaz = spec_counts[j,]$Ext.Station
    D = int_splits[i,]$rxy_dist
    Dtaz = int_splits[i,]$NO
    
    spec = c(20,13)
    spec_mtx_j = spec[j] # row index in spec_IE
    rxytrips = spec_IE[D,spec_mtx_j]
    Ofactor = int_splits[i,]$A_split
    
    matrix_j = j
    matrix_i = match(Dtaz,model_tazs) # index of Dtaz (SCOG Model Internal TAZ No.) in model_tazs
    spec_disagg_IE[matrix_i,matrix_j] = rxytrips * Ofactor
  }
}

# EI Split into Ext TAZs
spec_disagg_EE1 = matrix(nrow = 2, ncol = length(model_exts))
for(i in 1:2) { # Special Externals 20, 13
  for(j in seq_along(disagg_table$rxyRow)) { # External districts -> External TAZs
    O = i
    Otaz = spec_counts[i,]$Ext.Station
    D = disagg_table[j,]$rxyRow
    Dtaz = disagg_table[j,]$dest
    
    spec = c(20,13)
    spec_mtx_i = spec[i] # row index in spec_EI
    rxytrips = spec_EI[spec_mtx_i,D]
    Dfactor = disagg_table[j,]$factor_ei # Using E-I count for E-Es (E-I-E)
    
    matrix_i = i
    matrix_j = match(Dtaz,model_exts) # index of Dtaz (SCOG Model Internal TAZ No.) in model_tazs
    spec_disagg_EE1[is.na(spec_disagg_EE1)] = 0
    spec_disagg_EE1[matrix_i,matrix_j] = spec_disagg_EE1[matrix_i,matrix_j] + (rxytrips * Dfactor)
  }
}



# IE Split into Ext TAZs
spec_disagg_EE2 = matrix(nrow = length(model_exts), ncol = 2)
for(j in 1:2) { # Special Externals 20, 13
  for(i in seq_along(disagg_table$rxyRow)) { # External districts -> External TAZs
    O = j
    Otaz = spec_counts[j,]$Ext.Station
    D = disagg_table[i,]$rxyRow
    Dtaz = disagg_table[i,]$dest
    
    spec = c(20,13)
    spec_mtx_j = spec[j] #col index in spec_IE
    rxytrips = spec_IE[O,spec_mtx_j]
    Dfactor = disagg_table[i,]$factor_ei # Using E-I count for E-Es (E-I-E)
    
    matrix_i = match(Dtaz,model_exts) # index of Dtaz (SCOG Model Internal TAZ No.) in model_tazs
    matrix_j = j
    spec_disagg_EE2[is.na(spec_disagg_EE2)] = 0
    spec_disagg_EE2[matrix_i,matrix_j] = spec_disagg_EE2[matrix_i,matrix_j] + (rxytrips * Dfactor)
  }
}



#-------------------------------
# Combine I-E, E-I, E-E Matrices
model_exts_2 = sort(c(model_exts,905,909))
int_ext_tazs = c(model_tazs, model_exts_2)

final_mtx = matrix(nrow = length(int_ext_tazs), ncol = length(int_ext_tazs))
rownames(final_mtx) = sort(int_ext_tazs)
colnames(final_mtx) = sort(int_ext_tazs)

for(i in seq_along(int_ext_tazs)) {
  for(j in seq_along(int_ext_tazs)) {
    i_ext = match(int_ext_tazs[[i]], model_exts)
    i_int = match(int_ext_tazs[[i]], model_tazs)
    j_ext = match(int_ext_tazs[[j]], model_exts)
    j_int = match(int_ext_tazs[[j]], model_tazs)
    i_spec_ext = match(int_ext_tazs[[i]], c(905,909))
    j_spec_ext = match(int_ext_tazs[[j]], c(905,909))
    
    if(!is.na(i_int) && !is.na(j_int)) { #I-I
      final_mtx[i,j] = 0
    } else if(!is.na(i_int) && !is.na(j_ext)) { #I-E
      final_mtx[i,j] = disagg_IE[i_int, j_ext]
    } else if(!is.na(i_ext) && !is.na(j_int)) { #E-I
      final_mtx[i,j] = disagg_EI[i_ext, j_int]
    } else if(!is.na(i_ext) && !is.na(j_ext)) { #E-E
      final_mtx[i,j] = agg_EE[i_ext, j_ext]
    } else if(!is.na(i_spec_ext) && !is.na(j_ext)) {
      final_mtx[i,j] = spec_disagg_EE1[i_spec_ext, j_ext] # Special EE rows
    } else if(!is.na(i_ext) && !is.na(j_spec_ext)) {
      final_mtx[i,j] = spec_disagg_EE2[i_ext, j_spec_ext] # Special EE cols
    } else if(!is.na(j_spec_ext)) {
      final_mtx[i,j] = spec_disagg_IE[i_int, j_spec_ext] # Special I-E
    } else if(!is.na(i_spec_ext)) {
      final_mtx[i,j] = spec_disagg_EI[i_spec_ext, j_int] # Special E-I
    } 
  }
}

# 905 to 905,909 and 909 to 905,909 left NA -> assume 0 trips
final_mtx[is.na(final_mtx)] = 0

# Clear Intrazonals
diag(final_mtx) = 0

# write.csv(final_mtx, "./output/full_externals.csv")

#-------------------------------
# Adjust for PM Peak hour
# rXY data is for 3-6pm; target is peak hour average between 4-6pm
pm_factor = 0.4

adjusted_mtx = final_mtx * pm_factor

#write.csv(adjusted_mtx, "./output/full_externals.csv")


#-------------------------------
# Adjust 912-922
index912 = match(912,int_ext_tazs)
index922 = match(922,int_ext_tazs)

adjusted_mtx[index912,index922] = adjusted_mtx[index912,index922] * (1 - 0.25)
adjusted_mtx[index922,index912] = adjusted_mtx[index922,index912] * (1 - 0.25)
 
#-------------------------------
# Fratar - fit to external counts

# Vectors
iter = 1
do_iter = 1000
row_factor = rep(1,length(int_ext_tazs))
col_factor = rep(1,length(int_ext_tazs))

ext_index = 330:340
int_index = 1:329

row_ei_pct = rowSums(adjusted_mtx[ext_index, int_index]) / rowSums(adjusted_mtx[ext_index,])
col_ie_pct = colSums(adjusted_mtx[int_index, ext_index]) / colSums(adjusted_mtx[,ext_index])
row_ee_pct = rowSums(adjusted_mtx[ext_index, ext_index]) / rowSums(adjusted_mtx[ext_index,])
col_ee_pct = colSums(adjusted_mtx[ext_index, ext_index]) / colSums(adjusted_mtx[,ext_index])
  
#working_mtx = adjusted_mtx

ei_mtx = adjusted_mtx[ext_index,int_index]
ie_mtx = adjusted_mtx[int_index,ext_index]
ee_mtx = adjusted_mtx[ext_index,ext_index]

ext_tazs = int_ext_tazs[ext_index]
int_tazs = int_ext_tazs[int_index]

while(iter < do_iter){
  # E-I / I-E
  
  # Rows
  for(i in seq_along(ext_index)) {
    count = ext_counts %>% filter(Ext.Station == ext_tazs[i]) %>% pull(EI)
    ei_pct = row_ei_pct[[toString(ext_tazs[i])]]
    row_factor[i] = (count * ei_pct) / sum(ei_mtx[i,])
  }
  
  # Adjust rows
  for(i in seq_along(ext_index)) {
    ei_mtx[i,] = ei_mtx[i,] * row_factor[i]
  }
  
  # cols
  for(i in seq_along(ext_index)) {
    count = ext_counts %>% filter(Ext.Station == ext_tazs[i]) %>% pull(IE)
    ie_pct = col_ie_pct[[toString(ext_tazs[i])]]
    col_factor[i] = (count * ie_pct) / sum(ie_mtx[,i])
  }
  
  
  # Adjust cols
  for(i in seq_along(ext_index)) {
    ie_mtx[,i] = ie_mtx[,i] * col_factor[i]
  }
  
  # E-E
  
  # Rows
  for(i in seq_along(ext_index)) {
    count = ext_counts %>% filter(Ext.Station == ext_tazs[i]) %>% pull(EI)
    ee_row_pct = row_ee_pct[[toString(ext_tazs[i])]]
    row_factor[i] = (count * ee_row_pct) / sum(ee_mtx[i,])
  }
  
  # Adjust rows
  for(i in seq_along(ext_index)) {
    ee_mtx[i,] = ee_mtx[i,] * row_factor[i]
  }
  
  # cols
  for(i in seq_along(ext_index)) {
    count = ext_counts %>% filter(Ext.Station == ext_tazs[i]) %>% pull(IE)
    ee_col_pct = col_ee_pct[[toString(ext_tazs[i])]]
    col_factor[i] = (count * ee_col_pct) / sum(ee_mtx[,i])
  }
  
  
  # Adjust cols
  for(i in seq_along(ext_index)) {
    ee_mtx[,i] = ee_mtx[,i] * col_factor[i]
  }
  
  iter = iter + 1
}

working_mtx = adjusted_mtx
working_mtx[ext_index,int_index] = ei_mtx
working_mtx[int_index,ext_index] = ie_mtx
working_mtx[ext_index,ext_index] = ee_mtx

# finalize
write_mtx = working_mtx

# write.csv(write_mtx, paste0("./output/external_trips_eefix.csv"))

#-------------------------------
# Write Matrix to OMX
omxfile = "./output/external_trips_eefix_0106.omx"
createFileOMX(omxfile, Numrows = nrow(write_mtx), Numcols = ncol(write_mtx))
writeMatrixOMX(omxfile, write_mtx, MatrixSaveName = "XXod")
writeLookupOMX(omxfile,int_ext_tazs,"TAZ")


#-------------------------------
# Future Externals (2045)

# Internal Splits:  TAZ Productions and Attractions
PAtable_2045 = read.csv("./input/CalculatedTrips_PA_by_TAZ_Future.csv", stringsAsFactors = FALSE) %>%
  filter(!is.na(TAZ))

future_p_factor = sum(PAtable_2045$P) / sum(PAtable$P)
future_a_factor = sum(PAtable_2045$A) / sum(PAtable$A)

# Global factor
future_factor = 1.3
future_mtx = write_mtx * future_factor

# ext_index = 330:340
# int_index = 1:329
# 
# # EI growth
# future_mtx[ext_index,int_index] = future_mtx[ext_index,int_index] * future_p_factor
# 
# # IE growth
# future_mtx[int_index,ext_index] = future_mtx[int_index,ext_index] * future_a_factor
# 
# # EE growth
# future_ee_factor = (future_p_factor + future_a_factor)/2 # (avg. of P and A factors)
# future_mtx[ext_index,ext_index] = future_mtx[ext_index,ext_index] * future_ee_factor
# 
# Write FutureMatrix to OMX
omxfile = "./output/external_trips_2045_0106_eefix.omx"
createFileOMX(omxfile, Numrows = nrow(future_mtx), Numcols = ncol(future_mtx))
writeMatrixOMX(omxfile, future_mtx, MatrixSaveName = "XXod")
writeLookupOMX(omxfile,int_ext_tazs,"TAZ")

