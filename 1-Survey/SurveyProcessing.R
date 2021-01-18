# SCOG Survey Processing for Trip Generation
# Michael McCarthy, RSG - michael.mccarthy@rsginc.com

library(dplyr)
library(tidyr)

# Folder Structure
#  /
#   |- SurveyProcessing.R         this script
#   |- tables/                    survey tables (1_Household.tsv ... 7_Location.tsv)
#   |- output/                    output summaries
#

output_dir = "./output/"
household_table = "./tables/1_Household.tsv"
linked_trip_table = "./tables/6_Linked_Trip.tsv"

households = read.delim(household_table, stringsAsFactors = FALSE)
linked_trips = read.delim(linked_trip_table, stringsAsFactors = FALSE)

################
# Categorize Households by Number of Persons and Workers
#   `hh_id`           = unique household ID
#   `hh_size`         = household total size
#   `num_workers`     = number of workers in household
#   `hh_weight`       = survey household weight

# Creates a table with one row per Household, plus its associated cross-classification category by size and number of workers 
households_cat = households %>%
  filter(hh_size > 0) %>%
  transmute(hh_id = hh_id, hhcat_size = ifelse(hh_size >= 4,4,hh_size), hhcat_workers = ifelse(num_workers >=3, 3, num_workers), hhcat = paste0("P",hhcat_size,"_W",hhcat_workers), hh_weight = hh_weight, hh_act_size = hh_size)

# Creates a summary table of weighted households, unweighted households (count), and average HH weight, weighted # of persons, and unweighted # of persons 
hh_by_cat = households_cat %>%
  group_by(hhcat,hhcat_size,hhcat_workers) %>%
  summarise(weighted_hh = sum(hh_weight), unweighted_hh = n(), avg_hh_weight = mean(hh_weight), weighted_persons = sum(hh_weight * hh_act_size), unweighted_persons = sum(hh_act_size)) %>%
  ungroup()

# Save CSV output file
#write.csv(hh_by_cat,paste0(output_dir,"categorized_households.csv"))


#################
# Code Trip Purpose (Linked Trips)
#   `o_purp_cat`  = survey trip purpose from origin (ex: Home = 1)
#   `d_purp_cat`  = survey trip purpose to destination (ex: Work = 2)

linked_trips$trip_purp_type = ifelse(
  linked_trips$o_purp_cat ==  1 & linked_trips$d_purp_cat ==  1, "Loop", #Loop trip
  ifelse(linked_trips$o_purp_cat == 1 | linked_trips$d_purp_cat ==  1, #Home based
         ifelse((linked_trips$o_purp_cat %in% c(2,3) | linked_trips$d_purp_cat %in% c(2,3)) & linked_trips$o_purp_cat != linked_trips$d_purp_cat, "HBW","HBO") # If O or D is work, HBW, else HBO
         ,"NHB")) # else Non-home Based

# Person Trips by Household Category
linked_trip_hhcat = inner_join(linked_trips,households_cat, by=c("hh_id","hh_id"))

# Summary table of weighted person trips for PM peak hour, plus unweigthed trips (number of records)
pm_peak_trips = linked_trip_hhcat %>%
  mutate(depart_hour = as.POSIXlt(depart_time)$hour) %>% # Extract hour from time
  # Trips for complete HHs, on Weekdays, not presumed air trips, not Loop trips, 4-6pm departures
  filter(hh_day_complete == 1 & day_of_week < 6 & trip_path_distance_linked <= 250 & trip_purp_type != "Loop" & depart_hour >= 16 & depart_hour < 18 ) %>%
  group_by(hhcat,trip_purp_type) %>%
  summarize(person_trips = sum(trip_weight)/2, unweighted = n()) # 4-6pm trips / 2 = avg. peak hour trip rate

pm_peak_trips = pm_peak_trips %>%
  left_join(hh_by_cat, by = "hhcat") %>%
  mutate(hh_trip_rate = person_trips / weighted_hh, person_trip_rate = person_trips / weighted_persons) %>%
  ungroup()

pm_peak_table_hbw = pm_peak_trips %>%
  filter(trip_purp_type == "HBW") %>%
  select(hhcat_size,hhcat_workers,hh_trip_rate) %>%
  pivot_wider(names_from = hhcat_workers, names_prefix = "W_", values_from = hh_trip_rate)

pm_peak_table_hbo = pm_peak_trips %>%
  filter(trip_purp_type == "HBO") %>%
  select(hhcat_size,hhcat_workers,hh_trip_rate) %>%
  pivot_wider(names_from = hhcat_workers, names_prefix = "W_", values_from = hh_trip_rate)

pm_peak_table_nhb = pm_peak_trips %>%
  filter(trip_purp_type == "NHB") %>%
  select(hhcat_size,hhcat_workers,hh_trip_rate) %>%
  pivot_wider(names_from = hhcat_workers, names_prefix = "W_", values_from = hh_trip_rate)

# Save CSV output files
write.csv(pm_peak_table_hbw,paste0(output_dir,"pm_hbw_triprates.csv"), row.names = FALSE)
write.csv(pm_peak_table_hbo,paste0(output_dir,"pm_hbo_triprates.csv"), row.names = FALSE)
write.csv(pm_peak_table_nhb,paste0(output_dir,"pm_nhb_triprates.csv"), row.names = FALSE)