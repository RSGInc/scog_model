# Household Survey Processing

This folder contains an R script for processing household travel survey data to generate PM peak trip rates for each household category and trip purpose. The script expects the same table and variable names from RSG's Whatcom COG survey (2018).

## Folder Structure

```
 |-- SurveyProcessing.R            R script
 |-- tables                        Copy survey tables here (1_Household.tsv and 6_Linked_Trip.tsv)
      | -- empty
 |- output/                                    
      | -- Survey_tables_to_TripGen.xlsx       Excel file to help format CSV outputs to the tables used in the Trip Generation worksheets
      | -- R Script outputs are also saved here
```

## Requirements

- Install [RStudio Desktop](https://rstudio.com/products/rstudio/) (free)
- Install `dplyr` and `tidyr` packages for R by running `install.packages(c("dplyr","tidyr"))` in the console
- Survey tables (tab-separated text files) are not included in this repo

## Running the R Script

Open the `SurveyProcessing.R` script in RStudio. Once the required packages are installed and survey tables are in place, run the script by clicking the Source button. The script should populate the output folder.

From the output folder, open `Survey_tables_to_TripGen.xlsx` in Excel. Also open `pm_hbw_triprates.csv` in Excel. Copy the contents of the second file into the first file under HBW Output. Repeat for HBO and NHB trip purposes. The result is an Excel table of trip rates that is compatible with those in the 2018 and 2045 Trip Generation workbooks.
