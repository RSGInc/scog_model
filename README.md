# Skagit Council of Governments (SCOG) Model

This repository contains files necessary to run the SCOG travel demand model, including household travel survey processing in R, trip generation in Excel, and Visum version files.

## Folder Structure

```
 |-- 1-Survey             HTS processing in R
 |-- 2-TripGeneration     Trip generation workbooks for 2018 and 2045
 |-- 3-Externals          External trips matrix in OMX format (2018 and 2045)
 |-- 4-Visum              Visum version files
```

## Running the model

_Note: Steps 1-3 have already been performed on the included version files._

1. Update the Trip Generation workbook, if needed
	- See instructions in `1-Survey` folder to update the trip rates using a new household travel survey
	- SCOG updates Land Use (HH) and Land Use (Emp) or SCOG Emp Input tabs
	
2. Copy Trip Generation results into Visum
	- Open the version file in Visum
	- Right-click on Zones, then click List
	- On the menu bar, click List (Zones), then Open Layout. Open the `ModelFiles` folder and select the `LanduseInput.lla` file.
	- Open the Trip Generation workbook and copy the contents of the Skagit Co Export tab 
	- At the top of the table, click the clipboard icon to paste contents into the Zones data
	
3. Add Externals matrix
	- Open the Matrices tab on the left side of the screen
	- Delete matrix number 4 XXod
	- Open the procedure sequence
	- On line 15, double click on the Reference Objects cell for OMX Import. Select the desired OMX file to import (included in the `3-Externals` folder).
	- On line 14, activate the Group Load External Trips
	- Right-click on line 14 and select Define as procedure to be run next
	- In the toolbar, use the Single-step button to run only steps 15 and 16. 15 imports the OMX file as matrix number 4. 16 copies this matrix to number 16, which is used in assignment.
	- Deactivate the Group Load External Trips
	- Right-click on line 1 and select Define as procedure to be run next

4. Run Procedure Sequence
	- Verify that the Variant/File column has correct paths to filter files, such as `TSysCar.fil`, which are located in `4-Visum/ModelFiles`.
	- Visum will perform Trip Distribution and Assignment
