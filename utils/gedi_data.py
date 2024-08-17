import geojson
from shapely.geometry import shape, box
import h5py
from shapely.geometry import Point
from datetime import datetime
import os
import h5py
import numpy as np
import pandas as pd
import geopandas as gp
from shapely.geometry import Point
import geoviews as gv
from geoviews import opts, tile_sources as gvts
import holoviews as hv
import matplotlib.pyplot as plt
gv.extension('bokeh', 'matplotlib')
import shapely
import warnings
from shapely.errors import ShapelyDeprecationWarning
import math
from utils.config import out_path_quality_shots2
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
# !pip install seaborn
import requests as r
import seaborn as sns
import geopandas as gpd

def gedi_finder_granules_links(product, bbox):
    cmr = "https://cmr.earthdata.nasa.gov/search/granules.json?pretty=true&provider=LPDAAC_ECS&page_size=2000&concept_id="

    # Set up dictionary where key is GEDI shortname + version
    concept_ids = {'GEDI01_B.002': 'C1908344278-LPDAAC_ECS',
                   'GEDI02_A.002': 'C1908348134-LPDAAC_ECS',
                   'GEDI02_B.002': 'C1908350066-LPDAAC_ECS'}

    # CMR uses pagination for queries with more features returned than the page size
    page = 1
    bbox = bbox.replace(' ', '')  # remove any white spaces
    try:
        # Send GET request to CMR granule search endpoint w/ product concept ID, bbox & page number, format return as json
        cmr_response = r.get(f"{cmr}{concept_ids[product]}&bounding_box={bbox}&pageNum={page}").json()['feed']['entry']
        # If 2000 features are returned, move to the next page and submit another request, and append to the response
        while len(cmr_response) % 2000 == 0:
            page += 1
            cmr_response += r.get(f"{cmr}{concept_ids[product]}&bounding_box={bbox}&pageNum={page}").json()['feed']['entry']
        # CMR returns more info than just the Data Pool links, below use list comprehension to return a list of DP links
        return [c['links'][0]['href'] for c in cmr_response]
    except:
        # If the request did not complete successfully, print out the response from CMR
        print(r.get(f"{cmr}{concept_ids[product]}&bounding_box={bbox.replace(' ', '')}&pageNum={page}").json())

def save_gedi_shots(transecDF):
    savefile = transecDF.copy()
    savefile = savefile.reset_index(drop=True)
    outName = out_path_quality_shots2
    savefile.to_file(outName, driver='GeoJSON')  # Export to GeoJSON

def load_gedi_shots(path):
    count = 0
    gediL2A_list = []

    # Loop through each file
    for file_name in os.listdir(path):
        if file_name.endswith('.h5') and file_name.startswith('processed_GEDI02'):
            count+=1
            file_path = os.path.join(path, file_name)
            if count == 1:
                gediL2A1 = h5py.File(file_path, 'r')  # Read file using h5py
                gediL2A_list.append(gediL2A1)
            elif count == 2:
                gediL2A2 = h5py.File(file_path, 'r')  # Read file using h5py
                gediL2A_list.append(gediL2A2)
            elif count == 3:
                gediL2A3 = h5py.File(file_path, 'r')  # Read file using h5py
                gediL2A_list.append(gediL2A3)
            elif count == 4:
                gediL2A4 = h5py.File(file_path, 'r')  # Read file using h5py
                gediL2A_list.append(gediL2A4)
            else:
                gediL2A5 = h5py.File(file_path, 'r')  # Read file using h5py
                gediL2A_list.append(gediL2A5)

    beamNames = ['BEAM0000', 'BEAM0001', 'BEAM0010', 'BEAM0011', 'BEAM0101', 'BEAM0110', 'BEAM1000', 'BEAM1011']

    # Initialize empty lists to store datasets and filtered DataFrames
    gediL2A_objs = []
    all_dataframes = []
    gediSDS = []


    # Loop through each file
    for file_name in os.listdir(path):
        if file_name.endswith('.h5'):
            file_path = os.path.join(path, file_name)
            # Open the HDF5 file
            with h5py.File(file_path, 'r') as gediL2A:
                print(gediL2A)
                # Initialize an empty DataFrame to store the filtered data
                filtered_dataframes = []

                # Loop through each beam
                for beam_name in beamNames:
                    try:
                        # Extract data from the HDF5 file
                        lats = gediL2A[f'{beam_name}/lat_lowestmode'][()]
                        lons = gediL2A[f'{beam_name}/lon_lowestmode'][()]
                        shots = gediL2A[f'{beam_name}/shot_number'][()]
                        quality = gediL2A[f'{beam_name}/quality_flag'][()]

                        # Initialize lists to store filtered data
                        lonSample, latSample, shotSample, qualitySample, beamSample = [], [], [], [], []

                        # Take every shot with good quality and append to lists
                        for i in range(len(shots)):
                            if quality[i] == 1:
                                shotSample.append(str(shots[i]))
                                lonSample.append(lons[i])
                                latSample.append(lats[i])
                                qualitySample.append(quality[i])
                                beamSample.append(beam_name)

                        # Create a DataFrame for the current beam
                        beam_dataframe = pd.DataFrame({
                            'Beam': beamSample,
                            'Shot Number': shotSample,
                            'Longitude': lonSample,
                            'Latitude': latSample,
                            'Quality Flag': qualitySample,
                        })

                        # Append the filtered DataFrame to the list
                        filtered_dataframes.append(beam_dataframe)
                    except KeyError:
                        print(f"Beam {beam_name} does not have all required attributes. Skipping...")
                    # Concatenate all DataFrames for beams in the current file into a single DataFrame
                if filtered_dataframes:
                    file_dataframe = pd.concat(filtered_dataframes, ignore_index=True)

                    # Append the concatenated DataFrame to the list of all DataFrames
                    all_dataframes.append(file_dataframe)

                # Optionally, collect relevant datasets from gediL2A
                gediL2A.visit(gediL2A_objs.append)
                # Loop through each object reference in gediL2A_objs
                for obj_ref in gediL2A_objs:
                    try:
                        # Check if the current object is an instance of h5py.Dataset
                        if isinstance(gediL2A[obj_ref], h5py.Dataset):
                            # If it is, append the object reference (obj_ref) to gediSDS
                            gediSDS.append(obj_ref)
                    except KeyError:
                        print(f"Object reference {obj_ref} does not exist or is invalid. Skipping...")

    # Initialize empty lists for each variable
    dem_list, srtm_list, zElevation_list, zHigh_list, zLat_list, zLon_list, rh_list, quality_list, degrade_list, sensitivity_list, shotNums_list = [], [], [], [], [], [], [], [], [], [], []

    # Iterate over gediL2A objects and append the values to the lists
    for gediL2A in gediL2A_list:
        for beamName in beamNames:
            try:
                matching_datasets = [g for g in gediSDS if g.endswith(f'{beamName}/digital_elevation_model')]
                if matching_datasets:
                    dem_list.append(gediL2A[matching_datasets[0]][()])

                matching_datasets2 = [g for g in gediSDS if g.endswith(f'{beamName}/digital_elevation_model_srtm')]
                if matching_datasets2:
                    srtm_list.append(gediL2A[matching_datasets2[0]][()])

                matching_datasets3 = [g for g in gediSDS if g.endswith(f'{beamName}/elev_lowestmode')]
                if matching_datasets3:
                    zElevation_list.append(gediL2A[matching_datasets3[0]][()])

                matching_datasets4 = [g for g in gediSDS if g.endswith(f'{beamName}/elev_highestreturn')]
                if matching_datasets4:
                    zHigh_list.append(gediL2A[matching_datasets4[0]][()])

                matching_datasets5 = [g for g in gediSDS if g.endswith(f'{beamName}/lat_lowestmode')]
                if matching_datasets5:
                    zLat_list.append(gediL2A[matching_datasets5[0]][()])

                matching_datasets6 = [g for g in gediSDS if g.endswith(f'{beamName}/lon_lowestmode')]
                if matching_datasets6:
                    zLon_list.append(gediL2A[matching_datasets6[0]][()])

                matching_datasets7 = [g for g in gediSDS if g.endswith(f'{beamName}/rh')]
                if matching_datasets7:
                    rh_list.append(gediL2A[matching_datasets7[0]][()])

                matching_datasets8 = [g for g in gediSDS if g.endswith(f'{beamName}/quality_flag')]
                if matching_datasets8:
                    quality_list.append(gediL2A[matching_datasets8[0]][()])

                matching_datasets9 = [g for g in gediSDS if g.endswith(f'{beamName}/degrade_flag')]
                if matching_datasets9:
                    degrade_list.append(gediL2A[matching_datasets9[0]][()])

                matching_datasets10 = [g for g in gediSDS if g.endswith(f'{beamName}/sensitivity')]
                if matching_datasets10:
                    sensitivity_list.append(gediL2A[matching_datasets10[0]][()])

                matching_datasets11 = [g for g in gediSDS if g.endswith(f'{beamName}/shot_number')]
                if matching_datasets11:
                    shotNums_list.append(gediL2A[matching_datasets11[0]][()])

            except KeyError:
                print(f"Dataset for beam {beamName} not found. Skipping...")

    # Concatenate the arrays along the specified axis
    dem = np.concatenate(dem_list, axis=-1)
    srtm = np.concatenate(srtm_list, axis=-1)
    zElevation = np.concatenate(zElevation_list, axis=-1)
    zHigh = np.concatenate(zHigh_list, axis=-1)
    zLat = np.concatenate(zLat_list, axis=-1)
    zLon = np.concatenate(zLon_list, axis=-1)
    quality = np.concatenate(quality_list, axis=-1)
    degrade = np.concatenate(degrade_list, axis=-1)
    rh = np.vstack(rh_list)
    sensitivity = np.concatenate(sensitivity_list, axis=-1)
    shotNums = np.concatenate(shotNums_list, axis=-1)

    # Create a shot index
    shotIndex = np.arange(shotNums.size)
    canopyHeight = [r[100] for r in rh]  # Grab RH100 (index 100 for each RH metrics)

    # Take the DEM, GEDI-produced Elevation, and RH Metrics and add to a Pandas dataframe
    transecTotal = pd.DataFrame({'Shot Index': shotIndex, 'Shot Number': shotNums, 'Latitude': zLat, 'Longitude': zLon,
                               'Tandem-X DEM': dem, 'SRTM DEM': srtm, 'Elevation (m)': zElevation, 'Canopy Elevation (m)': zHigh,
                               'Canopy Height (rh100)': canopyHeight, 'Quality Flag': quality, 'Degrade Flag': degrade,
                               'Sensitivity': sensitivity})

    # Quality Filtering:
    # Below, remove any shots where the quality_flag is set to 0 by defining those shots as nan.
    transectDF = transecTotal.where(transecTotal['Quality Flag'].ne(0))  # Set any poor quality returns to NaN
    transectDF = transectDF.dropna()  # Drop all the rows (shots) that did not pass the quality filtering above
    print(f"Quality filtering complete, {len(transectDF)} high quality shots remaining.")

    # Create a copy of the DataFrame to avoid SettingWithCopyWarning
    transectDF1 = transectDF.copy()
    transectDF1 = transectDF1.reset_index(drop=True)
    # Take the lat/lon dataframe and convert each lat/lon to a shapely
    transectDF1['geometry'] = transectDF1.apply(lambda row:
    Point(row['Longitude'], row['Latitude']), axis=1)
    # Convert to a Geodataframe
    transectDF1 = gp.GeoDataFrame(transectDF1)
    save_gedi_shots(transectDF1)

    return transectDF
