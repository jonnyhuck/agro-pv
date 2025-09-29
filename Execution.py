"""
Tool to evaluate PV potential by country

@author jonnyhuck
"""
import arcpy
from os import path
from arcpy.sa import ExtractByMask
from collections import defaultdict
from numpy.random import permutation
from os.path import join as path_join
from pandas import read_csv, DataFrame
from numpy import argsort, isnan, unravel_index, zeros_like, nan, nansum, nanmin, nanmax

# set environment
arcpy.CheckOutExtension("Spatial")


def validate_csv_path(filepath):
    """
    Ensure filepath is a valid .csv file in an existing directory.
    - If extension is not .csv, change it to .csv.
    - If directory does not exist, raise FileNotFoundError.
    """
    
    # Ensure it has .csv extension
    root, ext = path.splitext(filepath)
    if ext.lower() != ".csv":
        filepath = root + ".csv"
    
    # Ensure directory exists
    directory = path.dirname(filepath)
    arcpy.AddMessage(directory)
    if not path.isdir(directory):
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    
    return filepath


def output_raster(output_raster_path, output, lower_left, cell_width, cell_height, 
                  spatial_ref, add_to_workspace=False):
    '''
    * Convert numpy array to raster, write to disk, load into environment
    '''
    
    # convert result back to raster, set projection
    output_raster = arcpy.NumPyArrayToRaster(output, lower_left, cell_width, cell_height, value_to_nodata=0)
    arcpy.DefineProjection_management(output_raster, spatial_ref)
    
    # save to disk
    output_raster.save(output_raster_path)

    # add to current ArcGIS Pro Project
    if add_to_workspace:
        aprx = arcpy.mp.ArcGISProject("CURRENT")
        map = aprx.listMaps("Map")[0]
        map.addDataFromPath(output_raster_path)


def select_indices(sorted_indices, values, target):
    ''' Select values until sum is less than the limit '''  
    total = 0
    selected_indices = []
    for idx in sorted_indices:
        val = values[idx]

        # if it's a nan - skip
        if isnan(val):
            continue

        # if we have exceeded the limit, finish
        if total + val > target:
            continue
        else:
            # otherwise, add the value to the total and record the location
            total += val
            selected_indices.append(idx)
    
    # return the indices of the selected cells
    return (selected_indices, total)


def run_tool(countries, pvo_path, npp_path, km2_MW, density, output_raster_path, 
             targets, output_csv_path, verbose=False):
    """ Evaluate each country in turn, outputting rasters as we go"""

    arcpy.AddMessage(f"\nPreparing Datasets...")

    # init output CSV data dictionary
    output_csv_data = defaultdict(list)
    
    # report info on the process
    arcpy.AddMessage(f"\nConversion Factor: {km2_MW}")
    arcpy.AddMessage(f"Density: {density}")

    # load rasters
    npp = arcpy.Raster(npp_path)
    pvo = arcpy.Raster(pvo_path)

    # group into MultiPolygons
    target_isos = set(targets['ISO_3'])
    if verbose:
        arcpy.AddMessage(target_isos)
    geoms_by_iso = defaultdict(list)
    with arcpy.da.SearchCursor(countries, ["ISO_3DIGIT", "SHAPE@"]) as cursor:
        for iso, geom in cursor:
            if (geom) and (iso in target_isos):
                geoms_by_iso[iso].append(geom)

    # Union per ISO into a single multipolygon
    multi_geoms = {}
    for iso, geoms in geoms_by_iso.items():

        # Start with the first geometry and union with the rest
        merged = geoms[0]
        for g in geoms[1:]:
            merged = merged.union(g)
        multi_geoms[iso] = merged

    if len(multi_geoms.items()) == 0:
        arcpy.AddMessage("No valid geometries to process, exiting...")
        exit()

    # loop through countries
    for _, row in targets.iterrows():

        # get data for this country
        iso3 = row['ISO_3']
        target = float(row['Target'])

        # load into CSV
        output_csv_data['ISO3'].append(iso3)
        output_csv_data['Target'].append(target)
        
        try:
            geom = multi_geoms[iso3]
        except KeyError:
            arcpy.AddMessage(f'WARNING: No geometry for {iso3}')
            continue

        # for name, iso, geom in cursor:
        arcpy.AddMessage(f"\nProcessing {iso3} (target: {target:,})...")

        # Extract raster values using country geometry
        pvo_extract = ExtractByMask(pvo, geom)
        npp_extract = ExtractByMask(npp, geom)

        # get raster params
        lower_left = arcpy.Point(pvo_extract.extent.XMin, pvo_extract.extent.YMin)
        cell_width = pvo_extract.meanCellWidth
        cell_height = pvo_extract.meanCellHeight
        spatial_ref = pvo_extract.spatialReference

        # export extracted rasters to numpy arrays
        pvo_np = arcpy.RasterToNumPyArray(pvo_extract, nodata_to_value=nan) 
        npp_np = arcpy.RasterToNumPyArray(npp_extract, nodata_to_value=nan)
        
        # convert units for PVO dataset
        pvo_np = pvo_np * 365 / 100 * (10 / km2_MW) * density

        # flatten arrays for scenarios
        pvo_flat = pvo_np.flatten()
        npp_flat = npp_np.flatten()

        # this is overall quality on a scale of 0-1
        both_flat = ((pvo_flat / nanmax(pvo_flat)) + (1 - npp_flat / nanmax(npp_flat))) / 2


        ''' SCENARIO 1 '''

        arcpy.AddMessage(f"\n Scenario 1: Total Available Resource...")

        # calculate PVO total
        pvo_total = nansum(pvo_np)
        pvo_min = nanmin(pvo_np)
        output_csv_data
        if verbose:
            arcpy.AddMessage(f"PVO Total = {pvo_total}")
            arcpy.AddMessage(f"PVO Min = {pvo_min}")

        # sense check the target
        if target > pvo_total:
            arcpy.AddMessage(f"\nWARNING: The specified limit ({target:,.2f}) is greater than the sum of cell values ({pvo_total:,.2f}).")
            arcpy.AddMessage(f"Nothing to do, skipping {iso3}...")
            for n in range(1, 6):
                output_csv_data[f'S{n}_PVO'].append(nan)
                output_csv_data[f'S{n}_NPP_Loss'].append(nan)
            continue
        elif target < pvo_min:
            arcpy.AddMessage(f"\nWARNING: The specified limit ({target:,.2f}) is smaller than the smallest cell value ({pvo_min:,.2f}).")
            arcpy.AddMessage(f"Nothing to do, skipping {iso3}...")
            for n in range(1, 6):
                output_csv_data[f'S{n}_PVO'].append(nan)
                output_csv_data[f'S{n}_NPP_Loss'].append(nan)
            continue

        # update outputs 
        npp_loss = nansum(npp_np)
        output_csv_data['S1_PVO'].append(pvo_total)
        output_csv_data['S1_NPP_Loss'].append(npp_loss * 0.1)

        # report results
        if verbose:
            arcpy.AddMessage(f"\n Scenario 1: Theoretical Maximum Potential...")
            arcpy.AddMessage(f"  {'Cell Count:':<32} {npp_flat[~isnan(npp_flat)].size:,.2f}")
            arcpy.AddMessage(f"  {'PVO Sum:':<32} {pvo_total:,.2f}")
            arcpy.AddMessage(f"  {'NPP Sum:':<32} {npp_loss:,.2f}")

        # write result to raster, load into workspace
        output_raster(path_join(output_raster_path, f"{iso3}_scenario1.tif"), 
                    pvo_np, lower_left, cell_width, cell_height, spatial_ref)


        ''' SCENARIO 2 '''

        arcpy.AddMessage(f"\n Scenario 2: Prioritise Energy Production...")

        # flatten and sort array and select top N cells
        selected_indices, total = select_indices(argsort(pvo_flat)[::-1], pvo_flat, target)

        # convert back to 2D indices and construct output surface
        rows, cols = unravel_index(selected_indices, pvo_np.shape)
        output = zeros_like(pvo_np)
        for r, c in zip(rows, cols):
            output[r, c] = pvo_np[r, c]

        # update outputs
        npp_loss = nansum(npp_np[rows, cols])
        output_csv_data['S2_PVO'].append(total) # output.sum()
        output_csv_data['S2_NPP_Loss'].append(npp_loss * 0.1)

        # report results
        if verbose:
            arcpy.AddMessage(f"  {'Cell Count:':<32} {len(selected_indices)}")
            arcpy.AddMessage(f"  {'Sum of Cell Values:':<32} {total:,.2f}")
            arcpy.AddMessage(f"  {'Difference from target:':<32} {target - total:,.2f} ({(target - total) / target:.4f}%)")
            arcpy.AddMessage(f"  {'Loss of Agricultural Potential:':<32} {npp_loss:,.2f}")

        # write result to raster, load into workspace
        output_raster(path_join(output_raster_path, f"{iso3}_scenario2.tif"), 
                    output, lower_left, cell_width, cell_height, spatial_ref)

        
        ''' SCENARIO 3 '''

        arcpy.AddMessage(f"\n Scenario 3: Prioritise Agricultural Production...")

        # flatten and sort array and select top N cells
        selected_indices, total = select_indices(argsort(npp_flat), pvo_flat, target)

        # convert back to 2D indices and construct output surface
        rows, cols = unravel_index(selected_indices, pvo_np.shape)
        output = zeros_like(pvo_np)
        for r, c in zip(rows, cols):
            output[r, c] = pvo_np[r, c]

        # update outputs
        npp_loss = nansum(npp_np[rows, cols])
        output_csv_data['S3_PVO'].append(total) # output.sum()
        output_csv_data['S3_NPP_Loss'].append(npp_loss * 0.1)
        
        # report results
        if verbose:
            arcpy.AddMessage(f"  {'Cell Count:':<32} {len(selected_indices)}")
            arcpy.AddMessage(f"  {'Sum of Cell Values:':<32} {total:,.2f}")
            arcpy.AddMessage(f"  {'Difference from target:':<32} {target - total:,.2f} ({(target - total) / target:.4f}%)")
            arcpy.AddMessage(f"  {'Loss of Agricultural Potential:':<32} {npp_loss:,.2f}")

        # write result to raster, load into workspace
        output_raster(path_join(output_raster_path, f"{iso3}_scenario3.tif"), 
                    output, lower_left, cell_width, cell_height, spatial_ref)


        ''' SCENARIO 4 '''

        arcpy.AddMessage(f"\n Scenario 4: Balance Energy and Agricultural Production...")

        # flatten and sort array and select top N cells
        selected_indices, total = select_indices(argsort(both_flat)[::-1], pvo_flat, target)

        # convert back to 2D indices and construct output surface
        rows, cols = unravel_index(selected_indices, pvo_np.shape)
        output = zeros_like(pvo_np)
        for r, c in zip(rows, cols):
            output[r, c] = pvo_np[r, c]
        
        # update outputs
        npp_loss = nansum(npp_np[rows, cols])
        output_csv_data['S4_PVO'].append(total) # output.sum()
        output_csv_data['S4_NPP_Loss'].append(npp_loss * 0.1)

        # report results
        if verbose:
            arcpy.AddMessage(f"  {'Cell Count:':<32} {len(selected_indices)}")
            arcpy.AddMessage(f"  {'Sum of Cell Values:':<32} {output.sum():,.2f}")
            arcpy.AddMessage(f"  {'Difference from target:':<32} {target - total:,.2f} ({(target - total) / target:.4f}%)")
            arcpy.AddMessage(f"  {'Loss of Agricultural Potential:':<32} {npp_loss:,.2f}")

        # write result to raster, load into workspace
        output_raster(path_join(output_raster_path, f"{iso3}_scenario4.tif"), 
                    output, lower_left, cell_width, cell_height, spatial_ref)

        
        ''' SCENARIO 5 '''

        arcpy.AddMessage(f"\n Scenario 5: Randomised Locations...")

        # flatten and sort array and select top N cells
        selected_indices, total = select_indices(permutation(len(pvo_flat)), pvo_flat, target)

        # convert back to 2D indices and construct output surface
        rows, cols = unravel_index(selected_indices, pvo_np.shape)
        output = zeros_like(pvo_np)
        for r, c in zip(rows, cols):
            output[r, c] = pvo_np[r, c]
        
        # update outputs
        npp_loss = nansum(npp_np[rows, cols])
        output_csv_data['S5_PVO'].append(total) # output.sum()
        output_csv_data['S5_NPP_Loss'].append(npp_loss * 0.1)

        # report results
        if verbose:
            arcpy.AddMessage(f"  {'Cell Count:':<32} {len(selected_indices)}")
            arcpy.AddMessage(f"  {'Sum of Cell Values:':<32} {output.sum():,.2f}")
            arcpy.AddMessage(f"  {'Difference from target:':<32} {target - total:,.2f} ({(target - total) / target:.4f}%)")
            arcpy.AddMessage(f"  {'Loss of Agricultural Potential:':<32} {npp_loss:,.2f}")

        # write result to raster, load into workspace
        output_raster(path_join(output_raster_path, f"{iso3}_scenario5.tif"), 
                    output, lower_left, cell_width, cell_height, spatial_ref)

    # output CSV File
    DataFrame(output_csv_data).to_csv(output_csv_path)

    return


if __name__ == "__main__":

    # read in parameters
    countries_shp = arcpy.GetParameterAsText(0)
    pvo_raster = arcpy.GetParameterAsText(1)
    npp_raster = arcpy.GetParameterAsText(2)
    km2_MW = float(arcpy.GetParameterAsText(3))
    density = float(arcpy.GetParameterAsText(4))
    output_raster_path = arcpy.GetParameterAsText(5)
    output_csv = arcpy.GetParameterAsText(6)
    target_file = arcpy.GetParameterAsText(7)

    # validate raster directory
    if not path.isdir(output_raster_path):
        arcpy.AddError(f"Directory does not exist: {output_raster_path}")
        exit()

    # validate output file path
    try:
        output_csv = validate_csv_path(output_csv)
    except FileNotFoundError as e:
        arcpy.AddError ("Error:", e)

    # read in targets file
    targets = read_csv(target_file)

    # run the tool
    run_tool(countries_shp, pvo_raster, npp_raster, km2_MW, 
             density, output_raster_path, targets, output_csv, verbose=False)