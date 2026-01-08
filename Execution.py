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
from numpy import argsort, isnan, unravel_index, zeros_like, nan, nansum, nanmin, nanmax, count_nonzero, arange

# set environment
arcpy.CheckOutExtension("Spatial")


def validate_csv_path(filepath):
    """ Ensure filepath is a valid .csv file in an existing directory. """

    # ensure file has .csv extension
    root, ext = path.splitext(filepath)
    if ext.lower() != ".csv":
        filepath = root + ".csv"
    
    # ensure directory exists
    directory = path.dirname(filepath)
    if not path.isdir(directory):
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    
    # return corrected / validated path
    return filepath


def output_raster(output_raster_path, output, lower_left, cell_width, cell_height, 
                  spatial_ref, add_to_workspace=False):
    """ Convert numpy array to raster, write to disk, load into environment """

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

        # if we have exceeded the limit, skip
        if total + val > target:
            continue
        
        # otherwise, add the value to the total and record the location
        total += val
        selected_indices.append(idx)
    
    # return the indices of the selected cells
    return (selected_indices, total)


def run_tool(countries, field_name, pvo_path, npp_path, km2_MW, density, grid_proportion, 
             proportion, increment, output_rasters, output_path, targets, verbose=False):
    """ Evaluate each country in turn, outputting rasters as we go"""

    arcpy.AddMessage(f"\nPreparing Datasets...")

    # report info on the process
    arcpy.AddMessage(f"\nConversion Factor: {km2_MW}")
    arcpy.AddMessage(f"Density: {density}")
    arcpy.AddMessage(f"Grid Cell Proportion: {grid_proportion}")
    arcpy.AddMessage(f"Proportion: {proportion}" if not increment else f"Increment: {proportion}")

    # load rasters
    npp = arcpy.Raster(npp_path)
    pvo = arcpy.Raster(pvo_path)

    # get raster properties
    spatial_ref = pvo.spatialReference
    cell_width = pvo.meanCellWidth
    cell_height = pvo.meanCellHeight
    cell_area = cell_width * cell_height

    # group coutry data into MultiPolygons
    arcpy.AddMessage("Preparing vector data...")
    target_isos = set(targets['ISO_3'])
    if verbose:
        arcpy.AddMessage(target_isos)
    geoms_by_iso = defaultdict(list)
    with arcpy.da.SearchCursor(countries, [field_name, "SHAPE@"]) as cursor:
        for iso, geom in cursor:
            if (geom) and (iso in target_isos):
                geoms_by_iso[iso].append(geom)

    # union per ISO into a single multipolygon
    multi_geoms = {}
    for iso, geoms in geoms_by_iso.items():

        # start with the first geometry and union with the rest
        merged = geoms[0]
        for g in geoms[1:]:
            merged = merged.union(g)
        multi_geoms[iso] = merged

    if len(multi_geoms.items()) == 0:
        arcpy.AddMessage("No valid geometries to process, exiting...")
        exit()

    # preload raster data into dictionary
    arcpy.AddMessage("Preparing raster data...")
    raster_extracts = {}
    for _, row in targets.iterrows():

            # get data for this country
            iso3 = row['ISO_3']

            # get geometry (validate before loading anything into output)
            try:
                geom = multi_geoms[iso3]
            except KeyError:
                arcpy.AddMessage(f'\nWARNING: No geometry for {iso3}')
                continue

            # Extract raster values using country geometry
            pvo_extract = ExtractByMask(pvo, geom)
            npp_extract = ExtractByMask(npp, geom)

            # get raster params
            lower_left = arcpy.Point(pvo_extract.extent.XMin, pvo_extract.extent.YMin)

            # export extracted rasters to numpy arrays
            pvo_np = arcpy.RasterToNumPyArray(pvo_extract, nodata_to_value=nan) 
            npp_np = arcpy.RasterToNumPyArray(npp_extract, nodata_to_value=nan)

            # convert units for PVO dataset
            pvo_np = pvo_np * 365 / (100 * grid_proportion) * (10 / km2_MW) * density

            # store data for this country
            raster_extracts[iso3] = (pvo_np, npp_np, lower_left, cell_width, cell_height)


    ''' run scenarios '''


    # list for countries for which the target has exceeded the capacity 
    exceeded_countries = []

    # loop through increments if required
    arcpy.AddMessage(f"Running scenarios...")   # target to allow 1
    for prop in arange(proportion, 1.00001, proportion) if increment else [proportion]:

        # init output CSV data dictionary
        output_csv_data = defaultdict(list)

        # loop through countries
        for _, row in targets.iterrows():

            # get data for this country
            iso3 = row['ISO_3']

            # get target for this country and load into CSV
            target = float(row['Target'])
            
            # if we are in increment mode, then we need to update the target
            if increment:
                full_target = target
                target *= prop
            
            output_csv_data['ISO3'].append(iso3)
            output_csv_data['Target'].append(target)

            # if this country has already discounted itself, skip
            if len(exceeded_countries) > 0:
                if iso3 in [c['ISO_3'] for c in exceeded_countries]:
                    arcpy.AddMessage(f"Nothing to do, skipping {iso3} Scenarios 2-5...")
                    for n in range(1, 6):
                        output_csv_data[f'S{n}_PVO'].append(nan)
                        output_csv_data[f'S{n}_Area_Used'].append(nan)
                    continue

            # and only continue if valid
            if target <= 0:
                arcpy.AddMessage(f"\nSkipping {iso3} (invalid target: {target:})...")
                for n in range(1, 6):
                    output_csv_data[f'S{n}_PVO'].append(nan)
                    output_csv_data[f'S{n}_Area_Used'].append(nan)
                continue

            # for name, iso, geom in cursor:
            arcpy.AddMessage(f"\nProcessing {iso3} (target: {target:,}, proportion: {prop})...")

            # get raster data and properties for this country
            pvo_np, npp_np, lower_left, cell_width, cell_height = raster_extracts[iso3]

            # flatten arrays for scenarios
            pvo_flat = pvo_np.flatten()
            npp_flat = npp_np.flatten()

            # this is overall quality on a scale of 0-1
            both_flat = ((pvo_flat / nanmax(pvo_flat)) + (1 - npp_flat / nanmax(npp_flat))) / 2

            ''' SCENARIO 1 '''

            # calculate PVO total
            pvo_total = nansum(pvo_np)
            pvo_min = nanmin(pvo_np)
            area_used = count_nonzero(~isnan(pvo_np)) * cell_area * grid_proportion
            if verbose:
                arcpy.AddMessage(f"PVO Total = {pvo_total}")
                arcpy.AddMessage(f"PVO Min = {pvo_min}")

            # update outputs 
            output_csv_data['S1_PVO'].append(pvo_total)
            output_csv_data['S1_Area_Used'].append(area_used)

            # report results
            if verbose:
                arcpy.AddMessage(f"\n Scenario 1: Theoretical Maximum Potential...")
                arcpy.AddMessage(f"  {'Cell Count:':<32} {npp_flat[~isnan(npp_flat)].size:,.2f}")
                arcpy.AddMessage(f"  {'PVO Sum:':<32} {pvo_total:,.2f}")
                arcpy.AddMessage(f"  {'Area Used:':<32} {area_used:,.2f}")

            # write result to raster, load into workspace
            if output_rasters:
                output_raster(path_join(output_path, f"{iso3}_scenario1_{prop}.tif"), 
                            pvo_np, lower_left, cell_width, cell_height, spatial_ref)


            # is target too big?
            if target > pvo_total:
                arcpy.AddMessage(f"\nWARNING: The specified target ({target:,.2f}) is greater than the sum of cell values ({pvo_total:,.2f}).")
                arcpy.AddMessage(f"Nothing to do, skipping {iso3} Scenarios 2-5...")
                
                # record the last increment at which it was not excluded
                if increment:   # just leave it empty if not in increment mode
                    exceeded_countries.append({'ISO_3': iso3,
                                               'succeeded_at': max(prop - proportion, 0.0),
                                               'exceeded_at': prop,
                                               'increment':  proportion,
                                               'target':  target,
                                               'full_target':  full_target,
                                               'PVO total':  pvo_total,
                                               'balance': (pvo_total - full_target) / full_target})
                
                # populate data with nans
                for n in range(2, 6):
                    output_csv_data[f'S{n}_PVO'].append(nan)
                    output_csv_data[f'S{n}_Area_Used'].append(nan)
                continue
            
            # is target too small?
            elif target < pvo_min:
                arcpy.AddMessage(f"\nWARNING: The specified target ({target:,.2f}) is smaller than the smallest cell value ({pvo_min:,.2f}).")
                arcpy.AddMessage(f"Nothing to do, skipping {iso3} Scenarios 2-5...")
                
                # populate data with nans
                for n in range(2, 6):
                    output_csv_data[f'S{n}_PVO'].append(nan)
                    output_csv_data[f'S{n}_Area_Used'].append(nan)
                continue


            ''' SCENARIO 2 '''

            # flatten and sort array and select top N cells
            selected_indices, total = select_indices(argsort(pvo_flat)[::-1], pvo_flat, target)
            area_used = len(selected_indices) * cell_area * grid_proportion
            
            # convert back to 2D indices and construct output surface
            rows, cols = unravel_index(selected_indices, pvo_np.shape)
            output = zeros_like(pvo_np)
            for r, c in zip(rows, cols):
                output[r, c] = pvo_np[r, c]

            # update outputs
            output_csv_data['S2_PVO'].append(total)
            output_csv_data['S2_Area_Used'].append(area_used)

            # report results
            if verbose:
                arcpy.AddMessage(f"\n Scenario 2: Prioritise Energy Production...")
                arcpy.AddMessage(f"  {'Cell Count:':<32} {len(selected_indices)}")
                arcpy.AddMessage(f"  {'Sum of Cell Values:':<32} {total:,.2f}")
                arcpy.AddMessage(f"  {'Difference from target:':<32} {target - total:,.2f} ({(target - total) / target:.4f}%)")
                arcpy.AddMessage(f"  {'Area Used:':<32} {area_used:,.2f}")

            # write result to raster, load into workspace
            if output_rasters:
                output_raster(path_join(output_path, f"{iso3}_scenario2_{prop}.tif"), 
                        output, lower_left, cell_width, cell_height, spatial_ref)

            
            ''' SCENARIO 3 '''

            # flatten and sort array and select top N cells
            selected_indices, total = select_indices(argsort(npp_flat), pvo_flat, target)
            area_used = len(selected_indices) * cell_area * grid_proportion

            # convert back to 2D indices and construct output surface
            rows, cols = unravel_index(selected_indices, pvo_np.shape)
            output = zeros_like(pvo_np)
            for r, c in zip(rows, cols):
                output[r, c] = pvo_np[r, c]

            # update outputs
            output_csv_data['S3_PVO'].append(total)
            output_csv_data['S3_Area_Used'].append(area_used)
            
            # report results
            if verbose:
                arcpy.AddMessage(f"\n Scenario 3: Prioritise Agricultural Production...")
                arcpy.AddMessage(f"  {'Cell Count:':<32} {len(selected_indices)}")
                arcpy.AddMessage(f"  {'Sum of Cell Values:':<32} {total:,.2f}")
                arcpy.AddMessage(f"  {'Difference from target:':<32} {target - total:,.2f} ({(target - total) / target:.4f}%)")
                arcpy.AddMessage(f"  {'Area Used:':<32} {area_used:,.2f}")

            # write result to raster, load into workspace
            if output_rasters:
                output_raster(path_join(output_path, f"{iso3}_scenario3_{prop}.tif"), 
                        output, lower_left, cell_width, cell_height, spatial_ref)


            ''' SCENARIO 4 '''

            # flatten and sort array and select top N cells
            selected_indices, total = select_indices(argsort(both_flat)[::-1], pvo_flat, target)
            area_used = len(selected_indices) * cell_area * grid_proportion

            # convert back to 2D indices and construct output surface
            rows, cols = unravel_index(selected_indices, pvo_np.shape)
            output = zeros_like(pvo_np)
            for r, c in zip(rows, cols):
                output[r, c] = pvo_np[r, c]
            
            # update outputs
            output_csv_data['S4_PVO'].append(total)
            output_csv_data['S4_Area_Used'].append(area_used)

            # report results
            if verbose:
                arcpy.AddMessage(f"\n Scenario 4: Balance Energy and Agricultural Production...")
                arcpy.AddMessage(f"  {'Cell Count:':<32} {len(selected_indices)}")
                arcpy.AddMessage(f"  {'Sum of Cell Values:':<32} {output.sum():,.2f}")
                arcpy.AddMessage(f"  {'Difference from target:':<32} {target - total:,.2f} ({(target - total) / target:.4f}%)")
                arcpy.AddMessage(f"  {'area Used:':<32} {area_used:,.2f}")

            # write result to raster, load into workspace
            if output_rasters:
                output_raster(path_join(output_path, f"{iso3}_scenario4_{prop}.tif"), 
                        output, lower_left, cell_width, cell_height, spatial_ref)

            
            ''' SCENARIO 5 '''

            # flatten and sort array and select top N cells
            selected_indices, total = select_indices(permutation(len(pvo_flat)), pvo_flat, target)
            area_used = len(selected_indices) * cell_area * grid_proportion

            # convert back to 2D indices and construct output surface
            rows, cols = unravel_index(selected_indices, pvo_np.shape)
            output = zeros_like(pvo_np)
            for r, c in zip(rows, cols):
                output[r, c] = pvo_np[r, c]
            
            # update outputs
            output_csv_data['S5_PVO'].append(total)
            output_csv_data['S5_Area_Used'].append(area_used)

            # report results
            if verbose:
                arcpy.AddMessage(f"\n Scenario 5: Randomised Locations...")
                arcpy.AddMessage(f"  {'Cell Count:':<32} {len(selected_indices)}")
                arcpy.AddMessage(f"  {'Sum of Cell Values:':<32} {output.sum():,.2f}")
                arcpy.AddMessage(f"  {'Difference from target:':<32} {target - total:,.2f} ({(target - total) / target:.4f}%)")
                arcpy.AddMessage(f"  {'Area Used:':<32} {area_used:,.2f}")

            # write result to raster, load into workspace
            if output_rasters:
                output_raster(path_join(output_path, f"{iso3}_scenario5_{prop}.tif"), 
                        output, lower_left, cell_width, cell_height, spatial_ref)

        # output the CSV for this proportion
        DataFrame(output_csv_data).to_csv(path_join(output_path, f"{prop}.csv"))
    
    # if in increment mode, also export a _exceeded dataset
    if increment:

        # loop back through all of the countries
        for _, row in targets.iterrows():
            iso3 = row['ISO_3']
            
            # add any missing countries in with "1" to show that they met the target
            if iso3  not in [c['ISO_3'] for c in exceeded_countries]:
                target = row['Target']
                pvo_total = nansum(raster_extracts[iso3][0])
                exceeded_countries.append({ 'ISO_3': iso3,
                                            'succeeded_at': 1.0,
                                            'exceeded_at': nan,
                                            'increment':  proportion,
                                            'target':  target,
                                            'full_target':  target,
                                            'PVO total':  pvo_total,
                                            'balance': (pvo_total - target) / target if target > 0 else nan})
            
        # write the result to CSV file
        DataFrame(exceeded_countries).to_csv(path_join(output_path, f"exceeded_targets.csv"))

    return


if __name__ == "__main__":

    # read in parameters
    countries_shp = arcpy.GetParameterAsText(0)
    field_name = arcpy.GetParameterAsText(1)
    pvo_raster = arcpy.GetParameterAsText(2)
    npp_raster = arcpy.GetParameterAsText(3)
    km2_MW = float(arcpy.GetParameterAsText(4))
    density = float(arcpy.GetParameterAsText(5))
    grid_proportion = float(arcpy.GetParameterAsText(6))
    proportion = float(arcpy.GetParameterAsText(7))
    increment = bool(arcpy.GetParameter(8))
    output_rasters = bool(arcpy.GetParameter(9))
    output_path = arcpy.GetParameterAsText(10)
    target_file = arcpy.GetParameterAsText(11)

    # validate proportion
    if not 0 < proportion < 1:
        arcpy.AddError(f"Proportion must be between 0-1 ({proportion} given)")
        exit()

    # validate raster directory
    if not path.isdir(output_path):
        arcpy.AddError(f"Directory does not exist: {output_path}")
        exit()

    # read in targets file
    targets = read_csv(target_file)

    # run the tool
    run_tool(countries_shp, field_name, pvo_raster, npp_raster, km2_MW, density, grid_proportion, 
             proportion, increment, output_rasters, output_path, targets, verbose=False)