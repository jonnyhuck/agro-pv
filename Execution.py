"""
Tool to evaluate PV potential by country

@author jonnyhuck
"""
import arcpy
from pandas import read_csv
from arcpy.sa import ExtractByMask
from collections import defaultdict
from numpy.random import permutation
from os.path import join as path_join
from numpy import argsort, isnan, unravel_index, zeros_like, nan, nansum, nanmin, nanmax

# set environment
arcpy.CheckOutExtension("Spatial")


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
            break

        # otherwise, add the value to the total and record the location
        total += val
        selected_indices.append(idx)
    
    # return the indices of the selected cells
    return (selected_indices, total)


def run_tool(countries, pvo_path, npp_path, km2_MW, density, output_raster_path, targets, debug=False):
    """ Evaluate each country in turn, outputting rasters as we go"""
    
    # load rasters
    npp = arcpy.Raster(npp_path)
    pvo = arcpy.Raster(pvo_path)

    # group into MultiPolygons
    target_isos = set(targets['ISO_3'])
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
        
        try:
            geom = multi_geoms[iso3]
        except KeyError:
            arcpy.AddMessage(f'WARNING: No geometry for {iso3}')
            continue

        # Loop through countries
        # with arcpy.da.SearchCursor(countries, ["NAME", "ISO_3DIGIT", "SHAPE@"], 
        #                            where_clause=f"ISO_3DIGIT = '{row['ISO_3']}'") as cursor:

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

        # calculate PVO total
        pvo_total = nansum(pvo_np)
        pvo_min = nanmin(pvo_np)

        # sense check the target
        if target > pvo_total:
            arcpy.AddMessage(f"\nWARNING: The specified limit ({target:,.2f}) is greater than the sum of cell values ({pvo_total:,.2f}).")
            arcpy.AddMessage(f"Nothing to do, skipping {iso3}...")
            break
        elif target < pvo_min:
            arcpy.AddMessage(f"\nWARNING: The specified limit ({target:,.2f}) is smaller than the smallest cell value ({pvo_min:,.2f}).")
            arcpy.AddMessage(f"Nothing to do, skipping {iso3}...")
            break

        arcpy.AddMessage(f"\n Scenario 1: Theoretical Maximum Potential...")
        arcpy.AddMessage(f"  {'PVO Sum:':<32} {pvo_total:,.2f}")
        arcpy.AddMessage(f"  {'NPP Sum:':<32} {nansum(npp_np):,.2f}")

        # write result to raster, load into workspace
        output_raster(path_join(output_raster_path, f"{iso}_scenario1.tif"), 
                    pvo_np, lower_left, cell_width, cell_height, spatial_ref)


        ''' SCENARIO 2 '''

        arcpy.AddMessage(f"\n Scenario 2: Prioritise Energy Production...")

        # flatten and sort array and select top N cells
        selected_indices, total = select_indices(argsort(pvo_flat)[::-1], pvo_flat, target)
        if debug:
            arcpy.AddMessage(pvo_flat[selected_indices])
            arcpy.AddMessage(npp_flat[selected_indices])

        # convert back to 2D indices and construct output surface
        rows, cols = unravel_index(selected_indices, pvo_np.shape)
        output = zeros_like(pvo_np)
        for r, c in zip(rows, cols):
            output[r, c] = pvo_np[r, c]

        # report results
        arcpy.AddMessage(f"  {'Cell Count:':<32} {len(selected_indices)}")
        arcpy.AddMessage(f"  {'Sum of Cell Values:':<32} {output.sum():,.2f}")
        arcpy.AddMessage(f"  {'Difference from target:':<32} {target - total:,.2f} ({(target - total) / target:.4f}%)")
        arcpy.AddMessage(f"  {'Loss of Agricultural Potential:':<32} {nansum(npp_np[rows, cols]):,.2f}")

        # write result to raster, load into workspace
        output_raster(path_join(output_raster_path, f"{iso}_scenario2.tif"), 
                    output, lower_left, cell_width, cell_height, spatial_ref)

        
        ''' SCENARIO 3 '''

        arcpy.AddMessage(f"\n Scenario 3: Prioritise Agricultural Production...")

        # flatten and sort array and select top N cells
        selected_indices, total = select_indices(argsort(npp_flat), pvo_flat, target)
        if debug:
            arcpy.AddMessage(pvo_flat[selected_indices])
            arcpy.AddMessage(npp_flat[selected_indices])

        # convert back to 2D indices and construct output surface
        rows, cols = unravel_index(selected_indices, pvo_np.shape)
        output = zeros_like(pvo_np)
        for r, c in zip(rows, cols):
            output[r, c] = pvo_np[r, c]
        
        # report results
        arcpy.AddMessage(f"  {'Cell Count:':<32} {len(selected_indices)}")
        arcpy.AddMessage(f"  {'Sum of Cell Values:':<32} {output.sum():,.2f}")
        arcpy.AddMessage(f"  {'Difference from target:':<32} {target - total:,.2f} ({(target - total) / target:.4f}%)")
        arcpy.AddMessage(f"  {'Loss of Agricultural Potential:':<32} {nansum(npp_np[rows, cols]):,.2f}")

        # write result to raster, load into workspace
        output_raster(path_join(output_raster_path, f"{iso}_scenario3.tif"), 
                    output, lower_left, cell_width, cell_height, spatial_ref)


        ''' SCENARIO 4 '''

        arcpy.AddMessage(f"\n Scenario 4: Balance Energy and Agricultural Production...")

        # flatten and sort array and select top N cells
        selected_indices, total = select_indices(argsort(both_flat)[::-1], pvo_flat, target)
        if debug:
            arcpy.AddMessage(pvo_flat[selected_indices])
            arcpy.AddMessage(npp_flat[selected_indices])

        # convert back to 2D indices and construct output surface
        rows, cols = unravel_index(selected_indices, pvo_np.shape)
        output = zeros_like(pvo_np)
        for r, c in zip(rows, cols):
            output[r, c] = pvo_np[r, c]
        
        # report results
        arcpy.AddMessage(f"  {'Cell Count:':<32} {len(selected_indices)}")
        arcpy.AddMessage(f"  {'Sum of Cell Values:':<32} {output.sum():,.2f}")
        arcpy.AddMessage(f"  {'Difference from target:':<32} {target - total:,.2f} ({(target - total) / target:.4f}%)")
        arcpy.AddMessage(f"  {'Loss of Agricultural Potential:':<32} {nansum(npp_np[rows, cols]):,.2f}")

        # write result to raster, load into workspace
        output_raster(path_join(output_raster_path, f"{iso}_scenario4.tif"), 
                    output, lower_left, cell_width, cell_height, spatial_ref)

        
        ''' SCENARIO 5 '''

        arcpy.AddMessage(f"\n Scenario 5: Randomised Locations...")

        # flatten and sort array and select top N cells
        selected_indices, total = select_indices(permutation(len(pvo_flat)), pvo_flat, target)
        if debug:
            arcpy.AddMessage(pvo_flat[selected_indices])
            arcpy.AddMessage(npp_flat[selected_indices])

        # convert back to 2D indices and construct output surface
        rows, cols = unravel_index(selected_indices, pvo_np.shape)
        output = zeros_like(pvo_np)
        for r, c in zip(rows, cols):
            output[r, c] = pvo_np[r, c]
        
        # report results
        arcpy.AddMessage(f"  {'Cell Count:':<32} {len(selected_indices)}")
        arcpy.AddMessage(f"  {'Sum of Cell Values:':<32} {output.sum():,.2f}")
        arcpy.AddMessage(f"  {'Difference from target:':<32} {target - total:,.2f} ({(target - total) / target:.4f}%)")
        arcpy.AddMessage(f"  {'Loss of Agricultural Potential:':<32} {nansum(npp_np[rows, cols]):,.2f}")

        # write result to raster, load into workspace
        output_raster(path_join(output_raster_path, f"{iso}_scenario5.tif"), 
                    output, lower_left, cell_width, cell_height, spatial_ref)

    return


if __name__ == "__main__":

    # read in parameters
    countries_shp = arcpy.GetParameterAsText(0)
    pvo_raster = arcpy.GetParameterAsText(1)
    npp_raster = arcpy.GetParameterAsText(2)
    km2_MW = float(arcpy.GetParameterAsText(3))
    density = float(arcpy.GetParameterAsText(4))
    output_raster_path = arcpy.GetParameterAsText(5)
    target_file = arcpy.GetParameterAsText(6)

    # read in 
    targets = read_csv(target_file)

    # run the tool
    run_tool(countries_shp, pvo_raster, npp_raster, km2_MW, 
             density, output_raster_path, targets, debug=False)