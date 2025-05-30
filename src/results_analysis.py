import os
import numpy as np
import rasterio
import pandas as pd
import geopandas as gpd
import os 

src_dir = os.path.dirname(os.path.abspath(__file__))

def process_disease_prediction():

    # Configuration parameters
    file_path = f'{src_dir}/../results/compare_models_predictions/ensemble/'
    file_names = ['valsa2.tif', 'ring2.tif', 'alternaria2.tif']
    threshould_bins = [[0.184,0.456,0.728,1], [0.211,0.474,0.737,1],[0.186,0.457,0.728,1]]
    columns = ['valsa', 'ring', 'alternaria']
    pixel_area_km2 = 28.470686309 / 1000  # Pixel area in thousand square kilometers

    # Read cluster data
    with rasterio.open(f'{src_dir}/../data/apple_panlting_reagions6_5km.tif') as src:
        clusters = src.read(1)

    clusters[clusters < -1000] = np.nan
    clusters[np.isnan(clusters)] = -69993
    clusters = clusters.astype(np.int32).flatten()

    all_area_results = []
    all_ratio_results = []

    for i, file_name in enumerate(file_names):
        with rasterio.open(os.path.join(file_path, file_name)) as src:
            data = src.read(1)

        data[data < 0] = np.nan
        classes = np.digitize(data, threshould_bins[i], right=True)
        classes = np.where(np.isnan(data), np.nan, classes).flatten()

        data_combined = np.column_stack((clusters, classes))
        data_combined = data_combined[data_combined[:, 0] != -69993]  # Remove invalid rows

        df = pd.DataFrame(data_combined, columns=['cluster', columns[i]])
        counts = df.groupby(['cluster', columns[i]]).size().unstack(fill_value=0)

        # Keep total counts for each class and convert to area
        counts_area = counts.iloc[:, 1:] * pixel_area_km2
        counts_ratio = counts_area.div(counts_area.sum(axis=1), axis=0)

        # Add model column name prefix for merging
        counts_area.columns = [f"{columns[i]}_class{j+1}_area" for j in range(counts_area.shape[1])]
        counts_ratio.columns = [f"{columns[i]}_class{j+1}_ratio" for j in range(counts_ratio.shape[1])]

        all_area_results.append(counts_area)
        all_ratio_results.append(counts_ratio)

    # Merge all results
    final_df = pd.concat(all_area_results + all_ratio_results, axis=1)
    final_df.index.name = 'cluster'

    # Save as Excel
    final_df.to_csv(f"{src_dir}/../results/disease_prediction_summary.csv")


def combined_suitability_area():
    file_path = '../results/compare_models_predictions/ensemble/conbined_high_suitables_2models_2025529.tif'
    # Read tif file
    with rasterio.open(file_path) as src:
        data = src.read(1)
        bounds = src.bounds
    datas = data.flatten()

    # Read tif file
    with rasterio.open('../data/apple_panlting_reagions6_5km.tif') as src:
        data_c = src.read(1)
        data_c[data_c < 0] = np.nan
        data_c[np.isnan(data_c)] = -69993
        data_c = data_c.astype(np.int32)
        data_c = data_c.flatten()
        
    data_s = np.column_stack((datas, data_c))
    # Find rows containing -69993 or less than 0
    rows = np.where((data_s[:, 1] == -69993) | (data_s[:, 0] < 0))[0]
    # Delete rows containing -69993 or less than 0
    data_s = np.delete(data_s, rows, axis=0)

    # Convert numpy array to DataFrame
    df = pd.DataFrame(data_s, columns=['conbined', 'clusters'])
    counts = df.groupby('clusters')['conbined'].value_counts().reset_index(name='counts')

    # Reshape data into format needed for stacked bar chart
    counts_pivot = counts.pivot(index='clusters', columns='conbined', values='counts')
    counts_pivot *= 28.470686309/1000  # Pixel area size in 10^3 square kilometers
    # print(counts_pivot)
    counts_pivot.columns = ['No disease', 'VC', 'ARR', 'VC-ARR', 'AB', 'VC-AB', 'ARR-AB', 'VC-ARR-AB']

    # Only plot areas where disease occurs
    df = counts_pivot.iloc[:, 1:]
    # Calculate sum for each row and sort by sum
    df['sum'] = df.sum(axis=1)
    df = df.sort_values('sum', ascending=False)
    df.index = ["Ⅰ","Ⅱ","Ⅲ","Ⅳ","Ⅴ","Ⅵ"]
    df1 = df.drop('sum', axis=1)
    df1.to_csv('../results/combined_hight_suitability_area.csv', index=False)


def maxent_rf_difference():
    file_names = ['valsa.tif', 'ring.tif', 'alternaria.tif']

    absolute_difference = []
    for file_name in file_names:
        tif_path = os.path.join('../results/compare_models_predictions', file_name)
        output_tif_path = os.path.join('../results/compare_models_predictions/maxent_rf_difference', file_name)

        with rasterio.open(tif_path) as src:
            # Read MaxEnt (band 3) and RF (band 4) data
            maxent = src.read(3).astype(float)
            rf = src.read(4).astype(float)
            transform = src.transform
            crs = src.crs
            profile = src.profile

        # Calculate difference map between the two models
        difference = np.abs(maxent - rf).squeeze()
        absolute_difference.append(difference.mean())

        # # Update profile to single band, float32 type
        # profile.update({
        #     'count': 1,
        #     'dtype': 'float32',
        #     'compress': 'lzw'
        # })

        # # Save difference map as GeoTIFF
        # with rasterio.open(output_tif_path, 'w', **profile) as dst:
        #     dst.write(difference.astype('float32'), 1)
        
    print(f"mean absolute difference: {np.mean(absolute_difference)}")


if __name__ == "__main__":
    # process_disease_prediction()
    # combined_suitability_area()
    maxent_rf_difference()



