import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shapereader
import cartopy.mpl.ticker as cticker
import rasterio
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import geopandas as gpd
import pandas as pd
import os
import string
from utils.plot_map_tools import add_north, add_scalebar
from utils.scalebar import scale_bar
import matplotlib.patches as mpatches
import seaborn as sns
import matplotlib.colors as mcolors

src_dir = os.path.dirname(os.path.abspath(__file__))
# define font
plt.rcParams["font.family"] =  "Times New Roman"
plt.rcParams["font.size"] = 12

threshould_bins = [[0.184,0.456,0.728,1], [0.211,0.474,0.737,1],[0.186,0.457,0.728,1]]


def laod_china_shp():
    provinces_english_name = pd.read_csv(f'{src_dir}/../data/china_shp/provinces_name_english.csv', encoding='utf-8')
    china = gpd.read_file(f'{src_dir}/../data/china_shp/中国省级地图GS（2019）1719号.geojson')
    china = china.merge(provinces_english_name, on='CNAME', how='right')
    nine = gpd.read_file(f'{src_dir}/../data/china_shp/九段线GS（2019）1719号.geojson')
    return china, nine

def laod_disese_points():
    valsa_points = gpd.read_file(f'{src_dir}/../results/three_diseases_shp/valsa_canker.shp')
    ring_points = gpd.read_file(f'{src_dir}/../results/three_diseases_shp/apple_ring_rot.shp')
    alternaria_points =gpd.read_file(f'{src_dir}/../results/three_diseases_shp/alternaria_blotch.shp')
    return [valsa_points, ring_points, alternaria_points]

def load_apple_planting_area():
    apple_planting = gpd.read_file(f'{src_dir}/../data/apple_planting/except_apple_planting5_WGS_1.shp')
    return apple_planting

def crate_proj():
    # Define the projection
    proj = ccrs.LambertConformal(central_longitude=105, standard_parallels=(25, 47))
    return proj

def fig1_research_map():
    plt.rcParams["font.size"] = 8
    # creat colormap
    colors = ["#33A02C", "#B2DF8A", "#FDBF6F", "#1F78B4", "#999999", "#E31A1C", "#E6E6E6", "#A6CEE3"]
    cmap = ListedColormap(colors)

    # read data file
    china, nine = laod_china_shp()
    valsa_points = gpd.read_file(f'{src_dir}/../results/three_diseases_shp/valsa_canker.shp')
    ring_points = gpd.read_file(f'{src_dir}/../results/three_diseases_shp/apple_ring_rot.shp')
    alternaria_points =gpd.read_file(f'{src_dir}/../results/three_diseases_shp/alternaria_blotch.shp')

    proj = crate_proj()
    fig = plt.figure(figsize=[6.89, 6.89])
    ax = plt.axes(projection=proj)                                              
    ax.set_extent([80,130,18,53],crs = ccrs.PlateCarree())
    nine.plot(ax=ax,transform=ccrs.PlateCarree(),facecolor='none',edgecolor='k',lw=0.5,alpha=0.8)
    china.plot(ax=ax,transform=ccrs.PlateCarree(),facecolor='none',edgecolor='k',lw=0.3,zorder=3, alpha=0.8)
     # Add abbreviation names
    for idx, row in china.iterrows():
        # Use representative_point to ensure the point is inside the polygon
        point = row['geometry'].representative_point()
        ax.text(point.x, point.y, row['abb_name'], fontsize=8, transform=ccrs.PlateCarree(), ha='center', va='center', zorder=4)

    #three apple diseases points
    valsa_points.plot(ax=ax,transform=ccrs.PlateCarree(),marker='o',markersize=20,facecolor='red',lw=0.1,zorder=4,alpha=0.8)
    ring_points.plot(ax=ax,transform=ccrs.PlateCarree(),marker='s',markersize=20,facecolor='gold',lw=0.1,zorder=4,alpha=0.8)
    alternaria_points.plot(ax=ax,transform=ccrs.PlateCarree(),marker='^',markersize=20,facecolor='blue',lw=0.1,zorder=4,alpha=0.8)

    # create legends
    valsa_patch = plt.scatter([], [], s=25, color='red',label='Apple valsa canker (n=215)', marker='o')
    ring_patch = plt.scatter([], [], s=25, color='gold',label='Apple ring rot (n=255)', marker='s')
    alternaria_patch = plt.scatter([], [], s=25, color='blue',label='Alternaria blotch on apple (n=151)', marker='^')

    # add legend
    plt.legend(handles=[valsa_patch, ring_patch, alternaria_patch], 
            loc='lower left', ncol=1, fontsize=10, frameon=False)

    ax.add_feature(cfeature.OCEAN.with_scale('50m'))
    ax.add_feature(cfeature.LAND.with_scale('50m'))

    # add dem 
    with rasterio.open(f'{src_dir}/../data/dem/dem_5km.tif') as src:
        data = src.read(1)
        transform = src.transform
        bounds = src.bounds
        
    data[data<-1000] = np.nan
    tif_extent = [bounds.left,bounds.right,bounds.bottom,bounds.top]
    p = ax.imshow(data, origin='upper', transform=ccrs.PlateCarree(),extent=tif_extent ,cmap=cmap,zorder=2,alpha=0.55)

    # Add a colorbar to the figure
    cbar = plt.colorbar(p, ax=ax, orientation='vertical', fraction=0.001, pad=0.01, aspect=20)
    # Set specific ticks
    cbar.set_ticks([0, 3000, 6000])
    # Set tick label font size
    cbar.ax.tick_params(labelsize=8)
    # Set the label for the colorbar (horizontal)
    cbar.set_label('Elevation (m)', size=10, rotation=0, labelpad=5, ha='left')
    cbar.ax.set_position([0.15, 0.31,  1, 0.1])

    # Add scale bar and north arrow
    add_north(ax)
    # add_scalebar(ax, y=19, x=92, length_km=500, lw=3, size=12, lat_range=(20, 40), proj=proj)
    scale_bar(ax, (0.65, 0.05), 5_00)

    #nanhai
    sub_ax = fig.add_axes([0.78, 0.18, 0.1, 0.2],projection=proj) 
    nine.plot(ax=sub_ax,transform=ccrs.PlateCarree(),facecolor='w',edgecolor='k',lw=0.3,zorder=3,alpha=1)
    china.plot(ax=sub_ax,transform=ccrs.PlateCarree(),facecolor='w',edgecolor='k',lw=0.3,zorder=3, alpha=1)
    sub_ax.add_feature(cfeature.OCEAN.with_scale('50m'))
    sub_ax.add_feature(cfeature.LAND.with_scale('50m'))    
    p = sub_ax.imshow(data, origin='upper', transform=ccrs.PlateCarree(),extent=tif_extent ,cmap=cmap,zorder=3,alpha=0.55)
    sub_ax.set_extent([105, 122, 2, 22])

    gl = ax.gridlines(draw_labels=True, linewidth=0, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.x_inline = False
    gl.xlocator = mticker.FixedLocator([90, 100, 110, 120])
    gl.ylocator = mticker.FixedLocator([20,30,40])
    gl.xlabel_style = {'fontsize':10}
    gl.ylabel_style = {'fontsize':10}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.rotate_labels = False
    gl.xpadding = 10
    # save the figure
    fig.savefig(f'{src_dir}/../figs/fig1_research_map.jpg', dpi=300, bbox_inches='tight')


def fig2_different_sdms():

    # Define file paths and filenames
    file_path = f'{src_dir}/../results/compare_models_predictions'
    file_names = ['valsa.tif', 'ring.tif', 'alternaria.tif']

    # read data file
    china, nine = laod_china_shp()
    disease_points = laod_disese_points()

    #model names
    models_names = ['GLM','GAM','SVM','MaxEnt','RF']

    #diseases names 
    diseases_names = ['Valsa canker', 'Apple ring rot', 'Alternaria blotch']

    colors = ['#A9B8C6','#96C37D','#F3D266','#D8383A']
    cmap = ListedColormap(colors)
    proj = crate_proj()

    fig = plt.figure(figsize=(6.89, 10.2))#/2.54
    # Define plotting parameters
    nrows = 5
    ncols = 3

    # Read data and plot images
    for i in range(nrows):
        for j in range(ncols):
            # Read file
            file_name = file_names[j]
            file = os.path.join(file_path, file_name)
            ax = fig.add_subplot(5,3,i*3+j+1, projection=proj)
            ax.set_extent([80,130,18,53],crs = ccrs.PlateCarree())

            ax.text(0.05, 0.95,  f"({string.ascii_lowercase[i*3+j]})", transform=ax.transAxes, 
                    size=10, va='top')
            # Read tif file
            with rasterio.open(file) as src:
                data = src.read(i+1)#Load different bands
                transform = src.transform
                bounds = src.bounds
                
            # Filter data
            if i == 4:
                data[data<-10] = np.nan
            else:    
                data[data<0] = np.nan
                tif_extent = [bounds.left,bounds.right,bounds.bottom,bounds.top]
            
            bin_disease = threshould_bins[j]
            data_classes = np.digitize(data, bin_disease, right=True)
            
            # Set np.nan in classification results where original data is np.nan
            data_classes = np.where(np.isnan(data), np.nan, data_classes)
            
            # Plot data with classified colors
            p = ax.imshow(data_classes, origin='upper', transform=ccrs.PlateCarree(),
                    extent=tif_extent ,cmap=cmap, zorder=3,alpha=1)
            

            nine.plot(ax=ax,transform=ccrs.PlateCarree(),facecolor='none',edgecolor='k',lw=1,alpha=0.5)
            china.plot(ax=ax,transform=ccrs.PlateCarree(),facecolor='none',edgecolor='k',lw=0.4,zorder=3, alpha=0.5)
            # Add abbreviation names
            for idx, row in china.iterrows():
                # Use representative_point to ensure the point is inside the polygon
                point = row['geometry'].representative_point()
                ax.text(point.x, point.y, row['abb_name'], fontsize=4, transform=ccrs.PlateCarree(), ha='center', va='center', zorder=4)
            #three apple diseases points
            disease_points[j].plot(ax=ax,transform=ccrs.PlateCarree(),marker='*',markersize=2,facecolor='black',lw=0.1,zorder=4,alpha=1)

                
            if i == 0:
                ax.set_title(diseases_names[j], size=12)
                
            if j == 0:
                ax_row = ax
                fig.text(-0.1, 0.5, models_names[i], va='center', rotation='vertical', fontsize=12, transform=ax_row.transAxes)
                
            if (i,j) == (0,1):   
                import matplotlib.patches as mpatches

                labels = ['USEC','LSEC', 'MSEC', 'HSEC']

                # Create the legend handles and labels
                handles = [mpatches.Patch(color=color, label=str(i)) for i, color in enumerate(colors)]

                # Add the legend to the plot and set its properties
                plt.legend(handles=handles, labels=labels, loc='upper center', bbox_to_anchor=(0.3,0.9,0.5,0.5),
                        bbox_transform=ax.transAxes, labelspacing=0.3, ncol=5, frameon=False, prop={'size': 12}, 
                        handlelength=2, handleheight=1)
    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=-0.2, hspace=0)
    # save the figure
    fig.savefig(f'{src_dir}/../figs/fig2_compare_models.jpg', dpi=300, bbox_inches='tight')

def fig3_model_performance():
    """
    Plot bar charts of model performance (TSS and AUC) across different categories.

    Parameters:
        csv_path (str): Path to CSV file containing columns ["class", "tss", "auc", "model"]
        save_path (str): If provided, save the figure to this path (recommended to provide full path, e.g. .jpg/.png)
    """
    csv_path=f'{src_dir}/../results/auc_tss.csv'
    static_path = f'{src_dir}/../results/statistical_significance_results.csv'
    save_path=f'{src_dir}/../figs/fig3_model_evaluation.jpg'
    # Set theme and font
    custom_params = {'font.family': ['Times New Roman']}
    sns.set_theme(style="ticks", rc=custom_params)
    sns.set_context("talk")

    colors = ['#8ECFC9', '#FFBE7A', '#FA7F6F', '#82B0D2', '#BEB8DC']
    # Create figure and subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6.89, 3.38))

    # Read data
    df = pd.read_csv(csv_path)
    static_df = pd.read_csv(static_path)
    # Ensure consistent column names
    static_df.columns = static_df.columns.str.lower()

    # ------------ First plot: TSS ------------
    ax1 = axes[0]
    sns.barplot(
        data=df, x="class", y="tss", hue="model", palette=colors,
        alpha=1, ax=ax1, errorbar="sd", errwidth=0.9, capsize=0.09
    )
    ax1.set_xlabel("")
    ax1.set_ylabel("TSS Value", fontname='Times New Roman',fontsize=10)
    ax1.set_xticklabels(['AVC', 'ARR', 'ABA'], rotation=0, fontsize=10)
    # ax1.axhline(y=0.7, linestyle='--', color='gray')

    ax1.yaxis.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax1.yaxis.set_tick_params(which='major', length=2, width=1)
    ax1.yaxis.set_tick_params(which='minor', length=1, width=1, labelsize=6)
    ax1.xaxis.set_tick_params(which='major', length=2, width=1)
    ax1.tick_params(labelsize=10)

    group_stats = (
        df.groupby(["class", "model"])["tss"]
        .agg(["mean", "std"])
        .reset_index())
    class_order = ["Valsa canker", "Apple ring rot", "Alternaria blotch"]
    model_order = ["GLM", "GAM", "SVM", "MaxEnt","RF"]
    # Set as ordered categorical variables
    group_stats["class"] = pd.Categorical(group_stats["class"], categories=class_order, ordered=True)
    group_stats["model"] = pd.Categorical(group_stats["model"], categories=model_order, ordered=True)
    # Sort by specified order
    group_stats = group_stats.sort_values(["model","class"]).reset_index(drop=True)
    
    for i, bar in enumerate(ax1.patches):
        if i >= len(group_stats): break
        row = group_stats.iloc[i]
        x = bar.get_x() + bar.get_width() / 2
        y = row["mean"] + row["std"]

        letter_row = static_df[
            (static_df['class'] == row['class']) & (static_df['model'] == row['model'])
        ]
        if not letter_row.empty:
            letter = letter_row.iloc[0]["tss"]
            ax1.text(x, y + 0.01, letter, ha="center", va="bottom", fontsize=8)

    ax1.legend(
            frameon=False, title=None, bbox_to_anchor=(0.5, 1.03), loc="upper center",
            ncol=3, labelspacing=0.01, columnspacing=0.06, handlelength=2, handleheight=1,
            prop={'size': 8}, fontsize='large'
    )

    # ------------ Second plot: AUC ------------
    ax2 = axes[1]
    sns.barplot(
        data=df, x="class", y="auc", hue="model", palette=colors,
        alpha=1, ax=ax2, errorbar="sd", errwidth=0.9, capsize=0.09
    )
    ax2.set_xlabel("")
    ax2.set_ylabel("AUC Value", fontname='Times New Roman',fontsize=10)
    ax2.set_xticklabels(['AVC', 'ARR', 'ABA'], rotation=0,fontsize=10)
    # ax2.axhline(y=0.8, linestyle='--', color='gray')
    ax2.legend().set_visible(False)

    ax2.yaxis.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax2.yaxis.set_tick_params(which='major', length=2, width=1)
    ax2.yaxis.set_tick_params(which='minor', length=1, width=1, labelsize=8)
    ax2.xaxis.set_tick_params(which='major', length=2, width=1)
    ax2.tick_params(labelsize=10)


    group_stats = (
        df.groupby(["class", "model"])["auc"]
        .agg(["mean", "std"])
        .reset_index())
    class_order = ["Valsa canker", "Apple ring rot", "Alternaria blotch"]
    model_order = ["GLM", "GAM", "SVM", "MaxEnt","RF"]
    # Set as ordered categorical variables
    group_stats["class"] = pd.Categorical(group_stats["class"], categories=class_order, ordered=True)
    group_stats["model"] = pd.Categorical(group_stats["model"], categories=model_order, ordered=True)
    # Sort by specified order
    group_stats = group_stats.sort_values(["model","class"]).reset_index(drop=True)

    for i, bar in enumerate(ax2.patches):
        if i >= len(group_stats):
            break
        row = group_stats.iloc[i]
        x = bar.get_x() + bar.get_width() / 2
        y = row["mean"] + row["std"]  # Above error bar

        letter_row = static_df[
            (static_df['class'] == row['class']) & (static_df['model'] == row['model'])
        ]
        if not letter_row.empty:
            letter = letter_row.iloc[0]["auc"]
            ax2.text(x, y + 0.01, letter, ha="center", va="bottom", fontsize=8)

    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
    plt.subplots_adjust(wspace=0.3) 
    # Save figure (if path specified)
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')


def fig4_ensemble_results():

    # === Configuration section ===
    colors = ['#A9B8C6', '#96C37D', '#F3D266', '#D8383A']
    cmap = ListedColormap(colors)
    file_names = ['valsa2.tif', 'ring2.tif', 'alternaria2.tif']
    titles = ['AVC', 'ARR', 'ABA']
    labels = ['USEC', 'LSEC', 'MSEC', 'HSEC']
    df = pd.DataFrame(index=['Usuitable', 'Low suitable', 'Moderately suitable', 'High suitable'],
                    columns=titles)

    # Paths
    data_dir = f'{src_dir}/../results/compare_models_predictions/ensemble/'
    fig_path = f'{src_dir}/../figs/fig4_ensemble.jpg'
    fig_path_abstract = f'{src_dir}/../figs/abstract_ensemble.jpg'

    # Map data
    china, nine = laod_china_shp() 
    apple_planting = load_apple_planting_area()
    proj = crate_proj()

    # Create figure and layout
    fig = plt.figure(figsize=(6.89, 6.2))
    gs = gridspec.GridSpec(2, 2, figure=fig)

    def process_and_plot_tif(ax, file_path, title, label, bins, column_key, label_sides):
        with rasterio.open(file_path) as src:
            data = src.read(1)
            bounds = src.bounds
        data[data < 0] = np.nan
        data_classes = np.digitize(data, bins, right=True)
        data_classes = np.where(np.isnan(data), np.nan, data_classes)
        _, counts = np.unique(data_classes, return_counts=True)
        df[column_key] = counts[0:4]

        ax.set_extent([80, 130, 18, 53], crs=ccrs.PlateCarree())
        ax.imshow(data_classes, origin='upper', transform=ccrs.PlateCarree(),
                extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
                cmap=cmap, zorder=3, alpha=1)

        nine.plot(ax=ax, transform=ccrs.PlateCarree(), facecolor='none', edgecolor='k', lw=0.4, zorder=1, alpha=0.5)
        china.plot(ax=ax, transform=ccrs.PlateCarree(), facecolor='none', edgecolor='k', lw=0.4, zorder=3, alpha=0.5)
        # apple_planting(ax=ax, transform=ccrs.PlateCarree(), facecolor='none', edgecolor='k', lw=1, zorder=3, alpha=0.5)

        for _, row in china.iterrows():
            point = row['geometry'].representative_point()
            ax.text(point.x, point.y, row['abb_name'], fontsize=6, transform=ccrs.PlateCarree(),
                    ha='center', va='center', zorder=4)

        ax.text(0.05, 0.95, label, transform=ax.transAxes, size=12, va='top')
        ax.set_title(title, loc='center', y=0.9, size=12)

        gl = ax.gridlines(draw_labels=True, linewidth=0, color='gray', alpha=0.5, linestyle='--')
        gl.left_labels = 'left' in label_sides
        gl.right_labels = 'right' in label_sides
        gl.bottom_labels = 'bottom' in label_sides
        gl.top_labels = False
        gl.x_inline = False
        gl.xlocator = mticker.FixedLocator([90, 100, 110, 120])
        gl.ylocator = mticker.FixedLocator([20, 30, 40])
        gl.xlabel_style = {'fontsize': 10}
        gl.ylabel_style = {'fontsize': 10}
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.rotate_labels = False
        gl.xpadding = 5

    # Map plots (1~3)
    for i, (fname, title, label) in enumerate(zip(file_names, titles, ['(a)', '(b)', '(c)'])):
        row, col = divmod(i, 2)
        ax = fig.add_subplot(gs[row, col], projection=proj)
        bins = threshould_bins[i]

        if i == 0:
            label_sides = ['left']
        elif i == 1:
            label_sides = ['right']
        elif i == 2:
            label_sides = ['left', 'bottom']

        process_and_plot_tif(ax, f'{data_dir}{fname}', title, label, bins, title, label_sides)

    # Fourth plot (stacked bar chart)
    ax4 = fig.add_subplot(gs[1, 1])
    df_pl = df.iloc[1:].T * 28.470686309 / 1000
    bar_colors = colors[1:]
    df_pl.plot(ax=ax4, kind='bar', stacked=True, color=bar_colors, legend=False)

    ax4.grid(False)
    ax4.set_ylabel("Area(10$^3$ Km$^2$)", fontsize=10)
    ax4.set_yticks([0, 500, 1000, 1500, 2000, 2500])
    ax4.yaxis.set_ticks_position('right')
    ax4.yaxis.set_label_position('right')
    ax4.tick_params(which='both', length=0.1, labelsize=10)
    ax4.xaxis.set_tick_params(rotation=0)
    ax4.text(0.05, 0.95, "(d)", transform=ax4.transAxes, size=12, va='top')

    # Legend
    handles = [mpatches.Patch(color=color, label=label) for color, label in zip(colors, labels)]
    plt.legend(handles=handles, loc='upper center', bbox_to_anchor=(-0.2, 1.8, 0.5, 0.5),
            labelspacing=1.5, ncol=4, frameon=False, prop={'size': 10},
            handlelength=5, handleheight=2.5)

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')

def fig5_combined_hight_suitability():
    # Read combined area
    df = pd.read_csv('../results/combined_hight_suitability_area.csv')
    proj = crate_proj()
    fig = plt.figure(figsize=(6.89, 6.89))
    grid = plt.GridSpec(1, 1, wspace=0, hspace=0)
    conbined_ax = plt.subplot(grid[0, 0], projection=proj)
    # bar_ax = plt.subplot(grid[0,1])
    conbined_ax.set_extent([77,127,16,53], crs = ccrs.PlateCarree())
    bar_ax = fig.add_axes([0.15,0.175,0.18,0.18])
    colors = ['#f25941','#a4a64d','#00a2d4','#3f6fb4','#34426d','#8b589d','#d5e6af','#cccccc']

    new_palette = colors[::-1]
    cmap = ListedColormap(new_palette)
    china, nine = laod_china_shp()
    apple_planting = load_apple_planting_area()
    # Read tif file
    with rasterio.open('../results/compare_models_predictions/ensemble/conbined_high_suitables_2models_2025529.tif') as src:
        data = src.read(1)
        bounds = src.bounds
        
    tif_extent = [bounds.left,bounds.right,bounds.bottom,bounds.top]
    data = np.where(data < 0, np.nan, data)
    conbined_ax.imshow(data, origin='upper', transform=ccrs.PlateCarree(),extent=tif_extent,cmap=cmap,zorder=3,alpha=1)
    nine.plot(ax=conbined_ax,transform=ccrs.PlateCarree(),facecolor='none',edgecolor='k',lw=1,alpha=0.5,zorder=1)
    china.plot(ax=conbined_ax,transform=ccrs.PlateCarree(),facecolor='none',edgecolor='k',lw=0.4,zorder=3, alpha=0.5)
    apple_planting.plot(ax=conbined_ax,transform=ccrs.PlateCarree(),facecolor='none',edgecolor='k',lw=1,zorder=3, alpha=0.5)

    for idx, row in apple_planting.iterrows():
        geom = row.geometry
        if geom.geom_type == 'Polygon':
            centroid = geom.centroid
            conbined_ax.text(centroid.x, centroid.y, row['pr_name'], fontsize=10, weight='bold',
                            horizontalalignment='center', verticalalignment='center',
                            transform=ccrs.PlateCarree(), color='black', zorder=4)
        elif geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:  
                if poly.area > 30:  
                    centroid = poly.centroid
                    conbined_ax.text(centroid.x, centroid.y, row['pr_name'], fontsize=10, weight='bold',
                                    horizontalalignment='center', verticalalignment='center',
                                    transform=ccrs.PlateCarree(), color='black', zorder=4)

    gl = conbined_ax.gridlines(draw_labels=True,linewidth=0, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels =True
    gl.left_labels = True
    gl.x_inline = False
    gl.xlocator = mticker.FixedLocator([90, 100, 110, 120])
    gl.ylocator = mticker.FixedLocator([20,30,40])
    gl.xlabel_style = {'fontsize':10}
    gl.ylabel_style = {'fontsize':10}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.rotate_labels = False
    gl.xpadding = 12

    #bar
    df1_subset = df.iloc[:3]
    xtick_labels = ["Ⅰ", "Ⅱ", "Ⅲ"]

    df1_subset.plot(
        kind='bar',
        stacked=True,
        ax=bar_ax,
        color=new_palette[1:],  # Skip 'No disease'
        legend=False
    )

    # Set y-axis labels and format
    bar_ax.set_ylabel("Area (10$^3$ Km$^2$)", fontsize=10)
    bar_ax.tick_params(axis='y', labelsize=10)
    bar_ax.yaxis.set_ticks_position('right')
    bar_ax.yaxis.set_label_position('right')
    bar_ax.set_xticks(range(len(xtick_labels)))
    bar_ax.set_xticklabels(xtick_labels, fontsize=10, rotation=0)
    bar_ax.set_xlabel('')

    # Create the legend handles and labels
    handles = [mpatches.Patch(color=color, label=str(i)) for i, color in enumerate(new_palette)]
    labels = ['No HSEC', 'HAVC', 'HARR', 'HAVC-HARR', 'HABA', 'HAVC-HABA', 'HARR-HABA', 'HAVC-HARR-HABA']

    conbined_ax.legend(handles=handles, labels=labels, loc='upper center', bbox_to_anchor=(-0.12,-0.013,1,1),
                        bbox_transform=conbined_ax.transAxes, labelspacing=0.2,columnspacing=0.3,
                        ncol=3, frameon=False, prop={'size': 10}, 
                        handlelength=2, handleheight=1)

    fig.savefig('../figs/fig5_combined_hight_suitability.jpg', dpi=300, bbox_inches='tight')


def fig_s_maxent_rf_difference():

    file_names = ['valsa.tif', 'ring.tif', 'alternaria.tif']

    # Classification boundaries and colors
    bounds = [0, 0.2, 0.5, 1.01]  # Add a small range beyond 1 to avoid boundary errors
    colors = ['#d4f7c5', '#fff79a', '#f99b7d']  # Light green - Yellow - Orange
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Map projection
    proj = crate_proj()
    fig = plt.figure(figsize=(6.89, 6.89))
    grid = plt.GridSpec(1, 1, wspace=0, hspace=0)
    ax = plt.subplot(grid[0, 0], projection=proj)

    ax.set_extent([77, 127, 16, 53], crs=ccrs.PlateCarree())
    china, nine = laod_china_shp()
    apple_planting = load_apple_planting_area()

    for file_name in file_names[1:2]:
        print(file_name)
        tif_path = os.path.join('../results/compare_models_predictions/maxent_rf_difference/', file_name)

        output_path = os.path.join(f'../figs/fig_s_maxent_rf_difference_{file_name.replace(".tif", "")}_binned.jpg')

        with rasterio.open(tif_path) as src:
            data = src.read(1)
            raster_bounds = src.bounds

        tif_extent = [raster_bounds.left, raster_bounds.right, raster_bounds.bottom, raster_bounds.top]
        data = np.ma.masked_where(data == 0, data)

        # Render difference map with classification
        im = ax.imshow(
            data,
            origin='upper',
            transform=ccrs.PlateCarree(),
            extent=tif_extent,
            cmap=cmap,
            norm=norm,
            zorder=3,
            alpha=1
        )

        # Draw vector layers
        nine.plot(ax=ax, transform=ccrs.PlateCarree(), facecolor='none', edgecolor='k', lw=1, alpha=0.5, zorder=1)
        china.plot(ax=ax, transform=ccrs.PlateCarree(), facecolor='none', edgecolor='k', lw=0.4, zorder=3, alpha=0.5)
        apple_planting.plot(ax=ax, transform=ccrs.PlateCarree(), facecolor='none', edgecolor='k', lw=1, zorder=3, alpha=0.5)

        # Add labels
        for idx, row in apple_planting.iterrows():
            geom = row.geometry
            if geom.geom_type == 'Polygon':
                centroid = geom.centroid
                ax.text(centroid.x, centroid.y, row['pr_name'], fontsize=8, weight='bold',
                        horizontalalignment='center', verticalalignment='center',
                        transform=ccrs.PlateCarree(), color='black', zorder=4)
            elif geom.geom_type == 'MultiPolygon':
                for poly in geom.geoms:
                    if poly.area > 30:
                        centroid = poly.centroid
                        ax.text(centroid.x, centroid.y, row['pr_name'], fontsize=8, weight='bold',
                                horizontalalignment='center', verticalalignment='center',
                                transform=ccrs.PlateCarree(), color='black', zorder=4)

            # Create the legend handles and labels
        handles = [mpatches.Patch(color=color, label=str(i)) for i, color in enumerate(colors)]
        labels = ['0–0.2: High agreement','0.2–0.5: Moderate disagreement','>0.5: Low agreement']

        ax.legend(handles=handles, labels=labels, loc='lower left', bbox_to_anchor=(0,0,1,1),
                            bbox_transform=ax.transAxes, labelspacing=0.6,
                            ncol=1, frameon=False, prop={'size': 10}, title='Difference in Suitability (0–1)', 
                            handlelength=3, handleheight=1.5)

        # Add grid lines
        gl = ax.gridlines(draw_labels=True, linewidth=0, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.bottom_labels = True
        gl.left_labels = True
        gl.x_inline = False
        gl.xlocator = mticker.FixedLocator([90, 100, 110, 120])
        gl.ylocator = mticker.FixedLocator([20, 30, 40])
        gl.xlabel_style = {'fontsize': 10}
        gl.ylabel_style = {'fontsize': 10}
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.rotate_labels = False
        gl.xpadding = 12
        
    fig.savefig(output_path, dpi=300, bbox_inches='tight')


def fig_s_apple_area_yield():
    china, nine = laod_china_shp()
    apple_planting = load_apple_planting_area()
    colors = ['#4b65af', '#7fcba4', '#e9f5a1', '#fdd985', '#f46f44', '#a40545']
    #colormap
    cmap = ListedColormap(colors)

    apple_data = pd.read_csv(f'{src_dir}/../data/2019apple_yield.csv')
    # Group by 'regions' column and calculate mean values
    grouped_df = apple_data.groupby('pr_name')[['area(kha)','yield(kton)','yield(ton_per_ha)']].mean()
    apple_planting = apple_planting.merge(grouped_df, on='pr_name')

    # Set projection
    proj = ccrs.LambertConformal(central_longitude=105, standard_parallels=(25, 47))
    fig = plt.figure(figsize=(14, 12))

    # ---- Subplot 1: Background + TIFF layer + Polygon labels ----
    ax = fig.add_subplot(2, 2, 1, projection=proj)
    ax.set_extent([80, 130, 18, 53], crs=ccrs.PlateCarree())
    nine.plot(ax=ax, transform=ccrs.PlateCarree(), facecolor='none', edgecolor='k', lw=0.4, zorder=1, alpha=0.5)
    china.plot(ax=ax, transform=ccrs.PlateCarree(), facecolor='none', edgecolor='k', lw=0.4, zorder=1, alpha=0.5)

    with rasterio.open(f'{src_dir}/../data/apple_panlting_reagions6_5km.tif') as src:
        data = src.read(1)
        transform = src.transform
        bounds = src.bounds

    data[data < -1000] = np.nan
    tif_extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    ax.imshow(data, origin='upper', transform=ccrs.PlateCarree(), extent=tif_extent, cmap=cmap, zorder=2, alpha=0.5)
    # print(apple_planting)
    # for idx, row in apple_planting.iterrows():
    #     if row.geometry.geom_type == 'Polygon':
    #         ax.text(row.geometry.centroid.x, row.geometry.centroid.y, row['pr_name'],
    #                 fontsize=12, weight='bold', ha='center', va='center',
    #                 transform=ccrs.PlateCarree(), color='black', zorder=4)
    #     elif row.geometry.geom_type == 'MultiPolygon':
    #         for poly in row.geometry.geoms:
    #             if poly.area > 30:
    #                 ax.text(poly.centroid.x, poly.centroid.y, row['pr_name'],
    #                         fontsize=12, weight='bold', ha='center', va='center',
    #                         transform=ccrs.PlateCarree(), color='black', zorder=4)

    apple_planting.plot(ax=ax, transform=ccrs.PlateCarree(), facecolor='none', edgecolor='k', lw=1, zorder=3, alpha=0.5)
    add_north(ax)
    scale_bar(ax, (0.2, 0.05), 500)
    ax.text(0.05, 0.95, "(a)", transform=ax.transAxes, size=16, weight='bold', va='top')

    # ---- Subplot 2: Planting area ----
    ax1 = fig.add_subplot(2, 2, 2, projection=proj)
    ax1.set_extent([80, 130, 18, 53], crs=ccrs.PlateCarree())
    nine.plot(ax=ax1, transform=ccrs.PlateCarree(), facecolor='none', edgecolor='k', lw=0.4, zorder=1, alpha=0.5)
    
    apple_planting.plot(ax=ax1, column='area(kha)', cmap='YlGn', transform=ccrs.PlateCarree(),
                        edgecolor='k', lw=0.4, zorder=3, alpha=0.5, legend=False)
    
    cax1 = fig.add_axes([ax1.get_position().x0 - 0.03, ax1.get_position().y0 + 0.01, 0.12, 0.015])
    sm1 = plt.cm.ScalarMappable(cmap='YlGn', norm=plt.Normalize(vmin=0, vmax=300))
    cbar1 = fig.colorbar(sm1, cax=cax1, orientation='horizontal', label='Planting area (k ha)')
    cbar1.set_ticks([0, 150, 300])
    
    apple_planting.plot(ax=ax1, transform=ccrs.PlateCarree(), facecolor='none', edgecolor='k', lw=1, zorder=3, alpha=0.5)
    ax1.text(0.05, 0.95, "(b)", transform=ax1.transAxes, size=16, weight='bold', va='top')

    # ---- Subplot 3: Yield ----
    ax2 = fig.add_subplot(2, 2, 3, projection=proj)
    ax2.set_extent([80, 130, 18, 53], crs=ccrs.PlateCarree())
    nine.plot(ax=ax2, transform=ccrs.PlateCarree(), facecolor='none', edgecolor='k', lw=0.4, zorder=1, alpha=0.5)
    
    apple_planting.plot(ax=ax2, column='yield(kton)', cmap='YlOrBr', transform=ccrs.PlateCarree(),
                        edgecolor='k', lw=0.4, zorder=3, alpha=0.5)

    cax2 = fig.add_axes([ax2.get_position().x0 + 0.02, ax2.get_position().y0 + 0.04, 0.12, 0.015])
    sm2 = plt.cm.ScalarMappable(cmap='YlOrBr', norm=plt.Normalize(vmin=0, vmax=500))
    cbar2 = fig.colorbar(sm2, cax=cax2, orientation='horizontal', label='Yield (k ton)')
    cbar2.set_ticks([0, 250, 500])

    apple_planting.plot(ax=ax2, transform=ccrs.PlateCarree(), facecolor='none', edgecolor='k', lw=1, zorder=3, alpha=0.5)
    ax2.text(0.05, 0.95, "(c)", transform=ax2.transAxes, size=16, weight='bold', va='top')

    # ---- Subplot 4: Yield per unit area ----
    ax3 = fig.add_subplot(2, 2, 4, projection=proj)
    ax3.set_extent([80, 130, 18, 53], crs=ccrs.PlateCarree())
    nine.plot(ax=ax3, transform=ccrs.PlateCarree(), facecolor='none', edgecolor='k', lw=0.4, zorder=1, alpha=0.5)
    
    apple_planting.plot(ax=ax3, column='yield(ton_per_ha)', cmap='YlGnBu', transform=ccrs.PlateCarree(),
                        edgecolor='k', lw=0.4, zorder=3, alpha=0.5)

    cax3 = fig.add_axes([ax3.get_position().x0 - 0.03, ax3.get_position().y0 + 0.04, 0.12, 0.015])
    sm3 = plt.cm.ScalarMappable(cmap='YlGnBu', norm=plt.Normalize(vmin=0, vmax=3))
    cbar3 = fig.colorbar(sm3, cax=cax3, orientation='horizontal', label='Yield (ton per ha)')
    cbar3.set_ticks([0, 1, 2, 3])

    apple_planting.plot(ax=ax3, transform=ccrs.PlateCarree(), facecolor='none', edgecolor='k', lw=1, zorder=3, alpha=0.5)
    ax3.text(0.05, 0.95, "(d)", transform=ax3.transAxes, size=16, weight='bold', va='top')

    # ---- Layout & Save ----
    plt.subplots_adjust(wspace=-0.1, hspace=0)

    fig.savefig(f'{src_dir}/../figs/fig_s_apple_area_yield.jpg',
                dpi=300, bbox_inches='tight')



if __name__ == "__main__":
    # fig1_research_map()
    # fig2_different_sdms()
    # fig3_model_performance()
    # fig4_ensemble_results()
    # fig5_combined_hight_suitability()
    # fig_s_apple_area_yield()
    fig_s_maxent_rf_difference()