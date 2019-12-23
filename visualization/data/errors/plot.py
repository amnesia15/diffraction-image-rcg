import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# Boxplot data
boxplot_data_mae = [df1_relu['MAE test'],
                    df2_relu['MAE test'],
                    df3_relu['MAE test'],
                    df4_relu['MAE test']]
boxplot_data_rmse = [df1_relu['RMSE test'],
                     df2_relu['RMSE test'],
                     df3_relu['RMSE test'],
                     df4_relu['RMSE test']]

file_name = ['mae_test_bp.png', 'rmse_test_bp.png']
data = [boxplot_data_mae, boxplot_data_rmse]

for i in np.arange(len(file_name)):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlabel('Architectures')
    plt.ylabel('MAE')
    bp = ax.boxplot(data[i], patch_artist=True)
    ax.set_xticklabels(labels)

    # Change outline color, fill color and linewidth of the boxes
    for box in bp['boxes']:
        box.set(color='#7570b3', linewidth=2)
        box.set(facecolor='#1b9e77')

    # Change color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='#b2df8a', linewidth=1.5)

    # Change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(color='#7570b3', linewidth=2)

    # Change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='#b2df8a', linewidth=2)

    # Change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color='#e7298a', alpha=0.5)

    # Saving the figure
    fig.savefig(file_name[i], bbox_inches='tight')
