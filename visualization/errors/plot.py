import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

labels = ['108-55', '200-50-25', '202-101-50-20', '2048-1024-1024-512-512-256']

labelsize = 7
mpl.rcParams['xtick.labelsize'] = labelsize
mpl.rcParams['ytick.labelsize'] = labelsize


# Load the data about errors of the models for various architectures (tanh and relu)
df1_relu = pd.read_csv('108-55 (relu)\model_output\errors.csv')
df2_relu = pd.read_csv('200-50-25 (relu)\model_output\errors.csv')
df3_relu = pd.read_csv('202-101-50-20 (relu)\model_output\errors.csv')
df4_relu = pd.read_csv('2048-1024-1024-512-512-256 (relu)\model_output\errors.csv')

df1_tanh = pd.read_csv('108-55 (tanh)\model_output\errors.csv')
df2_tanh = pd.read_csv('200-50-25 (tanh)\model_output\errors.csv')
df3_tanh = pd.read_csv('202-101-50-20 (tanh)\model_output\errors.csv')
df4_tanh = pd.read_csv('2048-1024-1024-512-512-256 (tanh)\model_output\errors.csv')

# Boxplot data
boxplot_data_mae = [df1_relu['MAE test'], df2_relu['MAE test'], df3_relu['MAE test'], df4_relu['MAE test']]
boxplot_data_rmse = [df1_relu['RMSE test'], df2_relu['RMSE test'], df3_relu['RMSE test'], df4_relu['RMSE test']]

file_name = ['mae_test_bp.png', 'rmse_test_bp.png']
data = [boxplot_data_mae, boxplot_data_rmse]

for i in np.arange(len(file_name)):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlabel('Architectures')
    plt.ylabel('MAE')
    bp = ax.boxplot(data[i], patch_artist=True)
    ax.set_xticklabels(labels)

    ## Change outline color, fill color and linewidth of the boxes
    for box in bp['boxes']:
        box.set( color='#7570b3', linewidth=2)
        box.set( facecolor = '#1b9e77' )

    ## Change color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='#b2df8a', linewidth=1.5)

    ## Change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(color='#7570b3', linewidth=2)

    ## Change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='#b2df8a', linewidth=2)

    ## Change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color='#e7298a', alpha=0.5)

    # Saving the figure
    fig.savefig(file_name[i], bbox_inches='tight')


# Loading running times of models with different activation functions and architecture
run_relu_1 = pd.read_csv('108-55 (relu)\model_output\\running_times.csv')['running times']
run_tanh_1 = pd.read_csv('108-55 (tanh)\model_output\\running_times.csv')['running times']
run_relu_2 = pd.read_csv('200-50-25 (relu)\model_output\\running_times.csv')['running times']
run_tanh_2 = pd.read_csv('200-50-25 (tanh)\model_output\\running_times.csv')['running times']
run_relu_3 = pd.read_csv('202-101-50-20 (relu)\model_output\\running_times.csv')['running times']
run_tanh_3 = pd.read_csv('202-101-50-20 (tanh)\model_output\\running_times.csv')['running times']
run_relu_4 = pd.read_csv('2048-1024-1024-512-512-256 (relu)\model_output\\running_times.csv')['running times']
run_tanh_4 = pd.read_csv('2048-1024-1024-512-512-256 (tanh)\model_output\\running_times.csv')['running times']

# Calculating means of running times for relu NNs and tanh NNs
n_groups = 4
means_running_relu = [run_relu_1.mean(), run_relu_2.mean(), run_relu_3.mean(), run_relu_4.mean()]
means_running_tanh = [run_tanh_1.mean(), run_tanh_2.mean(), run_tanh_3.mean(), run_tanh_4.mean()]

# Creating bar plot of running times
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, means_running_relu, bar_width, alpha=opacity, color='b', label='Relu')

rects2 = plt.bar(index + bar_width, means_running_tanh, bar_width, alpha=opacity, color='g', label='Tanh')

plt.xlabel('Architectures')
plt.ylabel('Seconds (mean)')
plt.title('Running times')
plt.xticks(index + bar_width, labels)
plt.legend()

plt.tight_layout()
fig.savefig('running_times_bar.png', bbox_inches='tight')

# Calculating means of MAE test for relu NNs and tanh NNs
n_groups = 4
means_relu_mae = [df1_relu['MAE test'].mean(), df2_relu['MAE test'].mean(), 
    df3_relu['MAE test'].mean(), df4_relu['MAE test'].mean()]
means_tanh_mae = [df1_tanh['MAE test'].mean(), df2_tanh['MAE test'].mean(), 
    df3_tanh['MAE test'].mean(), df4_tanh['MAE test'].mean()]

# Creating bar plot of MAE TEST erros
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, means_relu_mae, bar_width, alpha=opacity, color='b', label='Relu')

rects2 = plt.bar(index + bar_width, means_tanh_mae, bar_width, alpha=opacity, color='g', label='Tanh')

plt.xlabel('Architectures')
plt.ylabel('MAE')
plt.title('Mean absolute errors (test)')
plt.xticks(index + bar_width, labels)
plt.legend()

plt.tight_layout()
fig.savefig('mae_bar.png', bbox_inches='tight')