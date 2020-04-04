import seaborn as sns
import matplotlib.pyplot as plt




def plot_categorical(table, columns_to_plot, target_column):
    """helps find distributiuons of bads in each categorical variable"""
    sns.set(style="ticks", color_codes=True)
    sns.catplot(x=columns_to_plot, col=target_column, kind='count', data=table, height=4, aspect=.7)

    sns.countplot(x=columns_to_plot, hue=target_column, data=table)


def plot_correlation(table):
    """takes only numerical data"""
    colormap = plt.cm.RdBu
    plt.figure(figsize=(16, 14))

    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    sns.heatmap(table.astype(float).corr(), linewidths=0.1, vmax=1.0,
                square=True, cmap=colormap, linecolor='white', annot=True)


def plot_numerical(table, target, columns_to_plot):
    """takes only numerical data" i.w. float64"""
    sns.pairplot(table, hue=target, vars=[columns_to_plot])


