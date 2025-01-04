import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_univariate(df, columns, hue=None, bins=50, bw_method=0.1, size=(20, 24), ncols=2, hspace=0.7, wspace=0.2, log_dict=None):
    """Function to visualize columns in df. Visualization type depends on data type of the column.
    
    Arguments
    ---------
    df : pandas.DataFrame
        Dataframe whose columns shall be visualized.
    columns : list
        Subset of columns which shall be considered only.
    hue: str
        Column according to which single visualization shall be grouped.
    bins : int
        Number of bins for the histogram plots.
    bw_method : float
        method for determining the smoothing bandwidth to use.
    size: tuple
        Size of the resulting grid.
    nclos: int
        Number of columns in the resulting grid.
    hspace : float
        Horizontal space between subplots.
    wspace : float
        Vertical space between subplots.
    log_dict : dict
        Dictionary listing whether a column's visualization should be 
        displayed in log scale on the vertical axis
            

    Returns
    -------
    Visualization of each variable in columns as barplot or histogram.

    """

    # Reduce df to relevant columns
    df = df[columns]
    
    # Calculate the number of rows and columns for the grid
    num_cols = len(df.columns)
    num_rows = int(num_cols / ncols) + (num_cols % ncols)
    
    # Create the subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=ncols, figsize=size)

    # Change the vertical and horizontal spacing between subplots
    plt.subplots_adjust(hspace=hspace, wspace=wspace)  
    
    # Flatten the axes array for easier iteration
    axes = axes.flatten()

    # Do not display vertical axis in log scale as default
    logy = False
    
    # Iterate over each column and plot accordingly
    for i, column in enumerate(df.columns):
        if log_dict!=None:
            logy=log_dict.get(column, False)
            
        ax = axes[i]
        # Barplots for categorical features or integers with few distinct values
        if (df[column].dtype == 'Int64' and df[column].value_counts().shape[0] < 40) or df[column].dtype == 'string[python]' or df[column].dtype == '<M8[ns]':
            if hue==None or hue==column:
                temp = df[column].value_counts().sort_index()
                if temp.shape[0] > 20:
                    fontsize = 'small'
                else:
                    fontsize = 'medium'
                temp.plot(kind='bar', ax=ax, ylabel='Count', xlabel='', title=column, logy=logy, fontsize=fontsize)
            else:
                temp = df[[column, hue]].groupby(hue).value_counts(normalize=True).sort_index().to_frame().reset_index()
                if temp.shape[0] > 20:
                    fontsize = 'small'
                else:
                    fontsize = 'medium'
                temp[hue] = temp[hue].astype(str)
                p = sns.barplot(temp, x=column, y='proportion', hue=hue, errorbar=None, ax=ax)
                # Add title and labels
                p.set_title(column)
                p.set_xlabel('')
                p.set_ylabel('Proportion')
                p.set_xticks(p.get_xticks())
                p.set_xticklabels(
                    p.get_xticklabels(), 
                    rotation=90, 
                    horizontalalignment='center', 
                    fontsize=fontsize)
                if logy:
                    p.set_yscale("log")

        # Histograms for floats or integers with many distinct values
        elif (df[column].dtype == 'Int64' and df[column].value_counts().shape[0] >= 10) or df[column].dtype == 'Float64':
            if hue==None:
                df[column].plot(kind='hist', ax=ax, bins=bins, title=column, logy=logy)
            else:
                hue_groups = np.sort(df[hue].unique())
                for hue_group in hue_groups:
                    p = sns.kdeplot(data=df[df[hue] == hue_group], x=column, fill=True, label=hue_group, ax=ax, bw_method=bw_method)
                # Add title and labels
                p.set_title(column)
                p.set_xlabel('')
                p.set_ylabel('Density')
                p.legend(title=hue)
                if logy:
                    p.set_yscale("log")
                                
        # For all other data types pass
        else:
            pass
        

dict_vin2year = {
    'A' : 1980 ,
    'B' : 1981 ,
    'C' : 1982 ,
    'D' : 1983 ,
    'E' : 1984 ,
    'F' : 1985 ,
    'G' : 1986 ,
    'H' : 1987 ,
    'J' : 1988 ,
    'K' : 1989 ,
    'L' : 1990 ,
    'M' : 1991 ,
    'N' : 1992 ,
    'P' : 1993 ,
    'R' : 1994 ,
    'S' : 1995 ,
    'T' : 1996 ,
    'V' : 1997 ,
    'W' : 1998 ,
    'X' : 1999 ,
    'Y' : 2000 ,
    '1' : 2001 ,
    '2' : 2002 ,
    '3' : 2003 ,
    '4' : 2004 ,
    '5' : 2005 ,
    '6' : 2006 ,
    '7' : 2007 ,
    '8' : 2008 ,
    '9' : 2009 ,
    'A' : 2010 ,
    'B' : 2011 ,
    'C' : 2012 ,
    'D' : 2013 ,
    'E' : 2014 ,
    'F' : 2015 ,
    'G' : 2016 ,
    'H' : 2017 ,
    'J' : 2018 ,
    'K' : 2019 ,
    'L' : 2020 ,
    'M' : 2021 ,
    'N' : 2022 ,
    'P' : 2023 ,
    'R' : 2024 ,
    'S' : 2025 ,
    'T' : 2026 ,
    'V' : 2027 ,
    'W' : 2028 ,
    'X' : 2029}