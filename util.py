import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class ConditionalImputer(BaseEstimator, TransformerMixin):
    """
    A custom imputer that fills missing values in a target column based on the most frequent 
    or mean value conditioned on one or more other columns.

    Parameters:
    ----------
    target_col : str
        The name of the column to impute.
    condition_cols : list[str]
        A list of column names to condition the imputation on.
    strategy : str, default='most_frequent'
        The imputation strategy to use. Can be 'most_frequent' or 'mean'.

    Methods:
    -------
    fit(X, y=None):
        Learns the imputation values based on the specified strategy.
    transform(X):
        Imputes missing values in the target column based on the learned values.
    """

    def __init__(self, target_col: str, condition_cols: list[str], strategy: str = 'most_frequent'):
        self.strategy = strategy
        self.condition_cols = condition_cols
        self.target_col = target_col
        self.impute_values = {}

    def fit(self, X: pd.DataFrame, y=None) -> 'ConditionalImputer':
        """
        Learns the imputation values based on the specified strategy.

        Parameters:
        ----------
        X : pd.DataFrame
            The input data containing the target and condition columns.
        y : None
            Ignored. This parameter is included for compatibility with scikit-learn.

        Returns:
        -------
        self : ConditionalImputer
            Fitted imputer instance.
        """
        if self.strategy == 'most_frequent':
            self.impute_values = (
                X.groupby(self.condition_cols)[self.target_col]
                .agg(lambda x: x.mode()[0] if not x.mode().empty else None)
                .to_dict()
            )
        elif self.strategy == 'mean':
            self.impute_values = (
                X.groupby(self.condition_cols)[self.target_col]
                .agg(lambda x: np.nanmean(x))
                .to_dict()
            )
        else: 
            raise ValueError('Invalid strategy')
        return self

    def transform(self, X: pd.DataFrame) -> pd.Series:
        """
        Imputes missing values in the target column based on the learned values and returns the entire column.

        Parameters:
        ----------
        X : pd.DataFrame
            The input data containing the target and condition columns.

        Returns:
        -------
        pd.Series
            The target column with missing values filled.
        """
        # Create a copy to avoid modifying the original DataFrame
        imputed_column = X[self.target_col].copy()
        
        # Iterate over the groups and their corresponding impute values
        for group, value in self.impute_values.items():
            # Create a boolean mask for all condition columns
            if len(self.condition_cols) == 1:
                mask = (X[self.condition_cols[0]] == group)
            else:
                mask = (X[self.condition_cols] == group).all(axis=1)
            
            # Fill in the imputed values where the target column is null
            imputed_column.loc[mask & imputed_column.isnull()] = value
        
        return imputed_column


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
        if (df[column].dtype == 'int64' and df[column].value_counts().shape[0] < 40) or df[column].dtype == 'object' or df[column].dtype == '<M8[ns]':
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
        elif (df[column].dtype == 'int64' and df[column].value_counts().shape[0] >= 10) or df[column].dtype == 'float64':
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
    'S' : 2025 
    }


dict_vin2manufacturer = {
    'AAA': 'audi',
    'AAK': 'faw',
    'AAM': 'man',
    'AAP': '',
    'AAV': 'volkswagen',
    'AAW': 'challenger-trailer',
    'AA9': 'tr-tec',
    'CN1': 'tr-tec',
    'ABJ': 'mitsubishi',
    'ABM': 'bmw',
    'ACV': 'isuzu',
    'AC5': 'hyundai',
    'ADB': 'mercedes-benz',
    'ADD': '',
    'ADM': 'general-motors',
    'ADN': 'nissan',
    'ADR': 'renault',
    'ADX': 'tata',
    'AFA': '',
    'AFB': 'mazda',
    'AFD': 'baic',
    'AHH': 'hino',
    'AHM': 'mercedes-benz',
    'AHT': 'toyota',
    'BF9/': 'kibo',
    'BUK': 'kiira-motors-corporation',
    'BR1': 'mercedes-benz',
    'EBZ': 'nizhekotrans',
    'DF9/': 'laraki',
    'HA0': 'wuxi-sundiro-electric-vehicle-co-ltd',
    'HA6': 'niu technologies',
    'HA7': 'jinan-qingqi-kr-motors-co-ltd',
    'HES': 'smart',
    'HGL': 'farizon-auto',
    'HGX': 'wuling',
    'HJR': 'jetour',
    'HL4': 'morini',
    'HRV': 'beijing-henrey',
    'HZ2': 'taizhou-zhilong-technology-co-ltd',
    'H0D': 'taizhou-qianxin-vehicle-co-ltd',
    'JAA': 'isuzu',
    'JAB': 'isuzu',
    'JAC': 'isuzu',
    'JAE': 'acura',
    'JAL': 'isuzu',
    'JAM': 'isuzu',
    'JA3': 'mitsubishi',
    'JA4': 'mitsubishi',
    'JA7': 'mitsubishi',
    'JB3': 'dodge',
    'JB4': 'dodge',
    'JB7': 'dodge',
    'JC0': 'ford',
    'JC1': 'fiat',
    'JC2': 'ford',
    'JDA': 'daihatsu',
    'JD1': 'daihatsu',
    'JD2': 'daihatsu',
    'JD4': 'daihatsu',
    'JE3': 'mitsubishi',
    'JE4': 'mitsubishi',
    'JF1': 'subaru',
    'JF2': 'subaru',
    'JF3': 'subaru',
    'JF4': 'saab',
    'JG1': 'chevrolet',
    'JG2': 'pontiac',
    'JG7': 'suzuki',
    'JGC': 'chevrolet',
    'JGT': 'gmc',
    'JHA': 'hino',
    'JHB': 'hino',
    'JHD': 'hino',
    'JHF': 'hino',
    'JHH': 'hino',
    'JHG': 'honda',
    'JHL': 'honda',
    'JHM': 'honda',
    'JHN': 'honda',
    'JHZ': 'honda',
    'JH1': 'honda',
    'JH2': 'honda',
    'JH3': 'honda',
    'JH4': 'honda',
    'JH5': 'honda',
    'JH6': 'hino',
    'JJ3': 'chrysler',
    'JKA': 'kawasaki',
    'JKB': 'kawasaki',
    'JKS': 'suzuki',
    'JK8': 'suzuki',
    'JLB': 'mitsubishi-fuso',
    'JLF': 'mitsubishi-fuso',
    'JLS': 'mitsubishi-fuso-truck-bus-corp',
    'JL5': 'mitsubishi-fuso',
    'JL6': 'mitsubishi-fuso',
    'JL7': 'mitsubishi-fuso',
    'JMA': 'mitsubishi',
    'JMB': 'mitsubishi',
    'JMF': 'mitsubishi',
    'JMP': 'mitsubishi',
    'JMR': 'mitsubishi',
    'JMY': 'mitsubishi',
    'JMZ': 'mazda',
    'JM0': 'mazda',
    'JM1': 'mazda',
    'JM2': 'mazda',
    'JM3': 'mazda',
    'JM4': 'mazda',
    'JM6': 'mazda',
    'JM7': 'mazda',
    'JNA': 'nissan-diesel/ud-trucks',
    'JNC': 'nissan-diesel/ud-trucks',
    'JNE': 'nissan-diesel/ud-trucks',
    'JNK': 'infiniti',
    'JNR': 'infiniti',
    'JNX': 'infiniti',
    'JN1': 'nissan',
    'JN3': 'nissan',
    'JN6': 'nissan',
    'JN8': 'nissan',
    'JPC': 'nissan-diesel/ud-trucks',
    'JP3': 'mitsubishi',
    'JP4': 'mitsubishi',
    'JP7': 'mitsubishi',
    'JR2': 'isuzu',
    'JSA': 'suzuki',
    'JSK': 'suzuki',
    'JSL': 'suzuki',
    'JST': 'suzuki',
    'JS1': 'suzuki',
    'JS2': 'suzuki',
    'JS3': 'suzuki',
    'JS4': 'suzuki',
    'JTB': 'toyota',
    'JTD': 'toyota',
    'JTE': 'toyota',
    'JTF': 'toyota',
    'JTG': 'toyota',
    'JTH': 'lexus',
    'JTJ': 'lexus',
    'JTK': 'toyota',
    'JTL': 'toyota',
    'JTM': 'toyota',
    'JTN': 'toyota',
    'JTP': 'toyota',
    'JT1': 'toyota',
    'JT2': 'toyota',
    'JT3': 'toyota',
    'JT4': 'toyota',
    'JT5': 'toyota',
    'JT6': 'lexus',
    'JT7': 'toyota',
    'JT8': 'lexus',
    'JW6': 'mitsubishi',
    'JYA': 'yamaha',
    'JYE': 'yamaha',
    'JY3': 'yamaha',
    'JY4': 'yamaha',
    'J81': 'chevrolet',
    'J87': 'pontiac',
    'J8B': 'chevrolet',
    'J8C': 'chevrolet',
    'J8D': 'gmc',
    'J8T': 'gmc',
    'J8Z': 'chevrolet',
    'KF3': 'merkavim',
    'KF6': 'automotive-industries',
    'KF9': 'tomcar',
    'KG9': 'charash-ashdod',
    'KL': 'daewoo',
    'KLA': 'daewoo',
    'KLP': 'ct&t united',
    'KLT': 'tata-daewoo',
    'KLU': 'tata-daewoo',
    'KLY': 'daewoo',
    'KL1': 'gm-daewoo/gm-korea',
    'KL2': 'daewoo',
    'KL3': 'gm-daewoo/gm-korea',
    'KL4': 'gm-korea',
    'KL5': 'gm-daewoo',
    'KL6': 'gm-daewoo',
    'KL7': 'daewoo',
    'KL8': 'gm-daewoo/gm-korea',
    'KM': 'hyundai',
    'KMC': 'hyundai',
    'KME': 'hyundai',
    'KMF': 'hyundai',
    'KMH': 'hyundai',
    'KMJ': 'hyundai',
    'KMT': 'genesis motor',
    'KMU': 'genesis-motor',
    'KMX': 'hyundai',
    'KMY': 'daelim',
    'KM1': 'hyosung',
    'KM4': 'hyosung',
    'KM8': 'hyundai',
    'KNA': 'kia',
    'KNC': 'kia',
    'KND': 'kia',
    'KNE': 'kia',
    'KNF': 'kia',
    'KNG': 'kia',
    'KNJ': 'ford',
    'KNM': 'renault-samsung',
    'KN1': 'asia motors',
    'KN2': 'asia motors',
    'KPA': 'ssangyong',
    'KPB': 'ssangyong',
    'KPH': 'mitsubishi',
    'KPT': 'ssangyong/kg-mobility',
    'LAA': 'shanghai-jialing-vehicle-co-ltd',
    'LAE': 'jinan-qingqi',
    'LAL': 'honda',
    'LAN': 'changzhou-yamasaki',
    'LAP': 'chongqing jianshe motorcycle co., ltd.',
    'LAT': 'luoyang-northern-ek-chor-motorcycle-co-ltd',
    'LA6': 'king long',
    'LA8': 'anhui-ankai',
    'LA7': 'radar-auto',
    'LA9': 'byd',
    'LC0': 'byd',
    'LM6': 'srm-shineray',
    'LBB': 'zhejiang-qianjiang-motorcycle',
    'LBE': 'hyundai',
    'LBM': 'zongshen-piaggio',
    'LBP': 'chongqing jianshe yamaha motor co. ltd.',
    'LBV': 'bmw-brilliance',
    'LB1': 'fujian-benz',
    'LB2': 'geely',
    'LB3': 'geely',
    'LB4': 'chongqing-yinxiang-motorcycle-group-co-ltd',
    'LB5': 'foshan city fosti motorcycle co., ltd.',
    'LB7': 'tibet-new-summit-motorcycle-co-ltd',
    'LCE': 'hangzhou-chunfeng-motorcycles',
    'LCR': 'gonow',
    'LC2': 'kymco',
    'LC6': 'suzuki',
    'LDC': 'dongfeng-peugeot-citroen',
    'LDD': 'dandong-huanghai-automobile',
    'LDK': 'faw',
    'LDN': 'soueast',
    'LDP': 'voyah',
    'LDY': 'zhongtong',
    'LD3': 'guangdong-tayo-motorcycle-technology-co',
    'LD5': 'benzhou',
    'LD9': 'sitech',
    'L3A': 'sitech',
    'LEC': 'tianjin-qingyuan-electric-vehicle-co-ltd',
    'LEF': 'jiangling-motors-corporation-ltd',
    'LEH': 'zhejiang riya motorcycle co. ltd.',
    'LET': 'jiangling-isuzu',
    'LEW': 'dongfeng',
    'LE4': 'beijing-benz',
    'LE8': 'guangzhou-panyu-huanan-motors-industry-co-ltd',
    'LFB': 'faw',
    'LFF': 'zhejiang taizhou wangye power co., ltd.',
    'LFG': 'taizhou-chuanl-motorcycle-manufacturing',
    'LFJ': 'fujian-motors-group',
    'LFM': 'toyota',
    'LFN': 'faw',
    'LFP': '',
    'LFT': 'faw',
    'LFU': 'lifeng',
    'LFV': 'faw-volkswagen',
    'LFW': 'faw-jiefang',
    'LFY': 'changshu-light-motorcycle-factory',
    'LFZ': 'leapmotor',
    'LF3': 'lifan',
    'LGA': 'dongfeng',
    'LGB': 'dongfeng-nissan',
    'LGG': 'dongfeng-liuzhou-motor',
    'LGJ': 'dongfeng-fengshen',
    'LGL': 'guilin-daewoo',
    'LGV': 'heshan-guoji-nanlian-motorcycle-industry-co-ltd',
    'LGW': 'great-wall-motor',
    'LGX': 'byd auto',
    'LGZ': 'guangzhou-denway',
    'LHA': 'shuanghuan',
    'LHB': 'beijing-automotive-industry-holding',
    'LHG': 'gac-honda',
    'LHJ': 'chongqing-astronautic-bashan-motorcycle-manufacturing-co-ltd',
    'LHW': 'crrc',
    'LH0': 'wm motor',
    'LH1': 'faw-haima',
    'LJC': 'jincheng',
    'LJD': 'yueda-kia',
    'LJM': 'sunlong',
    'LJN': 'zhengzhou-nissan',
    'LJS': 'yaxing',
    'LJU': 'shanghai-maple-automobile',
    'LJV': 'sinotruk',
    'LJX': 'jmc-ford',
    'LJ1': 'jac',
    'LJ4': 'shanghai-jmstar-motorcycle-co-ltd',
    'LJ8': 'zotye-auto',
    'LKC': 'baic',
    'LKG': 'youngman-lotus',
    'LKH': 'hafei',
    'LKL': 'higer',
    'LKT': 'yunnan-lifan-junma-vehicle-co-ltd',
    'LK2': 'anhui-jac',
    'LK6': 'wuling, baojun',
    'LK8': 'zhejiang-yule-new-energy-automobile-technology-co-ltd',
    'LLC': 'loncin',
    'LLJ': 'jiangsu-xinling-motorcycle-fabricate-co-ltd',
    'LLN': 'qoros',
    'LLP': 'zhejiang jiajue motorcycle manufacturing co., ltd.',
    'LLU': 'dongfeng-fengxing-jingyi',
    'LLV': 'lifan',
    'LLX': 'yudo',
    'LL0': 'sanmen-county-yongfu-machine-co-ltd',
    'LL2': 'wm motor',
    'LL3': 'xiamen golden dragon bus co. ltd.',
    'LL6': 'gac-mitsubishi',
    'LL8': 'jiangsu-linhai-yamaha-motor-co-ltd',
    'LMC': 'suzuki',
    'LME': 'skyworth',
    'LMF': 'jiangmen-zhongyu-motor-co-ltd',
    'LMG': 'gac-trumpchi',
    'LMH': 'jiangsu-guowei-motor',
    'LMV': 'haima',
    'LMW': 'gac',
    'LMX': 'forthing',
    'LM0': 'wangye',
    'LM8': 'seres',
    'LNA': 'gac-aion',
    'LNB': 'baic',
    'LND': 'jmev',
    'LNN': 'chery',
    'LNY': 'yuejin',
    'LPA': 'changan psa',
    'LPE': 'byd auto',
    'LPS': 'polestar',
    'LP6': 'guangzhou-panyu-haojian-motorcycle-industry-co-ltd',
    'LRB': 'buick',
    'LRD': 'beijing-foton-daimler-automotive-co-ltd-auman-trucks',
    'LRE': 'cadillac',
    'LRW': 'tesla',
    'LSC': 'changan',
    'LSF': 'saic-maxus',
    'LSG': 'saic-general-motors-chevrolet-buick',
    'LSH': 'saic-maxus',
    'LSJ': 'saic-mg',
    'LSK': 'saic-maxus',
    'LSV': 'saic-volkswagen',
    'LSY': 'brilliance',
    'LS4': 'changan',
    'LS5': 'changan',
    'LS6': 'changan',
    'LS7': 'jmc',
    'LTA': 'zx auto',
    'LTN': 'soueast',
    'LTP': 'national electric vehicle sweden ab',
    'LTV': 'faw-toyota',
    'LTW': '',
    'LUC': 'honda',
    'LUD': 'dongfeng nissan diesel motor co ltd.',
    'LUG': 'qiantu',
    'LUJ': 'zhejiang-shanqi-tianying-vehicle-industry-co-ltd',
    'LUR': 'chery',
    'LUX': 'dongfeng-yulon-motor-co-ltd',
    'LUZ': 'hozon',
    'LVA': 'foton motor',
    'LVB': 'foton motor',
    'LVC': 'foton-motor',
    'LVF': 'changhe-suzuki',
    'LVG': 'toyota',
    'LVH': 'dongfeng-honda',
    'LVM': 'chery',
    'LVP': 'dongfeng-sokon',
    'LVR': 'changan-mazda',
    'LVS': 'changan-ford',
    'LVT': 'chery',
    'LVU': 'chery',
    'LVV': 'chery',
    'LVX': 'landwind',
    'LVY': 'volvo',
    'LVZ': 'dongfeng-sokon-motor-company',
    'LV3': 'national electric vehicle sweden ab',
    'LV7': 'jinan-qingqi',
    'LWB': 'wuyang-honda',
    'LWG': 'chongqing-huansong',
    'LWL': 'qingling-isuzu',
    'LWV': 'fiat',
    'LW4': 'li-auto',
    'LXA': 'jiangmen-qipai-motorcycle-co-ltd',
    'LXG': 'xuzhou construction machinery group co., ltd.',
    'LXM': 'xiamen xiashing motorcycle co., ltd.',
    'LXN': 'link-tour',
    'LXV': 'borgward',
    'LXY': 'chongqing-shineray-motorcycle-co-ltd',
    'LX6': 'jiangmen-city-huari-group-co-ltd',
    'LX8': 'xgjao',
    'LYB': 'weichai',
    'LYD': 'taizhou-city-kaitong-motorcycle-co-ltd',
    'LYM': 'zhuzhou-jianshe-yamaha',
    'LYU': 'huansu',
    'LYV': 'volvo',
    'LY4': 'chongqing-yingang-science-technology-group-co-ltd',
    'LZE': 'isuzu',
    'LZF': 'saic-iveco-hongyan',
    'LZG': 'shaanxi-automobile-group-shacman-bus',
    'LZK': 'sinotruk',
    'LZL': 'zengcheng-haili-motorcycle-ltd',
    'LZM': 'man',
    'LZP': 'zhongshan-guochi',
    'LZS': 'zongshen',
    'LZU': 'isuzu',
    'LZW': 'saic-gm-wuling',
    'LZY': 'yutong',
    'LZZ': 'sinotruk',
    'LZ0': 'shandong wuzheng group co., ltd.',
    'LZ4': 'jiangsu-linzhi-shangyang-group-co-ltd',
    'LZ9': 'raysince',
    'LZX': 'raysince',
    'L1K': 'chongqing-hengtong-bus-co-ltd',
    'L1N': 'xpeng',
    'L10': 'geely',
    'L2B': 'jiangsu-baodiao-locomotive-co-ltd',
    'L2C': 'chery-jaguar-land-rover',
    'L3H': 'shanxi-victory',
    'L37': 'huzhou daixi zhenhua technology trade co., ltd.',
    'L4B': 'xingyue',
    'L4F': 'suzhou-eagle-electric-vehicle-manufacturing-co-ltd',
    'L4H': 'ningbo longjia motorcycle co., ltd.',
    'L4S': 'zhejiang xingyue vehicle co ltd.',
    'L4Y': '',
    'L5C': 'zhejiang-kangdi-vehicles',
    'L5E': 'zoomlion',
    'L5K': 'zhejiang-yongkang-easy-vehicle',
    'L5N': 'zhejiang-taotao',
    'L5Y': 'merato motorcycle taizhou zhongneng motorcycle co. ltd. (znen)',
    'L6F': 'shandong-liangzi-power-co-ltd',
    'L6J': 'kayo',
    'L6K': 'shanghai-hwh',
    'L6T': 'geely',
    'L66': 'zhuhai-granton',
    'L82': 'baotian',
    'L85': 'zhejiang-yongkang-huabao-electric-appliance',
    'L8A': 'jinhua youngman automobile manufacturing co., ltd.',
    'L8X': 'zhejiang-summit-huawin-motorcycle',
    'L8Y': 'zhejiang jonway motorcycle manufacturing co., ltd.',
    'L9G': 'zhuhai-guangtong-automobile',
    'L9N': 'zhejiang-taotao-vehicles',
    'MAB': 'mahindra & mahindra',
    'MAC': 'mahindra & mahindra',
    'MAH': 'fiat',
    'MAJ': 'ford',
    'MAK': 'honda',
    'MAL': 'hyundai',
    'MAN': 'eicher-polaris',
    'MAT': 'tata-motors',
    'MA1': 'mahindra & mahindra',
    'MA3': 'maruti-suzuki',
    'MA7': 'hindustan-motors',
    'MBF': 'royal enfield',
    'MBH': 'suzuki',
    'MBJ': 'toyota',
    'MBK': 'man',
    'MBL': 'hero-motocorp',
    'MBR': 'mercedes-benz',
    'MBU': 'swaraj',
    'MBV': 'premier automobiles ltd.',
    'MBX': 'piaggio',
    'MBY': 'asia motor works ltd.',
    'MB1': 'ashok leyland',
    'MB2': 'hyundai',
    'MB7': 'reva',
    'MB8': 'suzuki',
    'MCA': 'fca',
    'MCD': 'mahindra',
    'MCG': 'atul-auto',
    'MCL': 'international-cars-and-motors-ltd',
    'MC1': 'force-motors',
    'MC2': '',
    'MC4': 'dilip-chhabria-design-pvt-ltd',
    'MDE': 'kinetic-engineering-limited',
    'MDH': 'nissan',
    'MDT': 'kerala-automobiles',
    'MD2': 'bajaj',
    'MD6': 'tvs motor company',
    'MD7': 'lml',
    'MD9': 'shuttle-cars-india',
    'MEC': 'daimler',
    'MEE': 'renault',
    'MEG': 'harley-davidson',
    'MER': 'benelli',
    'MET': 'piaggio',
    'MEX': 'skoda-auto-volkswagen',
    'ME1': 'yamaha',
    'ME3': 'royal enfield',
    'ME4': 'honda',
    'MYH': 'ather energy',
    'MZB': 'kia',
    'MZD': 'jawa',
    'MZZ': 'citroen',
    'M3G': 'isuzu',
    'M6F': 'um-lohia',
    'MF3': 'hyundai',
    'MHD': 'suzuki',
    'MHF': 'toyota',
    'MHK': 'daihatsu',
    'MHL': 'mercedes-benz',
    'MHR': 'honda',
    'MHY': 'suzuki',
    'MH1': 'honda',
    'MH3': 'pt yamaha indonesia motor mfg.',
    'MH4': 'kawasaki',
    'MH8': 'suzuki',
    'MKF': 'pt sokonindo automobile (dfsk)',
    'MK2': 'mitsubishi',
    'MK3': 'wuling',
    'MLB': 'siam-yamaha',
    'MLC': 'suzuki',
    'MLE': 'yamaha',
    'MLH': 'honda',
    'MLY': 'harley-davidson',
    'ML0': 'ducati',
    'ML3': 'mitsubishi',
    'ML5': 'kawasaki',
    'MMA': 'mitsubishi',
    'MMB': 'mitsubishi',
    'MMC': 'mitsubishi',
    'MMD': 'mitsubishi',
    'MME': 'mitsubishi',
    'MMF': 'bmw',
    'MMM': 'chevrolet',
    'MMR': 'subaru',
    'MMS': 'suzuki',
    'MMT': 'mitsubishi',
    'MMU': 'holden',
    'MM0': 'mazda',
    'MM6': 'mazda',
    'MM7': 'mazda',
    'MM8': 'mazda',
    'MNA': 'ford',
    'MNB': 'ford',
    'MNC': 'ford',
    'MNK': 'hino motors manufacturing thailand co ltd.'
 }