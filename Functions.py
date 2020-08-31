import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.stats.api as sms
import statsmodels.formula.api as smf
import scipy.stats as stats
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
import seaborn as sns
import scipy.stats as stats


def plot_scatter ():
    ''' This function will plot scatter plots.
    Inputs are independent variable (X), dependent variable (Y).
    output will be the fg and axes. '''
    
    fig ,ax = plt.subplots(x,y,xlabel,ylabel='SalePrice')
    
    ax.scatter(x,y, marker = '.')
    ax.set(xlabel=xlabel,ylabel=ylabel)
    ax.set_title(f'{xlabel} vs {ylabel}')
    
    return fig, ax


def corr_map(df):
    ''' Generates correlation mask to hide the unwanted cells from a correlation matrix. 
    required input is the df (DataFrame)'''
    
    # Set a new fig and its size
    fig, ax= plt.subplots(figsize = (12,12))
    
    # Create a corrrelation matrix for each df columns and round it to 3 sig-figs.
    corr = np.abs(df.corr().round(3))
    
    # Create a mask to hide the duplicate half of the matrix for easy comparison
    mask = np.zeros_like(corr, dtype=np.bool)
    idx = np.triu_indices_from(mask)
    mask[idx] = True
    
    # create a heat map with the help of the correlation values
    sns.heatmap(corr,annot=True,square=True,mask=mask,cmap='Blues',
            center=0,ax=ax,linewidths=.5, cbar_kws={"shrink": .5}, cbar=True)
    
    plt.set_ylim(len(corr),-0.5,0.5)


def multi_linear_reg (df, drop_cols, target):
    ''' Generate a multilinear model from a DataFrame and without the drop_cols. 
    
    @params:
    df is a pd.DataFrame
    drop_cols is a list of columns to not include in the model fit
    target is the str() of our predicted columns name.
    
    @Output:
    generated linear model
    Columns used to generate the model
    '''
    # generate the columns used to the 
    cols = df.drop(drop_cols,axis=1).columns
    str_cols = ' + '.join(cols)
    str_cols
    
    # join our taget with our predictors str
    f = str(target)+'~'+str_cols
    
    #generate our linear model
    model = smf.ols(f,df).df
    model.summary()
    
    # Checking normality of our residule errors
    resids = model.resid
    sm.graphics.qqplot(resids,stats.norm,line='45',fit=True)
    
    return cols, model


def remove_dupes(df,col):
    x = df.shape[0]
    df.sort_values(col, inplace=True) 
    df.drop_duplicates(subset=col, 
                     keep='first', inplace=True)
    print(x-df.shape[0], 'duplicates removed. \n')
    return df


def percent_null_df(df):
    ''' Prints the percentage of null values in the entire dataframe.
    
    @params:
    df is a pd.DataFrame
    
    @output
    a float percenatge describing null values in the data frame
    '''
    x = len(df.isna().sum())/len(df)*100
    print(round(x, 3))
    
    
def percent_null_col(df):
    ''' Prints the percentage of null values in each column of a dataframe
    
    @params
    df is a pd.DataFrame
    x is a list of strings containing the column names for missing_data
    missing_data is a pd.Dataframe containing x columns
    columns is a list of strings that contain the names of the columns in df
    col is an instance of the list of strings columns
    icolumn_name is a string containing the name of the column
    imissing_data is the sum of null values in col
    imissing_in_percentage returns a percentage of null values in col as a float
    missing_data.loc[len(missing_data)] creates a row containing the column name and percent null
    
    @output
    a pd.DataFrame containing the names of each col in df and their percent null values
    '''
    
    x = ['column_name','missing_data', 'missing_in_percentage']
    missing_data = pd.DataFrame(columns=x)
    columns = df.columns
    for col in columns:
        icolumn_name = col
        imissing_data = df[col].isnull().sum()
        imissing_in_percentage = round((df[col].isnull().sum()/df[col].shape[0])*100, 3)
        missing_data.loc[len(missing_data)] = [icolumn_name, imissing_data, imissing_in_percentage]
    missing_data = missing_data.sort_values(by = 'missing_in_percentage', ascending=False)
    print(missing_data)

    
def df_snapshot(df):
    ''' Generates and prints quick descriptive stats for any pandas dataframe
    
    @params
    df is a pd.DataFrame
    
    
    @output
    a list of column names
    the shape of the df as a tuple
    the percentage of null values in the entire df as a float
    the percentage of null values per each column as a float
    the results .info()
    '''
    
    print('\n <----- Columns ----->')
    print(list(df.columns),'\n')
    print('\n <----- Shape ----->')
    print(df.shape)
    print('\n <----- Total Percentage of Missing Data ----->') 
    print(percent_null_df(df))
    print('\n <----- Percentage of Missing Data Per Column ----->') 
    print(percent_null_col(df))
    print('\n <----- Info ----->')
    print(df.info())

def find_nans(df):
    nan_cols = []
    columns = df.columns
    for col in columns:
        if df[col].isna().sum() > 0:
            nan_cols.append(col)
    for x in nan_cols:
        print(x, 'has ', df[x].isna().sum(), 'NaN values.')
        print(x, 'has ', len(list(df[x].unique())), 'unique values.')
        print(sorted(df[x].unique()), '\n')
        
def determine_dtype(df):
    col_list = list(df.columns)
    objs = []
    ints = []
    flts = []
    for column in df.columns:
        x = df[column].dtypes
        # print(df[column].dtypes)
        if x == object:
            objs.append(column)
        elif x == int:
            ints.append(column) 
        elif x == float:
            flts.append(column)
    print('Objects: \n', objs, '\n')
    print('Integers: \n', ints, '\n')
    print('Floats:\n', flts, '\n')
    
def make_ints(df,cols):
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna('0')
        df[col] = df[col].astype(int)
    return df