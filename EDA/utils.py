import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class UnivariateAnalysis:
    def __init__(self, data): 
        '''
        Constructor method to initialize the UnivariateAnalysis object.

        Parameters:
        - data: DataFrame, the input data for analysis.
        '''
        self.data = data

    def visualize(self, x = str, width=20, height=8, rotate = None, create_other=False):
        '''
        Visualize the distribution of a categorical variable.

        Parameters:
        - x: str, the name of the categorical column to visualize.
        - width: int, width of the figure.
        - height: int, height of the figure.
        - rotate: int, rotation angle for x-axis labels.
        - create_other: bool, whether to combine small categories into an 'other' category.
        '''
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width, height))
        self.cnt = self.data[x].value_counts()
        total_count = self.cnt.sum()

        # Create 'other' category if specified
        if create_other == True:
          threshold_percentage = 2
          mask = self.cnt / total_count * 100 < threshold_percentage
          other_count = self.cnt[mask].sum()
          self.cnt = self.cnt[~mask]
          self.cnt['other'] = other_count

        # Plot pie chart
        colors = sns.color_palette("Reds", len(self.cnt))
        ax1.pie(self.cnt, autopct='%.2f%%', labels=self.cnt.index, colors=colors)

        #Plot bar chart
        ax2.bar(self.cnt.index, self.cnt.values, color= colors)

        # Customize x-axis labels rotation
        if rotate != None:
            for tick in ax2.get_xticklabels():
                tick.set_rotation(rotate)

        # Adjust x-axis ticks for binary variables
        if self.data[x].nunique() == 2:
            if 0 in self.data[x].unique() and 1 in self.data[x].unique():
                ax2.set_xticks([0,1])
                ax2.set_xticklabels(self.cnt.index)
        plt.suptitle(f'Distribution of {x}')
        plt.show()

    def visualize_numeric(self, x = str, width=20, height=10, kde=False, common_bins=False):
        '''
        Visualize the distribution of a numeric variable.

        Parameters:
        - x: str, the name of the numeric column to visualize.
        - width: int, width of the figure.
        - height: int, height of the figure.
        - kde: bool, whether to plot kernel density estimation.
        - common_bins: bool, whether to use common bin edges in histograms.
        '''
        print(self.data[x].describe())
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(width, height))
        sns.histplot(data=self.data, x=x, ax=ax1, kde=kde, common_bins=common_bins, color='#d62728')
        sns.boxplot(data=self.data, x=x, ax=ax2, color ='#d62728')
        fig.suptitle(f'Distribution of {x}')
        plt.show()


class StatisticAnalysis:
    def __init__(self, data):
        '''
        Constructor method to initialize the StatisticAnalysis object.

        Parameters:
        - data: DataFrame, the input data for analysis.
        '''
        self.data = data

    def check_null(self, width=12, height=5):
        '''
        Check and visualize missing values in the dataset.

        Parameters:
        - width: int, width of the bar plot.
        - height: int, height of the bar plot.
        '''
        # Display the count of missing values for each column
        print(self.data.isnull().sum())

        # Calculate the percentage of missing values for each column
        null_df = self.data.isnull().sum() / self.data.shape[0] * 100
        null_df = null_df[null_df != 0].sort_values(ascending=False).reset_index()
        null_df.columns = ['feature', 'null_percentage']

        # Create a bar plot showing the percentage of missing values per column
        plt.figure(figsize=(width, height))
        sns.barplot(x=null_df['null_percentage'], y=null_df['feature'], palette='coolwarm')
        plt.title('Null percentage per column', fontsize=20)
        plt.show()

    def correlation(self, width=12, height=10, drop_cols=None):
        '''
        Visualize the correlation matrix for numeric columns in the dataset.

        Parameters:
        - width: int, width of the heatmap.
        - height: int, height of the heatmap.
        - drop_cols: list, columns to be excluded from the correlation analysis.
        '''
        # Extract numeric columns from the dataset
        num_df = self.data._get_numeric_data()

        # Drop specified columns from the numeric dataframe
        if drop_cols is not None:
            num_df.drop(columns=drop_cols, inplace=True)

        # Calculate the correlation matrix
        corr_matrix = num_df.corr()

        # Create a heatmap of the correlation matrix
        plt.figure(figsize=(width, height))
        sns.heatmap(corr_matrix, cmap='coolwarm', annot=True,
                    fmt=".2f", linewidths=0.5,
                    mask=np.triu(np.ones_like(corr_matrix, dtype=bool)))
        plt.title('Correlation Heatmap')
        plt.show()


class BivariateAnalysis:
    def __init__(self):
        pass

    def scatter_plot(self, df1 = pd.DataFrame,
                     df2=pd.DataFrame, name_x= str, name_y=str,
                     width=12, height=6):
        '''
        Create a scatter plot for two dataframes based on specified columns.

        Parameters:
        - df1: DataFrame, data for non-defaulters (target = 0).
        - df2: DataFrame, data for defaulters (target = 1).
        - name_x: str, the column name for the x-axis.
        - name_y: str, the column name for the y-axis.
        - width: int, width of the figure.
        - height: int, height of the figure.
        '''
        fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(width, height))
        for ax in (ax1, ax2):
            ax.set_xlabel(name_x)
            ax.set_ylabel(name_y)

        ax1.scatter(df1[name_x], df1[name_y])
        ax1.set_title('Non-defaulter')
        ax1.set_xlim(list(ax1.get_xlim()))
        ax1.set_xticks([x+50000 for x in range(int(ax1.get_xlim()[0]), int(ax1.get_xlim()[1]), 50000)])
        ax1.set_xticklabels([str(x//1000) + 'k' for x in ax1.get_xticks()])

        ax2.scatter(df2[name_x], df2[name_y],color= 'red')
        ax2.set_title('Defaulter')
        ax2.set_xlim(list(ax2.get_xlim()))
        ax2.set_xticks([x+50000 for x in range(int(ax2.get_xlim()[0]), int(ax2.get_xlim()[1]), 50000)])
        ax2.set_xticklabels([str(x//1000) + 'k' for x in ax2.get_xticks()]);

        plt.show()

    def hist_plot(self, x=str, y = None, df1=pd.DataFrame, df2=pd.DataFrame, width=12, height=6, bins='auto', kde=False,
                  color='#eb0524'):
        '''
        Create histogram plots for non-defaulters and defaulters.

        Parameters:
        - x: str, the column name for the x-axis.
        - y: str, the column name for the y-axis.
        - df1: DataFrame, data for non-defaulters.
        - df2: DataFrame, data for defaulters.
        - width: int, width of the figure.
        - height: int, height of the figure.
        - bins: int or str, number of bins or binning strategy.
        - kde: bool, whether to plot kernel density estimation.
        - color: str, color of the histograms.
        '''
        fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(width, height))
        sns.histplot(df1, x=x, y=y, ax=ax1, bins=bins, kde=kde, color=color)
        ax1.set_title('Non-defaulter')
        sns.histplot(df2, x=x, y=y, ax=ax2, bins=bins, kde=kde, color=color)
        ax2.set_title('Defaulter')
        plt.show()

    def bar_plot(self, x=str, df1=pd.DataFrame, df2=pd.DataFrame, width=12, height=6, rotation=None):
        '''
        Create bar plots for non-defaulters and defaulters.

        Parameters:
        - x: str, the column name for the x-axis.
        - df1: DataFrame, data for non-defaulters.
        - df2: DataFrame, data for defaulters.
        - width: int, width of the figure.
        - height: int, height of the figure.
        - rotation: int, rotation angle for x-axis labels.
        '''
        fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(width, height))
        nd = df1[x].value_counts().sort_values()
        d = df2[x].value_counts().sort_values()

        ax1.bar(nd.index, nd.values, color = 'grey')
        ax1.set_title('Non-defaulter')

        ax2.bar(d.index, d.values, color='red')
        ax2.set_title('Defaulter')

        if rotation != None:
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=rotation)
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=rotation)

        plt.show()

   
    def box_plot(self, x='TARGET', y=str, data_to_plot=pd.DataFrame):
      '''
        Create a box plot for a specified column and target variable.

        Parameters:
        - x: str, the target variable column name.
        - y: str, the column name for the y-axis.
        - data_to_plot: DataFrame, data for plotting.
      '''
      sns.boxplot(x='TARGET', y=y, data=data_to_plot, palette='Reds')
      plt.title("Box-Plot of {}".format(y))
      plt.show()

    def pie_plot(self, x=str, y=None, df1=pd.DataFrame,df2=pd.DataFrame,
                 width=12, height=6):
        '''
        Create pie charts for non-defaulters and defaulters.

        Parameters:
        - x: str, the column name for creating the pie chart.
        - y: None, not used in this method.
        - df1: DataFrame, data for non-defaulters.
        - df2: DataFrame, data for defaulters.
        - width: int, width of the figure.
        - height: int, height of the figure.
        '''

        fig, (ax1, ax2) = plt.subplots(1,2, figsize= (width,height))
        cnt1 = df1[x].value_counts()
        colors1 = sns.color_palette("Reds", len(cnt1))
        ax1.pie(cnt1, autopct='%.2f%%', labels=cnt1.index, colors=colors1)
        ax1.set_title('Non-defaulter')

        cnt2 = df2[x].value_counts()
        colors2 = sns.color_palette("Reds", len(cnt2))
        ax2.pie(cnt2, autopct='%.2f%%', labels=cnt2.index, colors=colors2)
        ax2.set_title('Defaulter')


    def percentage_of_defauter_per_cat(self, df=pd.DataFrame, col_name=str):
        '''
        Calculate and display the percentage of defaulters for each category.

        Parameters:
        - df: DataFrame, the input data.
        - col_name: str, the column for which to calculate the percentage.
        '''
        
        summary = []
        for cat in df[col_name].unique():
            default_count = df[(df[col_name] == cat) & (df.TARGET == 1)].shape[0]
            total_count = df[df[col_name] == cat].shape[0]
            if total_count == 0:
                pass
            else:
                summary.append([cat ,default_count * 100 / total_count])

        report_df = pd.DataFrame(summary)
        report_df.columns = ["Categories", "Percentage_Of_Default"]
        report_df.sort_values(by='Percentage_Of_Default', ascending=False, inplace=True)

        sns.barplot(report_df, x='Percentage_Of_Default', y='Categories', palette='coolwarm')
        plt.show()