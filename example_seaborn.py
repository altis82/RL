# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 14:04:51 2022

@author: Chuan Pham
"""

#example with seaborn 
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

#create data 
data = np.array([[1, 12, 'Monday','Sunny'], 
                 [2, 13, 'Tuesday','Sunny'],
                 [3, 16, 'Wednesday','Cloudy'], 
                 [4, 13, 'Thursday','Rainy'], 
                 [5, 10, 'Friday','Rainy'], 
                 [6, 15, 'Saturday','Sunny'], 
                 [7, 12, 'Sunday','Cloudy'] 
                 ])
# Creating a data frame with the raw data
dataset = pd.DataFrame(data, columns=['id', 'temperature', 'Day','Weather'])

print(data)
print(dataset)
    
def draw_data(palette_name, custom_colors):
    """
    Parameters
    ----------
    palette_name : String
        for example palette_name="paired"
    custom_colors : Array of colors
        for example custom_colors=["#FF0B04", "#4374B3"]

    
    -------
    None.

    """
    #using color palette
    #sn.set_palette(sn.color_palette("Paired"))
    if len(palette_name)>0:
        sn.set_palette(sn.color_palette(palette_name))
    else:
        sn.set_palette(sn.color_palette(custom_colors))
    # Plot the data, specifying a different color for data points in
    # each of the day weather
    ax = sn.scatterplot(x='id', y='temperature', data=dataset, hue='Weather')
    
    # Customize the axes and title
    ax.set_title("Temperature")
    ax.set_xlabel("Day")
    ax.set_ylabel("Average Temperature")
    
    #  top and right borders
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    plt.show()

draw_data("Paired",[])
draw_data("", ["#FF0B04", "#4374B3","yellow"])
