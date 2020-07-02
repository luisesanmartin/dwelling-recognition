#!/usr/bin/env python
# coding: utf-8

# In[1]:


import geopandas as gpd


# In[2]:


file = '../../data/raw/train_tier_1/kam/4e7c7f-labels/4e7c7f.geojson'
gdf = gpd.read_file(file)


# In[3]:


gdf = gdf.to_crs(epsg=32636)


# In[4]:


gdf.head()


# In[5]:


gdf['is_building'] = 1


# In[6]:


output = '../../data/clean/train/kam/labels/kampala_buildings32636.shp'


# In[7]:


gdf.to_file(output)

