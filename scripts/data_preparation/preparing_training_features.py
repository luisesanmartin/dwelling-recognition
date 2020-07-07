#!/usr/bin/env python
# coding: utf-8


import numpy as np
from osgeo import gdal


# In[4]:


file = '../../data/raw/train_tier_1/kam/4e7c7f/4e7c7f.tif'


# In[6]:


image = gdal.Open(file)


# In[8]:


x = image.ReadAsArray()
x = np.moveaxis(x, 0, -1)


# In[9]:


print('Shape of training features tif:')
print(x.shape)


# In[14]:


def segment_image(image, width):
    size_x, size_y, _ = image.shape
    x = 0
    y = 0
    rv = []

    while x + width <= size_x:

        while y + width <= size_y:

            sub_image = image[x:x+width, y:y+width]
            rv.append(sub_image)
            y += width

        y = 0
        x += width

    return rv


# In[41]:


images = segment_image(x, 512)


# In[42]:


print('Number of features cropped images:')
print(len(images))


# In[47]:


path = '../../data/clean/train/kam/features/'
prefix = 'kam_train'

for i, image in enumerate(images):

    np.save(path + prefix + '_' + str(i) + '.npy', np.moveaxis(image, -1, 0))
