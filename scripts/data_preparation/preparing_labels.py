#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from osgeo import gdal
import os


# In[2]:


file = '../../data/clean/train/kam/labels/kampala_labels.tif'


# In[3]:


image = gdal.Open(file)


# In[ ]:


x = image.ReadAsArray()


# In[ ]:


print('Shape of labels tif image:')
print(x.shape)


# In[ ]:


def segment_image(image, width):
    size_x, size_y = image.shape
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


# In[ ]:


images = segment_image(x, 128)


# In[ ]:


print('Number of label images after cropping:')
print(len(images))


# In[ ]:


path = '../../data/clean/train/kam/labels/'
prefix = 'kam_label'
extension = '.npy'
files = os.listdir(path)

for file in files:
    if file.endswith(extension):
        os.remove(os.path.join(path, file))

for i, image in enumerate(images):

    np.save(path + prefix + '_' + str(i) + extension, image)
