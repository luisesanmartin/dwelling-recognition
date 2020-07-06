#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image
import numpy as np
from osgeo import gdal


# In[2]:


file = '../../data/clean/train/kam/labels/kampala_labels.tif'


# In[3]:


image = gdal.Open(file)


# In[ ]:


x = image.ReadAsArray()


# In[ ]:


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


images = segment_image(x, 512)


# In[ ]:


print(len(images))


# In[ ]:


path = '../../data/clean/train/kam/labels/'
prefix = 'kam_label'

for i, image in enumerate(images):

    np.save(path + prefix + '_' + str(i) + '.npy', np.moveaxis(image, -1, 0))
