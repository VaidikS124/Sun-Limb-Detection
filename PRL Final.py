#!/usr/bin/env python
# coding: utf-8

# In[14]:


import astropy.io
import cv2  
import numpy as np  
import matplotlib.pyplot as plt  
file_path = 'C:\\Users\\Vaidik Sharma\\Downloads\\SIP_USO_PRL\\SIP_USO_PRL\\UDAI.FDGB.03062019.080021.864.fits'
hdul = fits.open(file_path)
imgd = hdul[0].data  
N_data = (imgd - np.min(imgd)) / (np.max(imgd) - np.min(imgd)) * 255
N_data = N_data.astype(np.uint8)
B_data = cv2.GaussianBlur(N_data, (5, 5), 0)
edges = cv2.Canny(B_data, threshold1=30, threshold2=100) 
cont, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if len(cont) > 0:
    l_cont = max(cont, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(l_cont)
    center = (int(x), int(y))
    radius = int(radius)
    s_edge = cv2.cvtColor(N_data, cv2.COLOR_GRAY2BGR)
    cv2.circle(s_edge, center, radius, (0, 255, 0), 2)  # Green circle
    plt.figure(figsize=(8, 8))
    plt.imshow(s_edge)
    plt.title('Sun Edge')
    plt.show()
    print(f"Center Coordinates (x, y): {center}")
    print(f"Radius: {radius}")
else:
    print("No edge found")


# In[15]:


import astropy.io 
import cv2  
import numpy as np 
import matplotlib.pyplot as plt  

file_path = 'C:\\Users\\Vaidik Sharma\\Downloads\\SIP_USO_PRL\\SIP_USO_PRL\\UDAI.FDGB.03062019.080021.864.fits'
hdul = fits.open(file_path)
img1 = hdul[0].data 
N_data = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
B_data = cv2.GaussianBlur(N_data, (5, 5), 0)
edges = cv2.Canny(B_data, threshold1=30, threshold2=100)  
b_mask = np.zeros_like(edges) 
b_mask[edges > 0] = 1
plt.figure(figsize=(12, 6)) 
plt.subplot(1, 2, 1) 
plt.imshow(N_data) 
plt.title('Original Image') 

plt.subplot(1, 2, 2)  
plt.imshow(b_mask, cmap='gray') 
plt.title('Binary Mask for Sunspots') 
plt.tight_layout() 
plt.show() 


# In[ ]:




