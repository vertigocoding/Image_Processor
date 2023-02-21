# Image_Processor
The process of working through line detection in openCV. Both using provided methods and custom kernels to both understand how the process works and to be able to compare the results of each method and style

## Edge Detection 
### Grey Scale
Simple conversion to Grey Scale. 8 bit colour 

### Smoothing 
usinga a kernel to smooth out the image reducing noise giving a general blur


### Line detection 
Process the image with line detectors to find veritcal and horizontal lines seperately.
    -1 0 1       1 2 1
Kx =-2 0 2 ,Ky = 0 0 0
    -1 0 1      -1-2-1
    
### Combination
Combine the vertical and horizontal images together and calculate their intensity and angle. 
Sqrt(Ximage^2, YImage^2)

Angle(x,y) = arctan(Iy/Ix) 

### Non maximum supression
Thin out thicker lines

### Thresholding
Single or Double
