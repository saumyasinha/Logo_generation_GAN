import numpy as np

def rgb2lab ( inputColor ) :

   num = 0
   RGB = [0, 0, 0]

   for value in inputColor :
       value = float(value) / 255

       if value > 0.04045 :
           value = ( ( value + 0.055 ) / 1.055 ) ** 2.4
       else :
           value = value / 12.92

       RGB[num] = value * 100
       num = num + 1

   XYZ = [0, 0, 0,]

   X = RGB [0] * 0.4124 + RGB [1] * 0.3576 + RGB [2] * 0.1805
   Y = RGB [0] * 0.2126 + RGB [1] * 0.7152 + RGB [2] * 0.0722
   Z = RGB [0] * 0.0193 + RGB [1] * 0.1192 + RGB [2] * 0.9505
   XYZ[ 0 ] = round( X, 4 )
   XYZ[ 1 ] = round( Y, 4 )
   XYZ[ 2 ] = round( Z, 4 )

   XYZ[ 0 ] = float( XYZ[ 0 ] ) / 95.047         # ref_X =  95.047   Observer= 2°, Illuminant= D65
   XYZ[ 1 ] = float( XYZ[ 1 ] ) / 100.0          # ref_Y = 100.000
   XYZ[ 2 ] = float( XYZ[ 2 ] ) / 108.883        # ref_Z = 108.883

   num = 0
   for value in XYZ :

       if value > 0.008856 :
           value = value ** ( 0.3333333333333333 )
       else :
           value = ( 7.787 * value ) + ( 16 / 116 )

       XYZ[num] = value
       num = num + 1

   Lab = [0, 0, 0]

   L = ( 116 * XYZ[ 1 ] ) - 16
   a = 500 * ( XYZ[ 0 ] - XYZ[ 1 ] )
   b = 200 * ( XYZ[ 1 ] - XYZ[ 2 ] )

   Lab [ 0 ] = round( L, 4 )
   Lab [ 1 ] = round( a, 4 )
   Lab [ 2 ] = round( b, 4 )

   return tuple(Lab)

def classify(rgb):
    # eg. rgb_tuple = (2,44,300)
    rgb_tuple = tuple(rgb)

    # add as many colors as appropriate here, but for
    # the stated use case you just want to see if your
    # pixel is 'more red' or 'more green'
    colors = {'red': rgb2lab((255,0,0)),
          'green': rgb2lab((0,255,0)),
          'blue': rgb2lab((0,0,255)),
          'yellow': rgb2lab((255,255,0)),
          'orange': rgb2lab((255,127,0)),
          'white': rgb2lab((255,255,255)),
          'black': rgb2lab((0,0,0)),
          'pink': rgb2lab((255,127,127)),
          'purple': rgb2lab((127,0,255))}

    manhattan = lambda x,y : abs(x[0] - y[0]) + abs(x[1] - y[1]) + abs(x[2] - y[2])
    distances = {k: manhattan(v, rgb_tuple) for k, v in colors.items()}
    color = min(distances, key=distances.get)
    return color


if __name__ == "__main__":
    print(classify(rgb2lab((244, 241, 65))))
    X = np.load('/Users/shivendra/Desktop/CU/HCML/Logo_generation_GAN/icon_dataset.npy')
    X = X[:10]
    average_color = np.average(X, axis=(1,2))
    color_labels = []
    for rgb in average_color:
        color_labels.append(classify(rgb2lab(rgb)))
    # print(color_labels)
    color_labels = np.array(color_labels)
    print(color_labels.shape)
    np.save('icon_color_label.npy', color_labels)
