#resize image set to x y values and save to enother dir. 

from PIL import Image
import os, sys

path = "C:\\Users\\moroz\\Documents\\Projects\\Pytorch Classifier\data\\plants\\dataset\\images\\lab\\"
dirs = os.listdir( path )

def resize():
    for item in dirs:
        #print(item)
        for folder in item:
            
            if os.path.isfile(path+folder):
                im = Image.open(path+folder)
                f, e = os.path.splitext(path+folder)
                imResize = im.resize((600,500), Image.ANTIALIAS)
                imResize.save(f + ' resized.jpg', 'JPEG', quality=100)

resize()