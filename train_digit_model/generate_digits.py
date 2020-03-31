from PIL import Image, ImageDraw, ImageFont
from sklearn.model_selection import train_test_split
import numpy as np
import cv2



def elastic_distortion(image, alpha, sigma, random_state=None): #Create an elastic distortion of the image to account for camera iregularities
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = np.zeros_like(dx)

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
    distored_image = map_coordinates(image, indices, order=1, mode='reflect')
    return distored_image.reshape(image.shape)

    
def add_noise(img): #Add noise to the image
    row, col=np.shape(img)
    mean = 0
    var = 2
    sigma = var**0.3
    gauss = np.array((row, col))
    gauss = np.random.normal(mean,sigma,(row,col))
    gauss = gauss.reshape(row,col)
    noisy = img + gauss
    return noisy.astype('uint8')

def add_side_bars(img):
    
    for i in range(0, 4): #sides bars 4 pixel wide
        img[i, :]=np.random.randint(0, 255)
        img[-i, :]=np.random.randint(0, 255)
        img[:, i]=np.random.randint(0, 255)
        img[:, -i]=np.random.randint(0, 255)
    return img

def get_image_part(img, cell_size):
    x=np.random.randint(19, 40)
    y=np.random.randint(19, 40)
    img=img[x:x+cell_size, y:y+cell_size]
    return img


def create_data(length, list_of_font, threshold=True, consider_empty=True):
    cell_size=45
    y_label=np.zeros(length, dtype=int)
    
    x_img=np.zeros((length, cell_size, cell_size), dtype=float)
    for i in range(0, length):
        
        if consider_empty:
            y=np.random.randint(0, 9+1)
        
        else:
            y=np.random.randint(1, 9+1)
        
        y_label[i]=y
        
        strip_width, strip_height = 100, 100

        background =Image.new('RGB', (strip_width, strip_height), color = (np.random.randint(80, 255), np.random.randint(80, 255), np.random.randint(80, 255)))
        font1=list_of_font[np.random.randint(0, len(list_of_font))]
        font = ImageFont.truetype(font1, 40)
        draw = ImageDraw.Draw(background)
        text_width, text_height = draw.textsize(text, font)
        position = ((strip_width-text_width)/2,(strip_height-text_height)/2)
        color=(np.random.randint(0, 10), np.random.randint(0, 10), np.random.randint(0, 10))
        
        if y>0: #If label is greater than 0, draw number
            draw.text(position, str(y), color, font=font)
        #add bars of differents width around the cell:
        img=np.array(background)
        img=0.2989*img[:, :, 0]+0.5870*img[:, :, 1]+0.1140*img[:, :, 2]
        bar_w=np.random.randint(1, 4)
        img[:, 28:28+bar_w]=1
        bar_w=np.random.randint(1, 4)
        img[:, 74:74+bar_w]=1
        bar_w=np.random.randint(1, 4)
        img[28:28+bar_w, :]=1
        bar_w=np.random.randint(1, 4)
        img[74:74+bar_w, :]=1
        
        #Get images of different centers around the original cells center
        img=get_image_part(img, cell_size)
        
        img=add_noise(img)
        img = cv2.GaussianBlur(img, (3,3), 0)
        img=elastic_distortion(img, np.random.randint(0, 100), np.random.randint(5, 10), random_state=None)
        x_img[i, :, :]=img
        
    return x_img, y_label
        

def create_image_data(length):

    list_of_font=['/Library/Fonts/Arial.ttf', '/Library/Fonts/Courier New.ttf', '/Library/Fonts/Times New Roman.ttf']
    #ll=80000
    xx, yy=generate_digit_img(length, list_of_font)
    return xx, yy

