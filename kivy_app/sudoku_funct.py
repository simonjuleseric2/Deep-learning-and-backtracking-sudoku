
import numpy as np
import operator
import numpy as np
import keras
from keras.models import load_model
from PIL import Image, ImageFont, ImageDraw
import cv2


#import matplotlib.pyplot as plt

#%matplotlib inline


#reagange list in list of list for  each row:
def rearange_list(nlist):
    dlist=[]
    for i in range(0, 9):
        k=9*i
        l=k+9
        sub_list=nlist[k:l]
        dlist.append(sub_list)
    return np.matrix(dlist)


#check if a value is a possible fix for an empty slot
def verify_possile_value(x, y, n, grid):
    for i in range(0, 9):
        if grid[x, i]==n:
            return False
        
    for i in range(0, 9):
        if grid[i, y]==n:
            return False
    x0=(x//3)*3
    y0=(y//3)*3
    for i in range(0, 3):
        for j in range(0, 3):
            if grid[x0+i, y0+j]==n:
                return False
            
    return True



#fill in the empty slots:
def grid_solver(grid):
    for x in range(9):
        for y in range(9):
            if grid[x, y]==0:
                for n in range(1, 10):
                    if verify_possile_value(x, y, n, grid):
                        grid[x, y]=n
                        result=grid_solver(grid)
                        
                        if result is not None:
                            return result
                        
                        grid[x, y]=0 #Backtracking
                return None               
    return grid


        

def extract_cells_coordinates(square_img):
    cells_coodinates = np.zeros((81, 4), dtype=float)
    grid_side = square_img.shape[:1]
    cell_side = grid_side[0]/9
    k=0
    for i in range(9):
        for j in range(9):
            x1, y1 = (i * cell_side, j * cell_side) 
            x2, y2 = ((i + 1) * cell_side, (j + 1) * cell_side)
            cells_coodinates[k, :]=[x1, y1, x2, y2]
            k=k+1
    return cells_coodinates


def extract_clean_digit_area(img, coord):
    x1, y1, x2, y2=coord
    dig=img[int(x1):int(x2), int(y1):int(y2)]
    #dig=dig[:, :, 1]
    dig=dig[:, :]
    dig3=cv2.resize(dig, (45, 45), interpolation=cv2.INTER_CUBIC)
    digit=dig3/np.max(dig3)
    digit=np.expand_dims(digit, axis=-1)
    return digit


def measure_distance(pt1, pt2):
    x = pt2[0] - pt1[0]
    y = pt2[1] - pt1[1]
    return np.sqrt((x ** 2) + (y ** 2))
    
    

def create_digit_img(digit, side):
    background =Image.new('RGB', (side, side), color = (255, 255, 255))
    font = ImageFont.truetype('/Library/Fonts/Arial.ttf', 20)
    text=str(digit)
    draw = ImageDraw.Draw(background)
    if digit>0:
        text_width, text_height = draw.textsize(text, font)
        position = ((side-text_width)/2,(side-text_height)/2)
        color=(0, 0, 0)
        draw.text(position, text, color, font=font)
    img=np.array(background)
    img=0.2989*img[:, :, 0]+0.5870*img[:, :, 1]+0.1140*img[:, :, 2]
    
    return img

def draw_lines(img, side):
    
    ll=np.shape(img)[0]
    
    for i in range(0, 10):
        
        if i in [0, 3, 6]:
            img[side*i+1, 0:ll]=0
            img[0:ll, side*i+1]=0
        elif i==9:
            img[side*i-1, 0:ll]=0
            img[0:ll, side*i-1]=0
            
        img[side*i, 0:ll]=0
        img[0:ll, side*i]=0
    return img
    

def get_reconstructed_grid(matrix_grid):
    side=50 #Size of one cell

    backgr=np.ones((9*side+1, 9*side+1))*255

    #put digits:
    for i in range(0, 9):
        for j in range(0, 9):
        
            digit=matrix_cell[i, j]
            img=create_digit_img(digit, side)
            backgr[i*side:(i+1)*side, j*side:(j+1)*side]=img


    reconstructed=draw_lines(backgr, side)
    return reconstructed

  
    
    
def crop_grid(sudoku_img):
    #sudoku_img = cv2.imread(image_path)
    gray = cv2.cvtColor(sudoku_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    new_img, contours, hier = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    bigest_contour=contours[0]
    polygon = bigest_contour
    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))

    corners=polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]
    top_left, top_right, bottom_right, bottom_left = corners
    
    #Crop image
    src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')

    # Get the longest side in the rectangle
    side = max([
    measure_distance(bottom_right, top_right),
    measure_distance(top_left, bottom_left),
    measure_distance(bottom_right, bottom_left),
    measure_distance(top_left, top_right)
    ])

    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')
    m = cv2.getPerspectiveTransform(src, dst)

    cropped=cv2.warpPerspective(sudoku_img, m, (int(side), int(side)))
    
    
    return cropped



def get_digit(img_path):
    cropped=crop_grid(img_path)
    cells_coor=extract_cells_coordinates(cropped)
    total_dig=np.zeros((81, 45, 45, 1), dtype=float)#raw images
    
    for i in range(0, np.shape(cells_coor)[0]):
        digit=extract_clean_digit_area(cropped, cells_coor[i, :])
        total_dig[i, :, :, :]=digit
    
    model_path='models/digit_recognition_model_9951.h5'
    model = load_model(model_path)
    pred=model.predict(total_dig)
    list_values=list(pred.argmax(1))
    matrix_cell=rearange_list(list_values)
    
    return (matrix_cell)



def create_digit_img2(digit, side, original):
    background =Image.new('RGB', (side, side), color = (255, 255, 255))
    font = ImageFont.truetype('/Library/Fonts/Arial.ttf', 28)
    text=str(digit)
    draw = ImageDraw.Draw(background)
    #if digit>0:
    text_width, text_height = draw.textsize(text, font)
    position = ((side-text_width)/2,(side-text_height)/2)
        
    if original is True:
        color=(0, 0, 0)
            
    else:
        color=(0, 0, 255)
            
    draw.text(position, text, color, font=font)
        
    img=np.array(background)
    #img=0.2989*img[:, :, 0]+0.5870*img[:, :, 1]+0.1140*img[:, :, 2]
    
    return img

def draw_lines2(img, side):
    
    ll=np.shape(img)[0]
    
    for i in range(0, 10):
        
        if i in [0, 3, 6]:
            img[side*i+1, 0:ll, :]=(0, 0, 0)
            img[0:ll, side*i+1, :]=(0, 0, 0)
        elif i==9:
            img[side*i-1, 0:ll, :]=(0, 0, 0)
            img[0:ll, side*i-1, :]=(0, 0, 0)
            
        img[side*i, 0:ll, :]=(0, 0, 0)
        img[0:ll, side*i, :]=(0, 0, 0)
    return img
    
    
    
def get_solved_grid_img2(sudoku_img):
    #sudoku_img = cv2.imread(grid_img_path)
    gray = cv2.cvtColor(sudoku_img, cv2.COLOR_BGR2GRAY)
    cells_coor=extract_cells_coordinates(gray)
    total_dig=np.zeros((81, 45, 45, 1), dtype=float)#raw images
    
    for i in range(0, np.shape(cells_coor)[0]):
        digit=extract_clean_digit_area(gray, cells_coor[i, :])
        total_dig[i, :, :, :]=digit
    
    model_path='models/digit_recognition_model_9951.h5'
    model = load_model(model_path)
    pred=model.predict(total_dig)
    list_values=list(pred.argmax(1))
    original_grid=rearange_list(list_values)
    
    
    #croped_grid=get_digit(grid_img_path)
    original_grid2=original_grid.copy()
    print(original_grid2)
    solution=grid_solver(original_grid)
    
    print(original_grid2)
    side=50
    backgr=np.ones((9*side+1, 9*side+1, 3))*255
    
    print('------------')
    print(solution)
    print('------------')

    if solution is None:  # No solution found
        solved=None
    
    else:
        for i in range(0, 9):
            for j in range(0, 9):
                digit=original_grid2[i, j]
            #print(digit)
                if digit>0:
                    original=True
                else:
                    original=False
                img=create_digit_img2(solution[i, j], side, original)
        
                backgr[i*side:(i+1)*side, j*side:(j+1)*side]=img

        backgr=draw_lines2(backgr, side)
        solved=backgr.astype(np.uint8)
        
    return solved
  
    
    
def get_solved_grid_img(grid_img_path):
    original_grid=get_digit(grid_img_path)
    original_grid2=original_grid.copy()
    print(original_grid2)
    solution=grid_solver(original_grid)
    
    print(original_grid2)
    side=50
    backgr=np.ones((9*side+1, 9*side+1, 3))*255
    
    print('------------')
    print(solution)
    print('------------')
    #iput digits:
    for i in range(0, 9):
        for j in range(0, 9):
        
        
            digit=original_grid2[i, j]
            #print(digit)
            if digit>0:
                original=True
            else:
                original=False
            #print(original)
            #print(solution[i, j])
            img=create_digit_img2(solution[i, j], side, original)
        
            backgr[i*side:(i+1)*side, j*side:(j+1)*side]=img

    #data=backgr.astype(np.uint8)

    backgr=draw_lines2(backgr, side)
    solved=backgr.astype(np.uint8)
    
    
    return solved
