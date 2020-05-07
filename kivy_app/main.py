from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout

from kivy.uix.image import Image, AsyncImage

from kivy.uix.widget import Widget
from kivy.properties import StringProperty, ObjectProperty

from sudoku_funct import *
import numpy as np
from PIL import Image as ImgP
import shutil
import time
import cv2
from kivy.clock import Clock
from kivy.uix.image import Image
from kivy.graphics.texture import Texture

import matplotlib.pyplot as plt

Builder.load_file('test.kv')


class KivyCamera(Image):

    def __init__(self, **kwargs):
        super(KivyCamera, self).__init__(**kwargs)
        self.capture = None
        
    def start(self, capture, fps=30):
        self.capture = capture
        Clock.schedule_interval(self.update, 1.0 / fps)

    def stop(self):
        Clock.unschedule_interval(self.update)
        self.capture = None

    def update(self, dt):
        return_value, frame = self.capture.read()
        frame=self.get_cont(frame)
        #print(type(frame))
        if return_value:
            texture = self.texture
            w, h = frame.shape[1], frame.shape[0]
            if not texture or texture.width != w or texture.height != h:
                self.texture = texture = Texture.create(size=(w, h))
                texture.flip_vertical()
            texture.blit_buffer(frame.tobytes(), colorfmt='bgr')
            self.canvas.ask_update()
            
            
    def get_cont(self, img):
        #ret, frame = capture.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(5,5),0)
        thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
        new_img, contours, hier = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        if len(contours)>0:
            bigest_contour=contours[0]
        #cv2.drawContours(img, bigest_contour, -1, (0,255,0), 5)
            polygon = bigest_contour
            bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
            top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
            bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
            top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
            corners=polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]
            top_left, top_right, bottom_right, bottom_left = corners
       
            #a = [tuple(top_left), tuple(top_right), tuple(bottom_right), tuple(bottom_left), tuple(top_left)]
            #if np.sum(bottom_right-top_left)>(np.shape(img)[0]+np.shape(img)[1])/2:
            #if bottom_right[1]- top_right[1]> np.shape(img)[1]/4 and top_right[0]-top_left[0]>np.shape(img)[0]/4:
            right_side=bottom_right[1]- top_right[1]
            left_side=bottom_left[1]- top_left[1]
            
            top_side=top_right[0]-top_left[0]
            bottom_side=bottom_right[0]-bottom_left[0]
            
           # bottom_l2=bottom_right[0]- top_right[0]
           #print(bottom_l)
            #print(bottom_l2)
            #print('--------')
            
            ratio=1/45
            if np.abs(right_side-left_side)<np.shape(img)[1]*ratio and np.abs(right_side-top_side)<np.shape(img)[1]*ratio:
        
                img = cv2.line(img, tuple(top_left), tuple(top_right), (0,255,0), 5)
                img = cv2.line(img, tuple(top_right), tuple(bottom_right), (0,255,0), 5)
                img = cv2.line(img, tuple(bottom_right), tuple(bottom_left), (0,255,0), 5)
                img = cv2.line(img, tuple(bottom_left), tuple(top_left), (0,255,0), 5)
                
        return img

capture = None


class CameraClick(Screen):
    def __init__(self, **kwargs):
        super(CameraClick, self).__init__(**kwargs)
        global capture
        capture = cv2.VideoCapture(0)
        self.ids.qrcam.start(capture)
        
        
        
        
    def capture(self):
       ret, frame = capture.read()
       #cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)
       #cv2.imwrite('test_img.png', frame)
       settings = self.manager.get_screen("settings")
       #settings.callback_image('test_img.png')
       settings.callback_image2(frame)




class ImgDisplay(Image):

    def __init__(self, **kwargs):
        super(ImgDisplay, self).__init__(**kwargs)
        #self.capture = None
        

    def update(self, frame):
        return_value=True
        #frame=self.get_cont(frame)
        #print(type(frame))
        if return_value:
            texture = self.texture
            w, h = frame.shape[1], frame.shape[0]
            if not texture or texture.width != w or texture.height != h:
                self.texture = texture = Texture.create(size=(w, h))
                texture.flip_vertical()
            texture.blit_buffer(frame.tobytes(), colorfmt='bgr')
            self.canvas.ask_update()




class SettingsScreen(Screen):
    img = ObjectProperty(None)
    
    def get_cont2(self, img):
        #ret, frame = capture.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(5,5),0)
        thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
        new_img, contours, hier = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        if len(contours)>0:
            bigest_contour=contours[0]
        #cv2.drawContours(img, bigest_contour, -1, (0,255,0), 5)
            polygon = bigest_contour
            bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
            top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
            bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
            top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
            corners=polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]
            top_left, top_right, bottom_right, bottom_left = corners
            img = cv2.line(img, tuple(top_left), tuple(top_right), (0,255,0), 5)
            img = cv2.line(img, tuple(top_right), tuple(bottom_right), (0,255,0), 5)
            img = cv2.line(img, tuple(bottom_right), tuple(bottom_left), (0,255,0), 5)
            img = cv2.line(img, tuple(bottom_left), tuple(top_left), (0,255,0), 5)
                
        return img
        
    def callback_image2(self, img):
        sm.current = "settings"
        global img2
        img2=img.copy()
        
        img=self.get_cont2(img)
        self.ids.grid.update(img)
        #self.crop_grid_img(img2)

    def go_back(self):
        
        reco = self.manager.get_screen("camera")
        sm.current = "camera"
    
    
    def crop_grid_img(self):
       
        cropped_img=crop_grid(img2)
        #solved_grid=get_solved_grid_img('test_img.png')
        #im = ImgP.fromarray(cropped_img)
        #im.save("cropped_img.png")
        cropped_screen = self.manager.get_screen("crop")
        cropped_screen.goto_crop(cropped_img)
        


class CropScreen(Screen):
    img = ObjectProperty(None)
    
    
    def goto_crop(self, cropped_img):
        sm.current = "crop"
        #print(new_image_address)
        global cropped
        cropped=cropped_img.copy()
        #print(np.shape(cropped_img))
        cropped_img = cv2.resize(cropped_img, (700,700), interpolation=cv2.INTER_CUBIC)
        self.ids.crop.update(cropped_img)
        
        
        
    
    def go_to_solve(self):
       
        solved_img=get_solved_grid_img2(cropped)
        
        if solved_img is None:
            sm.current = "error_screen"
        else:
        #solved_grid=get_solved_grid_img('test_img.png')
            #im = ImgP.fromarray(solved_img)
            #im.save("solved_grid.png")
            solved_screen = self.manager.get_screen("solved")
            solved_screen.goto_solve(solved_img)
    
    
    def go_back(self):
        
        settings = self.manager.get_screen("settings")
        settings.go_back()


class SolvedScreen(Screen):
    img = ObjectProperty(None)
    
    def goto_solve(self, solved_img):
        sm.current = "solved"
        solved_img = cv2.resize(solved_img, (700,700), interpolation=cv2.INTER_CUBIC)
        self.ids.solved.update(solved_img)

    def select_new(self):
        
        crop = self.manager.get_screen("crop")
        crop.go_back()
        #sm.current = "menu"

class NoSolScreen(Screen):
     
    def select_new(self):
       
        crop = self.manager.get_screen("crop")
        crop.go_back()
    
# Create the screen manager
sm = ScreenManager()
sm.add_widget(CameraClick(name='camera'))
#sm.add_widget(MenuScreen(name='menu'))
sm.add_widget(CropScreen(name='crop'))
#sm.add_widget(SelectScreen(name='select'))
sm.add_widget(SettingsScreen(name='settings'))
sm.add_widget(SolvedScreen(name='solved'))
sm.add_widget(NoSolScreen(name='error_screen'))


class TestApp(App):

    def build(self):
        return sm
        
        #self.manager.current = 'crop'
        
if __name__ == '__main__':
    TestApp().run()