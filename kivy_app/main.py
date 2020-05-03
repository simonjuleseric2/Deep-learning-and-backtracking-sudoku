from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout

from kivy.uix.image import Image, AsyncImage

from kivy.uix.widget import Widget
from kivy.properties import StringProperty, ObjectProperty

from sudoku_funct import *
import numpy as np
from PIL import Image
import shutil
import time
import cv2

Builder.load_file('test.kv')






class CameraClick(Screen):
    #def __init__(self,**kwargs):
    #    super().__init__()
    #    self.ids['rec'].play=True
    #def __init__(self, **kwargs):
     #   self.ids['camera'].play=True
        
    
    def capture(self):
       
        reco = self.ids['rec']
        
        #timestr = time.strftime("%Y%m%d_%H%M%S")
        #camera.export_as_image("test_img.jpg")
        reco.export_to_png("test_img.png")
        #print("Captured")
        
        #print(dir(camera))
        #print(camera.play)
        reco.play=False
        #print(dir(camera.play))
        settings = self.manager.get_screen("settings")
        settings.callback_image('test_img.png')
        #sm.current = "settings"


class MenuScreen(Screen):
    pass

        

class SettingsScreen(Screen):
    img = ObjectProperty(None)
    #################
    #def __init__(self, **kwargs):
    #        super(Screen, self).__init__(**kwargs)

    def callback_image(self, new_image_address):
        sm.current = "settings"
        print(self.img)
        print(new_image_address)
        print(self.ids.grid.source)
        self.img = new_image_address
        self.ids.grid.source = self.img
        
    def go_back(self):
        self.img = ''
        self.ids.grid.source =  self.img
        #print(dir(self.ids.grid))
        self.ids.grid.reload()
        
        reco = self.manager.get_screen("camera")
        #reco.play=True
        reco.ids['rec'].play=True
        sm.current = "camera"
    
    def crop_grid_img(self):
       
        cropped_img=crop_grid('test_img.png')
        #solved_grid=get_solved_grid_img('test_img.png')
        im = Image.fromarray(cropped_img)
        im.save("cropped_img.png")
        
        #solved_screen = self.manager.get_screen("solved")
        #solved_screen.goto_solve("solved_grid.jpg")
        cropped_screen = self.manager.get_screen("crop")
        cropped_screen.goto_crop("cropped_img.png")
        


class CropScreen(Screen):
    img = ObjectProperty(None)
    def goto_crop(self, new_image_address):
        sm.current = "crop"
        #print(new_image_address)
        self.img = new_image_address
        print(self.img)
        print(new_image_address)
        print(self.ids)
        self.ids.croped_img.source = self.img
    
    def go_to_solve(self):
       
        solved_img=get_solved_grid_img2("cropped_img.png")
        
        if solved_img is None:
            sm.current = "error_screen"
        else:
        #solved_grid=get_solved_grid_img('test_img.png')
            im = Image.fromarray(solved_img)
            im.save("solved_grid.png")
        
            solved_screen = self.manager.get_screen("solved")
            solved_screen.goto_solve("solved_grid.png")
    
    
    def go_back(self):
        self.img = ''
        self.ids.croped_img.source =  self.img
        #print(dir(self.ids.grid))
        self.ids.croped_img.reload()
        
        settings = self.manager.get_screen("settings")
        settings.go_back()


class SolvedScreen(Screen):
    img = ObjectProperty(None)
    
    def goto_solve(self, new_image_address):
        sm.current = "solved"
        print(new_image_address)
        self.img = new_image_address
        self.ids.solved_img.source = self.img

    def select_new(self):
        self.img = ''
        self.ids.solved_img.source =  self.img
        #print(dir(self.ids.grid))
        self.ids.solved_img.reload()
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
sm.add_widget(MenuScreen(name='menu'))
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