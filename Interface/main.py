import wx
import os
from inference import forgery_detector
import numpy as np
import cv2
from termcolor import colored

import wx.lib.agw.gradientbutton as GB

class MyFrame(wx.Frame):    
    def __init__(self):
        super().__init__(parent=None, title='Hello World')

        self.detector = forgery_detector()

        # Interface
        self.panel = wx.Panel(self)
        self.panel.SetBackgroundColour(wx.Colour(50, 51, 51))  # Recod.ai background black color

        self.my_sizer = wx.BoxSizer(wx.VERTICAL)

        # Load the image file
        head_path = os.path.join(os.path.dirname(__file__), "assets/Head-Interface-ImageForgery.png")
        image = wx.Image(head_path, wx.BITMAP_TYPE_ANY)
        #image = image.Scale(500, 100, wx.IMAGE_QUALITY_HIGH)
        bitmap = wx.Bitmap(image)
        self.head_static_bitmap = wx.StaticBitmap(self.panel, bitmap=bitmap)
        #self.panel.Bind(wx.EVT_SIZE, self.on_panel_size)
        #self.Bind(wx.EVT_SIZE, self.on_frame_size)

        self.my_sizer.Add(self.head_static_bitmap, 0, wx.ALL | wx.EXPAND, 5) 

        # Including original image, localization map and confidence map in the interface
        self.images_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.image_display = wx.StaticBitmap(self.panel)
        self.images_sizer.Add(self.image_display, flag=wx.ALL | wx.EXPAND, border=10)

        self.localization_map_display = wx.StaticBitmap(self.panel)
        self.localization_map_display.Show(False)
        self.images_sizer.Add(self.localization_map_display, flag=wx.ALL | wx.EXPAND, border=10)

        self.confidence_map_display = wx.StaticBitmap(self.panel)
        self.confidence_map_display.Show(False)
        self.images_sizer.Add(self.confidence_map_display, flag=wx.ALL | wx.EXPAND, border=10)

        self.my_sizer.Add(self.images_sizer, flag=wx.ALL | wx.CENTER, border=10)

        # Inserting probability text
        self.text_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.expert01_probability_text = wx.StaticText(self.panel, label="")
        self.expert01_probability_text.Show(False)
        font = wx.Font(17, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        self.expert01_probability_text.SetFont(font)
        self.expert01_probability_text.SetForegroundColour(wx.Colour(255, 255, 255))
        self.text_sizer.Add(self.expert01_probability_text, flag=wx.LEFT|wx.RIGHT|wx.EXPAND, border=20)

        self.expert02_probability_text = wx.StaticText(self.panel, label="")
        self.expert02_probability_text.Show(False)
        font = wx.Font(17, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        self.expert02_probability_text.SetFont(font)
        self.expert02_probability_text.SetForegroundColour(wx.Colour(255, 255, 255))
        self.text_sizer.Add(self.expert02_probability_text, flag=wx.LEFT|wx.RIGHT|wx.EXPAND, border=20)

        self.expert03_probability_text = wx.StaticText(self.panel, label="")
        self.expert03_probability_text.Show(False)
        font = wx.Font(17, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        self.expert03_probability_text.SetFont(font)
        self.expert03_probability_text.SetForegroundColour(wx.Colour(255, 255, 255))
        self.text_sizer.Add(self.expert03_probability_text, flag=wx.LEFT|wx.RIGHT|wx.EXPAND, border=20)



        self.my_sizer.Add(self.text_sizer, flag=wx.ALL | wx.CENTER, border=10)

        # Inserting buttons
        self.button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        button_font = wx.Font(13, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)

        self.upload_button = GB.GradientButton(self.panel, label="Upload Image")
        self.upload_button.SetTopStartColour(wx.Colour(0, 102, 102))  # Green 
        self.upload_button.SetTopEndColour(wx.Colour(0, 102, 102))   # Red
        self.upload_button.SetBottomStartColour(wx.Colour(0, 102, 102))  # Green
        self.upload_button.SetBottomEndColour(wx.Colour(0, 102, 102))   # Red
        #self.upload_button = wx.Button(self.panel, label="Upload Image")
        self.upload_button.Bind(wx.EVT_BUTTON, self.on_upload)
        self.button_sizer.Add(self.upload_button, 1, wx.ALL | wx.CENTER, 5)
        self.upload_button.SetMinSize((160, 40))  # Set the minimum size of the button
        self.upload_button.SetFont(button_font)

        self.detector_button = GB.GradientButton(self.panel, label="Detect")
        self.detector_button.SetTopStartColour(wx.Colour(0, 102, 102))  # Green 
        self.detector_button.SetTopEndColour(wx.Colour(0, 102, 102))   # Red
        self.detector_button.SetBottomStartColour(wx.Colour(0, 102, 102))  # Green
        self.detector_button.SetBottomEndColour(wx.Colour(0, 102, 102))   # Red
        #self.detector_button = wx.Button(self.panel, label="Detect forgery")
        self.detector_button.Bind(wx.EVT_BUTTON, self.predict_forgery)
        self.button_sizer.Add(self.detector_button, 1, wx.ALL | wx.CENTER, 5)
        self.detector_button.SetMinSize((160, 40))  # Set the minimum size of the button
        self.detector_button.SetFont(button_font)

        self.my_sizer.Add(self.button_sizer, flag=wx.ALL | wx.CENTER, border=10)

        self.panel.SetSizer(self.my_sizer) 
        self.SetTitle("Recod.ai - Image Forgery Detection System")
        self.Center()       
        self.Show()
        

    def on_upload(self, event):
        
        with wx.FileDialog(self, "Choose an image file", wildcard="Image files (*.png;*.jpg;*.jpeg;*.heic);|*.png;*.jpg;*.jpeg;*.heic",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as file_dialog:
            if file_dialog.ShowModal() == wx.ID_CANCEL:
                return

            self.localization_map_display.Show(False)
            self.confidence_map_display.Show(False)
            self.expert01_probability_text.Show(False)
            self.expert02_probability_text.Show(False)
            self.expert03_probability_text.Show(False)

            # Get the selected file path
            self.image_path = file_dialog.GetPath()
            print(f"Image uploaded: {self.image_path}")
            image = wx.Image(self.image_path, wx.BITMAP_TYPE_ANY)

            self.display_image(image, self.image_display)

    def display_image(self, image, displayer):
    
        image_width, image_height = image.GetSize()
        panel_width, panel_height = self.panel.GetSize()

        image_aspect_ratio = image_width/image_height

        new_height = int(panel_height*0.5)
        new_width = int(new_height*image_aspect_ratio)

        # Checking if three images can be arranged in row fitting into the interface
        width_factor = 3*(new_width + 10)/panel_width # 10 is for the border between images

        # Reduces the width of the images to 1/3 of the panel width
        if width_factor >= 1.0:
            new_width = int(panel_width*0.3)

        
        image = image.Scale(new_width, new_height, wx.IMAGE_QUALITY_HIGH)

        bitmap = wx.Bitmap(image)
        displayer.SetBitmap(bitmap)

        self.panel.Layout()

    def predict_forgery(self, event):
        prob_recodai, prob, prob_e2e, mapped_pred, mapped_conf = self.detector(self.image_path)

        self.expert01_probability_text.SetLabel("Prob. of Forgery for Expert #1 (Recod): {:.2%}".format(prob_recodai))
        self.expert01_probability_text.Show(True)

        self.expert02_probability_text.SetLabel("Prob. of Forgery for Expert #2 (TruFor): {:.2%}".format(prob))
        self.expert02_probability_text.Show(True)

        if prob_e2e is None:
            self.expert03_probability_text.SetLabel("Prob. of Forgery for Expert #3 (E2E): Image too small")
        else:
            self.expert03_probability_text.SetLabel("Prob. of Forgery for Expert #3 (E2E): {:.2}".format(prob_e2e) + "%")
        self.expert03_probability_text.Show(True)
        
        # Replace these paths with the paths to your image files
        #ela_image_opencv, superimposed_img = self.detector.get_ela_and_superposed()
        cv2.imwrite("Noise_image.png", cv2.cvtColor(mapped_conf, cv2.COLOR_RGB2BGR))
        cv2.imwrite("XAI_image.png", cv2.cvtColor(mapped_pred, cv2.COLOR_RGB2BGR))

        superimposed_img = self.numpy_to_wx_image(mapped_pred[:,:,:3])
        confidence_image = self.numpy_to_wx_image(mapped_conf[:,:,:3])

        #self.reinitialize_grid()
        self.display_image(superimposed_img, self.localization_map_display)
        self.localization_map_display.Show(True)
        self.display_image(confidence_image, self.confidence_map_display)
        self.confidence_map_display.Show(True)

        #width_ref = confidence_image.GetWidth()
        #height_ref = confidence_image.GetHeight()

        #new_width = min(300, width_ref)
        #new_height = min(300, height_ref)

        #confidence_image = confidence_image.Rescale(new_width, new_height)
        #bitmap = wx.Bitmap(confidence_image)
        #scaled_bitmap = wx.Bitmap(bitmap)
        #static_bitmap = wx.StaticBitmap(self.panel, wx.ID_ANY, scaled_bitmap)

        # Add the static bitmap to the grid sizer
        #self.grid_sizer.Add(static_bitmap, 0, wx.ALL | wx.EXPAND, 2)
        #self.localization_map_display.SetBitmap(scaled_bitmap)
        #self.localization_map_display.Show(True)

        #new_width = min(300, superimposed_img.GetWidth())
        #new_height = min(300, superimposed_img.GetHeight())

        #superimposed_img = superimposed_img.Rescale(new_width, new_height)

        #bitmap = wx.Bitmap(superimposed_img)
        #scaled_bitmap = wx.Bitmap(bitmap)
        #static_bitmap = wx.StaticBitmap(self.panel, wx.ID_ANY, scaled_bitmap)

        # Add the static bitmap to the grid sizer
        #self.grid_sizer.Add(static_bitmap, 0, wx.ALL | wx.EXPAND, 2)
        #self.confidence_map_display.SetBitmap(scaled_bitmap)
        #self.confidence_map_display.Show(True)

        # Set the frame properties
        self.panel.Layout()

    def numpy_to_wx_image(self, numpy_image):

        # Ensure that the input array has the correct shape (channels, height, width)
        if len(numpy_image.shape) != 3 or numpy_image.shape[0] < 3:
            raise ValueError("Invalid input shape. Expected (channels, height, width) with at least 3 channels.")

        # Transpose the array to channels-last format (height, width, channels)
        # image_channels_last = np.transpose(numpy_image, (1, 2, 0))

        # Convert the NumPy array to a wx.Image
        wx_image = wx.ImageFromBuffer(numpy_image.shape[1], numpy_image.shape[0], numpy_image.tobytes())

        return wx_image
       
    def reinitialize_grid(self):
        
        for child in self.grid_sizer.GetChildren():
            child.GetWindow().Destroy()

        # Clear the grid sizer
        self.grid_sizer.Clear()

        # Update the layout
        self.panel.Layout()

        

if __name__ == '__main__':
    app = wx.App()
    frame = MyFrame()
    app.MainLoop()
