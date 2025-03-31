# Import required libraries
from tkinter import *
from PIL import ImageTk, Image
import time
from main import App # Import the App class from main.py

class SplashScreen:
    def __init__(self):
        self.window = Tk()
        self.width = 400
        self.height = 220
        self.logo_width = 156
        self.logo_height = 46
        
        self._initialize_window()
        self._create_ui()
        self._run_animation()
        self._cleanup_and_show_main()
        self.window.mainloop()
    
    def _initialize_window(self):
        """Initialize window properties and position"""
        self.window.overrideredirect(1)
        self._center_window()
    
    def _center_window(self):
        """Center the window on the screen"""
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        x = int((screen_width/2) - (self.width/2))
        y = int((screen_height/2) - (self.height/2))
        self.window.geometry(f"{self.width}x{self.height}+{x}+{y}")
    
    def _create_ui(self):
        """Create all UI elements"""
        self._create_background()
        self._create_logo()
        self._create_loading_text()
        self._load_animation_images()
    
    def _create_background(self):
        """Create the main background frame"""
        Frame(self.window, width=self.width, height=self.height, bg='#272727').place(x=0, y=0)
    
    def _create_logo(self):
        """Create and position the ALPACA logo"""
        self.logo_text = Label(self.window, text="ALPACA", fg="white", bg='#272727')
        self.logo_text.configure(font=("Game Of Squids", 24, "bold"))
        
        x = (self.width - self.logo_width) / 2
        y = (self.height - self.logo_height) / 2
        self.logo_text.place(x=x, y=y)
    
    def _create_loading_text(self):
        """Create the loading text at the bottom"""
        loading_text = Label(self.window, text="Loading...", fg="white", bg='#272727')
        loading_text.configure(font=("Calibri", 11))
        loading_text.place(x=20, y=180)
    
    def _load_animation_images(self):
        """Load the circle animation images"""
        self.image_off = ImageTk.PhotoImage(Image.open("application/windows/assets/images/frontend/splash/c2_off.png"))
        self.image_on = ImageTk.PhotoImage(Image.open("application/windows/assets/images/frontend/splash/c1_on.png"))
    
    def _create_circle(self, x_offset, image):
        """Create a circle at the specified x-offset from the logo"""
        x = ((self.width - self.logo_width) / 2) + x_offset
        y = ((self.height - self.logo_height) / 2) + self.logo_height + 5
        return Label(self.window, image=image, border=0, relief=SUNKEN).place(x=x, y=y)
    
    def _run_animation(self):
        """Run the loading animation sequence"""
        circle_positions = [45, 65, 85, 105]
        for _ in range(3):
            for active_pos in circle_positions:
                for pos in circle_positions:
                    self._create_circle(pos, self.image_on if pos == active_pos else self.image_off)
                self.window.update()
                time.sleep(0.3)
    
    def _cleanup_and_show_main(self):
        """Clean up splash screen and show main application"""
        self.window.destroy()
        app = App()
        app.app.mainloop()

if __name__ == "__main__":
    SplashScreen()