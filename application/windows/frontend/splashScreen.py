# Import required libraries
from tkinter import *
from PIL import ImageTk, Image
import time
from main import App # Import the App class from main.py

class SplashScreen:
    def __init__(self):
        # Initialize main window
        self.window = Tk()
        
        # Window configuration
        self.width = 400
        self.height = 220
        self.setup_window_position()
        self.window.overrideredirect(1)  # Hide title bar
        
        # Create UI elements
        self.create_background()
        self.create_logo()
        self.create_loading_text()
        self.load_animation_images()
        
        # Run animation
        self.run_loading_animation()
        
        # Clean up and show main window
        self.window.destroy()
        self.show_main_window()
        self.window.mainloop()
    
    def setup_window_position(self):
        """Set up the window position on screen"""
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        x_coordinate = (screen_width/2) - (self.width/2)
        y_coordinate = (screen_height/2) - (self.height/2)
        self.window.geometry("%dx%d+%d+%d" % (self.width, self.height, x_coordinate, y_coordinate))
    
    def create_background(self):
        """Create the background frame"""
        Frame(self.window, width=self.width, height=self.height, bg='#272727').place(x=0, y=0)
    
    def create_logo(self):
        """Create and position the logo text"""
        self.logo_text = Label(self.window, text="ALPACA", fg="white", bg='#272727')
        self.logo_text.configure(font=("Game Of Squids", 24, "bold"))
        
        # Logo dimensions
        self.logo_width = 156
        self.logo_height = 46
        
        # Center the logo
        self.logo_text.place(
            x=(self.width - self.logo_width) / 2,
            y=(self.height - self.logo_height) / 2
        )
    
    def create_loading_text(self):
        """Create the loading text"""
        loading_text = Label(self.window, text="Loading...", fg="white", bg='#272727')
        loading_text.configure(font=("Calibri", 11))
        loading_text.place(x=20, y=180)
    
    def load_animation_images(self):
        """Load the animation images"""
        self.image_off = ImageTk.PhotoImage(Image.open("application/windows/assets/images/frontend/splash/c2_off.png"))
        self.image_on = ImageTk.PhotoImage(Image.open("application/windows/assets/images/frontend/splash/c1_on.png"))
    
    def create_circle(self, x_pos, image):
        """Create a circle label at the specified position"""
        return Label(
            self.window, 
            image=image, 
            border=0, 
            relief=SUNKEN
        ).place(
            x=((self.width - self.logo_width) / 2) + x_pos, 
            y=((self.height - self.logo_height) / 2) + self.logo_height + 5
        )
    
    def run_loading_animation(self):
        """Run the loading animation sequence"""
        for _ in range(3):
            # First circle active
            self.create_circle(45, self.image_on)
            self.create_circle(65, self.image_off)
            self.create_circle(85, self.image_off)
            self.create_circle(105, self.image_off)
            self.window.update()
            time.sleep(0.3)

            # Second circle active
            self.create_circle(45, self.image_off)
            self.create_circle(65, self.image_on)
            self.create_circle(85, self.image_off)
            self.create_circle(105, self.image_off)
            self.window.update()
            time.sleep(0.3)

            # Third circle active
            self.create_circle(45, self.image_off)
            self.create_circle(65, self.image_off)
            self.create_circle(85, self.image_on)
            self.create_circle(105, self.image_off)
            self.window.update()
            time.sleep(0.3)

            # Fourth circle active
            self.create_circle(45, self.image_off)
            self.create_circle(65, self.image_off)
            self.create_circle(85, self.image_off)
            self.create_circle(105, self.image_on)
            self.window.update()
            time.sleep(0.3)
    
    def show_main_window(self):
        """Show the main application window"""
        app = App()
        app.app.mainloop()
if __name__ == "__main__":
    splash = SplashScreen()