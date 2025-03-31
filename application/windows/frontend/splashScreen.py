from abc import ABC, abstractmethod
from tkinter import *
from PIL import ImageTk, Image
import time
from main import App

class WindowConfig:
    """Configuration class for window properties"""
    def __init__(self):
        self.width = 400
        self.height = 220
        self.logo_width = 156
        self.logo_height = 46
        self.background_color = '#272727'
        self.text_color = 'white'

class Animation(ABC):
    """Abstract base class for animations"""
    @abstractmethod
    def run(self):
        pass

class LoadingAnimation(Animation):
    """Concrete implementation of loading animation"""
    def __init__(self, window, circle_positions, image_on, image_off):
        self.window = window
        self.circle_positions = circle_positions
        self.image_on = image_on
        self.image_off = image_off

    def run(self):
        for _ in range(3):
            for active_pos in self.circle_positions:
                for pos in self.circle_positions:
                    self._create_circle(pos, self.image_on if pos == active_pos else self.image_off)
                self.window.root.update()
                time.sleep(0.3)

    def _create_circle(self, x_offset, image):
        x = ((self.window.config.width - self.window.config.logo_width) / 2) + x_offset
        y = ((self.window.config.height - self.window.config.logo_height) / 2) + self.window.config.logo_height + 5
        return Label(self.window.root, image=image, border=0, relief=SUNKEN).place(x=x, y=y)

class UIComponent:
    """Base class for UI components"""
    def __init__(self, window):
        self.window = window

class Background(UIComponent):
    """Background frame component"""
    def create(self):
        Frame(self.window.root, 
              width=self.window.config.width, 
              height=self.window.config.height, 
              bg=self.window.config.background_color).place(x=0, y=0)

class Logo(UIComponent):
    """Logo component"""
    def create(self):
        self.text = Label(self.window.root, 
                         text="ALPACA", 
                         fg=self.window.config.text_color, 
                         bg=self.window.config.background_color)
        self.text.configure(font=("Game Of Squids", 24, "bold"))
        
        x = (self.window.config.width - self.window.config.logo_width) / 2
        y = (self.window.config.height - self.window.config.logo_height) / 2
        self.text.place(x=x, y=y)

class LoadingText(UIComponent):
    """Loading text component"""
    def create(self):
        text = Label(self.window.root, 
                    text="Loading...", 
                    fg=self.window.config.text_color, 
                    bg=self.window.config.background_color)
        text.configure(font=("Calibri", 11))
        text.place(x=20, y=180)

class SplashScreen:
    """Main splash screen class"""
    def __init__(self):
        self.config = WindowConfig()
        self.root = Tk()
        self._initialize_window()
        self._create_ui()
        self._run_animation()
        self._cleanup_and_show_main()
        self.root.mainloop()
    
    def _initialize_window(self):
        """Initialize window properties and position"""
        self.root.overrideredirect(1)
        self._center_window()
    
    def _center_window(self):
        """Center the window on the screen"""
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = int((screen_width/2) - (self.config.width/2))
        y = int((screen_height/2) - (self.config.height/2))
        self.root.geometry(f"{self.config.width}x{self.config.height}+{x}+{y}")
    
    def _create_ui(self):
        """Create all UI elements"""
        Background(self).create()
        Logo(self).create()
        LoadingText(self).create()
        self._load_animation_images()
    
    def _load_animation_images(self):
        """Load the circle animation images"""
        self.image_off = ImageTk.PhotoImage(Image.open("application/windows/assets/images/frontend/splash/c2_off.png"))
        self.image_on = ImageTk.PhotoImage(Image.open("application/windows/assets/images/frontend/splash/c1_on.png"))
    
    def _run_animation(self):
        """Run the loading animation sequence"""
        animation = LoadingAnimation(
            self,
            circle_positions=[45, 65, 85, 105],
            image_on=self.image_on,
            image_off=self.image_off
        )
        animation.run()
    
    def _cleanup_and_show_main(self):
        """Clean up splash screen and show main application"""
        self.root.destroy()
        app = App()
        app.app.mainloop()

if __name__ == "__main__":
    SplashScreen()