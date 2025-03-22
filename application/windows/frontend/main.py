# Import required libraries
import tkinter as tk
from customtkinter import set_appearance_mode, CTk, CTkFrame, CTkButton
import customtkinter
from tkinter import filedialog, messagebox
from customtkinter import *
import tkinter
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox

class App:
    """
    Main application class that handles the GUI and all its functionality.
    Uses customtkinter for modern-looking UI elements.
    """
    def __init__(self):
        """Initialize the main application window and its components"""
        self.app = CTk()
        self.setup_window()
        self.current_page = None
        self.buttons = {}

        self.create_option_menu()  # Create the top navigation menu
        self.main_frame = self.create_main_frame()  # Create the main content area

        self.create_pages()
        self.switch_page("home")  # Set home as the default page

    def setup_window(self):
        """Configure the main window properties (size, position, theme, etc.)"""
        # Calculate center position for the window
        screen_width = self.app.winfo_screenwidth()
        screen_height = self.app.winfo_screenheight()
        app_width = 1100
        app_height = 645
        x = (screen_width / 2) - (app_width / 2)
        y = (screen_height / 2) - (app_height / 2)
        
        # Set window properties
        self.app.geometry(f"{app_width}x{app_height}+{int(x)}+{int(y)}")
        self.app.title("Mohammad Hossein")
        self.app.resizable(0, 0)  # Make window non-resizable
        set_appearance_mode("dark")  # Set dark theme
        #self.app.iconbitmap("img/logo.ico")  # Set window icon

    def create_main_frame(self):
        """Create the main content frame where pages will be displayed"""
        main_frame_bg_color = "#242424"
        main_frame = tk.Frame(self.app, bg=main_frame_bg_color)
        main_frame.pack(expand=True, fill="both")
        return main_frame

    def create_option_menu(self):
        """Create the top navigation menu with buttons for different pages"""
        # Create frame for option menu
        option_menu_frame = CTkFrame(master=self.app, width=176, height=35, fg_color="#242424", corner_radius=0)
        option_menu_frame.pack_propagate(0)
        option_menu_frame.pack(fill="x", side="top")

        # Create navigation buttons
        button_names = ["home", "runner", "config", "tools", "settings", "about"]
        for name in button_names:
            button_text = name.capitalize()
            button_width = len(button_text) + 2
            button = CTkButton(master=option_menu_frame, text=name.capitalize(), fg_color="#242424",
                               font=("Arial Bold", 14), text_color="#FFFFFF", hover_color="#242424",
                               anchor="center", corner_radius=0, width=button_width,
                               command=lambda name=name: self.switch_page(name))
            button.pack(side="left", padx=(15, 0), pady=(5, 0), ipady=3)
            self.buttons[name] = button

    def switch_page(self, page_name):
        """Handle switching between different pages"""
        self.current_page = page_name
        self.update_button_colors()
        self.show_page(page_name)

    def update_button_colors(self):
        """Update navigation button colors to show active/inactive state"""
        active_color = "#242424"
        inactive_color = "#242424"
        active_text_color = "#1E90FF"
        inactive_text_color = "#FFFFFF"

        for name, button in self.buttons.items():
            if name == self.current_page:
                button.configure(fg_color=active_color, text_color=active_text_color)
            else:
                button.configure(fg_color=inactive_color, text_color=inactive_text_color)

    def create_pages(self):
        """Map page names to their corresponding creation methods"""
        self.pages = {
            "home": self.home_optionMenu_page,
            "runner": self.runner_optionMenu_page,
            "config": self.config_optionMenu_page,
            "tools": self.tools_optionMenu_page,
            "settings": self.settings_optionMenu_page,
            "about": self.about_optionMenu_page,
        }

    def show_page(self, page_name):
        """Display the selected page and clear previous content"""
        self.clear_current_page()
        if page_name in self.pages:
            self.pages[page_name]()

    def clear_current_page(self):
        """Remove all widgets from the main frame"""
        for widget in self.main_frame.winfo_children():
            widget.destroy()

    # Page creation methods
    def home_optionMenu_page(self):
        """Create and display the home page content"""
        print("Home Page")
        home_page_frame = tk.Frame(self.main_frame, bg="#242424")
        home_page_frame.pack(fill=tk.BOTH, expand=True)

    def runner_optionMenu_page(self):
        """Create and display the runner page with file handling functionality"""
        print("Runner Page")

        # Create main frame for runner page
        runner_page_frame = tk.Frame(self.main_frame, bg="#242424")
        runner_page_frame.pack(fill=tk.BOTH, expand=True)

        # Configure grid layout
        runner_page_frame.grid_rowconfigure(0, weight=1)
        runner_page_frame.grid_columnconfigure(0, weight=1)

        # Load button images
        self.edit_img_data = Image.open("application/windows/assets/images/frontend/original/editSquare_img.png")
        self.edit_img = CTkImage(dark_image=self.edit_img_data, light_image=self.edit_img_data, size=(15, 15))
        
        self.upload_img_data = Image.open("application/windows/assets/images/frontend/original/upload_img.png")
        self.upload_img = CTkImage(dark_image=self.upload_img_data, light_image=self.upload_img_data, size=(15, 15))
        
        self.browse_img_data = Image.open("application/windows/assets/images/frontend/original/browse_img.png")
        self.browse_img = CTkImage(dark_image=self.browse_img_data, light_image=self.browse_img_data, size=(15, 15))

        self.remove_img_data = Image.open("application/windows/assets/images/frontend/original/remove_img.png")
        self.remove_img = CTkImage(dark_image=self.remove_img_data, light_image=self.remove_img_data, size=(15, 15))

        # Create file path entry
        self.entry_file_path = customtkinter.CTkEntry(runner_page_frame, placeholder_text="   Set File Path Address", 
                                                     font=("Arial Bold", 12), fg_color="#000000", 
                                                     border_width=2, border_color="#FFFFFF", corner_radius=0)
        self.entry_file_path.grid(row=3, column=0, padx=(20, 0), pady=(20, 10), sticky="nsew")

        # Create browse button
        self.browse_button = customtkinter.CTkButton(runner_page_frame, text="Browse", image=self.browse_img,
                                                    font=("Arial Bold", 12), fg_color="transparent",
                                                    border_width=2, border_color="#FFFFFF", text_color="#FFFFFF",
                                                    hover_color="#000000", corner_radius=0,
                                                    command=self.browse_button_callback)
        self.browse_button.grid(row=3, column=4, padx=(20, 20), pady=(20, 10), ipady=4, sticky="nsew")

        # Create disabled button
        self.disabled_button = customtkinter.CTkButton(runner_page_frame, text="Disabled",
                                                      font=("Arial Bold", 12), state="disabled",
                                                      fg_color="transparent", border_width=0,
                                                      border_color="#FFFFFF", text_color=("gray10", "#DCE4EE"),
                                                      hover_color="#000000", corner_radius=0)
        self.disabled_button.grid(row=3, column=6, padx=(0, 20), pady=(20, 10), ipady=4, sticky="nsew")
        
        # Create remove button (initially hidden)
        self.remove_button = customtkinter.CTkButton(runner_page_frame, text="Remove",
                                                    font=("Arial Bold", 12), fg_color="transparent",
                                                    border_width=2, text_color=("gray10", "#DCE4EE"),
                                                    hover_color="#000000", corner_radius=0,
                                                    command=self.remove_button_callback)
        self.remove_button.grid(row=3, column=5, padx=(20, 10), pady=(20, 10), ipady=4, sticky="nsew")
        self.remove_button.grid_remove()  # Initially hide the remove button

    def browse_button_callback(self):
        """Handle browse button click to select an image file"""
        print("Browse button clicked")

        file_path = filedialog.askopenfilename(title="Select File", filetypes=[("Image Files", "*.png;*.jpg")])

        if file_path:
            # Update UI elements after file selection
            self.entry_file_path.delete(0, "end")
            self.entry_file_path.insert(0, file_path)
            print(f"Selected file: {file_path}")

            # Update button states and appearance
            self.browse_button.configure(text="Edit", fg_color="red", image=self.edit_img)
            self.disabled_button.configure(state="normal", text="OK", fg_color="green",
                                         border_width=2, border_color="#FFFFFF", image=self.upload_img,
                                         command=self.ok__button_callback)
            self.remove_button.grid()
            self.remove_button.configure(state="normal", text="Remove", fg_color="#BB8B0C",
                                       text_color="#FFFFFF", image=self.remove_img)
        else:
            print("No file selected")

    def remove_button_callback(self):
        """Handle remove button click to clear file selection"""
        # Reset UI elements to initial state
        self.browse_button.configure(text="Browse", state="normal", fg_color="transparent",
                                   border_width=2, text_color=("gray10", "#DCE4EE"), image=self.browse_img)
        self.disabled_button.configure(state="disabled", text="Disabled", fg_color="transparent",
                                     border_color="#000000", border_width=0,
                                     text_color=("gray10", "#DCE4EE"), image=None)
        self.entry_file_path.delete(0, "end")
        self.entry_file_path.configure(placeholder_text="Set File Path Address")
        self.remove_button.grid_remove()

    def ok__button_callback(self):
        """Handle OK button click (placeholder for future functionality)"""
        print("OK button clicked")

    def config_optionMenu_page(self):
        """Create and display the configuration page"""
        print("Config Page")
        config_page_frame = tk.Frame(self.main_frame, bg="#1CBB0F")
        config_page_frame.pack(fill=tk.BOTH, expand=True)
        label_test = tk.Label(config_page_frame, text="Config Page", font=("Arial Bold", 25), fg="#E8006D")
        label_test.pack(pady=80)

    def tools_optionMenu_page(self):
        """Create and display the tools page"""
        print("Tools Page")
        tools_page_frame = tk.Frame(self.main_frame, bg="#8F0FBB")
        tools_page_frame.pack(fill=tk.BOTH, expand=True)
        label_test = tk.Label(tools_page_frame, text="Tools Page", font=("Arial Bold", 25), fg="#E8006D")
        label_test.pack(pady=80)

    def settings_optionMenu_page(self):
        """Create and display the settings page with customization options"""
        print("Settings Page")
        settings_page_frame = tk.Frame(self.main_frame, bg="#242424")
        settings_page_frame.pack(fill=tk.BOTH, expand=True)

        settings_text = ("Customization" + "\n\n")

        # Create settings text display
        settings_text_widget = tk.Text(settings_page_frame, font=("Consolas", 24, "bold"),
                                     fg="#FFFFFF", bg="#242424", wrap="word",
                                     borderwidth=0, highlightthickness=0)
        settings_text_widget.pack(side="top", fill="both", expand=True, padx=20, pady=20)
        settings_text_widget.insert("1.0", settings_text)
        settings_text_widget.configure(state="disabled",
                                     inactiveselectbackground=settings_text_widget.cget("selectbackground"))

        # Add sample checkbox
        check_button_var = tk.BooleanVar()
        check_button = tk.Checkbutton(settings_page_frame, text="Sample Checkbox Text",
                                    variable=check_button_var, font=("Arial", 14),
                                    bg="#242424", fg="#FFFFFF", selectcolor="#242424")
        check_button.pack(pady=(0, 20))

    def about_optionMenu_page(self):
        """Create and display the about page with project information"""
        print("About Page")
        about_page_frame = tk.Frame(self.main_frame, bg="#242424")
        about_page_frame.pack(fill=tk.BOTH, expand=True)
        
        # Project description text
        about_text = (" This project aims to develop an AI-powered system capable of generating descriptive text for images. " +
                     "Given an image as input, the system will analyze its content and produce a meaningful and contextually " +
                     "relevant textual description as output. The task, known as image captioning, is a complex challenge that " +
                     "lies at the intersection of computer vision and natural language processing (NLP). The system must not only " +
                     "recognize objects, actions, and scenes within an image but also structure this information into a fluent, " +
                     "human-like description." + "\n\n\n")
                    
        # Create about text widget
        about_text_widget = tk.Text(about_page_frame, font=("Consolas", 14, "bold"),
                                  fg="#FFFFFF", bg="#242424", wrap="word",
                                  borderwidth=0, highlightthickness=0)
        about_text_widget.pack(side="top", fill="both", expand=True, padx=20, pady=20)
        about_text_widget.insert("1.0", about_text)
        
        # Add developer information
        red_text = ("Delevoper: Mohammad Hossein Mohammadi (Frontend)" + "\n" +
                   "Collaborators: Mr. Mehdi Kazemi" + "\n" +
                   "Mentor: Mr. Mehdi Kazemi" + "\n\n\n" +
                   "Contact US:" + "\n" +
                   "Mohammad Hossein Mohammadi -> Mohammadimir2017@gmail.com")
        about_text_widget.insert("end", red_text)
        
        # Style developer information text
        about_text_widget.tag_add("red_text", "3.0", "end")
        about_text_widget.tag_config("red_text", foreground="green")
        about_text_widget.configure(state="disabled",
                                  inactiveselectbackground=about_text_widget.cget("selectbackground"))

        # Create button container
        button_container = tk.Frame(about_page_frame, bg="#242424", height=40)
        button_container.pack(side="bottom", fill="x", padx=20, pady=20)
        button_container.pack_propagate(False)

        # Load button images
        self.licence_img_data = Image.open("application/windows/assets/images/frontend/original/licence.png")
        self.licence_img = CTkImage(dark_image=self.licence_img_data, light_image=self.licence_img_data, size=(15, 15))
        
        self.donate_img_data = Image.open("application/windows/assets/images/frontend/original/donate.png")
        self.donate_img = CTkImage(dark_image=self.donate_img_data, light_image=self.donate_img_data, size=(15, 15))
        
        self.repository_img_data = Image.open("application/windows/assets/images/frontend/original/repository.png")
        self.repository_img = CTkImage(dark_image=self.repository_img_data, light_image=self.repository_img_data, size=(15, 15))
    
        # Create footer buttons
        self.licence_button = CTkButton(button_container, image=self.licence_img, text="Licence",
                                      font=("Arial Bold", 12), fg_color="transparent",
                                      border_width=2, border_color="#FFFFFF", text_color="#FFFFFF",
                                      hover_color="#000000", corner_radius=0,
                                      command=self.licence_button_callback)
        self.licence_button.pack(side="left", padx=(0, 10), ipady=4)

        self.donate_button = CTkButton(button_container, image=self.donate_img, text="Donate",
                                     font=("Arial Bold", 12), fg_color="transparent",
                                     border_width=2, border_color="#FFFFFF", text_color="#FFFFFF",
                                     hover_color="#000000", corner_radius=0,
                                     command=self.donate_button_callback)
        self.donate_button.pack(side="left", padx=(0, 10), ipady=4)

        self.repository_button = CTkButton(button_container, image=self.repository_img, text="Repository",
                                         font=("Arial Bold", 12), fg_color="transparent",
                                         border_width=2, border_color="#FFFFFF", text_color="#FFFFFF",
                                         hover_color="#000000", corner_radius=0,
                                         command=self.repository_button_callback)
        self.repository_button.pack(side="left", ipady=4)

    def licence_button_callback(self):
        """Create and display license information window"""
        extra_window = tk.Toplevel()
        screen_width = extra_window.winfo_screenwidth()
        screen_height = extra_window.winfo_screenheight()
        app_width = 800
        app_height = 500
        x = (screen_width / 2) - (app_width / 2)
        y = (screen_height / 2) - (app_height / 2)
        
        # Configure license window
        extra_window.geometry(f"{app_width}x{app_height}+{int(x)}+{int(y)}")
        extra_window.title("Licence")
        extra_window.resizable(0, 0)
        extra_window.iconbitmap("img/repository.png")
        extra_window.overrideredirect(False)

        background_color = "#242424"    
        extra_window.configure(bg=background_color)

        label = tk.Label(extra_window, text="Custom Title Bar", bg=background_color, fg="white")
        label.pack(fill="both")

    def donate_button_callback(self):
        """Handle donate button click (placeholder)"""
        print("Donate button clicked")

    def repository_button_callback(self):
        """Handle repository button click (placeholder)"""
        print("Repository button clicked")

# Entry point of the application
if __name__ == "__main__":
    App().app.mainloop()