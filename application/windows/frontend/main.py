import tkinter as tk
from customtkinter import set_appearance_mode, CTk, CTkFrame, CTkButton
import customtkinter
from tkinter import filedialog, messagebox
from customtkinter import *
import tkinter
from PIL import Image, ImageTk  # type: ignore
from tkinter import filedialog, messagebox
import webbrowser  # Add this import at the top


import numpy as np
import time
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as TT
from torchvision.transforms import functional as TF
from torchvision.models import resnet50, ResNet50_Weights
from torchtext.data.utils import get_tokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = 'D:/EmKay/ImageCaptioning/image-caption-generator-pytorch/application/windows/backend/ptFiles/model.pt'
vocab = torch.load('D:/EmKay/ImageCaptioning/image-caption-generator-pytorch/application/windows/backend/ptFiles/vocab.pt')
tokenizer = get_tokenizer('basic_english')
embed_size=256
hidden_size=512
num_layers = 2
dropout_embd = 0.5
dropout_rnn = 0.5

class Encoder(nn.Module):
  def __init__(self, embed_size):
    super().__init__()
    self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    self.resnet.requires_grad_(False)
    feature_size = self.resnet.fc.in_features

    self.resnet.fc = nn.Identity()
    self.fc = nn.Linear(feature_size, embed_size)
    self.bn = nn.BatchNorm1d(embed_size)

  def forward(self, x):
    self.resnet.eval()
    with torch.no_grad():
      features = self.resnet(x)
    y = self.bn(self.fc(features))
    return y
  
class Decoder(nn.Module):
  def __init__(self, embed_size, hidden_size, vocab_size, num_layers, dropout_embd, dropout_rnn, max_seq_length=20):
    super(Decoder, self).__init__()

    self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=vocab['<pad>'])
    self.dropout_embd = nn.Dropout(dropout_embd)

    self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout=dropout_rnn, batch_first=True)

    self.linear = nn.Linear(hidden_size, vocab_size)

    self.max_seq_length = max_seq_length

  def init_weights(self):
    self.embedding.weight.data.uniform_(-0.1, 0.1)
    self.linear.bias.data.fill_(0)
    self.linear.weight.data.uniform_(-0.1, 0.1)

  def forward(self, features, captions):
    embeddings = self.dropout_embd(self.embedding(captions[:, :-1]))
    inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1)
    outputs, _ = self.lstm(inputs)
    outputs = self.linear(outputs)
    return outputs

  def generate(self, features, captions):
    if len(captions)!=0:
      embeddings = self.dropout_embd(self.embedding(captions))
      inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1)
    else:
      inputs = features.unsqueeze(1)
    outputs, _ = self.lstm(inputs)
    outputs = self.linear(outputs)
    return outputs


class ImageCaptioning(nn.Module):
  def __init__(self, embed_size, hidden_size, vocab_size, num_layers, dropout_embd, dropout_rnn, max_seq_length=20):
    super(ImageCaptioning, self).__init__()
    self.encoder = Encoder(embed_size)
    self.decoder = Decoder(embed_size, hidden_size, vocab_size, num_layers, dropout_embd, dropout_rnn, max_seq_length)

  def forward(self, images, captions):
    features = self.encoder(images)
    output = self.decoder(features, captions)
    return output

  def generate(self, images, captions):
    features = self.encoder(images)
    output = self.decoder.generate(features, captions)
    return output
  

def generate(image, transform, model_path, vocab, max_seq_length, device):
    img_array = np.frombuffer(image, dtype=np.uint8)
    image_ = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    model = torch.load(model_path, map_location=torch.device(device))
    model.eval()
    image_transformed = transform(TF.to_pil_image(image_)).unsqueeze(0)
    image = image_transformed.to(device)
    src, indices = [], []

    caption = ''
    itos = vocab.get_itos()

    for i in range(max_seq_length):
        with torch.no_grad():
            predictions = model.generate(image, src)

        idx = predictions[:, -1, :].argmax(1)
        token = itos[idx]
        caption += token + ' '

        if idx == vocab['<eos>']:
            break

        indices.append(idx)
        src = torch.LongTensor([indices]).to(device)

    return caption


ImageCaptioning(embed_size, hidden_size, len(vocab), num_layers, dropout_embd, dropout_rnn, 20)

test_transform = TT.Compose([TT.Resize((224, 224)),
                TT.ToTensor(),
                TT.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])])


class App:
    def __init__(self):
        self.app = CTk()
        self.setup_window()
        self.current_page = None
        self.buttons = {}

        self.create_option_menu()  # Create the option menu
        self.main_frame = self.create_main_frame()  # Then create the main content frame

        self.create_pages()
        self.switch_page("home")  # Default to home page

    def setup_window(self):
        screen_width = self.app.winfo_screenwidth()
        screen_height = self.app.winfo_screenheight()
        app_width = 1100
        app_height = 645
        x = (screen_width / 2) - (app_width / 2)
        y = (screen_height / 2) - (app_height / 2)
        self.app.geometry(f"{app_width}x{app_height}+{int(x)}+{int(y)}")
        self.app.title("Alpaca -- v1.0.0")
        self.app.resizable(0, 0)
        set_appearance_mode("dark")  # Set the app theme
        #self.app.iconbitmap("img/logo.ico")  # Change the logo/icon

    def create_main_frame(self):
        main_frame_bg_color = "#242424"
        main_frame = tk.Frame(self.app, bg=main_frame_bg_color)
        main_frame.pack(expand=True, fill="both")
        return main_frame

    def create_option_menu(self):
        option_menu_frame = CTkFrame(master=self.app, width=176, height=35, fg_color="#242424", corner_radius=0)
        option_menu_frame.pack_propagate(0)
        option_menu_frame.pack(fill="x", side="top")  # Ensuring it is at the top

        button_names = ["home", "runner", "settings", "about"]
        for name in button_names:
            button_text = name.capitalize()
            button_width = len(button_text) + 2  # Add padding
            button = CTkButton(master=option_menu_frame, text=name.capitalize(), fg_color="#242424",
                               font=("Arial Bold", 14), text_color="#FFFFFF", hover_color="#242424",
                               anchor="center", corner_radius=0, width=button_width,
                               command=lambda name=name: self.switch_page(name))
            button.pack(side="left", padx=(15, 0), pady=(5, 0), ipady=3)
            self.buttons[name] = button

    def switch_page(self, page_name):
        self.current_page = page_name
        self.update_button_colors()
        self.show_page(page_name)

    def update_button_colors(self):
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
        self.pages = {
            "home": self.home_optionMenu_page,
            "runner": self.runner_optionMenu_page,
            "config": self.config_optionMenu_page,
            "tools": self.tools_optionMenu_page,
            "settings": self.settings_optionMenu_page,
            "about": self.about_optionMenu_page,
        }

    def show_page(self, page_name):
        self.clear_current_page()
        if page_name in self.pages:
            self.pages[page_name]()

    def clear_current_page(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()

# "Home" option menu page methods
    def home_optionMenu_page(self):
        print("Home Page")
        home_page_frame = tk.Frame(self.main_frame, bg="#242424")
        home_page_frame.pack(fill=tk.BOTH, expand=True)

        # Create title label
        title_label = tk.Label(
            home_page_frame,
            text="Welcome to Image Caption Generator",
            font=("Arial Bold", 24),
            fg="#FFFFFF",
            bg="#242424"
        )
        title_label.pack(pady=(20, 10))

        # Create documentation text widget
        doc_text = tk.Text(
            home_page_frame,
            font=("Consolas", 12),
            fg="#FFFFFF",
            bg="#242424",
            wrap=tk.WORD,
            padx=20,
            pady=10,
            borderwidth=0,
            highlightthickness=0
        )
        doc_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

        # Insert documentation content
        doc_content = """How to Use This Software:

1. Getting Started
   - Launch the application
   - Navigate to the "Runner" page using the menu bar at the top

2. Image Selection
   - Click the "Browse" button to select an image file
   - Supported formats: PNG and JPG
   - The selected image will be displayed in the preview area
   - The file path will be shown in the input field

3. Image Processing
   - After selecting an image, the "Browse" button changes to "Edit"
   - The "Disabled" button becomes "OK" and turns green
   - Click "OK" to process the image and generate a caption
   - The caption will be displayed below the image

4. Managing Images
   - Use the "Remove" button to clear the current image
   - This will reset all buttons to their initial state
   - You can then select a new image to process

5. Settings
   - Access the "Settings" page to configure API settings
   - Note: API settings are currently under construction

6. Additional Information
   - Visit the "About" page for more details about the project
   - Check the license information using the "Unlicense" button
   - Support the project by visiting our GitHub repository

For best results:
- Use clear, well-lit images
- Ensure images are not too large (recommended size: under 5MB)
- Wait for the processing to complete before selecting a new image

__________________________________
Author: Mohammad Hossein Mohammadi"""
        
        doc_text.insert("1.0", doc_content)
        doc_text.configure(state="disabled")  # Make text read-only

# "Runner" option menu page methods
    def runner_optionMenu_page(self):
        print("Runner Page")

        # Create runner page frame
        runner_page_frame = tk.Frame(self.main_frame, bg="#242424")
        runner_page_frame.pack(fill=tk.BOTH, expand=True)

        # Configure the frame to expand the last row and column
        runner_page_frame.grid_rowconfigure(0, weight=1)  # Makes row 0 expand
        runner_page_frame.grid_columnconfigure(0, weight=1)  # Makes column 0 expand

        # Create image display frame
        self.image_frame = tk.Frame(runner_page_frame, bg="#242424", width=800, height=400)
        self.image_frame.grid(row=0, column=0, columnspan=7, padx=20, pady=(20, 10), sticky="nsew")
        self.image_frame.grid_propagate(False)  # Prevent frame from shrinking
        
        # Create label for displaying image
        self.image_label = tk.Label(self.image_frame, bg="#242424")
        self.image_label.pack(expand=True, fill="both")

    # Create image button
        # Edit image button
        self.edit_img_data = Image.open("application/windows/assets/images/frontend/original/editSquare_icon.png")
        self.edit_img = CTkImage(dark_image=self.edit_img_data, light_image=self.edit_img_data, size=(15, 15))
      
        # Upload image button
        self.upload_img_data = Image.open("application/windows/assets/images/frontend/original/upload_icon.png")
        self.upload_img = CTkImage(dark_image=self.upload_img_data, light_image=self.upload_img_data, size=(15, 15))
      
        # Browse image button
        self.browse_img_data = Image.open("application/windows/assets/images/frontend/original/browse_icon.png")
        self.browse_img = CTkImage(dark_image=self.browse_img_data, light_image=self.browse_img_data, size=(15, 15))

        # Remove image button
        self.remove_img_data = Image.open("application/windows/assets/images/frontend/original/remove_icon.png")
        self.remove_img = CTkImage(dark_image=self.remove_img_data, light_image=self.remove_img_data, size=(15, 15))

    # Create widget
        # Create the entry
        self.entry_file_path = customtkinter.CTkEntry(runner_page_frame, placeholder_text="   Set File Path Address", font=("Arial Bold", 12), fg_color="#000000", border_width=2, border_color="#FFFFFF", corner_radius=0)
        self.entry_file_path.grid(row=3, column=0, padx=(20, 0), pady=(20, 10), sticky="nsew")

        # Create the Browse button
        self.browse_button = customtkinter.CTkButton(runner_page_frame, text="Browse", image=self.browse_img, font=("Arial Bold", 12), fg_color="transparent", border_width=2, border_color="#FFFFFF", text_color="#FFFFFF", hover_color="#000000", corner_radius=0, command=self.browse_button_callback)
        self.browse_button.grid(row=3, column=4, padx=(20, 20), pady=(20, 10),  ipady=4, sticky="nsew")

        # Create the Disabled button
        self.disabled_button = customtkinter.CTkButton(runner_page_frame, text="Disabled", font=("Arial Bold", 12), state="disabled", fg_color="transparent", border_width=0, border_color="#FFFFFF", text_color=("gray10", "#DCE4EE"), hover_color="#000000", corner_radius=0)
        self.disabled_button.grid(row=3, column=6, padx=(0, 20), pady=(20, 10), ipady=4, sticky="nsew")
        
        # Create the Remove button
        self.remove_button = customtkinter.CTkButton(runner_page_frame, text="Remove", font=("Arial Bold", 12), fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"), hover_color="#000000", corner_radius=0, command=self.remove_button_callback)
        self.remove_button.grid(row=3, column=5, padx=(20, 10), pady=(20, 10), ipady=4, sticky="nsew")
        self.remove_button.grid_remove()  # Hide the Remove button

# Define Browse button callback
    def browse_button_callback(self):
        print("Browse button clicked")

        # Open a file dialog to select a file
        file_path = filedialog.askopenfilename(title="Select File", filetypes=[("Image Files", "*.png;*.jpg")])

        if file_path:

            # Update the entry file path widget with the selected file path
            self.entry_file_path.delete(0, "end")
            self.entry_file_path.insert(0, file_path)
            print(f"Selected file: {file_path}")

            # Load and display the image
            try:
                # Open the image
                image = Image.open(file_path)
                
                # Calculate the aspect ratio
                aspect_ratio = image.width / image.height
                
                # Set maximum dimensions
                max_width = 780  # Slightly smaller than frame width
                max_height = 380  # Slightly smaller than frame height
                
                # Calculate new dimensions maintaining aspect ratio
                if aspect_ratio > 1:
                    new_width = min(max_width, image.width)
                    new_height = int(new_width / aspect_ratio)
                    if new_height > max_height:
                        new_height = max_height
                        new_width = int(new_height * aspect_ratio)
                else:
                    new_height = min(max_height, image.height)
                    new_width = int(new_height * aspect_ratio)
                    if new_width > max_width:
                        new_width = max_width
                        new_height = int(new_width / aspect_ratio)
                
                # Resize the image
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Convert to PhotoImage
                photo = ImageTk.PhotoImage(image)
                
                # Update the image label
                self.image_label.configure(image=photo)
                self.image_label.image = photo  # Keep a reference

                # Remove any existing path label
                runner_frame = self.main_frame.winfo_children()[0]  # Get the runner page frame
                for widget in runner_frame.winfo_children():
                    if isinstance(widget, tk.Label) and widget.cget("text").startswith("Image Path:"):
                        widget.destroy()
                '''
                # Create a label to display the file path
                path_label = tk.Label(
                    runner_frame,
                    text=f"Image Path: {file_path}",
                    font=("Arial", 10),
                    fg="#FFFFFF",
                    bg="#242424",
                    wraplength=780  # Wrap text to fit within the frame width
                )
                path_label.grid(row=1, column=0, columnspan=7, padx=20, pady=(0, 20), sticky="nsew")
                '''
            except Exception as e:
                print(f"Error loading image: {e}")
                messagebox.showerror("Error", "Failed to load the image")

            # Update the Browse button to say "Edit" and change its color
            self.browse_button.configure(text="Edit", fg_color="red", image=self.edit_img)

            # Enable the previously disabled button, change its text and color, and assign its command
            self.disabled_button.configure(state="normal", text="OK", fg_color="green", border_width=2, border_color="#FFFFFF", image=self.upload_img)
            # Assign the button callback method to the button command
            self.disabled_button.configure(command=self.ok__button_callback)

            # Show the Remove button with updated text and color
            self.remove_button.grid(padx=(0, 20))
            self.remove_button.configure(state="normal", text="Remove", fg_color="#BB8B0C", text_color="#FFFFFF", image=self.remove_img)
        else:
            print("No file selected")

# Define Remove button callback
    def remove_button_callback(self): 
        # Reset the state of the buttons
        self.browse_button.configure(text="Browse", state="normal", fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"), image=self.browse_img)
        self.disabled_button.configure(state="disabled", text="Disabled", fg_color="transparent", border_color="#000000", border_width=0, text_color=("gray10", "#DCE4EE"), image=None)
        
        # Clear the file path entry file path
        self.entry_file_path.delete(0, "end")
        self.entry_file_path.configure(placeholder_text="Set File Path Address")
        
        # Clear the image display
        self.image_label.configure(image="")
        
        # Remove the path label and caption label if they exist
        runner_frame = self.main_frame.winfo_children()[0]  # Get the runner page frame
        for widget in runner_frame.winfo_children():
            if isinstance(widget, tk.Label) and (widget.cget("text").startswith("Image Path:") or widget.cget("text").startswith("Caption:")):
                widget.destroy()
        
        # Hide the Remove button again
        self.remove_button.grid_remove()  # Hide the Remove button

# Define Ok button callback
    def ok__button_callback(self):
        print("OK button clicked")
        
        try:
            # Get the file path from the entry field
            file_path = self.entry_file_path.get()
            
            # Read the image file as bytes
            with open(file_path, 'rb') as f:
                image_bytes = f.read()
            
            # Generate caption using the image bytes
            caption_result = generate(image_bytes, test_transform, model, vocab, 20, device)[5:-6]
            print("Caption Result: ", caption_result)
            
            # Remove any existing caption label
            runner_frame = self.main_frame.winfo_children()[0]  # Get the runner page frame
            for widget in runner_frame.winfo_children():
                if isinstance(widget, tk.Label) and widget.cget("text").startswith("Caption:"):
                    widget.destroy()
            
            # Create a label to display the caption
            caption_label = tk.Label(
                runner_frame,
                text=f"Caption: {caption_result}",
                font=("Arial", 18),
                fg="#FFFFFF",
                bg="#242424",
                wraplength=780  # Wrap text to fit within the frame width
            )
            caption_label.grid(row=2, column=0, columnspan=7, padx=20, pady=(0, 20), sticky="nsew")
            
        except Exception as e:
            print(f"Error generating caption: {e}")
            messagebox.showerror("Error", "Failed to generate caption for the image")

# "Config" option menu page methods
    def config_optionMenu_page(self):
        print("Config Page")

        # Create config page frame
        config_page_frame = tk.Frame(self.main_frame, bg="#1CBB0F")
        config_page_frame.pack(fill=tk.BOTH, expand=True)

        label_test = tk.Label(config_page_frame, text="Config Page", font=("Arial Bold", 25), fg="#E8006D")
        label_test.pack(pady=80)

# "Tools" option menu page methods
    def tools_optionMenu_page(self):
        print("Tools Page")

        # Create tools page frame
        tools_page_frame = tk.Frame(self.main_frame, bg="#8F0FBB")
        tools_page_frame.pack(fill=tk.BOTH, expand=True)

        label_test = tk.Label(tools_page_frame, text="Tools Page", font=("Arial Bold", 25), fg="#E8006D")
        label_test.pack(pady=80)

# "Settings" option menu page methods
    def settings_optionMenu_page(self):
        print("Settings Page")

        # Create settings page frame
        settings_page_frame = tk.Frame(self.main_frame, bg="#242424")
        settings_page_frame.pack(fill=tk.BOTH, expand=True)

        # Create label for API settings
        api_label = tk.Label(
            settings_page_frame,
            text="API Settings (Under construction)",
            font=("Arial Bold", 24),
            fg="#FFFFFF",
            bg="#242424"
        )
        api_label.pack(pady=(20, 10))

        # Create container frame for input and button
        input_container = tk.Frame(settings_page_frame, bg="#242424")
        input_container.pack(fill="x", padx=20, pady=(0, 20))

        # Create the entry for API key
        self.api_entry = customtkinter.CTkEntry(
            input_container,
            placeholder_text="   Set API Key",
            font=("Arial Bold", 12),
            fg_color="#000000",
            border_width=2,
            border_color="#FFFFFF",
            corner_radius=0,
            height=35,  # Match the height of runner page input
            state="disabled"  # Set input field to disabled state
        )
        self.api_entry.pack(side="left", fill="x", expand=True, padx=(0, 10), ipady=4)

        # Create the Set API button
        self.set_api_button = customtkinter.CTkButton(
            input_container,
            text="Set API",
            font=("Arial Bold", 12),
            fg_color="transparent",
            border_width=2,
            border_color="#FFFFFF",
            text_color=("gray10", "#DCE4EE"),  # Gray text for disabled state
            hover_color="#000000",
            corner_radius=0,
            command=self.set_api_callback,
            state="disabled",  # Set button to disabled state
            height=35  # Match the height of runner page buttons
        )
        self.set_api_button.pack(side="right", padx=(10, 0), ipady=4)

    def set_api_callback(self):
        api_key = self.api_entry.get()
        if api_key:
            print(f"API Key set: {api_key}")
            # Add your API key handling logic here
        else:
            messagebox.showwarning("Warning", "Please enter an API key")

# "About" option menu page methods
    def about_optionMenu_page(self):
        print("About Page")

        # Create about page frame
        about_page_frame = tk.Frame(self.main_frame, bg="#242424")
        about_page_frame.pack(fill=tk.BOTH, expand=True)
        
        '''
        # Define the new about text
        about_text = ("Plate Detection 1.0.0" +
                    " is an automation suite powered by customtkinter python. This software can be used for processing and parsing data and much more." +
                    "\n\n\n\n")

        # Create a Text widget for the new about text within the about_page_frame
        about_text_widget = tk.Text(about_page_frame, font=("Consolas", 14, "bold"), fg="#FFFFFF", bg="#242424", wrap="word", borderwidth=0, highlightthickness=0)
        about_text_widget.pack(side="top", fill="both", expand=True, padx=20, pady=20)
        about_text_widget.insert("1.0", about_text)
        about_text_widget.configure(state="disabled", inactiveselectbackground=about_text_widget.cget("selectbackground"))
        '''
        
        # Define the new about text
        about_text = (" This project aims to develop an AI-powered system capable of generating descriptive text for images. Given an image as input, the system will analyze its content and produce a meaningful and contextually relevant textual description as output. The task, known as image captioning, is a complex challenge that lies at the intersection of computer vision and natural language processing (NLP). The system must not only recognize objects, actions, and scenes within an image but also structure this information into a fluent, human-like description." +
                    "\n\n\n")
                    
        # Create a Text widget for the new about text within the about_page_frame
        about_text_widget = tk.Text(about_page_frame, font=("Consolas", 14, "bold"), fg="#FFFFFF", bg="#242424", wrap="word", borderwidth=0, highlightthickness=0)
        about_text_widget.pack(side="top", fill="both", expand=True, padx=20, pady=20)
        about_text_widget.insert("1.0", about_text)
        
        # Add the specified text in red
        red_text = ("Delevoper: Mohammad Hossein Mohammadi" +
                    "\n" +
                    "Collaborators: Mr. Mehdi Kazemi" +
                     "\n" +
                    "Mentor: Mr. Mehdi Kazemi" +
                    "\n\n\n" +
                    "Contact US:" +
                    "\n" +
                    "Mohammad Hossein Mohammadi -> Mohammadimir2017@gmail.com")
        about_text_widget.insert("end", red_text)
        
        # Apply tag to make text red
        about_text_widget.tag_add("red_text", "3.0", "end")
        about_text_widget.tag_config("red_text", foreground="green")
        
        about_text_widget.configure(state="disabled", inactiveselectbackground=about_text_widget.cget("selectbackground"))

    # Create a container frame for the buttons to control their positions more precisely
        button_container = tk.Frame(about_page_frame, bg="#242424", height=40)  # Height is optional, adjust as needed
        button_container.pack(side="bottom", fill="x", padx=20, pady=20)
        button_container.pack_propagate(False)  # Prevents the container from shrinking to fit the buttons

    # Create image button
        # Unlicence image button
        self.unlicence_img_data = Image.open("application/windows/assets/images/frontend/original/unlicence.png")
        self.unlicence_img = CTkImage(dark_image=self.unlicence_img_data, light_image=self.unlicence_img_data, size=(15, 15))
        
        # Donate image button
        self.donate_img_data = Image.open("application/windows/assets/images/frontend/original/donate.png")
        self.donate_img = CTkImage(dark_image=self.donate_img_data, light_image=self.donate_img_data, size=(15, 15))
        
        # Repository image button
        self.repository_img_data = Image.open("application/windows/assets/images/frontend/original/repository.png")
        self.repository_img = CTkImage(dark_image=self.repository_img_data, light_image=self.repository_img_data, size=(15, 15))
    
    # Create widget
        # Create the Unlicence button at the left side of the button_container
        self.unlicence_button = CTkButton(button_container, image=self.unlicence_img, text="Unlicence", font=("Arial Bold", 12), fg_color="transparent", border_width=2, border_color="#FFFFFF", text_color="#FFFFFF", hover_color="#000000", corner_radius=0, command=self.unlicence_button_callback)
        self.unlicence_button.pack(side="left", padx=(0, 10), ipady=4)  # Added padding to create space between the buttons

        # Create the Donate button to the right of the Licence button
        self.donate_button = CTkButton(button_container, image=self.donate_img, text="Donate", font=("Arial Bold", 12), fg_color="transparent", border_width=2, border_color="#FFFFFF", text_color="#FFFFFF", hover_color="#000000", corner_radius=0, command=self.donate_button_callback)
        self.donate_button.pack(side="left", padx=(0, 10), ipady=4)  # This places it immediately to the right of the Licence button

        # Create the Repository button to the right of the Donate button
        self.repository_button = CTkButton(button_container, image=self.repository_img, text="Repository", font=("Arial Bold", 12), fg_color="transparent", border_width=2, border_color="#FFFFFF", text_color="#FFFFFF", hover_color="#000000", corner_radius=0, command=self.repository_button_callback)
        self.repository_button.pack(side="left", ipady=4)  # This places it immediately to the right of the Donate button

    def unlicence_button_callback(self):
        # Create a new window
        license_window = tk.Toplevel()
        
        # Set window size and position
        screen_width = license_window.winfo_screenwidth()
        screen_height = license_window.winfo_screenheight()
        window_width = 800
        window_height = 600
        x = (screen_width / 2) - (window_width / 2)
        y = (screen_height / 2) - (window_height / 2)
        license_window.geometry(f"{window_width}x{window_height}+{int(x)}+{int(y)}")
        license_window.title("The Unlicense")
        license_window.resizable(0, 0)
        
        # Configure window background
        license_window.configure(bg="#242424")
        
        # Create main frame
        main_frame = tk.Frame(license_window, bg="#242424")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Create title label
        title_label = tk.Label(
            main_frame,
            text="The Unlicense",
            font=("Arial Bold", 24),
            fg="#FFFFFF",
            bg="#242424"
        )
        title_label.pack(pady=(0, 20))
        
        # Create text widget for license content
        license_text = tk.Text(
            main_frame,
            font=("Consolas", 12),
            fg="#FFFFFF",
            bg="#242424",
            wrap=tk.WORD,
            padx=10,
            pady=10,
            borderwidth=0,
            highlightthickness=0
        )
        license_text.pack(fill=tk.BOTH, expand=True)
        
        # Insert license text
        license_content = """This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or distribute this software, either in source code form or as a compiled binary, for any purpose, commercial or non-commercial, and by any means.

In jurisdictions that recognize copyright laws, the author or authors of this software dedicate any and all copyright interest in the software to the public domain. We make this dedication for the benefit of the public at large and to the detriment of our heirs and successors. We intend this dedication to be an overt act of relinquishment in perpetuity of all present and future rights to this software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE."""
        
        license_text.insert("1.0", license_content)
        license_text.configure(state="disabled")  # Make text read-only
        
        # Create close button
        close_button = CTkButton(
            main_frame,
            text="Close",
            font=("Arial Bold", 12),
            fg_color="#1E90FF",
            hover_color="#1873CC",
            command=license_window.destroy
        )
        close_button.pack(pady=(20, 0))
        
        # Make window modal
        license_window.transient(self.app)
        license_window.grab_set()

    def donate_button_callback(self):
        # Create a new window
        donate_window = tk.Toplevel()
        
        # Set window size and position
        screen_width = donate_window.winfo_screenwidth()
        screen_height = donate_window.winfo_screenheight()
        window_width = 800
        window_height = 350
        x = (screen_width / 2) - (window_width / 2)
        y = (screen_height / 2) - (window_height / 2)
        donate_window.geometry(f"{window_width}x{window_height}+{int(x)}+{int(y)}")
        donate_window.title("Support Our Project")
        donate_window.resizable(0, 0)
        
        # Configure window background
        donate_window.configure(bg="#242424")
        
        # Create main frame
        main_frame = tk.Frame(donate_window, bg="#242424")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Create title label
        title_label = tk.Label(
            main_frame,
            text="🚀 Support Our Project! 🚀",
            font=("Arial Bold", 24),
            fg="#FFFFFF",
            bg="#242424"
        )
        title_label.pack(pady=(0, 20))
        
        # Create text widget for donate content
        donate_text = tk.Text(
            main_frame,
            font=("Consolas", 12),
            fg="#FFFFFF",
            bg="#242424",
            wrap=tk.WORD,
            padx=10,
            pady=10,
            borderwidth=0,
            highlightthickness=0
        )
        donate_text.pack(fill=tk.BOTH, expand=True)
        
        # Insert donate text
        donate_content = """If you like our software and find it useful, the best way to support us is by visiting our GitHub repository and giving it a ⭐️ star!

Every star helps us stay motivated and improve the project even more. So, if you think this project is awesome, head over to the link below and show your support!

🔗 GitHub Repository

Thank you for being part of our journey! ❤️"""
        
        donate_text.insert("1.0", donate_content)
        
        # Add clickable link
        donate_text.tag_add("link", "4.2", "4.25")  # Only tag the "GitHub Repository Link" text
        donate_text.tag_config("link", foreground="#1E90FF", underline=True)
        donate_text.tag_bind("link", "<Button-1>", lambda e: webbrowser.open("https://github.com/Mohammadimh76/image-caption-generator-pytorch"))
        donate_text.tag_bind("link", "<Enter>", lambda e: donate_text.config(cursor="hand2"))
        donate_text.tag_bind("link", "<Leave>", lambda e: donate_text.config(cursor=""))
        
        donate_text.configure(state="disabled")  # Make text read-only
        
        # Create repository button
        repository_button = CTkButton(
            main_frame,
            text="Repository",
            image=self.repository_img,
            font=("Arial Bold", 12),
            fg_color="transparent",
            border_width=2,
            border_color="#FFFFFF",
            text_color="#FFFFFF",
            hover_color="#000000",
            corner_radius=0,
            command=lambda: webbrowser.open("https://github.com/Mohammadimh76/image-caption-generator-pytorch")
        )
        repository_button.pack(pady=(10, 20))
        
        # Create close button
        close_button = CTkButton(
            main_frame,
            text="Close",
            font=("Arial Bold", 12),
            fg_color="#1E90FF",
            hover_color="#1873CC",
            command=donate_window.destroy
        )
        close_button.pack(pady=(0, 0))
        
        # Make window modal
        donate_window.transient(self.app)
        donate_window.grab_set()

    def repository_button_callback(self):
        """Open the repository link in the default web browser"""
        repository_url = "https://github.com/Mohammadimh76/image-caption-generator-pytorch"  # Replace with your actual repository URL
        webbrowser.open_new(repository_url)

if __name__ == "__main__":
    App().app.mainloop()
