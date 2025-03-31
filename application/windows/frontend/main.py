import tkinter as tk
from tkinter import filedialog, messagebox
import webbrowser
from PIL import Image, ImageTk
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as TT
from torchvision.transforms import functional as TF
from torchvision.models import resnet50, ResNet50_Weights
from torchtext.data.utils import get_tokenizer
from customtkinter import (
    set_appearance_mode, CTk, CTkFrame, CTkButton, CTkImage,
    CTkEntry
)

# Constants
APP_TITLE = "Alpaca -- v1.0.0"
APP_WIDTH = 1100
APP_HEIGHT = 645
BACKGROUND_COLOR = "#242424"
TEXT_COLOR = "#FFFFFF"
BUTTON_COLORS = {
    "active": "#242424",
    "inactive": "#242424",
    "active_text": "#1E90FF",
    "inactive_text": "#FFFFFF"
}

# Model Configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = 'D:/EmKay/ImageCaptioning/image-caption-generator-pytorch/application/windows/backend/ptFiles/model.pt'
VOCAB_PATH = 'D:/EmKay/ImageCaptioning/image-caption-generator-pytorch/application/windows/backend/ptFiles/vocab.pt'

# Model Parameters
EMBED_SIZE = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 2
DROPOUT_EMBD = 0.5
DROPOUT_RNN = 0.5
MAX_SEQ_LENGTH = 20

# Image Transform
IMAGE_TRANSFORM = TT.Compose([
    TT.Resize((224, 224)),
    TT.ToTensor(),
    TT.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# Load model resources
vocab = torch.load(VOCAB_PATH)
tokenizer = get_tokenizer('basic_english')

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
        return self.bn(self.fc(features))

class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, dropout_embd, dropout_rnn, max_seq_length=20):
        super().__init__()
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
        return self.linear(outputs)

    def generate(self, features, captions):
        if captions:
            embeddings = self.dropout_embd(self.embedding(captions))
            inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        else:
            inputs = features.unsqueeze(1)
        outputs, _ = self.lstm(inputs)
        return self.linear(outputs)

class ImageCaptioning(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, dropout_embd, dropout_rnn, max_seq_length=20):
        super().__init__()
        self.encoder = Encoder(embed_size)
        self.decoder = Decoder(embed_size, hidden_size, vocab_size, num_layers, dropout_embd, dropout_rnn, max_seq_length)

    def forward(self, images, captions):
        features = self.encoder(images)
        return self.decoder(features, captions)

    def generate(self, images, captions):
        features = self.encoder(images)
        return self.decoder.generate(features, captions)

def generate_caption(image_bytes, transform, model_path, vocab, max_seq_length, device):
    img_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    model = torch.load(model_path, map_location=torch.device(device))
    model.eval()
    
    image_transformed = transform(TF.to_pil_image(image)).unsqueeze(0)
    image = image_transformed.to(device)
    src, indices = [], []
    
    caption = ''
    itos = vocab.get_itos()

    for _ in range(max_seq_length):
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

class ImageCaptioningApp:
    def __init__(self):
        self.app = CTk()
        self.current_page = None
        self.buttons = {}
        self.setup_window()
        self.create_option_menu()
        self.main_frame = self.create_main_frame()
        self.create_pages()
        self.switch_page("home")

    def setup_window(self):
        screen_width = self.app.winfo_screenwidth()
        screen_height = self.app.winfo_screenheight()
        x = (screen_width / 2) - (APP_WIDTH / 2)
        y = (screen_height / 2) - (APP_HEIGHT / 2)
        self.app.geometry(f"{APP_WIDTH}x{APP_HEIGHT}+{int(x)}+{int(y)}")
        self.app.title(APP_TITLE)
        self.app.resizable(0, 0)
        set_appearance_mode("dark")

    def create_main_frame(self):
        main_frame = tk.Frame(self.app, bg=BACKGROUND_COLOR)
        main_frame.pack(expand=True, fill="both")
        return main_frame

    def create_option_menu(self):
        option_menu_frame = CTkFrame(
            master=self.app,
            width=176,
            height=35,
            fg_color=BACKGROUND_COLOR,
            corner_radius=0
        )
        option_menu_frame.pack_propagate(0)
        option_menu_frame.pack(fill="x", side="top")

        button_names = ["home", "runner", "settings", "about"]
        for name in button_names:
            button = CTkButton(
                master=option_menu_frame,
                text=name.capitalize(),
                fg_color=BACKGROUND_COLOR,
                font=("Arial Bold", 14),
                text_color=TEXT_COLOR,
                hover_color=BACKGROUND_COLOR,
                anchor="center",
                corner_radius=0,
                width=len(name) + 2,
                command=lambda name=name: self.switch_page(name)
            )
            button.pack(side="left", padx=(15, 0), pady=(5, 0), ipady=3)
            self.buttons[name] = button

    def switch_page(self, page_name):
        self.current_page = page_name
        self.update_button_colors()
        self.show_page(page_name)

    def update_button_colors(self):
        for name, button in self.buttons.items():
            if name == self.current_page:
                button.configure(
                    fg_color=BUTTON_COLORS["active"],
                    text_color=BUTTON_COLORS["active_text"]
                )
            else:
                button.configure(
                    fg_color=BUTTON_COLORS["inactive"],
                    text_color=BUTTON_COLORS["inactive_text"]
                )

    def create_pages(self):
        self.pages = {
            "home": self.home_page,
            "runner": self.runner_page,
            "config": self.config_page,
            "tools": self.tools_page,
            "settings": self.settings_page,
            "about": self.about_page,
        }

    def show_page(self, page_name):
        self.clear_current_page()
        if page_name in self.pages:
            self.pages[page_name]()

    def clear_current_page(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()

    def home_page(self):
        home_page_frame = tk.Frame(self.main_frame, bg=BACKGROUND_COLOR)
        home_page_frame.pack(fill=tk.BOTH, expand=True)

        title_label = tk.Label(
            home_page_frame,
            text="Welcome to Image Caption Generator",
            font=("Arial Bold", 24),
            fg=TEXT_COLOR,
            bg=BACKGROUND_COLOR
        )
        title_label.pack(pady=(20, 10))

        doc_text = tk.Text(
            home_page_frame,
            font=("Consolas", 12),
            fg=TEXT_COLOR,
            bg=BACKGROUND_COLOR,
            wrap=tk.WORD,
            padx=20,
            pady=10,
            borderwidth=0,
            highlightthickness=0
        )
        doc_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

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
        doc_text.configure(state="disabled")

    def runner_page(self):
        runner_page_frame = tk.Frame(self.main_frame, bg=BACKGROUND_COLOR)
        runner_page_frame.pack(fill=tk.BOTH, expand=True)
        runner_page_frame.grid_rowconfigure(0, weight=1)
        runner_page_frame.grid_columnconfigure(0, weight=1)

        self.image_frame = tk.Frame(runner_page_frame, bg=BACKGROUND_COLOR, width=800, height=400)
        self.image_frame.grid(row=0, column=0, columnspan=7, padx=20, pady=(20, 10), sticky="nsew")
        self.image_frame.grid_propagate(False)
        
        self.image_label = tk.Label(self.image_frame, bg=BACKGROUND_COLOR)
        self.image_label.pack(expand=True, fill="both")

        self.setup_runner_buttons(runner_page_frame)

    def setup_runner_buttons(self, runner_page_frame):
        # Load button images
        self.edit_img = self.load_button_image("editSquare_icon.png")
        self.upload_img = self.load_button_image("upload_icon.png")
        self.browse_img = self.load_button_image("browse_icon.png")
        self.remove_img = self.load_button_image("remove_icon.png")

        # Create entry field
        self.entry_file_path = CTkEntry(
            runner_page_frame,
            placeholder_text="   Set File Path Address",
            font=("Arial Bold", 12),
            fg_color="#000000",
            border_width=2,
            border_color=TEXT_COLOR,
            corner_radius=0
        )
        self.entry_file_path.grid(row=3, column=0, padx=(20, 0), pady=(20, 10), sticky="nsew")

        # Create buttons
        self.browse_button = CTkButton(
            runner_page_frame,
            text="Browse",
            image=self.browse_img,
            font=("Arial Bold", 12),
            fg_color="transparent",
            border_width=2,
            border_color=TEXT_COLOR,
            text_color=TEXT_COLOR,
            hover_color="#000000",
            corner_radius=0,
            command=self.browse_button_callback
        )
        self.browse_button.grid(row=3, column=4, padx=(20, 20), pady=(20, 10), ipady=4, sticky="nsew")

        self.disabled_button = CTkButton(
            runner_page_frame,
            text="Disabled",
            font=("Arial Bold", 12),
            state="disabled",
            fg_color="transparent",
            border_width=0,
            border_color=TEXT_COLOR,
            text_color=("gray10", "#DCE4EE"),
            hover_color="#000000",
            corner_radius=0
        )
        self.disabled_button.grid(row=3, column=6, padx=(0, 20), pady=(20, 10), ipady=4, sticky="nsew")
        
        self.remove_button = CTkButton(
            runner_page_frame,
            text="Remove",
            font=("Arial Bold", 12),
            fg_color="transparent",
            border_width=2,
            text_color=("gray10", "#DCE4EE"),
            hover_color="#000000",
            corner_radius=0,
            command=self.remove_button_callback
        )
        self.remove_button.grid(row=3, column=5, padx=(20, 10), pady=(20, 10), ipady=4, sticky="nsew")
        self.remove_button.grid_remove()

    def load_button_image(self, image_name):
        image_data = Image.open(f"application/windows/assets/images/frontend/original/{image_name}")
        return CTkImage(dark_image=image_data, light_image=image_data, size=(15, 15))

    def browse_button_callback(self):
        file_path = filedialog.askopenfilename(
            title="Select File",
            filetypes=[("Image Files", "*.png;*.jpg")]
        )

        if file_path:
            self.entry_file_path.delete(0, "end")
            self.entry_file_path.insert(0, file_path)
            self.load_and_display_image(file_path)
            self.update_buttons_for_image_selection()

    def load_and_display_image(self, file_path):
        try:
            image = Image.open(file_path)
            aspect_ratio = image.width / image.height
            max_width = 780
            max_height = 380
            
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
            
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=photo)
            self.image_label.image = photo

        except Exception as e:
            messagebox.showerror("Error", "Failed to load the image")

    def update_buttons_for_image_selection(self):
        self.browse_button.configure(text="Edit", fg_color="red", image=self.edit_img)
        self.disabled_button.configure(
            state="normal",
            text="OK",
            fg_color="green",
            border_width=2,
            border_color=TEXT_COLOR,
            image=self.upload_img,
            command=self.ok_button_callback
        )
        self.remove_button.grid(padx=(0, 20))
        self.remove_button.configure(
            state="normal",
            text="Remove",
            fg_color="#BB8B0C",
            text_color=TEXT_COLOR,
            image=self.remove_img
        )

    def remove_button_callback(self):
        self.browse_button.configure(
            text="Browse",
            state="normal",
            fg_color="transparent",
            border_width=2,
            text_color=("gray10", "#DCE4EE"),
            image=self.browse_img
        )
        self.disabled_button.configure(
            state="disabled",
            text="Disabled",
            fg_color="transparent",
            border_color="#000000",
            border_width=0,
            text_color=("gray10", "#DCE4EE"),
            image=None
        )
        self.entry_file_path.delete(0, "end")
        self.entry_file_path.configure(placeholder_text="Set File Path Address")
        self.image_label.configure(image="")
        self.remove_button.grid_remove()

    def ok_button_callback(self):
        try:
            file_path = self.entry_file_path.get()
            with open(file_path, 'rb') as f:
                image_bytes = f.read()
            
            caption_result = generate_caption(
                image_bytes,
                IMAGE_TRANSFORM,
                MODEL_PATH,
                vocab,
                MAX_SEQ_LENGTH,
                DEVICE
            )[5:-6]
            
            self.update_caption_display(caption_result)
            
        except Exception as e:
            messagebox.showerror("Error", "Failed to generate caption for the image")

    def update_caption_display(self, caption):
        runner_frame = self.main_frame.winfo_children()[0]
        for widget in runner_frame.winfo_children():
            if isinstance(widget, tk.Label) and widget.cget("text").startswith("Caption:"):
                widget.destroy()
        
        caption_label = tk.Label(
            runner_frame,
            text=f"Caption: {caption}",
            font=("Arial", 18),
            fg=TEXT_COLOR,
            bg=BACKGROUND_COLOR,
            wraplength=780
        )
        caption_label.grid(row=2, column=0, columnspan=7, padx=20, pady=(0, 20), sticky="nsew")

    def config_page(self):
        config_page_frame = tk.Frame(self.main_frame, bg="#1CBB0F")
        config_page_frame.pack(fill=tk.BOTH, expand=True)
        label = tk.Label(config_page_frame, text="Config Page", font=("Arial Bold", 25), fg="#E8006D")
        label.pack(pady=80)

    def tools_page(self):
        tools_page_frame = tk.Frame(self.main_frame, bg="#8F0FBB")
        tools_page_frame.pack(fill=tk.BOTH, expand=True)
        label = tk.Label(tools_page_frame, text="Tools Page", font=("Arial Bold", 25), fg="#E8006D")
        label.pack(pady=80)

    def settings_page(self):
        settings_page_frame = tk.Frame(self.main_frame, bg=BACKGROUND_COLOR)
        settings_page_frame.pack(fill=tk.BOTH, expand=True)

        api_label = tk.Label(
            settings_page_frame,
            text="API Settings (Under construction)",
            font=("Arial Bold", 24),
            fg=TEXT_COLOR,
            bg=BACKGROUND_COLOR
        )
        api_label.pack(pady=(20, 10))

        input_container = tk.Frame(settings_page_frame, bg=BACKGROUND_COLOR)
        input_container.pack(fill="x", padx=20, pady=(0, 20))

        self.api_entry = CTkEntry(
            input_container,
            placeholder_text="   Set API Key",
            font=("Arial Bold", 12),
            fg_color="#000000",
            border_width=2,
            border_color=TEXT_COLOR,
            corner_radius=0,
            height=35,
            state="disabled"
        )
        self.api_entry.pack(side="left", fill="x", expand=True, padx=(0, 10), ipady=4)

        self.set_api_button = CTkButton(
            input_container,
            text="Set API",
            font=("Arial Bold", 12),
            fg_color="transparent",
            border_width=2,
            border_color=TEXT_COLOR,
            text_color=("gray10", "#DCE4EE"),
            hover_color="#000000",
            corner_radius=0,
            command=self.set_api_callback,
            state="disabled",
            height=35
        )
        self.set_api_button.pack(side="right", padx=(10, 0), ipady=4)

    def set_api_callback(self):
        api_key = self.api_entry.get()
        if api_key:
            print(f"API Key set: {api_key}")
        else:
            messagebox.showwarning("Warning", "Please enter an API key")

    def about_page(self):
        about_page_frame = tk.Frame(self.main_frame, bg=BACKGROUND_COLOR)
        about_page_frame.pack(fill=tk.BOTH, expand=True)
        
        about_text = (
            "This project aims to develop an AI-powered system capable of generating descriptive text for images. "
            "Given an image as input, the system will analyze its content and produce a meaningful and contextually "
            "relevant textual description as output. The task, known as image captioning, is a complex challenge that "
            "lies at the intersection of computer vision and natural language processing (NLP). The system must not "
            "only recognize objects, actions, and scenes within an image but also structure this information into a "
            "fluent, human-like description.\n\n\n"
        )
                    
        about_text_widget = tk.Text(
            about_page_frame,
            font=("Consolas", 14, "bold"),
            fg=TEXT_COLOR,
            bg=BACKGROUND_COLOR,
            wrap="word",
            borderwidth=0,
            highlightthickness=0
        )
        about_text_widget.pack(side="top", fill="both", expand=True, padx=20, pady=20)
        about_text_widget.insert("1.0", about_text)
        
        red_text = (
            "Delevoper: Mohammad Hossein Mohammadi\n"
            "Collaborators: Mr. Mehdi Kazemi\n"
            "Mentor: Mr. Mehdi Kazemi\n\n\n"
            "Contact US:\n"
            "Mohammad Hossein Mohammadi -> Mohammadimir2017@gmail.com"
        )
        about_text_widget.insert("end", red_text)
        
        about_text_widget.tag_add("red_text", "3.0", "end")
        about_text_widget.tag_config("red_text", foreground="green")
        about_text_widget.configure(state="disabled", inactiveselectbackground=about_text_widget.cget("selectbackground"))

        self.setup_about_buttons(about_page_frame)

    def setup_about_buttons(self, about_page_frame):
        button_container = tk.Frame(about_page_frame, bg=BACKGROUND_COLOR, height=40)
        button_container.pack(side="bottom", fill="x", padx=20, pady=20)
        button_container.pack_propagate(False)

        self.unlicence_img = self.load_button_image("unlicence.png")
        self.donate_img = self.load_button_image("donate.png")
        self.repository_img = self.load_button_image("repository.png")

        self.unlicence_button = CTkButton(
            button_container,
            image=self.unlicence_img,
            text="Unlicence",
            font=("Arial Bold", 12),
            fg_color="transparent",
            border_width=2,
            border_color=TEXT_COLOR,
            text_color=TEXT_COLOR,
            hover_color="#000000",
            corner_radius=0,
            command=self.show_license_window
        )
        self.unlicence_button.pack(side="left", padx=(0, 10), ipady=4)

        self.donate_button = CTkButton(
            button_container,
            image=self.donate_img,
            text="Donate",
            font=("Arial Bold", 12),
            fg_color="transparent",
            border_width=2,
            border_color=TEXT_COLOR,
            text_color=TEXT_COLOR,
            hover_color="#000000",
            corner_radius=0,
            command=self.show_donate_window
        )
        self.donate_button.pack(side="left", padx=(0, 10), ipady=4)

        self.repository_button = CTkButton(
            button_container,
            image=self.repository_img,
            text="Repository",
            font=("Arial Bold", 12),
            fg_color="transparent",
            border_width=2,
            border_color=TEXT_COLOR,
            text_color=TEXT_COLOR,
            hover_color="#000000",
            corner_radius=0,
            command=self.open_repository
        )
        self.repository_button.pack(side="left", ipady=4)

    def show_license_window(self):
        license_window = tk.Toplevel()
        self.setup_modal_window(license_window, "The Unlicense", 800, 600)
        
        main_frame = tk.Frame(license_window, bg=BACKGROUND_COLOR)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        title_label = tk.Label(
            main_frame,
            text="The Unlicense",
            font=("Arial Bold", 24),
            fg=TEXT_COLOR,
            bg=BACKGROUND_COLOR
        )
        title_label.pack(pady=(0, 20))
        
        license_text = tk.Text(
            main_frame,
            font=("Consolas", 12),
            fg=TEXT_COLOR,
            bg=BACKGROUND_COLOR,
            wrap=tk.WORD,
            padx=10,
            pady=10,
            borderwidth=0,
            highlightthickness=0
        )
        license_text.pack(fill=tk.BOTH, expand=True)
        
        license_content = """This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or distribute this software, either in source code form or as a compiled binary, for any purpose, commercial or non-commercial, and by any means.

In jurisdictions that recognize copyright laws, the author or authors of this software dedicate any and all copyright interest in the software to the public domain. We make this dedication for the benefit of the public at large and to the detriment of our heirs and successors. We intend this dedication to be an overt act of relinquishment in perpetuity of all present and future rights to this software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE."""
        
        license_text.insert("1.0", license_content)
        license_text.configure(state="disabled")
        
        close_button = CTkButton(
            main_frame,
            text="Close",
            font=("Arial Bold", 12),
            fg_color="#1E90FF",
            hover_color="#1873CC",
            command=license_window.destroy
        )
        close_button.pack(pady=(20, 0))

    def show_donate_window(self):
        donate_window = tk.Toplevel()
        self.setup_modal_window(donate_window, "Support Our Project", 800, 350)
        
        main_frame = tk.Frame(donate_window, bg=BACKGROUND_COLOR)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        title_label = tk.Label(
            main_frame,
            text="🚀 Support Our Project! 🚀",
            font=("Arial Bold", 24),
            fg=TEXT_COLOR,
            bg=BACKGROUND_COLOR
        )
        title_label.pack(pady=(0, 20))
        
        donate_text = tk.Text(
            main_frame,
            font=("Consolas", 12),
            fg=TEXT_COLOR,
            bg=BACKGROUND_COLOR,
            wrap=tk.WORD,
            padx=10,
            pady=10,
            borderwidth=0,
            highlightthickness=0
        )
        donate_text.pack(fill=tk.BOTH, expand=True)
        
        donate_content = """If you like our software and find it useful, the best way to support us is by visiting our GitHub repository and giving it a ⭐️ star!

Every star helps us stay motivated and improve the project even more. So, if you think this project is awesome, head over to the link below and show your support!

🔗 GitHub Repository

Thank you for being part of our journey! ❤️"""
        
        donate_text.insert("1.0", donate_content)
        
        donate_text.tag_add("link", "4.2", "4.25")
        donate_text.tag_config("link", foreground="#1E90FF", underline=True)
        donate_text.tag_bind("link", "<Button-1>", lambda e: self.open_repository())
        donate_text.tag_bind("link", "<Enter>", lambda e: donate_text.config(cursor="hand2"))
        donate_text.tag_bind("link", "<Leave>", lambda e: donate_text.config(cursor=""))
        
        donate_text.configure(state="disabled")
        
        repository_button = CTkButton(
            main_frame,
            text="Repository",
            image=self.repository_img,
            font=("Arial Bold", 12),
            fg_color="transparent",
            border_width=2,
            border_color=TEXT_COLOR,
            text_color=TEXT_COLOR,
            hover_color="#000000",
            corner_radius=0,
            command=self.open_repository
        )
        repository_button.pack(pady=(10, 20))
        
        close_button = CTkButton(
            main_frame,
            text="Close",
            font=("Arial Bold", 12),
            fg_color="#1E90FF",
            hover_color="#1873CC",
            command=donate_window.destroy
        )
        close_button.pack(pady=(0, 0))

    def setup_modal_window(self, window, title, width, height):
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        x = (screen_width / 2) - (width / 2)
        y = (screen_height / 2) - (height / 2)
        window.geometry(f"{width}x{height}+{int(x)}+{int(y)}")
        window.title(title)
        window.resizable(0, 0)
        window.configure(bg=BACKGROUND_COLOR)
        window.transient(self.app)
        window.grab_set()

    def open_repository(self):
        webbrowser.open_new("https://github.com/Mohammadimh76/image-caption-generator-pytorch")

if __name__ == "__main__":
    ImageCaptioningApp().app.mainloop()
