# imagen-streamlit app with stability ai model

A *super minimal* Streamlit app for image generation with Stable Diffusion 2.1, XL 1.0 and Turbo.

Includes:
- Text to image (txt2img) (v2.1, XL 1.0 and Tubo)
- Image to image (img2img) (v2.1, XL 1.0 and Turbo)
- Inpainting (v2.0 and XL 1.0)
- Negative prompt input for all methods.

## Getting Started
To run this Streamlit app locally, follow these steps:

### Prerequisites
- Python 3.10+
- pip (Python package manager)

#### Installation
1. Clone this repository to your local machine:
```git clone https://```
2. Navigate to the project directory:
```cd imagen_streamlit_app```
3. Install the required Python dependencies:
To not mess up your environment, you can use virtualenv.
* create a virtualenv
```
python3 -m venv <path to your venv>
```
* activate the virtualenv
```
source venv/bin/activate
```
* for GPU users
```
pip install -r requirements.txt
```
* for CPU users (there's some packages are not required for the CPU)
```
pip install -r requirements_cpu.txt
```
#### Running the App
Once you've installed the dependencies, you can run the Streamlit app using the following command:

```
streamlit run main.py
```
This will start the Streamlit server and open the app in your default web browser.

Troubleshooting
If you encounter any "module not found" issues, try deactivating and reactivating the virtual environment:
```
deactivate
source venv/bin/activate  # On Unix/macOS
```

The first time it runs, it will download the model from Hugging Face automatically.

Images are automatically saved in an `outputs/` folder, along with their prompt.

## Usage
### **Text to Image (txt2img)**
1. Select the "Text to Image (txt2img)" tab.
2. Enter your desired prompt text in the "Prompt" text area. This is the text that will be used to generate the image.
3. Optionally, provide a negative prompt in the "Negative prompt" text area to guide the generation process.
4. Adjust the inference settings using the sliders:
  * `Number of inference steps`: Set the number of inference steps to control the complexity of the generated image.
  * `Guidance scale`: Adjust the guidance scale to influence the direction of the image generation.
  * `Number of generated images`: Specify the number of images to generate.
  * Enable/disable additional options like attention slicing and CPU offload as needed.
  * Click the "Generate image" button to initiate the image generation process.

### **Image to Image (img2img)**
1. Select the "Image to image (img2img)" tab.
2. Upload an image using the file uploader provided.
3. Select the desired model version from the dropdown menu.
4. Adjust the strength slider to refine the generated image (optional).
5. Enter your desired prompt text in the "Prompt" text area. This is the text that will be used to generate the image.
6. Optionally, provide a negative prompt in the "Negative prompt" text area to guide the generation process.
7. Click the "Generate image" button to start the image generation process.

### **Inpainting**
1. Select the "Inpainting" tab.
2. Upload an image using the file uploader provided.
3. Adjust the brush size using the `Brush Size` number input.
4. Use the canvas to draw on the uploaded image to create a mask for inpainting.
5. Adjust the model version and strength sliders as needed.
5. Click the "Generate image" button to begin the inpainting process.