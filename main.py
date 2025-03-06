import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, UnidentifiedImageError
import imageio.v3 as imageio  # Best for reading TIFF files
import imghdr
import io

# Initialize Flask app
app = Flask(__name__, template_folder="templates")

# Load trained model
MODEL_PATH = "water_segmentation.h5"
model = load_model(MODEL_PATH)
IMG_SIZE = (128, 128)  # Adjust based on model input size

# Function to read and preprocess image
def read_input_image(file):
    """
    Reads an uploaded image file, handles TIFF files, and ensures it's in 3-channel RGB format.

    Args:
        file (werkzeug.datastructures.FileStorage): Uploaded image file.

    Returns:
        Image: PIL Image in RGB mode.
    """
    try:
        # Read file bytes
        file_bytes = file.read()
        if not file_bytes:
            raise ValueError("Empty file")

        # Validate file format
        file_type = imghdr.what(None, file_bytes)
        allowed_formats = ["jpeg", "png", "bmp", "gif", "tiff"]
        if file_type not in allowed_formats:
            raise ValueError(f"Unsupported file format: {file_type}")

        # Handle TIFF separately
        if file_type == "tiff":
            image_array = imageio.imread(io.BytesIO(file_bytes))  # Read TIFF
            image = convert_to_rgb(image_array)  # Convert to 3-channel RGB
        else:
            image = Image.open(io.BytesIO(file_bytes))
            image = image.convert("RGB")  # Convert any other format to RGB

        return image

    except UnidentifiedImageError:
        raise ValueError("Failed to process image: Unrecognized or corrupted file")
    except ValueError as e:
        raise ValueError(str(e))
    except Exception as e:
        raise RuntimeError(f"Unexpected error: {str(e)}")

def convert_to_rgb(image_array):
    """
    Converts an image array to 3-channel RGB format.

    Args:
        image_array (numpy.ndarray): The image array.

    Returns:
        PIL.Image: Converted image in RGB format.
    """
    if len(image_array.shape) == 2:  # Grayscale (1 channel)
        image_array = np.stack((image_array,) * 3, axis=-1)  # Convert to RGB
    elif image_array.shape[-1] == 4:  # RGBA (4 channels)
        image_array = image_array[:, :, :3]  # Remove alpha channel
    elif image_array.shape[-1] > 3:  # CMYK or Extra Channels
        image_array = image_array[:, :, :3]  # Keep only RGB
    elif image_array.dtype != np.uint8:  # If 16-bit or 32-bit, convert to 8-bit
        image_array = (255 * (image_array / image_array.max())).astype(np.uint8)

    return Image.fromarray(image_array)

# Preprocess function
def preprocess_image(image):
    """
    Prepares the image for model prediction.
    Resizes, normalizes, and adds batch dimension.

    Args:
        image (PIL.Image): Input image.

    Returns:
        np.array: Processed image for model input.
    """
    image = image.resize(IMG_SIZE)  # Resize to model input size
    image = img_to_array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension 

    return image

# Home route to render HTML form
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]

    try:
        # Read and process image
        image = read_input_image(file)
        processed_image = preprocess_image(image)

        # Model Prediction
        prediction = model.predict(processed_image)
        prediction = (prediction[0] > 0.5).astype(np.uint8)  # Convert to binary mask

        # Convert array to image
        mask_img = Image.fromarray(prediction.squeeze() * 255)  # Scale mask to 0-255
        mask_io = io.BytesIO()
        mask_img.save(mask_io, format="PNG")
        mask_io.seek(0)
        
        return mask_io.read(), 200, {"Content-Type": "image/png"}

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

# Run app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
