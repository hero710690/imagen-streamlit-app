import streamlit as st
from sd import
# Main Streamlit app
def main():
    st.title("Image Generation App with Stability AI")

    # Model type selection
    model_type = st.selectbox("Select Model Type", ["Turbo", "XL", "XL Refiner"])

    # Module type selection
    module_type_options = ["Image to Image", "Text to Image", "Inpaint"]
    if model_type == "Turbo":
        module_type_options.remove("Inpaint")  # Turbo doesn't support Inpaint
    module_type = st.selectbox("Select Module Type", module_type_options)

    # Image upload (if applicable)
    if module_type in ["Inpaint", "Image to Image"]:
        uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    # Width and Height adjustment
    width = st.slider("Width", min_value=100, max_value=1000, value=256)
    height = st.slider("Height", min_value=100, max_value=1000, value=256)

    # Number of images to generate
    num_images = st.number_input("Number of Images to Generate", min_value=1, max_value=10, value=1, step=1)

    # Button to generate images
    if st.button("Generate Images"):
        if module_type in ["Inpaint", "Image to Image"]:
            if uploaded_image is None:
                st.error("Please upload an image.")
            else:
                # Call the generate function with user inputs
                generated_images = generate_image(model_type, module_type, image_path=uploaded_image, width=width, height=height, num_images=num_images)
                # Display the generated images
                for img in generated_images:
                    st.image(img, use_column_width=True, caption="Generated Image")
        else:
            # Call the generate function with user inputs
            generated_images = generate_image(model_type, module_type, width=width, height=height, num_images=num_images)
            # Display the generated images
            for img in generated_images:
                st.image(img, use_column_width=True, caption="Generated Image")

if __name__ == "__main__":
    main()
