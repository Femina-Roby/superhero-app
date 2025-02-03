# import streamlit as st
# from transformers import pipeline
# from PIL import Image
# from diffusers import StableDiffusionPipeline
# import torch

# torch.cuda.empty_cache()
# # torch.cuda.memory_summary(device=None, abbreviated=False)  # Optional: Check memory usage


# # Load the GPT-2 text generation pipeline
# generator = pipeline("text-generation", model="gpt2", framework="pt")

# # Streamlit app title and inputs
# st.title("Create Your Own Superhero")
# name = st.text_input("What's your name?")
# color = st.color_picker("Pick your superhero's favorite color:")
# animal = st.text_input("Favorite animal:")
# power = st.text_input("Superpower:")

# if st.button("Create Superhero", type="primary"):
#     # Refined prompt for the story
#     story_prompt = (
#         f"Name: {name}\n"
#     f"Superpower: {power}\n"
#     f"Favorite animal: {animal}\n"
#     f"Favorite color: {color}\n"
#     f"Write a 100-word superhero story about {name}. "
#     f"Include their origin story, how they discovered their powers, and their motivation for fighting for good. Make it fun, imaginative, and positive."
#     )
    
#     # Generate the story
#     story_output = generator(
#     story_prompt,
#     max_length=150,  # Avoid truncation
#     num_return_sequences=1,
#     temperature=0.8,  # Lower temperature for more focused output
#     top_p=0.9         # Top-p sampling for coherent storytelling
# )

#         # pad_token_id=50256,  # Avoid padding issues
#         # )[0]["generated_text"]
#     generated_text = story_output[0]["generated_text"]

#     # Remove the prompt part from the generated text
#     story_start = generated_text.lower().find("write a short and creative superhero story")
#     if story_start != -1:
#         story = generated_text[story_start + len("write a short and creative superhero story"):].strip()
#     else:
#         story = generated_text.strip()
    
#     # Truncate the story to 100 words
#     word_limit = 100
#     story_words = story.split()[:word_limit]
#     trimmed_story = " ".join(story_words)
    
#     # Display the story
#     st.write("### Your Superhero Story:")
#     st.write(trimmed_story)

# # Load the Stable Diffusion pipeline
# pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
# pipe.to("cuda")  # Use GPU if available

# if st.button("Generate Image", type="primary"):
#     with st.spinner("Generating your superhero image..."):
#         # # Load the Stable Diffusion pipeline
#         # pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
#         # pipe.to("cuda")  # Use GPU if available
        
#         # Image generation prompt
#         image_prompt = (
#             f"A superhero named {name}, inspired by a {color} {animal}, "
#             f"with the power of {power}. Vibrant comic-style art."
#         )
        
#         # Generate the image
#         image = pipe(image_prompt).images[0]
        
#         # Display the image
#         st.write("### Your Superhero Image")
#         st.image(image, caption=f"{name} - The Superhero", use_column_width=True)


from dotenv import load_dotenv
import streamlit as st
import os
from PIL import Image
from diffusers import StableDiffusionPipeline
import google.generativeai as genai
import gc
import torch

# Load environment variables
load_dotenv()

# if "GEMINI_API_KEY" in st.secrets:  # Check if the secret is available
#     api_key = st.secrets["GEMINI_API_KEY"]
#     genai.configure(api_key=api_key)
# else:  # Handle missing secret (e.g., during local development)
#     api_key = os.getenv("GEMINI_API_KEY")  # Fallback to .env
#     if api_key:
#         genai.configure(api_key=api_key)
#     else:
#         st.error("GEMINI_API_KEY not found. Set it in Streamlit secrets or .env file.")
#         st.stop()  # Stop execution if no API key is found


# Configure the Gemini API key securely
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("API key not found. Please set it in the .env file.")
else:
    genai.configure(api_key=api_key)


# Select the 'gemini-pro' model
model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])

# Function to generate a superhero story
def get_superhero_story(name, color, animal, power):
    prompt = (
        f"Create a fun, imaginative superhero story about {name}.\n"
        f"Superhero Name: {name}\n"
        f"Superhero Gender: {gender}\n"
        f"Favorite Color: {color}\n"
        f"Spirit Animal: {animal}\n"
        f"Superpower: {power}\n"
        f"Tell the origin story of {name}, how they discovered their powers, and their mission to fight for good. "
        f"Make it engaging, positive, and creative. Keep it within 150 words. It should be pg-rated and suitable for ages below 16. Do not add extra names. When, referring to the colour in the story, do not use the hex code. Instead use the name of colour or a name that is closest to the colour."
    )
    
    response = chat.send_message(prompt)
    return response.text

# Initialize Streamlit app
st.set_page_config(page_title="Superhero Story Generator")
st.header("Create Your Own Superhero")

# User inputs
name = st.text_input("What's your superhero's name?")
gender = st.radio("What's your gender?",
    [ "Male", "Female",":rainbow[Other]","Prefer not to say"],
)
color = st.color_picker("Pick your superhero's favorite color:")
animal = st.text_input("Favorite animal:")
power = st.text_input("Superpower:")


submit = st.button("Generate Superhero Story", type="primary")

# If button is clicked, generate story
if submit:
    if name and color and animal and power:
            story = get_superhero_story(name, color, animal, power)
            st.subheader("Your Superhero Story:")
            st.write(story)
            # Load Stable Diffusion model
            pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
            pipe.to("cpu")  # Use GPU if available

            # Generate the image prompt
            image_prompt = (
            f"A superhero named {name}, inspired by a {color} {animal}, "
            f"with the power of {power}. Vibrant comic-style art. The {animal} is the superhero's sidekick. It should be present in the image alongside the superhero. Ensure that the superhero is present and the animal is standing beside the superhero"
         )
        
    with st.spinner("Generating superhero image..."):
        # Generate and display the image
        image = pipe(image_prompt).images[0]
        st.write("### Your Superhero Image")
        st.image(image, caption=f"{name} - The Superhero", use_column_width=True)

        del image
        pipe.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
    pipe=None
    gc.collect
    torch.cuda.empty_cache()
else:
    st.error("Please fill in all the details before generating the story.")
