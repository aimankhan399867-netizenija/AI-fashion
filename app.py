import os
import torch
import gradio as gr
from PIL import Image
from groq import Groq
from diffusers import StableDiffusionPipeline

# ==============================
# 🔐 SET YOUR GROQ API KEY HERE
# ==============================

os.environ["GROQ_API_KEY"] = "PASTE_YOUR_GROQ_API_KEY_HERE"

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# ==============================
# 🤖 LOAD IMAGE GENERATION MODEL
# ==============================

model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(model_id)

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)

# ==============================
# 🧠 DESCRIBE DRESS USING LLM
# ==============================

def describe_dress():

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Describe a fashionable modern dress in detail for a professional photoshoot.",
            }
        ],
        model="llama-3.3-70b-versatile",
    )

    return chat_completion.choices[0].message.content


# ==============================
# 🎨 GENERATE FULL BODY MODEL
# ==============================

def generate_model(image):

    dress_description = describe_dress()

    prompt = f"""
    Luxury high-end fashion editorial photoshoot.

    Full body South Asian female professional model wearing:
    {dress_description}

    Standing confidently.
    Elegant posture.
    Minimal white studio background.
    Soft studio lighting.
    Professional photography.
    Ultra realistic skin texture.
    Sharp focus.
    Full body visible head to toe.
    Fashion magazine quality.
    """

    negative_prompt = """
    ugly, distorted face, bad hands, extra fingers,
    deformed body, blurry, cartoon, anime,
    low quality, cropped, duplicate body
    """

    result = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=30,
        guidance_scale=7.5
    ).images[0]

    return result


# ==============================
# 🎛️ GEN Z STYLE UI (HCI Focused)
# ==============================

with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
        # 👗 AI Fashion Try-On Studio
        Upload your dress and generate a full-body luxury fashion model wearing it ✨
        """
    )

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Dress Image")
        output_image = gr.Image(label="Generated Fashion Model")

    generate_btn = gr.Button("✨ Generate Fashion Model")

    generate_btn.click(
        generate_model,
        inputs=image_input,
        outputs=output_image
    )

# ==============================
# 🚀 LAUNCH APP
# ==============================

demo.launch()
