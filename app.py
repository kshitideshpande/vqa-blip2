"""
VQA with BLIP-2 — upload an image, ask a question, get an answer.
Model: Salesforce/blip2-opt-2.7b (ViT + Q-Former + OPT-2.7B)
"""

import logging

import gradio as gr
import torch
from PIL import Image
from transformers import Blip2ForConditionalGeneration, Blip2Processor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_ID = "Salesforce/blip2-opt-2.7b"
MAX_NEW_TOKENS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


def load_model() -> tuple[Blip2Processor, Blip2ForConditionalGeneration]:
    logger.info("Loading processor from %s ...", MODEL_ID)
    processor = Blip2Processor.from_pretrained(MODEL_ID)

    logger.info("Loading model on %s ...", DEVICE)
    model = Blip2ForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE
    ).to(DEVICE)
    model.eval()

    logger.info("Model ready.")
    return processor, model


processor, model = load_model()


def extract_answer(raw: str) -> str:
    if "Answer:" in raw:
        return raw.split("Answer:")[-1].strip()
    return raw.strip()


def answer_question(image: Image.Image | None, question: str) -> str:
    if image is None:
        return "Please upload an image first."
    if not question or not question.strip():
        return "Please type a question about the image."

    prompt = f"Question: {question.strip()} Answer:"

    try:
        inputs = processor(image, prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        raw_output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        answer = extract_answer(raw_output)
        logger.info("Q: %s | A: %s", question, answer)
        return answer if answer else "No answer generated — try rephrasing your question."
    except Exception as e:
        logger.error("Inference failed: %s", e)
        return "Something went wrong. Please try a different image or question."


def build_ui() -> gr.Interface:
    return gr.Interface(
        fn=answer_question,
        inputs=[
            gr.Image(type="pil", label="Image"),
            gr.Textbox(label="Question", placeholder="What is in this image?", lines=1),
        ],
        outputs=gr.Textbox(label="Answer", lines=2),
        title="Visual Question Answering — BLIP-2",
        description="Upload any image and ask a question about it in plain English.",
    )


if __name__ == "__main__":
    demo = build_ui()
    demo.launch()