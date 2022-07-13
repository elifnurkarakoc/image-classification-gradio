import gradio as gr
from transformers import pipeline

pipe = pipeline(task="image-classification",
                model="microsoft/beit-base-patch16-224-pt22k-ft22k")
gr.Interface.from_pipeline(pipe,
                           title="Image Classification",
                           description="Object Recognition using Microsoft BEIT",
                           ).launch()
