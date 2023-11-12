"""Model Inference via Gradio Front End"""
# pylint: disable=W0621,C0103
from typing import Tuple, Callable
import os
import sys
import numpy as np
import gradio as gr
sys.path.append("../")
par_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, par_dir)
from music_generation import MusicGenerator, MODELS

music_gen = None

def generate(
        prompt_:str,
        input_audio_:Tuple[int, np.ndarray],
        top_k_:int,
        top_p_:float,
        temp_:float,
        duration_:int,
        progress_:Callable=gr.Progress()
    ) -> Tuple:
    """Generates music based on the given prompt and input audio.

    Args:
        prompt_ (str): The prompt for generating music.
        input_audio_ (Tuple[int, np.ndarray]): The input audio as a tuple of sample
                                               rate and numpy array.
        top_k_ (int): The number of top tokens to consider for generation.
        top_p_ (float): The cumulative probability threshold for generating music.
        temp_ (float): The temperature for controlling the randomness of generation.
        duration_ (int): The duration of the generated music in seconds.
        progress_ (Callable, optional): A callable object to track the progress of
                                        generation. Defaults to gr.Progress().

    Returns:
        Tuple: A tuple containing the generated music output and waveform.

    Note:
        If the output is a string, it means an error occurred during generation.

    Example:
        output, waveform = generate("Piano", (44100, audio), 5, 0.8, 0.5, 60)
    """
    output = music_gen.generate(prompt_, input_audio_, True, top_k_, top_p_, temp_, duration_,\
                                progress=progress_)
    if isinstance(output, str):
        return None, None, output
    else:
        waveform = gr.make_waveform(output, bg_color="#f3f4f6", bg_image=None, fg_alpha=1.00,\
                                    bars_color=("#65B5FF", "#1B76FF"), bar_count=50, bar_width=0.6)
        return output, waveform

def load_model(model) -> str:
    """Load a model for music generation.

    Args:
        model: The model to be loaded.

    Returns:
        The loaded model name.

    Raises:
        gr.Error: If the model fails to load.
    """
    if music_gen is None:
        music_gen = MusicGenerator(model)
    if not music_gen.load_model(model):
        raise gr.Error('Could not load model!')
    return model

def unload_model() -> str:
    """Unload the music generation model.

    This function unloads the music generation model if it exists.

    Returns:
        str: An empty string indicating the successful unloading of the model.
    """
    if music_gen is not None:
        music_gen.delete_model()
    return ''

with gr.Blocks() as demo:
    gr.Markdown("# Music Generative AI App")
    gr.Markdown("First select and **load*** a model. Then fill other fields and click **Generate**"
                "to see the output.")

    with gr.Row():
        with gr.Row():
            selected = gr.Dropdown(MODELS, value='medium', label='Model')
            with gr.Column(elem_classes='smallsplit'):
                load = gr.Button('🚀 Load', variant='tool secondary')
                unload = gr.Button('💣 Delete', variant='tool primary')
            load.click(load_model, selected, selected)
            unload.click(unload_model, outputs=selected)
        with gr.Row():
            gen_button = gr.Button('🎶 Generate 🎶', variant='primary')
    with gr.Row():
        with gr.Column():
            prompt = gr.TextArea(label='Prompt', info='Put the audio you want here.',
                                     placeholder='Something like: "happy rock", "energetic EDM"'
                                        'or "sad jazz"\nLonger descriptions are also supported.')
            duration = gr.Number(5, label='Duration (s)',\
                                 info='Duration for the generation in seconds.')
            input_audio = gr.Audio(label='Input audio (structure for melody, continuation for '
                                         'others)')
            with gr.Row():
                top_k = gr.Slider(label='top_k', minimum=0, value=250, maximum=10000, step=1,
                                  info='Higher number = more possible tokens, 0 to disable')
                top_p = gr.Slider(label='top_p', minimum=0, value=0, maximum=1, step=0.01,
                                  info='Higher number = more possible tokens, 0 to use top_k '
                                       'instead')
            temp = gr.Slider(label='temperature', minimum=0, value=1, maximum=2,
                             info='Higher number = more randomness for picking the next token')
        with gr.Column():
            with gr.Row():
                audio_out = gr.Audio(label='Generated audio', interactive=False)
            with gr.Row():
                video_out = gr.Video(label='Waveform video', interactive=False)
    gen_button.click(generate, inputs=[prompt, input_audio, top_k, top_p, temp, duration],
                     outputs=[audio_out, video_out])

demo.queue().launch(server_name='0.0.0.0', share=True, auth=None, inbrowser=True)
