"""Demo Script #3"""
import os
import sys
import warnings
import argparse
sys.path.append("../")
par_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, par_dir)
from music_generation import MusicGenerator, save_audio_file, read_audio_file
warnings.filterwarnings('ignore')

DEFAULT_TOP_K = 250
DEFAULT_TOP_P = 0.0
DEFAULT_TEMP = 1
DEFAULT_DURATION = 5
DEFAULT_CFG_COEF = 3

def main(
        model:str,
        prompt:str,
        input_file:str,
        duration:int,
        temp:float,
        name:str
    ) -> None:
    """Main function for generating music continuation based on a prompt.

    Args:
        model (str): Name of the model.
        prompt (str): Prompt for generating music continuation.
        input_file (str): Path to the input audio file or name of the file in the input folder.
        duration (int): Duration of the generated music in seconds.
        temp (float): Temperature parameter for controlling the randomness of the generated music.
        name (str): Path to the output audio file or name of the file in the output folder.

    Returns:
        None

    Raises:
        FileNotFoundError: If the model file or input audio file is not found.
        ValueError: If the duration or temperature is invalid.

    Example:
        main("small, "Start playing a piano melody", "input.wav", 60, 0.8, "output.wav")
    """

    input_audio = read_audio_file(input_file)

    music_gen = MusicGenerator(model)
    if music_gen.load_model():
        music_gen.set_model_params(duration=duration, temp=temp,\
                                   top_k=DEFAULT_TOP_K, top_p=DEFAULT_TOP_P,\
                                    cfg_coef=DEFAULT_CFG_COEF)
        music_gen.set_progress_bar_cb()
        # Generate music continuation to existing clip (`wav`) based on prompt
        sample_rate, wav = music_gen.cont_generate(input_audio, prompt)
        save_audio_file(sample_rate, wav, name=name)
        music_gen.delete_model()
    else:
        print(f"Could not load model! {music_gen.load_error}")


# Main execution block
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prompt', required=True,\
                        help='Prompt to use to generate audio clip')
    parser.add_argument('-i', '--input-file', required=True,\
                        help='Prompt to use to begin the audio clip')
    parser.add_argument('-d', '--duration', type=int, default=DEFAULT_DURATION,\
                        help='Duration of the audio clip to generate')
    parser.add_argument('-m', '--model', type=str, default='small',\
                        help='Generative model version to use')
    parser.add_argument('-t', '--temp', type=float, default=DEFAULT_TEMP,\
                        help='Temperature to use to generate the audio clip')
    parser.add_argument('-n', '--name',\
                        help='Name to use when saving the audio file')

    args = parser.parse_args()

    if args.name is None:
        args.name = args.prompt

    main(
        model=args.model,
        input_file=args.input_file,
        prompt=args.prompt,
        duration=args.duration,
        temp=args.temp,
        name=args.name
    )
