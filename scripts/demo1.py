"""Demo Script #1"""
import os
import sys
import warnings
import argparse
sys.path.append("../")
par_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, par_dir)
from music_generation import MusicGenerator, save_audio_file
warnings.filterwarnings('ignore')

DEFAULT_TOP_K = 250
DEFAULT_TOP_P = 0.0
DEFAULT_TEMP = 1
DEFAULT_DURATION = 5
DEFAULT_CFG_COEF = 3

def main(
        model:str,
        prompt:str,
        duration:int,
        temp:float,
        name:str
    ) -> None:
    """Main function to generate music based on a given prompt.

    Args:
        model (str): Name of the model.
        prompt (str): Prompt for generating music.
        duration (int): Duration of the generated music in seconds.
        temp (float): Temperature parameter for controlling randomness in music generation.
        name (str): Name of the output audio file.

    Returns:
        None

    Raises:
        None

    Example:
        main("small", "I want a happy melody", 60, 0.8, "output.wav")
    """

    music_gen = MusicGenerator(model)
    if music_gen.load_model():
        music_gen.set_model_params(duration=duration, temp=temp,\
                                   top_k=DEFAULT_TOP_K, top_p=DEFAULT_TOP_P,\
                                    cfg_coef=DEFAULT_CFG_COEF)
        music_gen.set_progress_bar_cb()

        # Generate music based on prompt
        sample_rate, wav = music_gen.text_cond_generate(prompt)
        save_audio_file(sample_rate, wav, name=name)
        music_gen.delete_model()
    else:
        print(f"Could not load model! {music_gen.load_error}")


# Main execution block
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prompt', required=True,\
                        help='Prompt to use to generate audio clip')
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
        prompt=args.prompt,
        duration=args.duration,
        temp=args.temp,
        name=args.name
    )
