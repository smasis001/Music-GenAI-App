"""Demo Script #2"""
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
        prompt:str,
        input_file:str,
        duration:int,
        temp:float,
        name:str
    ) -> None:
    """Main function for generating music based on a melody and a prompt.

    Args:
        prompt (str): The prompt for generating the music.
        input_file (str): The path to the input audio file.
        duration (int): The duration of the generated music in seconds.
        temp (float): The temperature parameter for controlling the randomness of the generated
                      music.
        name (str): The name of the output audio file.

    Returns:
        None

    Raises:
        None
    """

    model = 'melody'
    input_audio = read_audio_file(input_file)

    music_gen = MusicGenerator(model)
    if music_gen.load_model():
        music_gen.set_model_params(duration=duration, temp=temp,\
                                   top_k=DEFAULT_TOP_K, top_p=DEFAULT_TOP_P,\
                                    cfg_coef=DEFAULT_CFG_COEF)
        music_gen.set_progress_bar_cb()
        # Generate music based on melody of existing clip (`wav`) and prompt
        sample_rate, wav = music_gen.melody_cond_generate(input_audio, prompt)
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
                        help='Prompt to use to generate audio clip')
    parser.add_argument('-d', '--duration', type=int, default=DEFAULT_DURATION,\
                        help='Duration of the audio clip to generate')
    parser.add_argument('-t', '--temp', type=float, default=DEFAULT_TEMP,\
                        help='Temperature to use to generate the audio clip')
    parser.add_argument('-n', '--name',\
                        help='Name to use when saving the audio file')

    args = parser.parse_args()

    if args.name is None:
        args.name = args.prompt

    main(
        prompt=args.prompt,
        input_file=args.input_file,
        duration=args.duration,
        temp=args.temp,
        name=args.name
    )
