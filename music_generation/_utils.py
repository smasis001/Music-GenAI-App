"""Utility Functions"""
from typing import Tuple
import os
import pytube
import ffmpeg
import numpy as np
import librosa
from scipy.io import wavfile

INPUT_DIR_PATH = os.path.join(os.path.dirname(__file__),\
                                     '../data/inputs/')
OUTPUT_DIR_PATH = os.path.join(os.path.dirname(__file__),\
                                     '../data/outputs/')

def download_audio_from_youtube(
        url:str
    ) -> str:
    """Downloads audio from a YouTube video given its URL.

    Args:
        url (str): The URL of the YouTube video.

    Returns:
        str: The path of the downloaded audio file.

    Example:
        download_audio_from_youtube("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    """

    print(f"Downloading {url}")
    yt = pytube.YouTube(url)
    video = yt.streams.filter(only_audio=True).first()
    out_file = video.download(output_path=INPUT_DIR_PATH)
    filename, _ = os.path.splitext(out_file)
    ffmpeg_out_file = f'{filename}.wav'
    ffmpeg.input(out_file).output(ffmpeg_out_file).run(overwrite_output=1)
    print(f"Downloaded to {INPUT_DIR_PATH}/{ffmpeg_out_file}")
    return ffmpeg_out_file

def save_audio_file(
        sample_rate:int,
        wav:np.ndarray,
        name:str = 'audio'
    ) -> str:
    """Saves an audio file in WAV format.

    Args:
        sample_rate (int): The sample rate of the audio.
        wav (np.ndarray): The audio data as a NumPy array.
        name (str, optional): The name of the audio file. Defaults to 'audio'.

    Returns:
        str: The path of the saved audio file.

    Example:
        >>> save_audio_file(44100, audio_data, 'my_audio')
        Saved to /path/to/output_dir/my_audio.wav
        '/path/to/output_dir/my_audio.wav'
    """

    output_path = os.path.join(OUTPUT_DIR_PATH,f"{name}.wav")
    wavfile.write(output_path, sample_rate, wav)
    print(f"Saved to {output_path}")

    return sample_rate, wav

def read_audio_file(
        file_path:str
    ) -> Tuple[int, np.ndarray]:
    """
    Reads an audio file and returns a tuple (sample_rate, waveform).

    Parameters:
    file_path (str): Path to the audio file.

    Returns:
    tuple: (sample_rate, waveform)

    Raises:
    FileNotFoundError: If the file does not exist.
    ValueError: If the file cannot be read as an audio file.
    """
    # Check if the file exists. Try finding file in input or output dir otherwise
    if not os.path.exists(file_path):
        input_file_path = os.path.join(INPUT_DIR_PATH, file_path)
        output_file_path = os.path.join(OUTPUT_DIR_PATH, file_path)
        if os.path.exists(input_file_path):
            file_path = input_file_path
        elif os.path.exists(output_file_path):
            file_path = output_file_path
        else:
            raise FileNotFoundError(f"File not found: {file_path}")

    try:
        # Read the audio file
        wav, sample_rate = librosa.load(file_path, sr=None)
        return sample_rate, wav
    except Exception as err:
        # Handle other exceptions (e.g., file cannot be read)
        raise ValueError("Error reading the audio file") from err
