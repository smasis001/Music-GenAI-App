"""Music Generation Functionality"""
# pylint: disable=W0718
from typing import Literal, Tuple, Callable, Optional
import gc
import warnings
import numpy as np
from tqdm.auto import tqdm
from audiocraft.models import MusicGen
import torch
warnings.filterwarnings('ignore')

DEFAULT_MODEL = 'medium'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODELS = ['small','medium','large','melody']

class MusicGenerator:
    """Class for Music Generation"""
    def __init__(
            self,
            pretrained:Literal['small','medium','large','melody'] = DEFAULT_MODEL,
            device:str = DEVICE
        ) -> None:
        self.pretrained = pretrained
        self.device = device
        self.model = None
        self.loaded = False
        self.load_error = None
        self.pbar = None

    def load_model(
            self,
            pretrained:Literal['small','medium','large','melody'] = None,
            device:str = None
        ) -> bool:
        """Load a pre-trained model for music generation.

        Args:
            pretrained (Literal['small','medium','large','melody'], optional): The size of
                                            the pre-trained model to load. Defaults to None.
            device (str, optional): The device to use for model inference. Defaults to None.

        Returns:
            bool: True if the model is successfully loaded, False otherwise.
        """
        if self.loaded is True:
            self.delete_model()

        if pretrained is not None:
            self.pretrained = pretrained
        if device is not None:
            self.device = device
        try:
            pt_fullname = f"facebook/musicgen-{self.pretrained}"
            self.model = MusicGen.get_pretrained(pt_fullname,\
                                                device=self.device)
            self.loaded = True
        except Exception as err:
            self.load_error = err
            self.loaded = False

        return self.loaded

    def delete_model(
            self
        ) -> None:
        """Deletes the model and frees up memory.

        This function deletes the model object, performs garbage collection, empties the
        CUDA cache, and resets the device, model, loaded flag, load error, and progress
        bar attributes.

        Returns:
            None
        """
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        self.device = None
        self.model = None
        self.loaded = False
        self.load_error = None
        self.pbar = None

    def reset_progress_bar(
            self
        ) -> None:
        """Resets the progress bar.

        This function resets the progress bar by setting the total value to 1 and updating the
        description to 'Generating'.

        Returns:
            None
        """
        self.pbar = tqdm(total=1, dynamic_ncols=True,\
                         desc='Generating')

    def set_progress_bar_cb(
            self,
            callback:Callable = None
        ) -> None:
        """Sets the progress bar callback function.

        Args:
            callback (Callable, optional): The callback function to be set. Defaults to None.

        Returns:
            None

        Example:
            set_progress_bar_cb(callback=my_callback_function)
        """
        if callback is None:
            self.reset_progress_bar()
            callback = self.cb_progress_bar
        self.model.set_custom_progress_callback(callback)

    def cb_progress_bar(
            self,
            p:int,
            t:int
        ) -> None:
        """Updates the progress bar with the given progress and total values.

        Args:
            p (int): The current progress value.
            t (int): The total value.

        Raises:
            ValueError: If the progress bar is not initialized.

        Returns:
            None
        """
        if self.pbar is None:
            raise ValueError("Progress Bar must be initialized first")
        if self.pbar.total != t:
            self.pbar.total = t
            self.pbar.reset(t)
        self.pbar.update(p)
        if self.pbar.total == p:
            self.pbar.close()
        else:
            self.pbar.refresh()

    def set_model_params(
            self,
            use_sample:bool=True,
            top_k:int = 250,
            top_p:float = 0.0,
            temp:float = 1,
            duration:int = 8,
            cfg_coef:int = 3
        ) -> None:
        """Sets the parameters for generating music using the model.

        Args:
            use_sample (bool): Use sampling if True, else do argmax decoding. Defaults to True.
            top_k (int): Top-k is a setting in text and music generation models that limits the
                         number of choices (tokens) the model considers at each step. A smaller
                         top-k leads to more predictable outputs, while a larger top-k allows
                         for more variety. Defaults to 250.
            top_p (float): Top-p, also called nucleus sampling, is a method in text generation
                           where the model chooses from a set of tokens based on their combined
                           likelihood, ensuring a balance between diversity and coherence in
                           the output. Unlike top-k, which picks a fixed number of tokens,
                           top-p's selection varies based on their probabilities. Defaults to
                           0.0, when set to 0 top_k is used.
            temp (float): Softmax temperature parameter. In music generation, the temperature
                          setting controls how predictable or varied the music is. A higher
                          temperature results in more random and diverse music, while a lower
                          temperature creates more consistent and less varied music. Defaults
                          to 1.0.
            duration (int): Duration of the generated waveform. Defaults to 30.0.
            cfg_coef (int): Coefficient used for classifier free guidance. This technique in
                            music generation uses an additional classifier network to guide
                            the music creation process towards specific characteristics or
                            styles. It provides more detailed control, allowing users to
                            influence the style and features of the generated music. Defaults
                            to 3.0.

        Raises:
            ValueError: If the model is not loaded.

        Returns:
            None
        """
        if self.model is None:
            raise ValueError("Model must be loaded first")
        self.model.set_generation_params(use_sample, top_k, top_p, temp,\
                                            duration, cfg_coef)

    def text_cond_generate(
            self,
            prompt:str
        ) -> Tuple[int, np.ndarray]:
        """Text conditional music generation using the loaded model.

        Args:
            prompt (str): The prompt text for generating the waverform.

        Returns:
            Tuple[int, np.ndarray]: A tuple containing the sample rate and the generated
                                    waveform as a numpy array.

        Raises:
            ValueError: If the model is not loaded.
        """
        if self.model is None:
            raise ValueError("Model must be loaded first")

        # Generate music based on prompt
        wav = self.model.generate([prompt], progress=True)
        wav = wav.cpu().flatten().numpy()

        return self.model.sample_rate, wav

    @staticmethod
    def prepare_input_audio(
            input_audio:Tuple[int, np.ndarray] = None
        ) -> Tuple[int, np.ndarray]:
        """Prepares the input audio for further processing.

        Args:
            input_audio (Tuple[int, np.ndarray], optional): A tuple containing the sample
                                                            rate and waveform of the input
                                                            audio. Defaults to None.

        Returns:
            Tuple[int, np.ndarray]: A tuple containing the sample rate and processed
                                    waveform.

        Note:
            - If `input_audio` is not provided, the function returns a tuple with sample
              rate 0 and waveform None.
            - If `input_audio` is provided, the function converts the waveform to a torch
              tensor and performs additional processing if necessary.
        """
        if input_audio is not None:
            sample_rate, wav = input_audio
            wav = torch.tensor(wav)
            if wav.dtype == torch.int16:
                wav = wav.float() / 32767.0
            if wav.dim() == 2 and wav.shape[1] == 2:
                wav = wav.mean(dim=1)
        else:
            sample_rate, wav = 0, None

        return sample_rate, wav

    def melody_cond_generate(
            self,
            input_audio:Tuple[int, np.ndarray],
            prompt:Optional[str] = ''
        ) -> Tuple[int, np.ndarray]:
        """Melody conditional music generation on an input audio with the loaded model.

        Args:
            input_audio (Tuple[int, np.ndarray]): The input audio as a tuple of sample
                                                  rate and waveform.
            prompt (Optional[str]): The prompt for generating the waveform. Defaults to
                                    an empty string.

        Returns:
            Tuple[int, np.ndarray]: A tuple containing the sample rate and the generated
                                    melody waveform.

        Raises:
            ValueError: If the model is not loaded.

        Note:
            - If no prompt is provided, it will be set to None.
            - The input audio will be prepared using the MusicGenerator class.
            - The generated melody will be conditioned on the input audio using the model's
              generate_with_chroma method.
            - The generated waveform will be converted to a numpy array.
        """
        if self.model is None:
            raise ValueError("Model must be loaded first")

        if not prompt:
            prompt = None
        sample_rate, wav = MusicGenerator.prepare_input_audio(input_audio)
        wav = wav[None].expand(1, -1, -1)
        # Generate music based on melody of existing clip (`wav`) and prompt
        wav = self.model.generate_with_chroma([prompt], wav,\
                                              sample_rate, progress=True)
        wav = wav.cpu().flatten().numpy()

        return self.model.sample_rate, wav

    def cont_generate(
            self,
            input_audio:Tuple[int, np.ndarray],
            prompt:Optional[str] = ''
        ) -> Tuple[int, np.ndarray]:
        """Generates a music continuation to an existing audio clip based on a prompt.

        Args:
            input_audio (Tuple[int, np.ndarray]): The input audio clip as a tuple of
                                                  sample rate and waveform.
            prompt (Optional[str], optional): The prompt for generating the music
                                              continuation. Defaults to ''.

        Returns:
            Tuple[int, np.ndarray]: The sample rate and waveform of the generated
                                    music continuation.

        Raises:
            ValueError: If the model is not loaded.

        Note:
            - If no prompt is provided, the prompt will be set to None.
            - The input audio clip is prepared using the `prepare_input_audio` method of the
              `MusicGenerator` class.
            - The generated music continuation is obtained by calling the `generate_continuation`
              method of the model.
            - The generated waveform is converted to a numpy array.
        """

        if self.model is None:
            raise ValueError("Model must be loaded first")

        if not prompt:
            prompt = None
        sample_rate, wav = MusicGenerator.prepare_input_audio(input_audio)
        wav = wav[None].expand(1, -1, -1)
        # Generate music continuation to existing clip (`wav`) based on prompt
        wav = self.model.generate_continuation(wav, sample_rate,\
                                               [prompt], progress=True)
        wav = wav.cpu().flatten().numpy()

        return self.model.sample_rate, wav

    def uncond_generate(
            self
        ) -> Tuple[int, np.ndarray]:
        """Generates music unconditionally.

        This function generates music unconditionally using a pre-loaded model. It returns the
        sample rate and the generated waveform.

        Returns:
            Tuple[int, np.ndarray]: A tuple containing the sample rate (int) and the generated
                                    waveform (np.ndarray).

        Raises:
            ValueError: If the model is not loaded.
        """

        if self.model is None:
            raise ValueError("Model must be loaded first")

        # Generate music unconditionally (based on nothing)
        wav = self.model.generate_unconditional(1, progress=True)
        wav = wav.cpu().flatten().numpy()

        return self.model.sample_rate, wav

    def generate(
            self,
            prompt:Optional[str] = '',
            input_audio:Optional[Tuple[int, np.ndarray]] = None,
            use_sample:Optional[bool] = True,
            top_k:Optional[int] = 250,
            top_p:Optional[float] = 0.0,
            temp:Optional[float] = 1,
            duration:Optional[int] = 8,
            cfg_coef:Optional[int] = 3,
            progress:Optional[Callable] = None
        ) -> Tuple[int, np.ndarray]:
        """Generate audio using the model.

        Args:
            prompt (Optional[str]): The prompt text to generate audio from. Default is an empty
                                    string.
            input_audio (Optional[Tuple[int, np.ndarray]]): The input audio to condition the
                                                            generation on. Default is None.
            use_sample (Optional[bool]): Whether to use sampling during generation. Default is
                                         True.
            top_k (Optional[int]): The number of top tokens to consider during sampling. Default
                                   is 250.
            top_p (Optional[float]): The cumulative probability threshold for top-k sampling.
                                     Default is 0.0.
            temp (Optional[float]): The temperature value for sampling. Default is 1.
            duration (Optional[int]): The duration of the generated audio in seconds. Default
                                      is 8.
            cfg_coef (Optional[int]): The coefficient for controlling the model configuration.
                                      Default is 3.
            progress (Optional[Callable]): A callback function to track the progress of
                                          generation. Default is None.

        Returns:
            Tuple[int, np.ndarray]: A tuple containing the sample rate and the generated
                                    audio waveform.
        """
        self.set_model_params(use_sample, top_k, top_p, temp, duration, cfg_coef)

        if progress is not None:
            def progress_callback(p, t):
                progress((p, t), desc='Generating')
        else:
            progress_callback = None

        self.set_progress_bar_cb(progress_callback)

        input_audio_not_none = input_audio is not None

        if input_audio_not_none and (self.pretrained == 'melody'):
            sample_rate, wav = self.melody_cond_generate(input_audio, prompt)
        elif input_audio_not_none:
            self.set_model_params(use_sample, top_k, top_p, temp, duration, cfg_coef)
            sample_rate, wav = self.cont_generate(input_audio, prompt)
        elif not prompt:
            sample_rate, wav = self.uncond_generate()
        else:
            sample_rate, wav = self.text_cond_generate(prompt)

        return sample_rate, wav
