# Music Gen AI App

## Project Structure

```
.
└── Music-GenAI-App/
    ├── music_generation/
    │   ├── __init__.py
    │   ├── _utils_.py
    │   └── music_gen.py
    ├── webapp/
    │   └── app.py
    ├── scripts/
    │   ├── extract_from_youtube.py
    │   ├── demo1.py
    │   ├── demo2.py
    │   └── demo3.py
    ├── data/
    │   ├── inputs
    │   └── outputs
    ├── Dockerfile
    ├── docker-compose.yaml
    ├── mkdocs.yaml
    ├── pyproject.toml
    └── README.md
```

## Documentation

For technical documentation of code under `music_generation` please refer to [github.io](https://smasis001.github.io/Music-GenAI-App/). All other code make use of `music_generation` and has internal documentation explaining how it uses it.

## Getting Started

### Install Basic Requirements

You'll need Python 3.10 and above, ffmpeg, and GIT. On Windows, the best way is via the [Python website](https://python.org/downloads/windows/), [ffpmeg website](https://ffmpeg.org/download.html) and [GIT website](https://git-scm.com/downloads/win). On other operating systems command line options might be the easiest:

**MacOs**:

```sh
brew install python git ffmpeg
```
(assuming you have brew installed)

**Linux**:

```sh
sudo apt install python3 python-is-python3 git ffmpeg
```

It is recommended that you open the folder where this repository will be cloned with an IDE like VSCode or PyCharm.

### Create Environment

First you need to clone this repository, and then create the environment. You can use `pyenv`, `conda` or whichever one you want but for this example we are using `venv` which is included in `python`.

```sh
cd /path/where/you/want/to/clone/repo
git clone https://github.com/smasis001/Music-GenAI-App.git
python -m venv .venv
```

Then for managing the packages, this repository comes with `pyproject.toml` for `poetry`, but you can use `requirements.txt` and `pip`. For poetry, this is how it goes:

```sh
pip install poetry
python -m poetry env use .venv/bin/python
python -m poetry install
source .venv/bin/activate
```

For installing the packages with `pip`, you must activate the virtual enviroment first.

```sh
source .venv/bin/activate
pip install -r requirements.txt
```

### Download Raw Data (optional)

The repo already comes with a couple of royalty free input clips under `data/inputs`. But you can add more with the youtube script like this:

```sh
python scripts/extract_from_youtube.py https://youtu.be/t139Vd83pgs
python scripts/extract_from_youtube.py https://youtu.be/bklB0Q13iNI
```

### Run Web App (locally) and through SSH tunnel

To see the music generation system work via browser, run:

```sh
python webapp/app.py
```

It should open a `gradio.live` address (via tunnel) in your default browser. But if it doesn't, try [http://127.0.0.1:7860/](http://127.0.0.1:7860/).


When done using it, press ctrl-C to stop the webapp in your command line.

### Run Scripts (locally)

To demo text-conditional music generation with Python, run:

```sh
python scripts/demo1.py --prompt "your prompt" --model "small" --duration 10 --name "name for the audio file saved"
```

It will generate music using your `prompt` with the chosen `model` and `duration` (in seconds). It will then save a file with the chosen `name` in the `data/outputs` directory. `model` can be "small", "medium" and "large".

And for the melody-conditional music generation demo:

```sh
python scripts/demo2.py --prompt "your prompt" --input-file "input.wav" --duration 10 --name "name for the audio file saved"
```

The arguments are the same as with "demo1" except you can't chose a `model` and you must chose an `input-file`.

Lastly, if you want text-conditional music continuation:

```sh
python scripts/demo3.py --prompt "your prompt" --input-file "input.wav" --model "small"  --duration 10 --name "name for the audio file saved"
```

For this one, you also use an `input-file` but you can chose a `model` ("small", "medium" or "large). Duration must be longer than the input audio file.
