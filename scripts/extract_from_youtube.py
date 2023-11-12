"""Extract Music from Youtube Script"""
import os
import sys
import argparse
sys.path.append("../")
par_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, par_dir)
from music_generation import download_audio_from_youtube

# Main execution block
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('URL',
                        help='URL of youtube video')

    args = parser.parse_args()

    download_audio_from_youtube(args.URL)
