# Copyright 2023 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pathlib
import random
from typing import Dict, List, Tuple, Union

import bark
import numpy as np
import pandas as pd
import torch
import torchaudio
import tqdm

# TODO:
# def generate_single_speaker_audio():
#     pass


def generate_multi_speakers_audio(
    data_path: str,
    output_directory: str,
    speakers: Union[List[str], Dict[str, int]],
    available_voices: List[str],
    use_gpu: bool = True,
    use_small_models: bool = False,
    offload_cpu: bool = False,
    sample_rate: int = 16000,
    file_format: str = "wav",
    verbose: bool = True,
) -> Tuple[str, pd.DataFrame, dict]:
    """

    :param data_path:
    :param output_directory:
    :param speakers:
    :param available_voices:
    :param use_gpu:
    :param use_small_models:
    :param offload_cpu:
    :param sample_rate:
    :param file_format:
    :param verbose:

    :return:
    """
    # Get the input text files to turn to audio:
    data_path = pathlib.Path(data_path).absolute()
    text_files = _get_text_files(data_path=data_path)

    # Load the bark models according to the given configurations:
    bark.preload_models(
        text_use_gpu=use_gpu,
        text_use_small=use_small_models,
        coarse_use_gpu=use_gpu,
        coarse_use_small=use_small_models,
        fine_use_gpu=use_gpu,
        fine_use_small=use_small_models,
        codec_use_gpu=use_gpu,
        force_reload=offload_cpu,
    )

    # Check for per channel generation:
    if isinstance(speakers, dict):
        speaker_per_channel = True
        # Sort the given speakers by channels:
        speakers = {
            speaker: channel
            for speaker, channel in sorted(speakers.items(), key=lambda item: item[1])
        }
    else:
        speaker_per_channel = False

    # Prepare the resampling module:
    resampler = torchaudio.transforms.Resample(
        orig_freq=bark.SAMPLE_RATE, new_freq=sample_rate, dtype=torch.float64
    )

    # Prepare the gap between each speaker:
    gap_between_speakers = np.zeros(int(0.5 * bark.SAMPLE_RATE))

    # Prepare the successes dataframe and errors dictionary to be returned:
    successes = []
    errors = {}

    # Create the output directory:
    output_directory = pathlib.Path(output_directory)
    output_directory.mkdir(exist_ok=True)

    # Start generating audio:
    # Go over the audio files and transcribe:
    for text_file in tqdm.tqdm(
        text_files, desc="Generating", unit="file", disable=not verbose
    ):
        try:
            # Randomize voices for each speaker:
            chosen_voices = {}
            available_voices_copy = available_voices.copy()
            for speaker in speakers:
                voice = random.choice(available_voices_copy)
                chosen_voices[speaker] = voice
                available_voices_copy.remove(voice)
            # Read text:
            with open(text_file, "r") as fp:
                text = fp.read()
            # Prepare a holder for all the generated pieces (if per channel each speaker will have its own):
            audio_pieces = (
                {speaker: [] for speaker in speakers}
                if speaker_per_channel
                else {"all": []}
            )
            # Generate audio per line:
            for line in text.splitlines():
                # Validate line is in correct speaker format:
                if ": " not in line:
                    # TODO: if verbose: logger.warning(f"Skipping line: {line}")
                    continue
                # Split line to speaker and his words:
                current_speaker, sentences = line.split(": ", 1)
                # Validate speaker is known:
                if current_speaker not in speakers:
                    raise ValueError(
                        f"Unknown speaker: {current_speaker}. Given speakers are: {speakers}"
                    )
                for sentence in _split_line(line=sentences):
                    # Generate words audio:
                    audio = bark.generate_audio(
                        sentence,
                        history_prompt=chosen_voices[current_speaker],
                        silent=True,
                    )
                    if speaker_per_channel:
                        silence = np.zeros_like(audio)
                        for speaker in audio_pieces.keys():
                            if speaker == current_speaker:
                                audio_pieces[speaker] += [audio, gap_between_speakers]
                            else:
                                audio_pieces[speaker] += [silence, gap_between_speakers]
                    else:
                        audio_pieces["all"] += [audio, gap_between_speakers]
            # Construct a single audio array from all the pieces and channels:
            audio = np.vstack(
                [np.concatenate(audio_pieces[speaker]) for speaker in speakers]
            )
            # Resample:
            audio = torch.from_numpy(audio)
            audio = resampler(audio)
            # Save to audio file:
            audio_file = output_directory / f"{text_file.stem}.{file_format}"
            torchaudio.save(
                uri=audio_file, src=audio, sample_rate=sample_rate, format=file_format
            )
            # Collect to the successes:
            successes.append([text_file, audio_file])
        except Exception as exception:
            # Note the exception as error in the dictionary:
            # TODO: if verbose: logger.warning(f"Error in file: '{text_file.name}'")
            print(exception)
            errors[str(text_file.name)] = str(exception)
            continue

        # Construct the translations dataframe:
        successes = pd.DataFrame(
            successes,
            columns=["text_file", "audio_file"],
        )

        # Print the head of the produced dataframe and return:
        # if verbose:
        #     _LOGGER.info(
        #         f"Done ({successes.shape[0]}/{len(text_files)})\n"
        #         f"Translations summary:\n"
        #         f"{successes.head()}"
        #     )
        return str(output_directory), successes, errors


def _get_text_files(
    data_path: pathlib.Path,
) -> List[pathlib.Path]:
    # Check if the path is of a directory or a file:
    if data_path.is_dir():
        # Get all files inside the directory:
        text_files = list(data_path.glob("*.*"))
    elif data_path.is_file():
        text_files = [data_path]
    else:
        raise ValueError(
            f"Unrecognized data path. The parameter `data_path` must be either a directory path or a file path. "
            f"Given: {str(data_path)} "
        )

    return text_files


def _split_line(line: str, max_length: int = 250) -> List[str]:
    if len(line) < max_length:
        return [line]

    sentences = [
        f"{sentence.strip()}." for sentence in line.split(".") if sentence.strip()
    ]

    splits = []
    current_length = len(sentences[0])
    split = sentences[0]
    for sentence in sentences[1:]:
        if current_length + len(sentence) > max_length:
            splits.append(split)
            split = sentence
            current_length = len(sentence)
        else:
            current_length += len(sentence)
            split += " " + sentence
    if split:
        splits.append(split)

    return splits
