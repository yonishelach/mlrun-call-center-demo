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
import os
from typing import List

import mlrun


def pipeline(
    output_directory: str,
    amount: int,
    generation_model: str,
    text_to_speech_model: str,
    language: str,
    available_voices: List[str],
    min_time: int,
    max_time: int,
    from_date: str,
    to_date: str,
    from_time: str,
    to_time: str,
):
    # Create the output directory:
    os.makedirs(output_directory, exist_ok=True)

    # Get the project:
    project = mlrun.get_current_project()

    # Generate conversations texts:
    conversations_generator_function = project.get_function("conversations-generator")
    generate_conversations_run = conversations_generator_function.run(
        handler="generate_conversations",
        params={
            "amount": amount,
            "output_directory": os.path.join(
                output_directory, "generated_conversations"
            ),
            "model_name": generation_model,
            "language": language,
            "min_time": min_time,
            "max_time": max_time,
            "from_date": from_date,
            "to_date": to_date,
            "from_time": from_time,
            "to_time": to_time,
        },
        returns=[
            "conversations: path",
            "metadata: dataset",
            "ground_truths: dataset",
        ],
    )

    # Text to audio:
    text_to_audio_generator_function = project.get_function("text-to-audio-generator")
    generate_multi_speakers_audio_run = text_to_audio_generator_function.run(
        handler="generate_multi_speakers_audio",
        inputs={"data_path": generate_conversations_run.outputs["conversations"]},
        params={
            "output_directory": os.path.join(output_directory, "audio_files"),
            "speakers": {"Agent": 0, "Client": 1},
            "available_voices": available_voices,
            "use_small_models": text_to_speech_model == "small",
        },
        returns=[
            "audio_files: path",
            "audio_files_dataframe: dataset",
            "text_to_speech_errors: file",
        ],
    )

    # Create the input batch:
    create_batch_for_analysis_run = conversations_generator_function.run(
        handler="create_batch_for_analysis",
        inputs={
            "conversations_data": generate_conversations_run.outputs["metadata"],
            "audio_files": generate_multi_speakers_audio_run.outputs[
                "audio_files_dataframe"
            ],
        },
        returns=["calls_batch: file"],
    )
