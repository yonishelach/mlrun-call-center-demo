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
import mlrun

from src.calls_analysis.db_management import create_tables
from src.common import ProjectSecrets


def setup(
    project: mlrun.projects.MlrunProject,
    openai_key: str,
    openai_base: str,
    huggingface_token: str,
    mysql_url: str,
    source: str = None,
    default_image: str = None,
    gpus: int = None,
    apply_auto_mount: bool = False,
) -> mlrun.projects.MlrunProject:
    """
    Creating the project for the demo. This function is expected to call automatically when calling the function
    `mlrun.get_or_create_project`.

    :param project:           The project to set up.
    :param openai_key:
    :param openai_base:
    :param huggingface_token: A HuggingFace token
    :param mysql_url:
    :param source:
    :param default_image:
    :param gpus:
    :param apply_auto_mount:

    :returns: A fully prepared project for this demo.
    """
    # Set the project git source:
    if source:
        print(f"Project Source: {source}")
        project.set_source(source=source, pull_at_runtime=True)

    # Set or build the default image:
    if default_image is None:
        print("Building default image for the demo:")
        _build_image(project=project)
    else:
        project.set_default_image(default_image)

    # Set the secrets:
    _set_secrets(
        project=project,
        openai_key=openai_key,
        openai_base=openai_base,
        huggingface_token=huggingface_token,
        mysql_url=mysql_url,
    )

    # Set the functions:
    _set_calls_generation_functions(
        project=project, gpus=gpus, apply_auto_mount=apply_auto_mount
    )
    _set_calls_analysis_functions(
        project=project,
        gpus=gpus,
        apply_auto_mount=apply_auto_mount,
    )

    # Set the workflows:
    _set_workflows(project=project)

    # Create the DB tables:
    create_tables(project=project)

    # Save and return the project:
    project.save()
    return project


def _build_image(project: mlrun.projects.MlrunProject):
    assert project.build_image(
        base_image="mlrun/mlrun-gpu",
        commands=[
            # Update apt-get to install ffmpeg (support audio file formats):
            "apt-get update -y",
            "apt-get install ffmpeg -y",
            # Install demo requirements:
            "pip install tqdm mpi4py",
            "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
            "pip install pyannote.audio faster-whisper bitsandbytes transformers accelerate datasets peft optimum",
            "pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/",
            "pip install langchain openai",
            "pip install git+https://github.com/suno-ai/bark.git",  # suno-bark
            "pip install streamlit st-annotated-text spacy librosa presidio-anonymizer presidio-analyzer nltk flair",
            "pip install -U SQLAlchemy",
            "pip uninstall -y onnxruntime-gpu",
            "pip uninstall -y onnxruntime",
            "pip install onnxruntime-gpu",
            "python -m spacy download en_core_web_lg",
        ],
        set_as_default=True,
    )


def _set_secrets(
    project: mlrun.projects.MlrunProject,
    openai_key: str,
    openai_base: str,
    huggingface_token: str,
    mysql_url: str,
):
    # Set the secrets in the project:
    project.set_secrets(
        secrets={
            ProjectSecrets.OPENAI_API_KEY: openai_key,
            ProjectSecrets.OPENAI_API_BASE: openai_base,
            ProjectSecrets.HUGGING_FACE_HUB_TOKEN: huggingface_token,
            ProjectSecrets.MYSQL_URL: mysql_url,
        }
    )


def _set_function(
    project: mlrun.projects.MlrunProject,
    func: str,
    name: str,
    kind: str,
    gpus: int = None,
    apply_auto_mount: bool = False,
):
    # Set the given function:
    mlrun_function = project.set_function(func=func, name=name, kind=kind)

    # Configure GPUs according to the given kind:
    if gpus:
        if kind == "mpijob":
            # 1 GPU for each rank:
            mlrun_function.with_limits(gpus=1)
            mlrun_function.spec.replicas = gpus
        else:
            # All GPUs for the single job:
            mlrun_function.with_limits(gpus=gpus)

    # Apply auto-mount if needed:
    if apply_auto_mount:
        mlrun_function.apply(mlrun.auto_mount())

    # Save:
    mlrun_function.save()


def _set_calls_generation_functions(
    project: mlrun.projects.MlrunProject, gpus: int, apply_auto_mount: bool
):
    # Conversation generator:
    _set_function(
        project=project,
        func="./src/calls_generation/conversations_generator.py",
        name="conversations-generator",
        kind="job",
        apply_auto_mount=apply_auto_mount,
    )

    # Text to audio generator:
    _set_function(
        project=project,
        func="./src/hub_functions/text_to_audio_generator.py",
        name="text-to-audio-generator",
        kind="job",  # TODO: MPI once MLRun supports it out of the box
        gpus=gpus,
        apply_auto_mount=apply_auto_mount,
    )


def _set_calls_analysis_functions(
    project: mlrun.projects.MlrunProject, gpus: int, apply_auto_mount: bool
):
    # DB management:
    _set_function(
        project=project,
        func="./src/calls_analysis/db_management.py",
        name="db-management",
        kind="job",
        apply_auto_mount=apply_auto_mount,
    )

    # Speech diarization:
    _set_function(
        project=project,
        func="./src/hub_functions/speech_diarization.py",
        name="speech-diarization",
        kind="mpijob" if gpus and gpus > 1 else "job",
        gpus=gpus,
        apply_auto_mount=apply_auto_mount,
    )

    # Transcription:
    _set_function(
        project=project,
        func="./src/hub_functions/transcribe.py",
        name="transcription",
        kind="mpijob" if gpus and gpus > 1 else "job",
        gpus=gpus,
        apply_auto_mount=apply_auto_mount,
    )

    # Translation:
    _set_function(
        project=project,
        func="./src/hub_functions/translate.py",
        name="translation",
        kind="mpijob" if gpus and gpus > 1 else "job",
        gpus=gpus,
        apply_auto_mount=apply_auto_mount,
    )

    # PII recognition:
    _set_function(
        project=project,
        func="./src/hub_functions/pii_recognizer.py",
        name="pii-recognition",
        kind="job",
        apply_auto_mount=apply_auto_mount,
    )

    # Question answering:
    _set_function(
        project=project,
        func="./src/hub_functions/question_answering.py",
        name="question-answering",
        kind="job",
        gpus=gpus,
        apply_auto_mount=apply_auto_mount,
    )

    # Postprocessing:
    _set_function(
        project=project,
        func="./src/calls_analysis/postprocessing.py",
        name="postprocessing",
        kind="job",
        apply_auto_mount=apply_auto_mount,
    )


def _set_workflows(project: mlrun.projects.MlrunProject):
    project.set_workflow(
        name="calls-generation", workflow_path="./src/workflows/calls_generation.py"
    )
    project.set_workflow(
        name="calls-analysis", workflow_path="./src/workflows/calls_analysis.py"
    )
