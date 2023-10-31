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
import datetime
import hashlib
import os
import pathlib
import random
from typing import Tuple

import mlrun
import pandas as pd
import tqdm
from langchain.chat_models import ChatOpenAI

from src.common import TONES, TOPICS, ProjectSecrets

#: The approximate amount of words in one minute.
WORDS_IN_1_MINUTE = 240


def generate_conversations(
    context: mlrun.MLClientCtx,
    amount: int,
    output_directory: str,
    model_name: str = "gpt-3.5-turbo",
    language: str = "en",
    min_time: int = 2,
    max_time: int = 5,
    from_date: str = "01.01.2023",
    to_date: str = "01.03.2023",
    from_time: str = "09:00",
    to_time: str = "17:00",
    # TODO: Remove for using a dedicated generation function in 'data_generator.py'.
    n_agents: int = 4,
    n_clients: int = 4,
) -> Tuple[str, pd.DataFrame, pd.DataFrame]:
    """
    Generates a list of conversations between an internet provider call center and a customer.

    :param context:
    :param amount: The number of conversations to generate.
    :param output_directory:
    :param model_name: The name of the model to use for conversation generation. You should choose one of GPT-4 or
                       GPT-3.5 from the list here: https://platform.openai.com/docs/models. Default: 'gpt-3.5-turbo'.
    :param language: The language to use for the generated conversation text.
    :param min_time: Minimum time of conversation in minutes. Will be used approximately to generate the minimum words
                       with the following assessment: 240 words are equal to one minute. Default: 2.
    :param max_time: Maximum time of conversation in minutes. Will be used approximately to generate the maximum words
                       with the following assessment: 240 words are equal to one minute. Default: 5.
    :param from_date:
    :param to_date:
    :param from_time:
    :param to_time:
    :param n_agents:
    :param n_clients:

    """
    # Get the minimum and maximum amount of words:
    min_words = WORDS_IN_1_MINUTE * min_time
    max_words = WORDS_IN_1_MINUTE * max_time

    # Get the minimum and maximum dates and times:
    min_time = datetime.datetime.strptime(from_time, "%H:%M")
    max_time = datetime.datetime.strptime(to_time, "%H:%M")
    min_date = datetime.datetime.strptime(from_date, "%m.%d.%Y").date()
    max_date = datetime.datetime.strptime(to_date, "%m.%d.%Y").date()

    # Create the concern addressed options:
    concern_addressed_options = {
        "yes": "",
        "no": "Don't",
    }

    # Generate agents and clients:
    # TODO: Remove once there are tables of agents and clients
    clients = [_generate_id() for _ in range(n_clients)]
    agents = [_generate_id() for _ in range(n_agents)]

    # Create the prompt structure:
    prompt_structure = (
        "Generate a conversation between an internet provider call center agent (“Iguazio-Mckinsey Internet”) and "
        "a client in {language}.\n"
        "Format the conversation as follow:\n"
        "Agent: <text here>\n"
        "Client: <text here>>\n"
        "The conversations has to include at least {min_words} words and no more than {max_words} words.\n"
        "It has to be about a client who is calling to discuss about {topic}.\n"
        "The agent {concern_addressed} address the client's concern.\n"
        "Do not add any descriptive tag and do not mark the end of the conversation with [End of conversation].\n"
        "Use ... for hesitation.\n"
        "The client needs to have a {client_tone} tone.\n"
        "The agent needs to have a {agent_tone}.\n"
        "Remove from the final output any word inside parentheses of all types."
    )

    # Load the OpenAI model using langchain:
    os.environ["OPENAI_API_KEY"] = context.get_secret(key=ProjectSecrets.OPENAI_API_KEY)
    os.environ["OPENAI_API_BASE"] = context.get_secret(
        key=ProjectSecrets.OPENAI_API_BASE
    )
    llm = ChatOpenAI(model=model_name)

    # Create the output directory:
    output_directory = pathlib.Path(output_directory)
    output_directory.mkdir(exist_ok=True)

    # Start generating conversations:
    conversations = []
    ground_truths = []
    for _ in tqdm.tqdm(range(amount), desc="Generating"):
        # Randomize the conversation metadata:
        conversation_id = _generate_id()
        date = _get_random_date(min_date=min_date, max_date=max_date)
        time = _get_random_time(min_time=min_time, max_time=max_time)
        # Randomly select the conversation parameters:
        concern_addressed = random.choice(list(concern_addressed_options.keys()))
        client_tone = random.choice(TONES)
        agent_tone = random.choice(TONES)
        topic = random.choice(TOPICS)
        # Create the prompt:
        prompt = prompt_structure.format(
            language=language,
            min_words=min_words,
            max_words=max_words,
            topic=topic,
            concern_addressed=concern_addressed_options[concern_addressed],
            client_tone=client_tone,
            agent_tone=agent_tone,
        )
        # Generate the conversation:
        conversation = llm.predict(text=prompt)
        # Remove redundant newlines and spaces:
        conversation = "".join(
            [
                line
                for line in conversation.strip().splitlines(keepends=True)
                if line.strip("\n").strip()
            ]
        )
        # Save to file:
        conversation_text_path = output_directory / f"{conversation_id}.txt"
        with open(conversation_text_path, "w") as fp:
            fp.write(conversation)
        # Collect to the conversations and ground truths lists:
        # TODO: Remove the clients and agents once are in separated tables
        conversations.append(
            [
                conversation_id,
                conversation_text_path.name,
                random.choice(clients),
                random.choice(agents),
                date,
                time,
            ]
        )
        ground_truths.append(
            [
                conversation_id,
                language,
                topic,
                concern_addressed,
                client_tone,
                agent_tone,
            ]
        )

    # Cast the conversations and ground truths into a dataframe:
    conversations = pd.DataFrame(
        conversations,
        columns=["call_id", "text_file", "client_id", "agent_id", "date", "time"],
    )
    ground_truths = pd.DataFrame(
        ground_truths,
        columns=[
            "call_id",
            "language",
            "topic",
            "concern_addressed",
            "client_tone",
            "agent_tone",
        ],
    )

    return str(output_directory), conversations, ground_truths


def _generate_id() -> str:
    return hashlib.md5(str(datetime.datetime.now()).encode("utf-8")).hexdigest()


def _get_random_time(
    min_time: datetime.datetime, max_time: datetime.datetime
) -> datetime.time:
    if max_time.hour <= min_time.hour:
        max_time += datetime.timedelta(days=1)
    return (
        min_time
        + datetime.timedelta(
            seconds=random.randint(0, int((max_time - min_time).total_seconds())),
        )
    ).time()


def _get_random_date(min_date, max_date) -> datetime.date:
    return min_date + datetime.timedelta(
        days=random.randint(0, int((max_date - min_date).days)),
    )


def create_batch_for_analysis(
    conversations_data: pd.DataFrame, audio_files: pd.DataFrame
) -> pd.DataFrame:
    conversations_data = conversations_data.join(
        other=audio_files.set_index(keys="text_file"), on="text_file"
    )
    conversations_data.drop(columns="text_file", inplace=True)
    conversations_data.dropna(inplace=True)
    return conversations_data
