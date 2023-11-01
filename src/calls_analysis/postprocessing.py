import pandas as pd


def postprocess_answers(answers: pd.DataFrame):
    answers["concern_addressed"] = answers["concern_addressed"].apply(
        lambda x: "yes" in x
    )

    return answers
