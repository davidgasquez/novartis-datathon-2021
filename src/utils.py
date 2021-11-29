"""
Functions to make our life easier.
"""

from datetime import datetime


def save_submission(dataframe, name="submission", path: str = "data/submissions/"):
    """Save submission to csv file with the current timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    file_name = str(name) + "_" + timestamp + ".csv"
    dataframe = dataframe[["month", "region", "brand", "sales", "lower", "upper"]]
    dataframe.to_csv(path + file_name, index=False)
    return file_name
