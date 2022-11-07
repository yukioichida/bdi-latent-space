import pandas as pd

goldpath_df = pd.read_csv("dataset_sentence_level.csv")

randomwalk_df = pd.read_csv("all_sentences_random_walk.csv")

join_df = pd.concat([randomwalk_df, goldpath_df])

join_df[~join_df['sentence'].isnull()].to_csv("all_sentences_join.csv", index=False)