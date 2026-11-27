import pandas as pd

df = pd.read_csv("acdc_output/chokam_dataset/weather_labels.csv")

print(df.head(20))
print(df.groupby("weather").count())
print(df.groupby(["weather","weather_id"]).size())