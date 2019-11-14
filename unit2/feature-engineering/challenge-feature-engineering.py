import pandas as pd

video_game_sales = pd.read_csv('vgsales.csv')

video_game_sales['Total_sales'] = video_game_sales['NA_Sales'] + video_game_sales['EU_Sales'] \
                                 + video_game_sales['JP_Sales'] + video_game_sales['Other_Sales']

print(video_game_sales.head())
