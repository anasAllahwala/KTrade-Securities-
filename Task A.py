#this code basically sorts out the data based on the reviews keeping the 1-star reviews on first, and then second, and then third, and so on
import pandas as pd
file_path = '/mnt/data/Year_2023_Ktrade_reviews.xlsx'
data = pd.read_excel(file_path)
sorted_data = data.sort_values(by='Rating', ascending=True)

# Saving the sorted data to a new file
sorted_file_path = '/mnt/data/Sorted_by_Rating_Year_2023_Ktrade_reviews.xlsx'
sorted_data.to_excel(sorted_file_path, index=False)

sorted_file_path

#----------------------------------------------------------------------------------------------------------------------------------
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Separating reviews based on ratings
negative_reviews = reviews_data[reviews_data['Rating'] <= 2]['Review Content']
positive_reviews = reviews_data[reviews_data['Rating'] >= 4]['Review Content']

# Generating word clouds for positive and negative reviews
wordcloud_neg = WordCloud(width = 800, height = 800, 
                          background_color ='white', 
                          min_font_size = 10).generate(" ".join(negative_reviews))

wordcloud_pos = WordCloud(width = 800, height = 800,
                          background_color ='white', 
                          min_font_size = 10).generate(" ".join(positive_reviews))

# Plotting the WordCloud images
plt.figure(figsize = (12, 12), facecolor = None) 
plt.subplot(1, 2, 1)
plt.imshow(wordcloud_neg) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.title("Word Cloud of Negative Reviews")

plt.subplot(1, 2, 2)
plt.imshow(wordcloud_pos)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.title("Word Cloud of Positive Reviews")

plt.show()
-----------------------------------------------------------------------------------------------------------------------------------
#this code basically visualises the trends of ratings 
import matplotlib.dates as mdates

# Preparing data for temporal trend analysis
reviews_data['Month'] = reviews_data['Date'].dt.to_period('M')
monthly_ratings = reviews_data.groupby(['Month', 'Rating']).size().unstack(fill_value=0)

# Plotting the temporal trend of reviews by ratings
plt.figure(figsize=(12, 6))
plt.plot(monthly_ratings)
plt.title('Monthly Ratings Trend')
plt.xlabel('Month')
plt.ylabel('Number of Reviews')
plt.legend(monthly_ratings.columns, title='Ratings')
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

---------------------------------------------------------------------------------------------------------------------
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns

def get_top_n_phrases(corpus, n=None):
    """
    Function to identify the top n phrases in a corpus of text
    """
    vec = CountVectorizer(ngram_range=(2, 3), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

# Extracting top phrases from negative and positive reviews
top_phrases_negative = get_top_n_phrases(negative_reviews_cleaned, 10)
top_phrases_positive = get_top_n_phrases(positive_reviews_cleaned, 10)

# Converting to DataFrame for easier visualization
df_top_phrases_neg = pd.DataFrame(top_phrases_negative, columns=['Phrase', 'Frequency'])
df_top_phrases_pos = pd.DataFrame(top_phrases_positive, columns=['Phrase', 'Frequency'])

# Plotting the top phrases in negative reviews
plt.figure(figsize=(10, 6))
sns.barplot(x='Frequency', y='Phrase', data=df_top_phrases_neg)
plt.title('Top Phrases in Negative Reviews')
plt.xlabel('Frequency')
plt.ylabel('Phrase')
plt.show()

# Plotting the top phrases in positive reviews
plt.figure(figsize=(10, 6))
sns.barplot(x='Frequency', y='Phrase', data=df_top_phrases_pos)
plt.title('Top Phrases in Positive Reviews')
plt.xlabel('Frequency')
plt.ylabel('Phrase')
plt.show()

-----------------------------------------------------------------------------------------------------------------
#Distribution of App Ratings 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Re-creating the DataFrame for the ratings and their counts
ratings_data = {
    'Rating': [1, 2, 3, 4, 5],
    'Count': [138, 22, 28, 2, 82]
}
df_ratings = pd.DataFrame(ratings_data)

# Plotting the distribution of ratings
plt.figure(figsize=(10, 6))
sns.barplot(x='Rating', y='Count', data=df_ratings, palette='viridis')
plt.title('Distribution of App Ratings')
plt.xlabel('Rating (Stars)')
plt.ylabel('Number of Reviews')
plt.show()
----------------------------------------------------------------------------------------------------------------------------
#this code is written for trend analysis 
file_path = '/mnt/data/Sorted_by_Rating_Year_2023_Ktrade_reviews.xlsx'
reviews_data = pd.read_excel(file_path)

reviews_data['Date'] = pd.to_datetime(reviews_data['Date'])
reviews_data['Month'] = reviews_data['Date'].dt.to_period('M').dt.to_timestamp()


monthly_ratings = reviews_data.groupby(['Month', 'Rating']).size().unstack(fill_value=0)

colors = {1: 'red', 2: 'blue', 3: 'green', 4: 'yellow', 5: 'orange'}

plt.figure(figsize=(12, 6))
for rating in monthly_ratings.columns:
    plt.plot(monthly_ratings.index, monthly_ratings[rating], label=f'{rating} Star', color=colors[rating])

plt.title('Monthly Ratings Trend (January to December)')
plt.xlabel('Month')
plt.ylabel('Number of Reviews')
plt.legend(title='Ratings')
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
--------------------------------------------------------------------------------------------------------------------------------------
