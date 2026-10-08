# --------------------------------------------------
# Load NLTK stopwords
# --------------------------------------------------
try:
	stop_words_en = stopwords.words('english')
except LookupError:
	nltk.download('stopwords')
	stop_words_en = stopwords.words('english')
