import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load ngram_summary.csv
ngram_summary = pd.read_csv('sl_data_for_dashboard//ngram_summary.csv')
ngram_summary['ngram_count'] = ngram_summary['count_real'] + ngram_summary['count_dubious']

# create histogram of ngram counts
ngram_summary['ngram_count'].hist(bins=100, range=(0, 1000))
plt.xlabel('ngram count')
plt.ylabel('frequency')
plt.title('Histogram of ngram counts')
plt.show()

# create histogram of ngram counts on log scale
ngram_summary['ngram_count'].hist(bins=100, range=(0, 1000), log=True)
plt.xlabel('ngram count')
plt.ylabel('frequency')
plt.title('Histogram of ngram counts on log scale')
plt.show()
