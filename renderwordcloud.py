import csv
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS


your_list = ""
bad_list = ""
stopwords = set(STOPWORDS)
with open(os.path.join("data", "train.csv"), newline='\n') as f:
    reader = csv.reader(f)
    for row in reader:
        your_list += row[1]
        if row[2]=="1" or row[3]=="1" or row[4]=="1" or row[5]=="1" or row[6]=="1" or row[7]=="1":
            bad_list += row[1]
    
mywordcl = WordCloud(width = 1920, height = 1080, stopwords = stopwords).generate(your_list)
badwordcl = WordCloud(width = 1920, height = 1080, stopwords = stopwords).generate(bad_list)
plt.figure()
plt.imshow(mywordcl, interpolation="bilinear")
plt.gcf().set_facecolor("black")
plt.axis("off")
plt.show()
plt.imshow(badwordcl, interpolation="bilinear")
plt.gcf().set_facecolor("black")
plt.axis("off")
plt.show()
