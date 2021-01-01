import nltk
nltk.download("book", quiet=True) # 말뭉치 자료 다운
nltk.corpus.gutenberg.fileids() # 저작권이 말소된 문학작품

# whitman-leaves
whitman_leaves = nltk.corpus.gutenberg.raw("whitman-leaves.txt")
print(whitman_leaves)

# 토큰 생성
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer

print(sent_tokenize(whitman_leaves[:100])) # 문장 단위로 토큰화
print(word_tokenize(whitman_leaves[:100])) # space 단위와 구두점을 기준으로 토큰화
retokenize = RegexpTokenizer("[\w]+") # 구두점 제외 -> 단어만을 가지고 토큰화 수행
print(retokenize.tokenize(whitman_leaves[:100]))

# 어간 추출
from nltk.stem import PorterStemmer, LancasterStemmer

s1 = PorterStemmer()
s2 = LancasterStemmer()

words = ["eat", "ate", "eaten", "eating"]

print("Porter Stemmer :", [s1.stem(w) for w in words])
print("Lancaster Stemmer:", [s2.stem(w) for w in words])

# 원형 복원
from nltk.stem import WordNetLemmatizer

lm = WordNetLemmatizer()
print("원형 복원 :", [lm.lemmatize(w, pos="v") for w in words])

# 품사 부착
from nltk.tag import pos_tag
sentence = "Grand is the seen, the light, to me--grand are the sky and stars"
tagged_list = pos_tag(word_tokenize(sentence))
print(tagged_list)

# 명사만 선택
nouns_list = [tag[0] for tag in tagged_list if tag[1] == "NN"]
print(nouns_list)

# 빈도
from nltk import Text
import matplotlib.pyplot as plt

text = Text(retokenize.tokenize(whitman_leaves))
text.plot(20)
plt.show() # 그래프

from nltk import FreqDist
fd_words = FreqDist(text)
print("출현 횟수가 많은 단어 5개 :", fd_words.most_common(5)) # 가장 출현 횟수가 많은 단어 5개

# 워드클라우드
from wordcloud import WordCloud
wc = WordCloud(width=1000, height=600, background_color="white", random_state=0)
plt.imshow(wc.generate_from_frequencies(fd_words))
plt.axis("off")
plt.show()