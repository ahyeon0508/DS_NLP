from konlpy.tag import Okt
text = '안녕하세요 음 저는 AI에 관심을 갖고 이 학교에 지원하게 되었는데요 많은 것을 배우고 싶습니다'

okt = Okt()
okt_noun = okt.nouns(text)
print(okt_noun)