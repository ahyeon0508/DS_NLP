from konlpy.tag import Okt

okt = Okt()
text = '2020년 12월 21일 본격적인 자연어 처리 공부 시작 화이팅! 다양한 예제 및 프로젝트를 진행할 예정입니다.'

# morphs() : 텍스트를 형태소 단위로 나눔
# 옵션 : 기본값은 둘 다 False
# - norm : 정규화
# - stem : 각 단어에서 어간을 추출
print(okt.morphs(text))
print(okt.morphs(text, norm = True))
print(okt.morphs(text, stem = True))

# nouns() : 명사 추출
print(okt.nouns(text))

# phrases() : 어절 추출
print(okt.phrases(text))

# pos() : 각 품사 태깅하는 역할
# 품사 태깅? 주어진 텍스트를 형태소 단위로 나누고, 나누어진 각 형태소의 품사를 형태소와 함께 리스트화
print(okt.pos(text))
