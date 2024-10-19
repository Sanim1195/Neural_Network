rounak_friends = [
    "Preson1",
    "Preson4",
    "Preson3",
    "Preson2",
    "Preson5"
    ]

friends_score = [
    23,
    38,
    40,
    50
]
                  
for friends, score in zip(rounak_friends,friends_score):
    print(friends, " : ", score)
# for i,y in enumerate(rounak_friends):
#     print("At " ,i,"th index", y,": Is rounkas Friend")
