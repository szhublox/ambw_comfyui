import transformers

def score(image):
    pipe = transformers.pipeline("image-classification",
                                 model="cafeai/cafe_waifu")
    result = pipe(image, top_k=5)
    for data in result:
        if data['label'] == "waifu":
            return data['score']
