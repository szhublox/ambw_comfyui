import transformers

def score(image):
    pipe = transformers.pipeline("image-classification",
                                 model="cafeai/cafe_aesthetic")
    result = pipe(image, top_k=2)
    for data in result:
        if data['label'] == "aesthetic":
            return data['score']
