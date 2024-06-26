import os
from functools import partial, lru_cache

import numpy as np
from fastapi import FastAPI, HTTPException
from starlette.responses import RedirectResponse

from .dataset import CharTokenizedDataset
from .transformer import Transformer

app = FastAPI(title="NLP Engineer Assignment", version="1.0.0")

device = "cpu"
model = None
vocabs = [chr(ord("a") + i) for i in range(0, 26)] + [" "]

"""
load model from pretrained
"""
async def load_model(device):
    global model
    #load pretrained model if it exists, raise error otherwise
    model_path = os.path.join(
        os.path.abspath(os.path.join(__file__, "../../../")), "data", "trained_model.ckpt"
    )
    if os.path.exists(model_path):
        model = Transformer.from_pretrained(model_path)
        model = model.to(device)
        model.eval()
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")


app.add_event_handler("startup", partial(load_model, device=device))


@app.get("/", include_in_schema=False)
async def index():
    """
    Redirects to the OpenAPI Swagger UI
    """
    return RedirectResponse(url="/docs")
"""
Cache response
"""
@lru_cache(maxsize=1000)
def get_prediction(encoded_text: tuple):
    global model
    if model is None:
        raise HTTPException(
            status_code=503, detail="Service Unavailable: Model not loaded"
        )
    logits, prediction = model.generate(np.array(encoded_text))
    prediction = prediction.to("cpu").numpy().tolist()
    return prediction

# TODO: Add a route to the API that accepts a text input and uses the trained
# model to predict the number of occurrences of each letter in the text up to
# that point.
@app.post("/predict")
async def predict(text: str):
    global vocabs
    #Error handing if model does not exist
    if model is None:
        raise HTTPException(
            status_code=503, detail="Service Unavailable: Model not loaded"
        )

    encoded_text, _ = CharTokenizedDataset.encode_sentence(sentence=text, vocab=vocabs)
    encoded_text = np.array(encoded_text)
    logits, prediction = model.generate(encoded_text)
    prediction = prediction.to("cpu").numpy().tolist()
    prediction_dict = dict(counts=prediction, counts_with_text=[(char, count) for char, count in zip(text, prediction)])
    return {"input": text, "prediction": prediction_dict}
