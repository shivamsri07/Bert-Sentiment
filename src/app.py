import config
import torch
import time
import flask
from flask import Flask
from model import  BertSentiment
import functools
import torch.nn as nn
import joblib
from flask import request


app = Flask(__name__)

MODEL = None
DEVICE = "cuda"



MODEL = None
DEVICE = "cuda"
PREDICTION_DICT = dict()
memory = joblib.Memory("../input/", verbose=0)


def predict_from_cache(sentence):
    if sentence in PREDICTION_DICT:
        return PREDICTION_DICT[sentence]
    else:
        result = sentence_prediction(sentence)
        PREDICTION_DICT[sentence] = result
        return result


@memory.cache
def sentence_prediction(sentence):
    tokenizer = config.TOKENIZER
    max_len = config.MAX_LEN
    review = str(sentence)
    review = " ".join(review.split())

    inputs = tokenizer.encode_plus(
        review,
        None,
        add_special_tokens=True,
        max_length=max_len,
        pad_to_max_length=True,
    )


    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]


    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0)

    ids = ids.to(DEVICE, dtype=torch.long)
    token_type_ids = token_type_ids.to(DEVICE, dtype=torch.long)
    mask = mask.to(DEVICE, dtype=torch.long)

    outputs = MODEL(ids=ids, mask=mask, token_type_ids=token_type_ids)

    outputs = torch.sigmoid(outputs).cpu().detach().numpy()
    return outputs[0][0]

@app.route("/predict")
def predict():
    sentence = request.args.get("sentence")
    start_time = time.time()
    positive_prediction = sentence_prediction(sentence)
    negative_prediction = 1 - positive_prediction
    response = {}
    response["response"] = {
        "positive": str(positive_prediction),
        "negative": str(negative_prediction),
        "sentence": str(sentence),
        "time_taken": str(time.time() - start_time),
    }
    return flask.jsonify(response)


if __name__ == "__main__":
    MODEL = BertSentiment()
    MODEL = nn.DataParallel(MODEL)
    MODEL.load_state_dict(torch.load(config.MODEL_PATH))
    MODEL.to(DEVICE)
    MODEL.eval()
    app.run()