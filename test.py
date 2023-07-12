from sources.bdi_components.inference import NLIModel

if __name__ == '__main__':
    a = "you are in the kitchen in it you see a glass cup"
    b = "you are in the art studio and you see a glass cup"

    print("Loading model")
    model = NLIModel(hg_model_name="alisawuffles/roberta-large-wanli")

    print("Predicting")
    logits = model._predict_nli([(a,b)])

    print(logits)
    print(logits.argmax(-1) == model.entailment_idx)
