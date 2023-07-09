class NLIModel:

    def __init__(self):
        self.llm = None

    def entailment_relation(self, obs: list[str], obs2: list[str]):
        pass


if __name__ == '__main__':
    import time

    start = time.time()
    print("Loading Model")

    p = ['This room is called the art studio.', 'you see the agent', 'you see a substance called air',
           'you see a large cupboard. The large cupboard door is closed.',
           'you see a table. On the table is: a glass cup (containing nothing).']

    p = "you see a table. on the table is: a glass cup containing water. table is a furniture"
    #p = "container"
    h = "a dog containing water on a furniture"
    #h = "glass cup"


    from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
    import torch

    max_length = 256

    #premise = "Two women are embracing while holding to go packages."
    #premise = "The men are embracing while holding to go packages."
    #hypothesis = "The men are fighting outside a deli."

    #hg_model_hub_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"

    hg_model_hub_name = "alisawuffles/roberta-large-wanli"

    config = AutoConfig.from_pretrained(hg_model_hub_name)
    tokenizer = AutoTokenizer.from_pretrained(hg_model_hub_name)

    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device='cpu'

    model = AutoModelForSequenceClassification.from_pretrained(hg_model_hub_name)
    model.to(device)

    end = time.time()
    print(f"Model loaded {end - start} - model {model.device}")

    start = time.time()
    tokenized_input_seq_pair = tokenizer.encode_plus(p,h,
                                                     max_length=max_length,
                                                     return_token_type_ids=True, truncation=True, padding=True)

    input_ids = torch.tensor(tokenized_input_seq_pair['input_ids'], device=device).long().unsqueeze(0)
    # remember bart doesn't have 'token_type_ids', remove the line below if you are using bart.
    token_type_ids = torch.tensor(tokenized_input_seq_pair['token_type_ids'], device=device).long().unsqueeze(0)
    attention_mask = torch.tensor(tokenized_input_seq_pair['attention_mask'], device=device).long().unsqueeze(0)

    #print(config)

    outputs = model(input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=None)
    # Note:
    # "id2label": {
    #     "0": "entailment",
    #     "1": "neutral",
    #     "2": "contradiction"
    # },
    end = time.time()

    print(f"Inference time: {end - start}")
    logits = outputs[0]
    predicted_probability = torch.softmax(logits, dim=1)[0].tolist()  # batch_size only one
    print("Premise:", p)
    print("Hypothesis:", h)
    print("Entailment:", predicted_probability[int(config.label2id['entailment'])])
    print("Neutral:", predicted_probability[int(config.label2id['neutral'])])
    print("Contradiction:", predicted_probability[int(config.label2id['contradiction'])])

