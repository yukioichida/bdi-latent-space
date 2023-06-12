from transformers import AutoTokenizer, AutoModelForSequenceClassification, RobertaForSequenceClassification, \
    RobertaTokenizer
import torch

if __name__ == '__main__':
    max_length = 512

#if plan.task entails belief.task:
	#if belief.look+inv entails plan.plan_context:
    premise = """
    Your task is to boil gallium.
 This room is called the art studio. In it, you see: 
	the agent
	a substance called air
	a large cupboard. The large cupboard door is closed. 
	a table. On the table is: a glass cup (containing nothing).
	a wood cup (containing blue paint)
	a wood cup (containing yellow paint)
	a wood cup (containing red paint)
You also see:
	A door to the hallway (that is closed)
"""#.replace('\t', '').replace('\n', ' ')


    premise = " ".join(premise.split()).lower()
    premise = "your task is to boil water"
    #hypothesis = 'your task is to boil galium AND you are not in the workshop'
    hypothesis = 'your task is to boil galium'

    hg_model_hub_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
    #hg_model_hub_name = "ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli"
    # hg_model_hub_name = "ynie/bart-large-snli_mnli_fever_anli_R1_R2_R3-nli"
    # hg_model_hub_name = "ynie/electra-large-discriminator-snli_mnli_fever_anli_R1_R2_R3-nli"
    # hg_model_hub_name = "ynie/xlnet-large-cased-snli_mnli_fever_anli_R1_R2_R3-nli"
    #hg_model_hub_name = 'alisawuffles/roberta-large-wanli'

    tokenizer = AutoTokenizer.from_pretrained(hg_model_hub_name)
    model = AutoModelForSequenceClassification.from_pretrained(hg_model_hub_name)

    tokenized_input_seq_pair = tokenizer.encode_plus(premise, hypothesis,
                                                     max_length=max_length,
                                                     return_token_type_ids=True, truncation=True)

    input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long().unsqueeze(0)
    # remember bart doesn't have 'token_type_ids', remove the line below if you are using bart.
    token_type_ids = torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long().unsqueeze(0)
    attention_mask = torch.Tensor(tokenized_input_seq_pair['attention_mask']).long().unsqueeze(0)


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

    predicted_probability = torch.softmax(outputs[0], dim=1)[0].tolist()  # batch_size only one

    label2id = {label: idx for idx, label in model.config.id2label.items()}

    entails_idx = label2id['entailment']
    neutral_idx = label2id['neutral']
    contradiction_idx = label2id['contradiction']

    print(f"{entails_idx} - {neutral_idx} - {contradiction_idx}")

    print("Premise:", premise)
    print("Hypothesis:", hypothesis)
    print("Entailment:", predicted_probability[entails_idx])
    print("Neutral:", predicted_probability[neutral_idx])
    print("Contradiction:", predicted_probability[contradiction_idx])

    print(model.config.id2label)

    label_id = torch.argmax(torch.softmax(outputs[0], dim=1)[0]).item()
    prediction = model.config.id2label[label_id]
    print(prediction)
    """
    model = RobertaForSequenceClassification.from_pretrained('alisawuffles/roberta-large-wanli')
    tokenizer = RobertaTokenizer.from_pretrained('alisawuffles/roberta-large-wanli')

    x = tokenizer(premise, hypothesis, return_tensors='pt', max_length=256,
                  truncation=True)
    logits = model(**x).logits
    probs = logits.softmax(dim=1).squeeze(0)
    label_id = torch.argmax(probs).item()
    prediction = model.config.id2label[label_id]

    predicted_probability = probs.tolist()
    print(prediction)
    print("Entailment:", predicted_probability[1])
    print("Neutral:", predicted_probability[2])
    print("Contradiction:", predicted_probability[0])
    print(model.config.id2label)
    """