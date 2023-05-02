from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

if __name__ == '__main__':
    max_length = 512

    premise = """
Your task is to boil gallium. This room is called the art studio. In it, you see: 
	the agent
	a substance called air
	a large cupboard. The large cupboard door is closed. 
	a table. On the table is: a glass cup (containing nothing).
	a wood cup (containing blue paint)
	a wood cup (containing red paint)
	a wood cup (containing yellow paint)
You also see:
	A door to the hallway (that is closed) In your inventory, you see:
	an orange
    
    """
    hypothesis = 'your goal is to boil an water AND your are seeing the kitchen door'

    hg_model_hub_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
    #hg_model_hub_name = "ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli"
    # hg_model_hub_name = "ynie/bart-large-snli_mnli_fever_anli_R1_R2_R3-nli"
    # hg_model_hub_name = "ynie/electra-large-discriminator-snli_mnli_fever_anli_R1_R2_R3-nli"
    # hg_model_hub_name = "ynie/xlnet-large-cased-snli_mnli_fever_anli_R1_R2_R3-nli"

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

    print("Premise:", premise)
    print("Hypothesis:", hypothesis)
    print("Entailment:", predicted_probability[0])
    print("Neutral:", predicted_probability[1])
    print("Contradiction:", predicted_probability[2])