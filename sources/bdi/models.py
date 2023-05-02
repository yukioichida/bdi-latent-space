from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


class NLIModel:

    def __init__(self,
                 hg_model_hub_name: str = 'ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli',
                 max_length: int = 512):
        """
        Natural language inference model
        :param hg_model_hub_name: Name of NLI model registered in Hugging face hub
        :param max_length: Maximum length of sequence which includes both premise and hypothesis
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.model = AutoModelForSequenceClassification.from_pretrained(hg_model_hub_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(hg_model_hub_name)
        
        self.max_length = max_length

    def entails(self, p: str, h: str) -> (bool, float):
        """
        Verify whether the hypothesis entails the premise
        :param p: premise text
        :param h: hypothesis text
        :return: True whether there is an entailment relation and the confidence level
        """

        tokenized_seq = self.tokenizer.encode_plus(p, h,
                                                   max_length=self.max_length,
                                                   return_token_type_ids=True,
                                                   truncation=True)

        input_ids = torch.tensor(tokenized_seq['input_ids'], device=self.device).long().unsqueeze(0)
        token_type_ids = torch.tensor(tokenized_seq['token_type_ids'], device=self.device).long().unsqueeze(0)
        attention_mask = torch.tensor(tokenized_seq['attention_mask'], device=self.device).long().unsqueeze(0)
        # model prediction
        outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=None)

        predicted_probability = torch.softmax(outputs[0], dim=1)
        probabilities = torch.softmax(outputs[0], dim=1)[0].tolist()  # batch_size only one

        entails = predicted_probability.argmax() == 0
        return entails, probabilities[0]
