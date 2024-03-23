import itertools
from typing import NamedTuple

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig


class NLIModel:

    def __init__(self, hg_model_name: str, device: str = 'cpu', labels2id: dict[str, int] = None):
        """
        Natural Language Inference component.
        :param hg_model_name: Hugging Face name of the pretrained NLI model
        """
        self.llm = AutoModelForSequenceClassification.from_pretrained(hg_model_name, torch_dtype=torch.float16).to(
            device).eval()
        self.model_name = hg_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(hg_model_name)
        self.device = device
        if labels2id is None:
            config = AutoConfig.from_pretrained(hg_model_name)
            labels2id = {k.lower(): v for k, v in config.label2id.items()}
        self.entailment_idx = labels2id['entailment']
        model_size = self.llm.num_parameters()
        print(f"model size: {model_size:,}")
        self.statistics = []

    def reset_statistics(self):
        self.statistics = []

    def _tracking_stats(self, combinations: list[(str, str)], predicted_classes: torch.tensor):
        for pair, predict in zip(combinations, predicted_classes):
            self.statistics.append({
                'p': pair[0],
                'h': pair[1],
                'output': predict.item(),
                'model': self.model_name
            })

    def check_context_entailment(self, beliefs: list[str], plan_contexts: list[str]) -> (bool, float):
        """
        Infer whether the plan context are entailed by the belief base
        :param beliefs: Beliefs contained in belief base
        :param plan_contexts: Statements contained in plan context
        :return: True whether the belief base entails the plan context with the entailment score
        """

        num_ctx_statements = len(plan_contexts)
        num_beliefs = len(beliefs)

        # combination with all (belief,context) available pairs
        combinations = list(itertools.product(beliefs, plan_contexts))
        # sort by the context statement to facilitate the entailment retrieval in the following steps
        combinations.sort(key=lambda x: x[1])

        logits = self._predict_nli(combinations)
        probs = torch.softmax(logits, dim=1)
        argmax_probs = probs.argmax(-1)  # predicted class indexes (neutral, entailment, contradiction)

        self._tracking_stats(combinations, argmax_probs)

        # True when a c_n is entailed by b_n
        entailment_mask = torch.where(argmax_probs == self.entailment_idx, True, False)
        # [B,c1:B,c2:...:B,cn]
        slice_idx = []
        idx = 0
        for i in range(num_ctx_statements):  # [c1, ..., cn]
            slice_idx.append(argmax_probs[idx:(idx + num_beliefs)])
            idx = num_beliefs

        # True if ANY context comparation is ENTAILED by an belief in belief base (OR)
        context_or = [torch.where(c == self.entailment_idx, True, False).any().unsqueeze(0) for c in slice_idx]
        or_tensor = torch.concatenate(context_or)  # at least one belief should entail a context (OR)
        and_result = or_tensor.all()  # all context must be entailed by the belief base (AND)
        entailment_result = and_result.item()  # boolean result

        entailment_probs = (probs[:, self.entailment_idx] * entailment_mask)  # retrieving only entailment predictions
        entailment_score = entailment_probs[entailment_probs != 0].mean()  # mean of all entailment predictions

        return entailment_result, entailment_score.item()

    def _predict_nli(self, sentence_pairs: list[(str, str)]) -> torch.tensor:
        """
        Generate a NLI prediction from a list of sentence pairs
        :param sentence_pairs: batch of sentence pairs
        :return: logits of each sentence pair contained in the input batch
        """
        tokenized_input_seq_pair = self.tokenizer.batch_encode_plus(sentence_pairs,
                                                                    return_token_type_ids=True,
                                                                    truncation=True,
                                                                    padding=True)

        input_ids = torch.tensor(tokenized_input_seq_pair['input_ids'], device=self.device).long()
        token_type_ids = torch.tensor(tokenized_input_seq_pair['token_type_ids'], device=self.device).long()
        attention_mask = torch.tensor(tokenized_input_seq_pair['attention_mask'], device=self.device).long()

        outputs = self.llm(input_ids,
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids,
                           labels=None)

        return outputs[0]
