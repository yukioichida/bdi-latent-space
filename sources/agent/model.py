import itertools

import torch


class NLIModel:

    def __init__(self):
        # TODO:
        self.llm = None
        self.tokenizer = None
        self.device = None
        self.entailment_idx = None

    def infer_entailment(self, beliefs: list[str], plan_contexts: list[str]) -> (bool, float):
        """
        Infer whether the plan context are entailed by the belief base
        :param beliefs: Beliefs contained in belief base
        :param plan_contexts: Statements contained in plan context
        :return: True whether the belief base entails the plan context with the entailment score
        """

        num_ctx_statements = len(plan_contexts)
        num_beliefs = len(beliefs)

        combinations = list(itertools.product(beliefs, plan_contexts))
        # sort by the context statement to facilitate the entailment retrieval in the following steps
        combinations.sort(key=lambda x: x[1])

        tokenized_input_seq_pair = self.tokenizer.batch_encode_plus(combinations,
                                                                    return_token_type_ids=True, truncation=True,
                                                                    padding=True)

        input_ids = torch.tensor(tokenized_input_seq_pair['input_ids'], device=self.device).long()
        token_type_ids = torch.tensor(tokenized_input_seq_pair['token_type_ids'], device=self.device).long()
        attention_mask = torch.tensor(tokenized_input_seq_pair['attention_mask'], device=self.device).long()

        outputs = self.llm(input_ids,
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids,
                           labels=None)

        logits = outputs[0]
        probs = torch.softmax(logits, dim=1)
        argmax_probs = probs.argmax(-1) # predicted class indexes (neutral, entailment, contradiction)

        # True when a c_n is entailed by b_n
        entailment_mask = torch.where(argmax_probs == self.entailment_idx, True, False)
        # [B,c1:B,c2:...:B,cn]
        slice_idx = [num_beliefs * i for i in range(num_ctx_statements)]

        # test for each context whether any belief entails it.
        or_results = []
        for i in slice_idx:
            context_comparison = entailment_mask[i: i + num_beliefs]
            or_results.append(context_comparison)

        # relative to AND operation
        entailment_result = torch.tensor(or_results).all()

        all_entailment_prob = (probs * entailment_mask).mean()

        return entailment_result, all_entailment_prob





