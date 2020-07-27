from collections import OrderedDict, Counter
from tqdm import tqdm
from typing import List
from pynori.korean_analyzer import KoreanAnalyzer

import torch
import torch.nn.utils.rnn as rnn_utils
import numpy as np
import re

class RasaIntentEntityDataset(torch.utils.data.Dataset):
    """
    RASA NLU markdown file lines based Custom Dataset Class

    Dataset Example in nlu.md

    ## intent:intent_데이터_자동_선물하기_멀티턴                <- intent name
    - T끼리 데이터 주기적으로 보내기                            <- utterance without entity
    - 인터넷 데이터 [달마다](Every_Month)마다 보내줄 수 있어?    <- utterance with entity
    
    """

    def __init__(self, markdown_lines: List[str], o_token_id=0, pad_token_id=1):
        self.o_token_id =  o_token_id
        self.pad_token_id = pad_token_id

        self.intent_dict = {}
        self.entity_dict = {}
        self.entity_dict["O"] = self.o_token_id 

        self.vocab_dict = {}
        self.vocab_dict["[UNK]"] = self.o_token_id 
        self.vocab_dict["[PAD]"] = self.pad_token_id 

        self.dataset = []
        self.max_seq_len = 0

        intent_value_list = []
        entity_type_list = []

        # dataset tokenizer setting
        self.tokenizer = KoreanAnalyzer(
            decompound_mode="DISCARD",  # DISCARD or MIXED or NONE
            infl_decompound_mode="DISCARD",  # DISCARD or MIXED or NONE
            discard_punctuation=True,
            output_unknown_unigrams=False,
            pos_filter=False,
            stop_tags=["JKS", "JKB", "VV", "EF"],
            synonym_filter=False,
            mode_synonym="NORM",
        )

        current_intent_focus = ""

        for line in tqdm(
            markdown_lines,
            desc="Organizing Intent & Entity dictionary in NLU markdown file ...",
        ):
            if len(line.strip()) < 2:
                current_intent_focus = ""
                continue

            if "## " in line:
                if "intent:" in line:
                    intent_value_list.append(line.split(":")[1].strip())
                    current_intent_focus = line.split(":")[1].strip()
                else:
                    current_intent_focus = ""

            else:
                if current_intent_focus != "":
                    text = line[2:].strip().lower()

                    for type_str in re.finditer(r"\([a-zA-Z_1-2]+\)", text):
                        entity_type = (
                            text[type_str.start() + 1 : type_str.end() - 1]
                            .replace("(", "")
                            .replace(")", "")
                        )
                        entity_type_list.append(entity_type)

                    text = re.sub(r"\([a-zA-Z_1-2]+\)", "", str(text))  # remove (...) str
                    text = text.replace("[", "").replace(
                        "]", ""
                    )  # remove '[',']' special char

                    if len(text) > 0:
                        vocab_list = self.tokenizer.do_analysis(text)['termAtt']
                        self.max_seq_len = max(self.max_seq_len, len(vocab_list))

                        for vocab in vocab_list:
                            if vocab not in self.vocab_dict.keys():
                                self.vocab_dict[vocab] = len(self.vocab_dict)

        intent_value_list = sorted(intent_value_list)
        for intent_value in intent_value_list:
            if intent_value not in self.intent_dict.keys():
                self.intent_dict[intent_value] = len(self.intent_dict)

        entity_type_list = sorted(entity_type_list)
        for entity_type in entity_type_list:
            if entity_type + "_B" not in self.entity_dict.keys():
                self.entity_dict[str(entity_type) + "_B"] = len(self.entity_dict)
            if entity_type + "_I" not in self.entity_dict.keys():
                self.entity_dict[str(entity_type) + "_I"] = len(self.entity_dict)

        current_intent_focus = ""

        for line in tqdm(
            markdown_lines, desc="Extracting Intent & Entity in NLU markdown files...",
        ):
            if len(line.strip()) < 2:
                current_intent_focus = ""
                continue

            if "## " in line:
                if "intent:" in line:
                    current_intent_focus = line.split(":")[1].strip()
                else:
                    current_intent_focus = ""
            else:
                if current_intent_focus != "":  # intent & entity sentence occur case
                    text = line[2:].strip().lower()

                    entity_value_list = []
                    for value in re.finditer(r"\[(.*?)\]", text):
                        entity_value_list.append(
                            text[value.start() + 1 : value.end() - 1]
                            .replace("[", "")
                            .replace("]", "")
                        )

                    entity_type_list = []
                    for type_str in re.finditer(r"\([a-zA-Z_1-2]+\)", text):
                        entity_type = (
                            text[type_str.start() + 1 : type_str.end() - 1]
                            .replace("(", "")
                            .replace(")", "")
                        )
                        entity_type_list.append(entity_type)

                    text = re.sub(r"\([a-zA-Z_1-2]+\)", "", text)  # remove (...) str
                    text = text.replace("[", "").replace(
                        "]", ""
                    )  # remove '[',']' special char

                    if len(text) > 0:
                        each_data_dict = {}
                        each_data_dict["text"] = text.strip()
                        each_data_dict["intent"] = current_intent_focus
                        each_data_dict["intent_idx"] = self.intent_dict[
                            current_intent_focus
                        ]
                        each_data_dict["entities"] = []

                        for value, type_str in zip(entity_value_list, entity_type_list):
                            for entity in re.finditer(value, text):
                                entity_tokens = self.tokenize(value)

                                for i, entity_token in enumerate(entity_tokens):
                                    if i == 0:
                                        BIO_type_str = type_str + "_B"
                                    else:
                                        BIO_type_str = type_str + "_I"

                                    each_data_dict["entities"].append(
                                        {
                                            "start": text[entity.start():].find(entity_token),
                                            "end": text[entity.start():].find(entity_token) + len(entity_token),
                                            "entity": type_str,
                                            "value": entity_token,
                                            "entity_idx": self.entity_dict[
                                                BIO_type_str
                                            ],
                                        }
                                    )

                        self.dataset.append(each_data_dict)

        print(f"Intents: {self.intent_dict}")
        print(f"Entities: {self.entity_dict}")
        print(f"vocab size: {len(self.vocab_dict)}")
        print(f"max_seq_len: {self.max_seq_len}")

    def tokenize(self, text: str):
        vocab_list = self.tokenizer.do_analysis(text)['termAtt']
        #return [ self.vocab_dict.get(vocab, self.pad_token_id) for vocab in vocab_list]
        return vocab_list

    def encode(self, text: str, return_tensor: bool = True):
        vocab_list = self.tokenize(text)
        tokens = [ self.vocab_dict.get(vocab, self.o_token_id) for vocab in vocab_list ]

        if type(tokens) == list:
            tokens = torch.tensor(tokens).long()
        else:
            tokens = tokens.long()

        if return_tensor:
            return tokens
        else:
            return tokens.numpy()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        tokens = self.tokenize(self.dataset[idx]["text"])
        token_ids = [ self.vocab_dict.get(vocab, self.pad_token_id) for vocab in tokens ]

        intent_idx = self.dataset[idx]["intent_idx"]

        entity_idx = np.array(len(tokens) * [self.o_token_id])  # O tag indicate 0(zero)

        for entity_seq, entity_info in enumerate(self.dataset[idx]["entities"]):
            ##check whether entity value is include in splitted token
            for token_seq, token_value in enumerate(tokens):
                if token_value in entity_info["value"]:
                    entity_idx[token_seq] = entity_info["entity_idx"]
                    break

        entity_idx = torch.from_numpy(entity_idx)

        return torch.tensor(token_ids), intent_idx, entity_idx

    def get_intent_idx(self):
        return self.intent_dict

    def get_entity_idx(self):
        return self.entity_dict

    def get_vocab_size(self):
        return len(self.vocab_dict)

def token_concat_collate_fn(batch, pad_token_id=1):
    """
    batch : tokens have various length, inten_idx, entity_indices have various length 
    """
    tokens = rnn_utils.pad_sequence([each_data[0] for each_data in batch], batch_first=True, padding_value=pad_token_id).long()
    intent_indices = torch.tensor([each_data[1] for each_data in batch]).long()
    entity_indices = rnn_utils.pad_sequence([each_data[2] for each_data in batch], batch_first=True, padding_value=pad_token_id).long()

    return (tokens, intent_indices, entity_indices)
