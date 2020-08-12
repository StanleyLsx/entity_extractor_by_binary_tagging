from transformers import BertTokenizer
from torch.utils.data import Dataset
import torch
import numpy as np
import re

BATCH_SIZE = 128
EPOCH_NUM = 50


class DataGenerator:
    def __init__(self, data, logger, batch_size=BATCH_SIZE):
        self.data = data
        self.batch_size = batch_size
        logger.info('train_data_length:{},batch_size:{},steps in each epoch:{}'
                    .format(len(data), BATCH_SIZE, len(data)//BATCH_SIZE))
        assert len(data) >= batch_size, '数据量不够一个批次'
        self.categories = {'company': 0, 'position': 1, 'detail': 2}
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    @staticmethod
    def get_index(text_token, token):
        text_token_str = '#'.join([str(index) for index in text_token])
        token_str = '#'.join([str(index) for index in token])
        start_str_index = re.search(token_str, text_token_str).start()
        start_index = text_token_str[:start_str_index].count('#')
        end_index = start_index + len(token)
        return start_index, end_index - 1

    def prepare_data(self):
        sentence_vectors = []
        segment_vectors = []
        attention_mask_vectors = []
        entity_vectors = []
        for item in self.data:
            text = item.get('text')
            token_results = self.tokenizer(text, padding='max_length')
            token_ids = token_results.get('input_ids')
            segment_ids = token_results.get('token_type_ids')
            attention_mask = token_results.get('attention_mask')
            entity_vector = np.zeros((len(token_ids), len(self.categories), 2))
            for classifier, classifier_id in self.categories.items():
                item_text = item.get(classifier)
                if item_text is not None:
                    item_token = self.tokenizer(item_text).get('input_ids')
                    item_token = item_token[1:-1]
                    start_index, end_index = self.get_index(token_ids, item_token)
                    entity_vector[start_index, classifier_id, 0] = 1
                    entity_vector[end_index, classifier_id, 1] = 1
            sentence_vectors.append(token_ids)
            segment_vectors.append(segment_ids)
            attention_mask_vectors.append(attention_mask)
            entity_vectors.append(entity_vector)
        sentence_vectors_np = np.array(sentence_vectors)
        segment_vectors_np = np.array(segment_vectors)
        attention_mask_vectors_np = np.array(attention_mask_vectors)
        entity_vectors_np = np.array(entity_vectors)
        return sentence_vectors_np, segment_vectors_np, attention_mask_vectors_np, entity_vectors_np


class MyDataset(Dataset):
    """
    下载数据、初始化数据，都可以在这里完成
    """

    def __init__(self, _sentence_vectors, _segment_vectors, _attention_mask_vectors, _entity_vectors):
        self.sentence_vectors = _sentence_vectors
        self.segment_vectors = _segment_vectors
        self.attention_mask_vectors = _attention_mask_vectors
        self.entity_vectors = _entity_vectors
        self.len = len(self.sentence_vectors)

    def __getitem__(self, index):
        return self.sentence_vectors[index], self.segment_vectors[index], self.attention_mask_vectors[index],\
               self.entity_vectors[index]

    def __len__(self):
        return self.len


def collate_fn(data):
    """
    如何取样本的，定义自己的函数来准确地实现想要的功能
    """
    sentence = np.array([item[0] for item in data], np.int32)
    segment = np.array([item[1] for item in data], np.int32)
    attention_mask = np.array([item[2] for item in data], np.int32)
    entity_vec = np.array([item[3] for item in data], np.int32)
    return {'sentence': torch.LongTensor(sentence),
            'segment': torch.LongTensor(segment),
            'attention_mask': torch.LongTensor(attention_mask),
            'entity_vec': torch.LongTensor(entity_vec)}
