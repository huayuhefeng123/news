import numpy as np
import torch
import time
import os

from transformers import (GPT2LMHeadModel, GPT2Tokenizer,
                          BertTokenizer, BertForMaskedLM,
                          AutoTokenizer, AutoModelForCausalLM,
                          GPT2TokenizerFast)


class AbstractLanguageChecker:
    """
    Abstract Class that defines the Backend API of GLTR.

    To extend the GLTR interface, you need to inherit this and
    fill in the defined functions.
    """

    def __init__(self):
        """
        In the subclass, you need to load all necessary components
        for the other functions.
        Typically, this will comprise a tokenizer and a model.
        """
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def check_probabilities(self, in_text, topk=40):
        """
        Function that GLTR interacts with to check the probabilities of words

        Params:
        - in_text: str -- The text that you want to check
        - topk: int -- Your desired truncation of the head of the distribution

        Output:
        - payload: dict -- The wrapper for results in this function, described below

        Payload values
        ==============
        bpe_strings: list of str -- Each individual token in the text
        real_topk: list of tuples -- (ranking, prob) of each token
        pred_topk: list of list of tuple -- (word, prob) for all topk
        """
        raise NotImplementedError

    def postprocess(self, token):
        """
        clean up the tokens from any special chars and encode
        leading space by UTF-8 code '\u0120', linebreak with UTF-8 code 266 '\u010A'
        :param token:  str -- raw token text
        :return: str -- cleaned and re-encoded token text
        """
        raise NotImplementedError


def top_k_logits(logits, k):
    """
    Filters logits to only the top k choices
    from https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_gpt2.py
    """
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1]
    return torch.where(logits < min_values,
                       torch.ones_like(logits, dtype=logits.dtype) * -1e10,
                       logits)


class ChineseLM(AbstractLanguageChecker):
    def __init__(self):
        super(ChineseLM, self).__init__()
        try:
            model_name = "uer/gpt2-chinese-cluecorpussmall"
            print(f"正在加载模型 {model_name}...")
            
            self.enc = BertTokenizer.from_pretrained(model_name)
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            self.start_token = torch.tensor([self.enc.cls_token_id], device=self.device)
            print(f"模型加载完成! 使用设备: {self.device}")
            
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            raise e

    def check_probabilities(self, in_text, topk=40):
        try:
            # 处理空白文本
            if not in_text or not in_text.strip():
                return {
                    'bpe_strings': [],
                    'real_topk': [],
                    'pred_topk': []
                }
            
            # 编码文本
            encoding = self.enc(in_text, 
                              return_tensors='pt',
                              add_special_tokens=True)
            
            token_ids = encoding['input_ids'][0].to(self.device)
            
            # 添加起始标记
            token_ids = torch.cat([self.start_token, token_ids])
            
            # 前向传播
            input_ids = token_ids.unsqueeze(0)
            
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits
                
            # 处理预测结果
            all_logits = logits[0, :-1, :]
            all_probs = torch.softmax(all_logits, dim=-1)
            y = token_ids[1:]
            
            real_topk_pos = []
            real_topk_probs = []
            
            # 计算每个位置的排名和概率
            for i in range(y.shape[0]):
                target_id = y[i].item()
                probs = all_probs[i]
                sorted_indices = torch.argsort(probs, descending=True)
                
                rank = (sorted_indices == target_id).nonzero().item()
                prob = probs[target_id].item()
                
                real_topk_pos.append(rank)
                real_topk_probs.append(round(prob, 5))

            real_topk = list(zip(real_topk_pos, real_topk_probs))
            
            # 获取分词结果
            bpe_strings = self.enc.convert_ids_to_tokens(token_ids)

            # 获取topk预测
            pred_topk = []
            for i in range(y.shape[0]):
                probs = all_probs[i]
                values, indices = torch.topk(probs, k=min(topk, probs.shape[0]))
                
                current_pred = list(zip(
                    self.enc.convert_ids_to_tokens(indices.cpu()),
                    values.cpu().numpy().tolist()
                ))
                pred_topk.append(current_pred)

            payload = {
                'bpe_strings': bpe_strings,
                'real_topk': real_topk,
                'pred_topk': pred_topk
            }
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return payload
            
        except Exception as e:
            print(f"处理文本时出错: {str(e)}")
            return {
                'bpe_strings': [],
                'real_topk': [],
                'pred_topk': []
            }

    def postprocess(self, token):
        return token


def _get_topk(text):
    print("测试中文模型...")
    try:
        # # 创建模型缓存目录
        # if not os.path.exists('./models'):
        #     os.makedirs('./models')
            
        lm = ChineseLM()  # 使用默认模型
        
        payload = lm.check_probabilities(text, topk=5)
        return payload
            
    except Exception as e:
        print(f"运行时错误: {str(e)}")

if __name__ == "__main__":
    _get_topk("人工智能是一个快速发展的领域，它正在改变我们的生活方式。")
