import torch
from typing import List, Union

from biglm import BIGLM
from data import Vocab, DataLoader, s2t


class SequenceCreation:
    def __init__(
        self,
        model_dir: str = "../checkpoint/GPT2_12L_10G/12L_10G.ckpt",
        vocab_dir: str = "../checkpoint/GPT2_12L_10G/12L_10G.vocab.txt",
        max_length: int = 50,
    ) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_length = max_length

        # init models
        self.lm_model, self.lm_vocab, self.lm_args = self._init_model(model_dir, vocab_dir, self.device)

    def get_sequence(self, words: Union[List[str], str]) -> Union[List[str], str]:
        if isinstance(words, List):
            seq = []
            for word in words:
                s = [[w for w in word]]
                result = self._greedy(s)
                seq.append(result)
            return seq
        else:
            s = [[w for w in words]]
            return self._greedy(s)

    def _init_model(self, model_dir: str, vocab_dir: str, device: str) -> tuple:
        ckpt = torch.load(model_dir, map_location="cpu")
        lm_args = ckpt["args"]
        lm_vocab = Vocab(vocab_dir, min_occur_cnt=lm_args.min_occur_cnt, specials=[])
        lm_model = BIGLM(device, lm_vocab, lm_args.embed_dim, lm_args.ff_embed_dim, lm_args.num_heads, lm_args.dropout, lm_args.layers, 0.1, lm_args.approx)
        lm_model.load_state_dict(ckpt["model"])
        lm_model = lm_model.to(device)
        lm_model.eval()
        return lm_model, lm_vocab, lm_args

    def _greedy(self, s: List[List[str]]) -> str:
        x, m = s2t(s, self.lm_vocab)
        x = x.to(self.device)
        res = []
        for l in range(self.max_length):
            probs, pred = self.lm_model.work(x)
            next_tk = []
            for i in range(len(s)):
                next_tk.append(self.lm_vocab.idx2token(pred[len(s[i]) - 1, i].item()))

            s_ = []
            for idx, (sent, t) in enumerate(zip(s, next_tk)):
                if t == "<eos>":
                    res.append(sent)
                else:
                    s_.append(sent + [t])
            if not s_:
                break
            s = s_
            x, m = s2t(s, self.lm_vocab)
            x = x.to(self.device)
        res += s_

        r = "".join(res[0])
        if "<bos>" in r:
            return r.split("<bos>")[1]
        else:
            return r
