#!/usr/bin/env python
# coding: utf-8

import json
import os.path

from itertools import permutations
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
from torch import nn
from tqdm import tqdm
import copy
import pickle
from torch.nn.functional import pad
from collections import defaultdict
from pathlib import Path
torch.cuda.empty_cache()

BATCH_SIZE = 512  # BS number, don't remember what it was, isn't used in the experiments. Remove after refactor.


# Borrowed the starter code from https://github.com/writerai/fitbert
# Heavily modified, assume there are some strange changes ahead
class FitBert:
    def __init__(
            self,
            model_name="bert-large-uncased",
            disable_gpu=False,
    ):
        # self.mask_token = mask_token
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not disable_gpu else "cpu"
        )
        # self._score = pll_score_batched
        print("device:", self.device)

        self.bert = AutoModelForMaskedLM.from_pretrained(model_name)
        self.bert.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space="roberta" in model_name)
        self.mask_token = self.tokenizer.mask_token
        self.pad_token = self.tokenizer.pad_token
        self.cls_token = self.tokenizer.cls_token
        self.sep_token = self.tokenizer.sep_token
        with torch.no_grad():
            self.mask_token_vector = self.bert.get_input_embeddings()(torch.LongTensor([self.tokenizer.mask_token_id]).to(self.device))[0]

    @staticmethod
    def top_k(x, k=10):
        tk = torch.topk(x, k, sorted=False)
        return torch.zeros_like(x).scatter_(-1, tk.indices, FitBert.softmax(tk.values))

    def get_vocab_output_dim(self):
        return self.bert.get_output_embeddings().out_features

    def __call__(self, data, is_split_into_words=False, use_softmax=True, *args, **kwds):
        if is_split_into_words:
            _tokens = self.tokenizer.convert_tokens_to_ids(data)
            _tokens = [self.tokenizer.cls_token_id] + _tokens + [self.tokenizer.sep_token_id]
            tokens = {'input_ids':torch.LongTensor([_tokens]).to(self.device)}
        else:
            tokens = self.tokenizer(data, add_special_tokens=True,padding=True, return_tensors='pt').to(self.device)
        
        # inp = torch.tensor(data, device=self.device)
        # if len(tokens.shape) == 1:
        #     inp = inp.unsqueeze(0)
        # print(tokens)
        # print(tokens.input_ids)
        # print(self.tokenizer.convert_ids_to_tokens(tokens.input_ids[0].tolist()))
        # print("=="*50)
        b = self.bert(**tokens)
        # print(b.logits.shape)
        if use_softmax:
            return self.softmax(b[0])[:, 1:-1, :]
        else:
            return b[0][:, 1:-1, :]
        # return self.softmax(self.bert_am(**tokens, **kwds)[0])[:, 1:, :]
#         return self.bert_am(torch.tensor(self._tokens(data, **kwds)), *args, **kwds)

    def bert_am(self, data, *args, **kwds):
        return self.bert(data, *args, attention_mask=(data!=self.tokenizer.pad_token_id), **kwds)

    def tokenize(self, *args, **kwds):
        return self.tokenizer.tokenize(*args, **kwds)

    def mask_tokenize(self, sent, keep_original=False, add_special_tokens=False, padding=False, return_full=False):
        tokens = self.tokenize(sent, add_special_tokens=add_special_tokens, padding=padding)
        # print(tokens)
        tlen = len(tokens)
        offset = 1 if add_special_tokens else 0
        token_mat = [tokens[:] for i in range(tlen - (2*offset))]
        for i in range(offset, tlen-offset):
            token_mat[i-offset][i] = self.tokenizer.mask_token
        if keep_original:
            token_mat = [tokens[:]] + token_mat

        if return_full:
            return token_mat, self.tokenizer(token_mat, add_special_tokens=(not add_special_tokens), is_split_into_words=True, return_tensors='pt')
        return token_mat

    def _tokens_to_masked_ids(self, tokens, mask_ind, pad=0):
        masked_tokens = tokens[:]
        masked_tokens[mask_ind] = self.mask_token
        masked_ids = self._tokens(masked_tokens, pad=pad)
        return masked_ids

    @staticmethod
    def softmax(x):
        # Break into two functions to minimize the memory impact of calling .exp() on very large tensors.
        return FitBert._inn_soft(x.exp())

    @staticmethod
    def _inn_soft(xexp):
        return xexp / (xexp.sum(-1)).unsqueeze(-1)

    @staticmethod
    def softmax_(x):
        # Break into two functions to minimize the memory impact of calling .exp() on very large tensors.
        # Further reduce memory impact by making it an in-place operation. Beware.
        return FitBert._inn_soft(x.exp_())

    @staticmethod
    def masked_softmax(x):
        return FitBert._inn_soft(x.exp() * (x > 0.0).float())

    def augment(self, vecs, nonlinearity, pooling):
        if nonlinearity in self.nonlins:
            nl = self.nonlins[nonlinearity]
        elif callable(nonlinearity):
            nl = nonlinearity
        else:
            nl = self.nonlins[None]
        if pooling:
            if pooling in ["mean", "avg"]:
                return nl(torch.mean(vecs, dim=0, keepdim=True))
            elif pooling == "max":
                # print(e.shape)
                # print(nl(e).shape)
                # print(torch.max(ent_vecs[nl(e)], dim=0, keepdim=True))
                # print(torch.max(ent_vecs[e], dim=0, keepdim=True))
                return nl(torch.max(vecs, dim=0, keepdim=True)[0])
            elif pooling == "sum":
                return nl(torch.sum(vecs, dim=0, keepdim=True))
            elif callable(pooling):
                return nl(pooling(vecs))
        else:
            return nl(vecs)

    def fuzzy_embed(self, vec):
        return vec.to(self.device)@self.bert.get_input_embeddings().weight

    nonlins = {None: lambda x:x,
               "softmax": softmax,
               "relu": torch.relu,
               "relmax": masked_softmax,
               "top10":top_k,
               "top20": lambda x: FitBert.top_k(x, 20),
               "top50": lambda x: FitBert.top_k(x, 50),
               "top100": lambda x: FitBert.top_k(x, 100)
              }

def read_punctuation(fb):
    punctuation = fb.tokenizer([".,-()[]{}_=+?!@#$%^&*\\/\"'`~;:|…）（•−"], add_special_tokens=False)['input_ids'][0]
    one_hot_punctuation = torch.ones(fb.bert.get_output_embeddings().out_features, dtype=torch.long)
    one_hot_punctuation[punctuation] = 0
    one_hot_punctuation[1232] = 0
    return one_hot_punctuation


class ScoringMethod(nn.Module):
    def __init__(self, label):
        super(ScoringMethod, self).__init__()
        self.label = label


class PllScoringMethod(ScoringMethod):
    def __init__(self, label):
        super(PllScoringMethod, self).__init__(label)

    def forward(self, probs, origids, return_all=False, **kwargs):
        mask = origids >= 0
        origids[~mask] = 0
        slen = len(probs) - 1
        dia = torch.diag(probs[1:].gather(-1, origids.unsqueeze(0).repeat(slen, 1).unsqueeze(-1)).squeeze(-1), diagonal=0)[mask]
        dia_list = dia.tolist()
        prob = torch.mean(torch.log_(dia), dim=-1).detach().item()
        if return_all:
            return prob, dia_list
        return prob


class ComparativeScoringMethod(ScoringMethod):
    def __init__(self, label):
        super(ComparativeScoringMethod, self).__init__(label)

    def forward(self, probs, return_all=False, **kwargs):
        slen = len(probs) - 1
        dia = self.calc(probs[0, :slen], probs[torch.arange(1, slen + 1), torch.arange(slen)])
        dia_list = dia.tolist()
        prob = torch.mean(torch.log_(dia), dim=-1).detach().item()
        if return_all:
            return prob, dia_list
        return prob

    def calc(self, p: torch.tensor, q: torch.tensor):
        raise NotImplementedError


class JSD(ComparativeScoringMethod):
    def __init__(self):
        super(JSD, self).__init__("jsd")
        self.kl = nn.KLDivLoss(reduction='none', log_target=True)

    def calc(self, p: torch.tensor, q: torch.tensor):
        m = torch.log_((0.5 * (p + q)))
        return 1 - (0.5 * (torch.sum(self.kl(m, p.log()), dim=-1) + torch.sum(self.kl(m, q.log()), dim=-1)))


class PLL(PllScoringMethod):
    def __init__(self):
        super(PLL, self).__init__("pll")


class CSD(ComparativeScoringMethod):
    def __init__(self):
        super(CSD, self).__init__("csd")
        self.csd = torch.nn.CosineSimilarity(dim=1)

    def calc(self, p: torch.tensor, q: torch.tensor):
        return self.csd(p, q)


class ESD(ComparativeScoringMethod):
    def __init__(self):
        super(ESD, self).__init__("esd")
        self.pwd = torch.nn.PairwiseDistance()
        self.sqrt = torch.sqrt(torch.tensor(2, requires_grad=False))

    def norm(self, dist):
        return (torch.relu(self.sqrt - dist) + 0.000001) / self.sqrt

    def calc(self, p: torch.tensor, q: torch.tensor):
        return self.norm(self.pwd(p, q))


class MSD(ComparativeScoringMethod):
    def __init__(self):
        super(MSD, self).__init__("msd")
        self.mse = torch.nn.MSELoss(reduction="none")

    def calc(self, p: torch.tensor, q: torch.tensor):
        return self.mse(p, q).mean(axis=-1)


class HSD(ComparativeScoringMethod):
    def __init__(self):
        super(HSD, self).__init__("hsd")
        self.sqrt = torch.sqrt(torch.tensor(2, requires_grad=False))

    def calc(self, p: torch.tensor, q: torch.tensor):
        return 1 - torch.sqrt_(torch.sum(torch.pow(torch.sqrt_(p) - torch.sqrt_(q), 2), dim=-1)) / self.sqrt


KNOWN_METHODS = [CSD(), ESD(), JSD(), MSD(), HSD(), PLL()]
KNOWN_METHODS = {m.label: m for m in KNOWN_METHODS}


def prompt(rel, xy=True, ensure_period=True):
    if xy:
        prompt = rel_info[rel]['prompt_xy']
    else:
        prompt = rel_info[rel]['prompt_yx']
    if ensure_period and prompt[-1] != '.':
        return prompt + "."
    else:
        return prompt


def candidates(prompt:str, choices, return_ments=False):
    for a, b in permutations(choices, 2):
        if return_ments:
            yield prompt.replace("?x", a, 1).replace("?y", b, 1), (a, b)
        else:
            yield prompt.replace("?x", a, 1).replace("?y", b, 1)


# scores equivalently to the old method, even with padding.
# Can be used to batch across examples.
def pll_score_batched(self, sents: list, return_all=False):
    self.bert.eval()
    key_to_sent = {}
    with torch.no_grad():
        data = {}
        for sent in sents:
            tkns = self.tokenizer.tokenize(sent)
            data[len(data)] = {
                'tokens': tkns,
                'len': len(tkns)
            }
        scores = {"pll": {}}
        all_plls = {"pll": {}}

        sents_sorted = list(sorted(data.keys(), key=lambda k: data[k]['len']))

        inds = []
        lens = []

        methods = [PLL()]

        for sent in sents_sorted:
            n_tokens = data[sent]['len']
            if sum(lens) <= BATCH_SIZE:
                inds.append(sent)
                lens.append(n_tokens)
            else:
                # There is at least one sentence.
                # If the count is zero, then its size is larger than the batch size.
                # Send it anyway.
                flag = (len(inds) == 0)
                if flag:
                    inds.append(sent)
                    lens.append(n_tokens)
                _inner_score_stuff(self, data, inds, lens, methods, scores, all_plls, return_all)
                inds = [sent]
                lens = [n_tokens]
            if sent == sents_sorted[-1]:
                _inner_score_stuff(self, data, inds, lens, methods, scores, all_plls, return_all)

        for d in data:
            data[d].clear()
        data.clear()
        # del all_probs
        if self.device == "cuda":
            torch.cuda.empty_cache()
        for method in scores:
            assert len(scores[method]) == len(sents_sorted)
        if return_all:
            return unsort_flatten(scores)["pll"], unsort_flatten(all_plls)["pll"]
        return unsort_flatten(scores)["pll"]


def unsort_flatten(mapping):
    # print(mapping.keys())
    return {f: list(mapping[f][k] for k in range(len(mapping[f]))) for f in mapping}


def cos_score_batched(self, sents: list, return_all=True):
    return score_batched(self, methods=[CSD()], sents=sents, return_all=return_all)


def euc_score_batched(self, sents: list, return_all=True):
    return score_batched(self, methods=[ESD()], sents=sents, return_all=return_all)


def jsd_score_batched(self, sents: list, return_all=True):
    return score_batched(self, methods=[JSD()], sents=sents, return_all=return_all)


def msd_score_batched(self, sents: list, return_all=True):
    return score_batched(self, methods=[MSD()], sents=sents, return_all=return_all)


def hel_score_batched(self, sents: list, return_all=True):
    return score_batched(self, methods=[HSD()], sents=sents, return_all=return_all)


def all_score_batched(self, sents: list,  return_all=True):
    return score_batched(self, methods=list(KNOWN_METHODS.values()), sents=sents, return_all=return_all)


# For this one:
# Take a sentence, tokenize it, return all needed information like number of tokens.
def _inner_tokenize_sentence(self, sent, keep_original):
    _, tkns = self.mask_tokenize(sent, keep_original=keep_original, add_special_tokens=True, return_full=True)
    # print(tkns)
    # print(f"TKNS:{len(tkns.input_ids[0]) - 2}")
    return tkns, len(tkns.input_ids[0]) - 2


def score_batched(self, methods, sents: list, return_all=True):
    # Enforce evaluation mode
    self.bert.eval()
    with torch.no_grad():
        data = {}
        for sent in sents:
            # Tokenize every sentence
            # print("S2:", sent)
            tkns, n_tkns = _inner_tokenize_sentence(self, sent, keep_original=True)
            data[len(data)] = {
                'tokens': tkns,
                'len': n_tkns
            }
        # print("Boo")

        scores = {m.label: {} for m in methods}
        all_plls = {m.label: {} for m in methods}

        sents_sorted = list(sorted(data.keys(), key=lambda k: data[k]['len']))

        inds = []
        lens = []

        for sent in sents_sorted:
            n_tokens = data[sent]['len']
            if sum(lens) <= BATCH_SIZE:
                inds.append(sent)
                lens.append(n_tokens)
            else:
                # There is at least one sentence.
                # If the count is zero, then its size is larger than the batch size.
                # Send it anyway.
                flag = (len(inds) == 0)
                if flag:
                    inds.append(sent)
                    lens.append(n_tokens)
                _inner_score_stuff(self, data, inds, lens, methods, scores, all_plls, return_all)
                inds = [sent]
                lens = [n_tokens]
            if sent == sents_sorted[-1]:
                _inner_score_stuff(self, data, inds, lens, methods, scores, all_plls, return_all)

        for d in data:
            data[d].clear()
        data.clear()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        for method in scores:
            assert len(scores[method]) == len(sents_sorted)
        if return_all:
            return unsort_flatten(scores), unsort_flatten(all_plls)
        return unsort_flatten(scores)


def bert_am(self, data, *args, **kwds):
    return self.bert(data, *args, attention_mask=(data!=self.tokenizer.pad_token_id), **kwds)

def _inner_score_stuff(self, data, inds, lens, methods, scores, all_plls, return_all):
    longest = max(lens)

    bert_forward = torch.concat([pad(data[d]['tokens'].input_ids, (0, longest - l), 'constant', self.tokenizer.pad_token_id ) for d,l in zip(inds, lens)], dim=0).to(self.device)
    token_type_ids = torch.concat([pad(data[d]['tokens'].token_type_ids, (0, longest - l), 'constant', 0) for d, l in zip(inds, lens)], dim=0).to(self.device)
    _probs = self.softmax(bert_am(self, bert_forward, token_type_ids=token_type_ids)[0])[:, 1:, :]
    
    del bert_forward

    use_pll = any(["pll" in method.label for method in methods])
    print(["pll" in method.label for method in methods])
    print(use_pll)

    for ind, slen in zip(inds, lens):
        origids = data[ind]['tokens'].input_ids[0][1:-1].to(self.device) if use_pll else None
        for method in methods:
            prob, alls = method(_probs[:slen + 1], origids=origids, return_all=True)
            if return_all:
                assert ind not in all_plls[method.label]
                all_plls[method.label][ind] = alls
            assert ind not in scores[method.label]
            scores[method.label][ind] = prob
            del alls, prob
        _probs = _probs[slen + 1:]
    del _probs


def extend_bert(fb):
    add_tokens = ['?x', '?y']
    add_tokens.append('[ENT_BEG]')
    add_tokens.append('[ENT_END]')
    fb.tokenizer.add_tokens(add_tokens, special_tokens=True)  # Add the tokens to the tokenizer.
    fb.bert.resize_token_embeddings(len(fb.tokenizer))  # Add the tokens to the embedding matrix, initialize with defaults. DO NOT TRAIN.
    fb.entity_tokens = fb.tokenizer("".join(add_tokens), add_special_tokens=False)['input_ids']
    return fb


class Document:
    def __init__(self, doc, num, mlm=None, use_blanks=False):
        self.doc = doc
        self.num = num
        self.mlm = mlm
        self.overlaps = {}
        self.mentions, self.mention_types, self.m_to_e  = self.read_mentions()
        self.entities = self.read_entities(self.doc['vertexSet'])
        self.relations = set([a[1] for a in self.answers(detailed=False)])
        self.width = 0
        self.use_blanks = use_blanks
        self._masked_doc = None

    @property
    def masked_doc(self):
        if not self._masked_doc and self.mlm:
            self._masked_doc = self.tokenize_entities()
        return self._masked_doc

    def contextualize_doc(self):
        self.unmasked_doc = self.contextualize(mask=False)
        return self.unmasked_doc

    def __getitem__(self, item):
        return self.doc[item]
    
    def __contains__(self, item):
        return item in self.doc

    def sentences(self):
        for sent in self['sents']:
            yield " ".join(sent)

    def text(self):
        return " ".join(self.sentences())

    def read_mentions(self):

        # Accumulate all mentions with their position (s, b, e) = (w, t)
        # avoid duplicates
        # Sort by key, ascending
        # return ordered list of mentions (w), mapping from index to type (t).

        mentions = dict()
        m_to_e = dict()
        for i, v in enumerate(self['vertexSet']):
            for m in v:
                s = m['sent_id']
                b, e = m['pos']
                w = self['sents'][s][b:e]
                t = m['type']
                if (s, b, e) not in mentions:
                    mentions[(s, b, e)] = (w, t, i)

        ments = list()
        types = dict()
        for i, (_, v) in enumerate(sorted(mentions.items())):
            w, t, e = v
            ments.append(w)
            m_to_e[i] = e
            types[i] = t
        return ments, types, m_to_e

    @staticmethod
    def read_entities(vertSet):
        ents = {}
        for i, ent in enumerate(vertSet):
            ents[i] = list(set(e['name'] for e in ent))
        return ents

    def tokenize_entities(self):
        # Step 1: Copy
        sents = copy.deepcopy(self['sents'])
        e_beg = '[ENT_BEG]'
        e_end = '[ENT_END]'

        seen_positions = []
        positions = []
        for i, v in enumerate(self['vertexSet']):
            for m in v:
                s = m['sent_id']
                b, e = m['pos']
                if (s, b, e) not in seen_positions:
                    seen_positions.append((s, b, e))
                    positions.append((s, b, e, i))
                else:
                    print(f"Duplicate at {(s, b, e)}")
        positions = list(sorted(positions, reverse=True))

        for s, b, e, _ in positions:
            sents[s][b:e] = [e_beg] + sents[s][b:e] + [e_end]
        sents = sum(sents, [])
        # print(sents, flush=True)
        tkns = self.mlm.tokenizer.tokenize(sents, add_special_tokens=False, is_split_into_words=True)
        e = 0
        mentions = []  # Handled.
        mention_mask = []  # Handled.
        mapp = {}
        e = -1
        m = len(positions)
        m_types = [None]*m
        ment_lens = [0]*m
        # m -= 1
        for i, w in enumerate(reversed(tkns)):
            if w == e_end:
                en = i
                e = positions.pop(0)[-1]
                if e not in mapp:
                    mapp[e] = []
                m -= 1
                mapp[e].append(m)
            elif w == e_beg:
                en = 0
                e = -1
            else:
                if e > -1:
                    ment_lens[m] += 1
                mention_mask.append(e > -1)
                mentions.append(m if mention_mask[-1] else -1)
        mentions = list(reversed(mentions))
        mention_mask = list(reversed(mention_mask))
        e_count = len(self['vertexSet'])
        tokens = [t for t in tkns if t.upper() not in [e_beg, e_end] ]
        
        return {
            "length": len(tokens),
            "tokens": tokens,
            "ments": mentions,
            "ment_mask":mention_mask,
            "ment_types":m_types,
            "ment_lens":ment_lens,
            "ents": mapp,
            "m_count": len(m_types),
            "e_count": e_count,
        }

    def answers(self, detailed=True):
        ans = []
        for an in self['labels']:
            
            if detailed:
                ents = self.entities
                hs = ents[an['h']]
                ts = ents[an['t']]
                r = an['r']
                trips = []
                for h in hs:
                    for t in ts:
                        trips.append((h, r, t))
                ans.append(trips)
            else:
                ans.append((an['h'], an['r'], an['t']))
        return ans

    def answer_prompts(self):
        ents = self.entities()
        if 'labels' in self:
            ans = []
            for an in self['labels']:
                _ans = []
                pmpt = prompt(an['r'])
                for h in ents[an['h']]:
                    for t in ents[an['t']]:
                        _ans.append(pmpt.replace("?x", h, 1).replace("?y", t, 1))
                ans.append(_ans)
            return ans

    def candidate_maps(self, rel:str=None, filt=True):
        if rel:
            rels = [rel]
        else:
            rels = rel_info
        for rel in rels:
            pmpt = prompt(rel)
            prompts = {}
            dom = rel_info[rel]['domain']
            ran = rel_info[rel]['range']
            for a, b in permutations(self.mentions, 2):
                ta, tb = self.mention_types[a], self.mention_types[b]
                if not filt or (ta in dom and tb in ran):
                    prompts[pmpt.replace("?x", a, 1).replace("?y", b, 1)] = ((a, ta), (b, tb))
        return prompts

    def entity_vecs(self, nonlinearity=lambda x:x, pooling=None, passes=0):
        ent_vecs = {}
        ment_inds = {}
        # Each ent_vecs[e] needs to be the correct corresponding set of vectors.
        # Lengths are available:
        ment_vecs = {}
        _s = 0
        for i, s in enumerate(self.masked_doc['ment_lens']):
            ment_vecs[i] = self.mlm.augment(self.ment_vecs[passes][_s:_s + s], nonlinearity, None)  # We can't pool here.
            ment_inds[i] = self.ment_inds[_s:_s + s]
            if passes > 0:
                # ment_inds[i] = [-1]*len(ment_inds[i])
                ment_inds[i] = torch.ones_like(ment_inds[i]) * -1
            _s += s
        for e, inds in self.masked_doc['ents'].items():
            ent_vecs[e] = [ment_vecs[m] for m in inds]
        return ent_vecs, ment_inds
        

def read_document(task_name: str = 'docred', dset: str = 'dev', *, mlm = None, path: str = 'data', doc=-1, verbose=False):
    if task_name == 'docred' and dset == 'train':
        dset = 'train_annotated'
    with open(f"{path}/{task_name}/{dset}.json") as datafile:
        jfile = json.load(datafile)
        if doc >= 0:
            yield Document(jfile[doc], doc, mlm=mlm)#, num_passes=num_passes)
        else:
            for i, doc in enumerate(jfile):
                yield Document(doc, i, mlm=mlm)#, num_passes=num_passes)


def mask_vectors(self, sent, keep_original=False, add_special_tokens=False, padding=False):
    # tokens = self.tokenize(sent, add_special_tokens=add_special_tokens, padding=padding)
    # print(tokens)
    sent.squeeze_(0)
    # print(sent.shape)
    tlen = len(sent)
    offset = 1 if add_special_tokens else 0
    token_mat = [torch.clone(sent) for i in range(tlen - (2*offset))]
    for i in range(offset, tlen-offset):
        # print("ti B:", token_mat[i-offset][i])
        token_mat[i-offset][i] = self.mask_token_vector
        # print("ti A:", token_mat[i-offset][i])
    if keep_original:
        token_mat = [torch.clone(sent)] + token_mat
    return torch.stack(token_mat)


def run_exp(fb:FitBert, resdir, task_name='docred', dset='dev', doc=0, num_passes=1, top_k=0, skip=[], model='bert-large-cased', start_at=0, end_at=1000):
    with torch.no_grad():  # Super enforce no gradients whatsoever.
        torch.cuda.empty_cache()
        if fb is None:
            fb = extend_bert(FitBert(model_name=model))
        qx, qy = fb.tokenizer(["?x?y"], add_special_tokens=False)['input_ids'][0]
        # Take a document
        # Find all the entities
        for d in read_document(task_name=task_name, dset=dset, doc=doc, mlm=fb, path='data'):
            docfile = f"{resdir}/{task_name}_{model}_{dset}_{d.num}_0b_{num_passes}p.pickle"
            # print(docfile)
            # if os.path.exists(docfile) or
            if d.num in skip or d.num < start_at:
                # print(f"Pass {p_doc.num}")
                print(f"Document {d.num} skipped.", flush=True)
                continue
            print(f"Document {d.num} started.", flush=True)
            md = d.masked_doc
            if len(md['tokens']) <= 510:
                mask = [False] + md['ment_mask'] + [False]
                # print("mt", len(md['tokens']))
                # ents = md['ents']
                _tokens = fb.tokenizer.convert_tokens_to_ids(md['tokens'])
                _tokens = [fb.tokenizer.cls_token_id] + _tokens + [fb.tokenizer.sep_token_id]
                _tokens = torch.LongTensor([_tokens])
                V = fb.get_vocab_output_dim()  # [tokens, vocab(29028)]

                d.ment_inds = _tokens.squeeze(0)[mask]
                # d.ment_inds_masked = d.ment_inds.clone()
                # d.ment_inds_masked[d.ment_inds >= min(fb.entity_tokens)] = -1
                out = dict()
                # Initial pass: Just one-hot vectors as inputs.
                # print("t", _tokens.shape)
                input_vecs = torch.nn.functional.one_hot(_tokens, V).float()
                out[0] = input_vecs.cpu()
                # if num_passes == 0:
                #     # Then take the input tokens and convert them to one-hot vectors.
                # else:
                #     input_vecs = fb.bert.get_input_embeddings()(_tokens.to(fb.device)).cpu()
                #     out[0] = fb.bert(inputs_embeds=input_vecs.to(fb.device)).logits.cpu()
                #     d.ment_inds = torch.LongTensor([-1]*sum(mask))
                # del _tokens
                np = num_passes
                cp = 1
                while np > 0:
                    # Second and further passes: Run through the MLM.
                    # Step 1: Take original input vecs and sub in the entities.
                    if cp > 1:
                        entities = out[cp - 1].squeeze(0)[mask]
                        # We don't actually experiment with this parameter.
                        # We only show the effectiveness of applying this to the output.
                        # For the inner steps, we kept everything softmaxed.
                        if top_k > 0:
                            entities = fb.fuzzy_embed(fb.top_k(entities, k=top_k))
                        elif top_k == 0:
                            entities = fb.fuzzy_embed(fb.softmax(entities))
                        else:
                            entities = fb.fuzzy_embed(entities)
                        input_embeds.squeeze(0)[mask] = entities.squeeze(0)
                    else:
                        # First pass: Just use normal input embeddings.
                        input_embeds = fb.bert.get_input_embeddings()(_tokens.to(fb.device))

                    # else it's the same vectors already.
                    # Then we do a forward pass, gather the new output logits.
                    out[cp] = fb.bert(inputs_embeds=input_embeds).logits.cpu()
                    print(cp, out[cp].shape)
                    cp += 1
                    np -= 1
                del _tokens
                
                d.ment_vecs = dict()
                for p in out:
                    # if fb.token_width > 0:
                    #     d.ment_vecs[p] = out[p].squeeze(0)[mask].view(-1, fb.token_width, V)
                    # else:
                    d.ment_vecs[p] = out[p].squeeze(0)[mask].view(-1, V)
                    # print(d.ment_vecs[p].shape)
                del out
                torch.cuda.empty_cache()
                print(f"Document {d.num} preprocessed.", flush=True)
                yield d, docfile
                # exit(0)
            else:
                print("skipped", d)


def replace_embeddings(x, y, prompt):
    ix = prompt['ix']
    iy = prompt['iy']
    embs = prompt['vecs'].clone()
    # tkns = prompt['input_ids']
    return torch.cat([embs[:,:ix], x, embs[:,ix+1:iy], y, embs[:,iy+1:]], dim=1)


def replace_ids(x, y, prompt):
    ix = prompt['ix']
    iy = prompt['iy']
    embs = prompt['input_ids']
    return torch.cat([embs[:ix], x, embs[ix+1:iy], y, embs[iy+1:]])


def output_to_fuzzy_embeddings(fb, v):
    return (v.to(fb.device)@fb.bert.get_input_embeddings().weight).cpu()


def meminfo():
    f, t = torch.cuda.mem_get_info()
    f = f / (1024 ** 3)
    t = t / (1024 ** 3)
    return f"{f:.2f}g/{t:.2f}g"


def run_many_experiments(task_name, dset, rel_info, nonlins, scorers, resdir, num_passes=1, max_batch=2000, skip=[], model='bert-large-cased', start_at=0, end_at=1000):
    with torch.no_grad():
        torch.cuda.empty_cache()
        fb = extend_bert(FitBert(model_name=model))
        model = model.split('/')[-1]
        # nls = {None: lambda x:x, "softmax": FitBert.softmax, "relu":torch.relu}
        # global one_hot_punctuation
        # one_hot_punctuation = read_punctuation(fb)
        qx, qy = fb.tokenizer(["?x?y"], add_special_tokens=False)['input_ids'][0]
        prompt_data = {}
        if task_name == "docred":
            # Top ten.
            prompts = ['P17', 'P27', 'P131', 'P150', 'P161', 'P175', 'P527', 'P569', 'P570', 'P577']
        else:
            prompts = list(sorted(rel_info.keys()))
        # prompts = ['P17']
        for prompt in prompts:
            pi = dict()
            tkns = fb.tokenizer(rel_info[prompt]['prompt_xy'], return_tensors='pt')['input_ids']
            pi['input_ids'] = tkns[0].cpu()
            pi['ix'] = torch.where(tkns[0] == qx)[0].item()
            pi['iy'] = torch.where(tkns[0] == qy)[0].item()
            # print("Cosine score:", score_batched(fb, [scorer], [prompt])[0]['csd'][0])
            pi['vecs'] = fb.bert.get_input_embeddings()(tkns.to(fb.device)).cpu()
            prompt_data[prompt] = pi
        
        for p_doc, docfile in run_exp(fb, resdir, task_name=task_name, dset=dset, doc=-1, num_passes=num_passes, skip=skip, model=model, start_at=start_at, end_at=end_at, top_k=0):
            all_scores = dict()
            for nps in range(0, num_passes):
                if os.path.exists(docfile.replace(f'_{num_passes}p', f'_{nps}p')):
                    print(f"Document {p_doc.num} at {nps} passes skipped.", flush=True)
                    continue
                all_scores = dict()
                for nonlin in nonlins:
                    # print(f"NL: {nonlin}")
                    all_scores[nonlin] = {}
                    nl = FitBert.nonlins[nonlin]
                    for pooler in [None]:
                        # print(f"PL: {pooler}")
                        all_scores[nonlin][pooler] = {}
                        evs, ev_tkns = p_doc.entity_vecs(nonlinearity=nl, pooling=pooler, passes=nps)
                        fuzzy_embeds = {e:[output_to_fuzzy_embeddings(fb, v1.unsqueeze(0)) for v1 in evs[e]] for e in evs}
                        e_to_m_map = p_doc.masked_doc['ents']
                        # print(f"A: {torch.cuda.mem_get_info()}")
                        # times = []
                        for prompt_id in prompts: # rel_info:
                            if prompt_id.split('|')[0] not in p_doc.relations:
                                continue
                            # nnow = time.time()
                            # print(f"PI: {prompt_id}")
                            ans = [(a[0], a[2]) for a in p_doc.answers(detailed=False) if a[1] == prompt_id.split('|')[0]]
                            # BioRED explicitly states that all relations are non-directional.
                            # This is honestly false, but we marked the ones that aren't clearly non-directional to avoid issues.
                            # For example, "Conversion" is a one-way process between chemicals.
                            # "Bind" is questionable in this regard. It feels one-directional in some circumstances, but I'm not an expert...
                            # The remaining relations are obviously symmetric:
                            # Association, Positive/Negative Correlation, Comparison, Co-Treament, and Drug Interaction.
                            # The only DocRED relation marked symmetric is "sister city".
                            # "spouse" and "sibling" should also be marked as such, though, so that might get updated.
                            # (Those relations aren't examined in these experiments)
                            if rel_info[prompt_id]["symmetric"] == "true":
                                ans.extend([(a[1], a[0]) for a in ans if (a[1], a[0]) not in ans])

                            # No sense in setting up a bunch of examples if none are correct.
                            # Maybe a retrieval system (RAG?) can make this selection in the wild?
                            if len(ans) == 0:
                                continue
                            all_scores[nonlin][pooler][prompt_id] = {}
                            for sc in scorers:
                                all_scores[nonlin][pooler][prompt_id][sc.label] = []
                            # prompt = rel_info[prompt_id]['prompt_xy']
                            # AAAAAAA
                            # tkns = prompt_data[prompt]['input_ids']
                            # print(len(tkns[0]), score_len)
                            torch.cuda.empty_cache()
                            # print(f"B: {torch.cuda.mem_get_info()}")
                            res = defaultdict(lambda:-float('inf'))
                            all_replaced_m = {}
                            all_labels_m = {}
                            for e1 in fuzzy_embeds:
                                for e2 in fuzzy_embeds:
                                    if e1 != e2:
                                        _seen = set()
                                        for v1, m1 in zip(fuzzy_embeds[e1], e_to_m_map[e1]):
                                            for v2, m2 in zip(fuzzy_embeds[e2], e_to_m_map[e2]):
                                                if nps == 0:
                                                    vals = tuple(ev_tkns[m1].tolist() + [None] + ev_tkns[m2].tolist())
                                                    if vals in _seen:
                                                        continue
                                                    else:
                                                        _seen.add(vals)

                                                # vx = output_to_fuzzy_embeddings(fb, v1.unsqueeze(0))
                                                # vy = output_to_fuzzy_embeddings(fb, v2.unsqueeze(0))
                                                rep_vecs = replace_embeddings(v1, v2, prompt_data[prompt_id])
                                                # assert v1.shape[1] == ev_tkns[m1].shape[0]
                                                # print(rep_vecs.shape)
                                                mv = mask_vectors(fb, rep_vecs, keep_original=True, add_special_tokens=True)
                                                size = mv.shape[0]
                                                if size not in all_replaced_m:
                                                    all_replaced_m[size] = []
                                                    all_labels_m[size] = []
                                                all_replaced_m[size].append(mv)
                                                all_labels_m[size].append((e1, e2, m1, m2))
                            for size in all_replaced_m:
                                _max_batch_resized = max_batch - (max_batch % size)
                                _sentences_per_batch = _max_batch_resized // size
                                all_labels = all_labels_m[size]
                                print(f"{len(all_labels)} candidate statements.", flush=True)
                                bert_forward = torch.cat(all_replaced_m[size], dim=0).cpu()
                                for v in all_replaced_m[size]:
                                    del v
                                # all_replaced_m[size] = None
                                torch.cuda.empty_cache()
                                # print(bert_forward.shape)
                                # fwd_pieces = []
                                print(f"Document {p_doc.num} scoring bert forward ({size}, {len(bert_forward)}) for {nonlin} {pooler} {prompt_id} at {nps} passes", flush=True)
                                while len(bert_forward) > 0:
                                    print(f"BF: {len(bert_forward)} ({min(_max_batch_resized, len(bert_forward))//size}/{len(all_labels)})", flush=True)
                                    sm_bert = fb.softmax_(fb.bert(inputs_embeds=bert_forward[:_max_batch_resized].to(fb.device)).logits)[:, 1:, :]
                                    bert_forward = bert_forward[_max_batch_resized:]
                                    torch.cuda.empty_cache()
                                    # sm_bert = fb.softmax_(fb.bert(inputs_embeds=bert_forward.to(fb.device)).logits)[:, 1:, :]
                                    # print(f"Document {p_doc.num} Past.", flush=True)
                                    # This will cause issues.
                                    # sm_bert = torch.cat(fwd_pieces) if len(fwd_pieces) > 1 else fwd_pieces[0]
                                    # print(len(sm_bert.view(-1, score_len, sm_bert.shape[1], sm_bert.shape[2])), len(all_labels))
                                    # print(all_labels)
                                    # print(f"SM: {sm_bert.shape}")
                                    # print(len(sm_bert.view(-1, score_len, sm_bert.shape[1], sm_bert.shape[2])), len(all_labels))
                                    for (e1, e2, m1, m2), s in zip(all_labels[:_sentences_per_batch], sm_bert.view(-1, size, sm_bert.shape[1], sm_bert.shape[2])):
                                        # print("S:", s.shape)
                                        for scorer in scorers:
                                            origids = None
                                            if scorer.label == "pll":
                                                # Then make the index from its parts, same as the other thing.
                                                origids = replace_ids(ev_tkns[m1], ev_tkns[m2], prompt_data[prompt_id])[1:-1].to(fb.device)
                                            all_scores[nonlin][pooler][prompt_id][scorer.label].append((e1, e2, (e1, e2) in ans, scorer(s, origids=origids)))
                                    all_labels = all_labels[_sentences_per_batch:]
                                    # print(f"Document {p_doc.num} Tick.", flush=True)
                                    # for scorer in scorers:
                                    #     print(scorer.label, len(all_scores[nonlin][pooler][prompt_id][scorer.label]))
                                    del s
                                    del sm_bert
                                    torch.cuda.empty_cache()
                                del bert_forward
                            # return
                with open(docfile.replace(f'_{num_passes}p', f'_{nps}p'), 'wb') as resfile:
                    pickle.dump(all_scores, resfile)


if __name__ == '__main__':
    # dset = sys.argv[1]
    import sys
    # data_path = sys.argv[3] if len(sys.argv) > 3 else "data"
    task_name = sys.argv[1] if len(sys.argv) > 1 else "docred"
    data_set = sys.argv[2] if len(sys.argv) > 2 else "dev"
    model = sys.argv[3] if len(sys.argv) > 3 else "bert-base-cased"
    num_passes = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    resdir = sys.argv[5] if len(sys.argv) > 5 else "res"
    max_batch = int(sys.argv[6]) if len(sys.argv) > 6 else 1000
    start_at = int(sys.argv[7]) if len(sys.argv) > 7 else 0
    # nonlins = ["relu", "softmax", None]
    # poolers = ["mean", "max", None]  # Pooling didn't really yield great results. Maybe as an additional representation along with the others?
    # scorers = [CSD(), ESD(), JSD(), MSD(), HSD()]

    models = {
        "bert": "bert-large-cased",
        "roberta": "roberta-large-cased",
        "biobert": "dmis-lab/biobert-large-cased-v1.1",
        "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract"
    }

    if model.lower() in models:
        model = models[model.lower()]

    data_path = 'data'

    with open(f'{data_path}/{task_name}/rel_info_full.json', 'r') as rel_info_file:
        rel_info = json.load(rel_info_file)
    
    skip = []
    if task_name == 'docred':
        skip=[723]
    # def run_many_experiments(data, dset, rel_info, nonlins, poolers, scorers, resdir, num_passes=1, max_batch=2000, skip=[], model='bert-large-cased'):

    run_many_experiments(task_name,
                         data_set,
                         rel_info,
                         nonlins=[None, "relu", "softmax", "relmax", "top10", "top50", "top100"],
                        #  poolers=[None],
                         scorers=[PLL()], # CSD(), HSD(), MSD(), JSD(), ESD()],
                         resdir=resdir,
                         num_passes=num_passes + 1,
                         max_batch=max_batch,
                         skip=skip,
                         model=model,
                         start_at=start_at
                        )
