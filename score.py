import pickle
import os
import json
import pandas as pd
import numpy as np
import matplotlib as mpl
import seaborn as sns
from datetime import datetime
import time

rel_info = dict()

for task in ["docred", "biored"]:
    with open(f'data/{task}/rel_info_full.json', 'r') as rel_info_file:
        rel_info[task] = json.load(rel_info_file)

# The input scores look like this, where P17 is the variable relation label.
# scores = {'relu': {'max': {'P17': {'hsd': [...]}}}}
# The output scores need to instead look like this:
# scores = {'relu': {'max': {'hsd': {'top_5': 0.0}}}}
# Easy enough.
def score_fixer(subscores, doc, rel, flipsort=False, dr=False, task_name='docred'):
    """
    No, we're not cheating here. By "fixing" the scores, we're actually trying to make things more fair.
    We want to try and present only one candidate per entity pair, but we generate candidates at the
    mention level, which can greatly inflate the number of true (and false) statements.
    So, this function filters the candidates by the highest-scoring version.
    It also filters candidates by domain and range restrictions, in case that didn't happen earlier.
    """
    seen = set()
    res_scores = []
    list_sort = list(sorted(subscores, key=lambda x:x[-1]) if flipsort else sorted(subscores, key=lambda x:-x[-1]))
    # print(list_sort)
    count_true = 0
    count_false = 0
    first_true = -1
    i = 0
    for score in list_sort:
        ex = score[0:2]
        # If we've seen this pair before, then it had a higher score, so skip it.
        # We include the tag because there are some documents which have the same mentions marked for multiple
        # entities, so the pairs could simultaneously be correct and incorrect... it's weird.
        if ex not in seen:
            if dr:
                # Check the domain and range calculated elsewhere to make sure this should be kept.
                if ex in dr[doc][rel]:
                    if score[2]:
                        count_true += 1
                        if first_true == -1:
                            first_true = i
                    else:
                        count_false += 1
                    seen.add(ex)
                    res_scores.append(score)
                    i += 1
            else:
                if score[2]:
                    count_true += 1
                    first_true = min(first_true, i)
                else:
                    count_false += 1
                seen.add(ex)
                res_scores.append(score)
                i += 1
    return res_scores, first_true, count_true, count_false, i


# Processing step.
# The first pass only counts how many times each top-k value is scored.
# This step converts that to percentage of all documents which scored that top-k value.
# These percentages are what are presented in the paper.
def counts_to_percent(fullscores):
    rel_len = {}
    for rel in fullscores:
        if rel != 'total_docs':
            for nlpl in fullscores[rel]:
                for tk in fullscores[rel][nlpl]:
                    for scr in fullscores[rel][nlpl][tk]:
                        if rel not in rel_len:
                            rel_len[rel] = fullscores[rel][nlpl][tk][scr]['total_docs']
                        if fullscores[rel][nlpl][tk][scr]['total_docs'] > 0:
                            if 'ratioB' == tk:
                                fullscores[rel][nlpl][tk][scr] = (fullscores[rel][nlpl]['#true'][scr]['correct'] / fullscores[rel][nlpl]['#false'][scr]['correct'])*100.0
                            elif '#' not in tk:
                                fullscores[rel][nlpl][tk][scr] = (fullscores[rel][nlpl][tk][scr]['correct'] / fullscores[rel][nlpl][tk][scr]['total_docs'])*100.0
                            else:
                                fullscores[rel][nlpl][tk][scr] = (fullscores[rel][nlpl][tk][scr]['correct'] / fullscores[rel][nlpl][tk][scr]['total_docs'])
                        else:
                            fullscores[rel][nlpl][tk][scr] = -1
    return fullscores, rel_len


# Reads the results from the experiments and counts how many times each document
# scores a point for each top-k value, broken down by relation, nonlinearity, pooling
# method (not used), and scoring method.
# The values reported are top-k counts, the average number of true/false statements,
# the ratio between those two values (ratioB), and the average ratio over all docs
# (ratioA).
# Returns the values as counts, so use counts_to_percent() above to get these as
# percentages.
def top_k(folder, task_name, data_set, max_k=5, num_blanks=2, dr=None, max_doc=1000, n_passes=1, model_name="bert-large-cased"):
    out_scores = {'total_docs':0}
    for d in range(max_doc):
        pth = f"res/{folder}{task_name}_{model_name}_{data_set}_{d}_{num_blanks}b_{n_passes}p.pickle"
        if os.path.exists(pth):
            if task_name == "biored" and model_name != "bert-large-cased":
                pth2 = f"res/{folder}{task_name}_bert-large-cased_{data_set}_{d}_{num_blanks}b_{n_passes}p.pickle"
                if not os.path.exists(pth2):
                    continue
            with open(pth, "rb") as pfile:
                out_scores['total_docs'] += 1
                in_scores = pickle.load(pfile)
                for nl in in_scores:
                    # if nl not in out_scores:
                    #     out_scores[nl] = {}
                    for pl in in_scores[nl]:
                        nlpl = f"{nl}+{pl}"
                        for rel in in_scores[nl][pl]:
                            if rel not in out_scores:
                                out_scores[rel] = {}    
                            if nlpl not in out_scores[rel]:
                                out_scores[rel][nlpl] = {}
                            for scr in in_scores[nl][pl][rel]:
                                sf, best, nt, nf, tot = score_fixer(in_scores[nl][pl][rel][scr], flipsort=(scr=="msd"), dr=dr, doc=d, rel=rel, task_name=task_name)
                                # There are some documents where all correct answers get filtered out due to the domain and range constraints.
                                # In this sense, the document does not represent the relation as defined by the schema/ontology, so we choose
                                # to filter it out anyway.
                                # The other conditions try to avoid "easy mode" documents which only inflate the scores.
                                if best == -1 or tot < 10 or nf < 5:
                                    continue
                                tks = [f"top_{k}" for k in range(1, max_k+1)]

                                for tk in ["ratioB", "mrr"] + tks + ["ratioA", "#true", "#false"]:
                                    if tk not in out_scores[rel][nlpl]:
                                        out_scores[rel][nlpl][tk] = {}
                                    if scr not in out_scores[rel][nlpl][tk]:
                                        out_scores[rel][nlpl][tk][scr] = {}
                                        out_scores[rel][nlpl][tk][scr]['total_docs'] = 1
                                        out_scores[rel][nlpl][tk][scr]['correct'] = 0
                                    else:
                                        out_scores[rel][nlpl][tk][scr]['total_docs'] += 1
                                # best = trues.index(True)
                                out_scores[rel][nlpl]["mrr"][scr]['correct'] += 1/(best + 1) if best != -1 else 0
                                # Because, of course, there are some documents with *only* correct answers.
                                if nf > 0:
                                    out_scores[rel][nlpl]["ratioA"][scr]['correct'] += nt/nf
                                else:
                                    out_scores[rel][nlpl]["ratioA"][scr]['total_docs'] -= 1
                                out_scores[rel][nlpl]["#true"][scr]['correct'] += nt
                                out_scores[rel][nlpl]["#false"][scr]['correct'] += nf
                                if best > -1 and best < max_k:
                                    for k in range(best, max_k):
                                        out_scores[rel][nlpl][f"top_{k+1}"][scr]['correct'] += 1
                                    
    return out_scores

flipsort = False
max_doc = 1000



cm = sns.light_palette("green", as_cmap=True)

def inner_work(task_name, data_set, max_doc, num_blanks, masks, n_passes, dr, easymode, model_name, folder=''):
    if not folder:
        folder = '/'
    elif folder[-1] != '/':
        folder = folder + '/'
    res, lens = counts_to_percent(top_k(folder=folder, task_name=task_name, data_set=data_set, dr=dr, num_blanks=num_blanks, max_doc=max_doc, masks=masks, n_passes=n_passes, easymode=easymode, model_name=model_name))
    output = []
    if res['total_docs'] > 0:
        output.append(f"<h1>{task_name.upper()} {data_set.upper()} Test: {model_name} {'MASK' if masks else 'Blanked Entities'}, {num_blanks} blanks, {n_passes} passes<br/>First {min(res['total_docs'], max_doc)} docs.<br/>")
        output.append(f"D&R restrictions are {'ON' if dr else 'OFF'}.</h1><br/>")
        rels = sorted((k for k in res.keys() if k != 'total_docs'), key=lambda x: int(x.split(' ')[0][1:])) if task_name == 'docred' else sorted(res.keys())
        for rel in rels:
            if rel == 'total_docs':
                continue
            rr = res[rel]
            rr2 = {(outerKey, innerKey): values for outerKey, innerDict in rr.items() for innerKey, values in innerDict.items()}
            df = pd.DataFrame(rr2).astype('float')
            #   .format_index(str.upper, axis=1) \
            #   .relabel_index(["row 1", "row 2"], axis=0)
            if rel in lens:
                output.append(f'<h2>{rel} ({rel_info[task_name][rel]["name"]}) after {lens[rel]} docs</h2><br/>')
                output.append(df.style.background_gradient(cmap=cm, vmin=0.0, vmax=100.0).format('{:.1f}').to_html())
                output.append('<br/>')
    return output

# t = tqdm.tqdm(total=(2*2*2*2*1*1*3))
res_map = {}

with open('dev_answers_domain_range.pickle', 'rb') as domain_range_pickle:
    dr_restricted_docred = pickle.load(domain_range_pickle)
    
with open('dev_answers_domain_range_biored.pickle', 'rb') as domain_range_pickle:
    dr_restricted_biored = pickle.load(domain_range_pickle)

# while True:
    # t.reset()
now = datetime.now()
for task_name, dr in [("docred", dr_restricted_docred)]:#, ("biored", dr_restricted_biored)]:
    for data_set in ["train", "dev"]:
        for num_passes in range(0, 3):  # range(0, 5):
            for num_blanks in range(0, 1):  # range(0, 5):
                for fs in [True]:
                    for em in [True]:
                        for model in ["bert-base-cased", "bert-large-cased", "BiomedNLP-PubMedBERT-large-uncased-abstract", "biobert-large-cased-v1.1", "roberta-large"]:
                            op = inner_work(task_name=task_name, data_set=data_set, max_doc=max_doc, num_blanks=num_blanks, masks=True, n_passes=num_passes, dr=dr, easymode=em, model_name=model)
                            if len(op) > 0:
                                res_map[(task_name, data_set, model, num_passes, num_blanks, fs)] = op
                                filename = "index.html" if task_name == "docred" else "biored.html"
                                with open(f'/home/riley/{filename}', 'w', encoding='utf8') as resfile:
                                    resfile.write(f'<!DOCTYPE html><html><head><title>Experimental Results</title><meta http-equiv="refresh" content="60"/></head><body>')
                                    resfile.write(f'<h0>Last updated: {now}</h0>')
                                    for res in res_map:
                                        if res[0] == task_name:
                                            resfile.writelines(res_map[res])
                                    resfile.write("</body></html>")
                                    resfile.flush()
    # time.sleep(60)