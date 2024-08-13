import json
# from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from spacy.attrs import ORTH, NORM

def reformat_biored(data_folder, data_set, verbose=False, document=-1, include_title=False):
    with open(f'{data_folder}/{data_set[0].upper()}{data_set[1:].lower()}.BioC.JSON') as bioredfile:
        bioredjson = json.load(bioredfile)
    nlp = English()
    nlp.add_pipe('sentencizer')
    
    for i in range(1000):
        tkn = f"[M{i}]"
        case = [{ORTH: tkn, NORM:tkn}]
        nlp.tokenizer.add_special_case(tkn, case)
    biored_reformatted = []
    
    skipped = 0
    total_rels = 0
    ix = -1
    for doc in bioredjson['documents']:
        ix += 1
        if document >=0 and document != ix:
            continue
        doc_obj = dict()
        pass_title, pass_text = doc['passages']
        doc_obj['title'] = doc['id']

        if include_title:
            orig_text = pass_title['text'] + " " + pass_text['text']
            main_off = 0
            annotations = pass_title['annotations'] + pass_text['annotations']
        else:
            orig_text = pass_text['text']
            main_off = pass_text['offset']
            annotations = pass_text['annotations']
    
        entities_to_inds = dict()
        entities = dict()
        mentions = dict()
        
        for ann in reversed(annotations):
            assert len(ann['locations']) == 1
            loc = ann['locations'][0]
            offs = loc['offset'] - main_off
            leng = loc['length']
            m_id = ann['id']
            men_tag = f"[M{m_id}]"
            ent = frozenset(ann['infons']['identifier'].split(','))
            if ent not in entities_to_inds:
                entities_to_inds[ent] = len(entities_to_inds)
                entities[entities_to_inds[ent]] = []
            orig_text = orig_text[:offs].rstrip() + f" {men_tag} {orig_text[offs:offs+leng]} {men_tag} " + orig_text[offs+leng:].lstrip()
            reps = ['RATIONALE', 'BACKGROUND', 'AIMS', 'AIM', 'CONCLUSIONS', 'CONCLUSION', 'DISCUSSION', 'OBJECTIVES', 'OBJECTIVE', 'METHODS', 'METHOD', 'MEASUREMENTS', 'RESULTS', 'PATIENT', 'PURPOSE']
            for rep in reps:
                orig_text = orig_text.replace(rep, rep[0] + rep[1:].lower())
            mentions[m_id] = {'name':ann['text'], 'pos':[], 'sent_id':0, 'type':ann['infons']['type'], 'entity':ent}

        tokenized_doc = nlp(orig_text.strip())
    
        sents = []
    
        for i, s in enumerate(tokenized_doc.sents):
            off = 0
            m_in = -1
            sents.append([])
            for j, token in enumerate(s):
                tk_txt = token.text
                if tk_txt.startswith("[M"):
                    if m_in == -1:
                        m_in = tk_txt[2:-1]
                        mentions[m_in]['pos'].append(j - off)
                        mentions[m_in]['sent_id'] = i
                    else:
                        assert m_in == tk_txt[2:-1]
                        mentions[m_in]['pos'].append(j - off - 1)
                        ent = mentions[m_in]['entity']
                        del mentions[m_in]['entity']
                        entities[entities_to_inds[ent]].append(mentions[m_in])
                        off += 2
                        m_in = -1
                else:
                    sents[-1].append(token.text)
        
        doc_obj['sents'] = sents
        vertexSet = list()
        for i, v in sorted(entities.items()):
            assert int(i) == len(vertexSet)
            vertexSet.append(v)
        
        doc_obj['vertexSet'] = vertexSet
        del mentions
    
        relations = list()
        grouped_entities = list(entities_to_inds.keys())
        head_facing = dict()

        for rel in doc['relations']:
            infons = rel['infons']
            h = infons['entity1']
            t = infons['entity2']
            r = infons['type']
            if (h, r) not in head_facing:
                head_facing[(h, r)] = set()
            head_facing[(h, r)].add(t)

        out_candidates = dict()
        
        for hr, t in head_facing.items():
            t = frozenset(t)
            candidates = set()
            for es in grouped_entities:
                s = len(es)
                # If we have an exact match.
                if len(es.intersection(t)) == len(es):
                    to_add = True
                    remove = set()
                    # Then we see if we have found an exact match before
                    for c in candidates:
                        # Is the entity a super set of an earlier match?
                        if len(es.intersection(c)) == len(c):
                            remove.add(c)
                        # Is it instead a subset of an earlier match?
                        elif len(es.intersection(c)) == len(es):
                            to_add = False
                            break
                    if remove:
                        candidates -= remove
                    if to_add:
                        candidates.add(es)
            out_candidates[hr] = candidates

        tail_facing = dict()
        for hr, ts in out_candidates.items():
            
            h,r = hr
            for t in ts:
                if (t, r) not in tail_facing:
                    tail_facing[(t, r)] = set()
                tail_facing[(t, r)].add(h)

        out_candidates = dict()
        
        for tr, h in tail_facing.items():
            h = frozenset(h)
            candidates = set()
            for es in grouped_entities:
                s = len(es)
                # If we have an exact match.
                if len(es.intersection(h)) == len(es):
                    to_add = True
                    remove = set()
                    # Then we see if we have found an exact match before
                    for c in candidates:
                        # Is the entity a super set of an earlier match?
                        if len(es.intersection(c)) == len(c):
                            remove.add(c)
                        # Is it instead a subset of an earlier match?
                        elif len(es.intersection(c)) == len(es):
                            to_add = False
                            break
                    if remove:
                        candidates -= remove
                    if to_add:
                        candidates.add(es)
            out_candidates[tr] = candidates

        relations = list()
        for tr, hs in out_candidates.items():
            t, r = tr
            for h in hs:
                relations.append((h, r, t))
        labels = list()
        for e1, rel, e2 in relations:
            total_rels += 1
            if e1 in entities_to_inds and e2 in entities_to_inds:
                labels.append({'r': rel, 'h': entities_to_inds[e1], 't': entities_to_inds[e2]})
            else:
                skipped += 1
                if verbose:
                    print(f"For doc {ix} skipping {e1} {rel} {e2}, can't match mention to entity.")

        doc_obj['labels'] = labels

        biored_reformatted.append(doc_obj)

    print(f"Skipped {100*skipped/total_rels:.2f}% of the relations.")
    if document == -1:
        with open(f"{data_folder}/{data_set.lower()}.json", 'w', encoding='utf8') as out_json:
            json.dump(biored_reformatted, out_json)

reformat_biored("data/git/understanding-pll/data/biored", "dev", verbose=True)
reformat_biored("data/git/understanding-pll/data/biored", "train", verbose=True)
reformat_biored("data/git/understanding-pll/data/biored", "test", verbose=True)