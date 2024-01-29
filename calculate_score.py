from bert_score import score as bert_score_compute
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from collections import defaultdict
import json


tokenizer = PTBTokenizer()

targets = []
preds = []
res = defaultdict(list)
gts = defaultdict(list)

gtfile = '/nfs/swang7/blip2_eval/spotlight/test_1k.json'
predfile = '/nfs/swang7/blip2_eval/spotlight/pred_20240118071.ndjson'

with open(gtfile) as fp:
    gt_list = json.load(fp)
    gt_map = {}
    for gt in gt_list:
        gt_map[gt['video']] = gt['caption']

pred_list = list(map(json.loads, open(predfile)))

cnt = 0
for pred in pred_list:
    if pred['video'] not in gt_map:
        continue
    cnt += 1

    video_name = pred['video']

    pred_caption = pred['caption'][0]
    gt_caption = gt_map[video_name]
    
    targets.append(gt_caption)
    preds.append(pred_caption)
    
    res[video_name].append({'image_id':video_name, 'caption':pred_caption})
    gts[video_name].append({'image_id':video_name, 'caption':gt_caption})

res = tokenizer.tokenize(res)
gts = tokenizer.tokenize(gts)
scorers = [(Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")]

_, _, score = bert_score_compute(preds, targets, lang='en', verbose=False)
eval_dict = {"BERTScore": score.mean()}

for scorer, method in scorers:
    score, scores = scorer.compute_score(gts, res)
    eval_dict[scorer.method()] = score

final = {
    "bert_score": "{:.3f}".format(eval_dict["BERTScore"]),
    "bleu4": "{:.3f}".format(eval_dict["Bleu"][-1]),
    "rougel": "{:.3f}".format(eval_dict["Rouge"]),
    "METEOR": "{:.3f}".format(eval_dict["METEOR"]),
    "CIDEr": "{:.3f}".format(eval_dict["CIDEr"]),
}

print("number of predictions in testing data:", cnt)
print(final)
