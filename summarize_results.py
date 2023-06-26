from sacrebleu import CHRF, BLEU
import evaluate


def get_metrics(file: str):
    print(file)
    bleu_values = []
    chrf_values = []
    p = []
    r = []
    with open(file, "r") as f:
        for l in f:
            a = l.strip().split(" ")
            true_comment = a[0].replace("|", " ")
            predict_comment = a[1].replace("|", " ")
            p.append(predict_comment)
            r.append(true_comment)
            chrf_values.append(chrf.sentence_score(predict_comment, [true_comment]).score)
            bleu_values.append(bleu.sentence_score(predict_comment, [true_comment]).score)
    print("BLEU:", sum(bleu_values) / len(bleu_values))
    print("CHRF:", sum(chrf_values) / len(chrf_values))
    print("METEOR:", meteor.compute(predictions=p, references=r)['meteor'])


bleu = BLEU(effective_order=True)
chrf = CHRF()
meteor = evaluate.load("meteor")
get_metrics("results/code2seq.txt")
get_metrics("results/codet5.txt")
