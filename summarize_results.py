from sacrebleu import CHRF
import evaluate


def sequential_f1(predicted_seq, target_seq):
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for predicted_token in predicted_seq:
        if predicted_token in target_seq:
            true_positive += 1
        else:
            false_positive += 1
    for target_token in target_seq:
        if target_token not in predicted_seq:
            false_negative += 1
    precision = true_positive
    if true_positive + false_positive > 0:
        precision = true_positive / (true_positive + false_positive)
    recall = true_positive
    if true_positive + false_negative > 0:
        recall = true_positive / (true_positive + false_negative)
    f1_score = 2 * precision * recall
    if precision + recall > 0:
        f1_score /= precision + recall
    return f1_score


def get_metrics(file: str):
    print(file)
    chrf_values = []
    f1_values = []
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
            f1_values.append(sequential_f1(predict_comment.split(), true_comment.split()))
    print("F1:", sum(f1_values) / len(f1_values))
    print("CHRF:", sum(chrf_values) / len(chrf_values))
    print("METEOR:", meteor.compute(predictions=p, references=r)["meteor"])


chrf = CHRF()
meteor = evaluate.load("meteor")
get_metrics("results/code2seq.txt")
get_metrics("results/codet5.txt")
