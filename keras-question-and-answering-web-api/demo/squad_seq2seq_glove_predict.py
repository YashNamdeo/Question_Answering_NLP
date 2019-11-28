from keras_question_and_answering_system.library.seq2seq_glove import Seq2SeqGloveQA
from keras_question_and_answering_system.library.utility.squad import SquADDataSet
import re
import string
from collections import Counter
from sklearn.metrics import label_ranking_average_precision_score



def f1_score(prediction, ground_truth):
    if prediction == ground_truth:
        return 1
    prediction_tokens = _normalize_answer(prediction).split()
    ground_truth_tokens = _normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def exact_match_score(prediction, ground_truth):
    return (_normalize_answer(prediction) == _normalize_answer(ground_truth))

def _normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def main():
    qa = Seq2SeqGloveQA()
    qa.load_glove_model('./very_large_data')
    qa.load_model(model_dir_path='./models')
    score=0
    data_set = SquADDataSet(data_path='./data/SQuAD/BioASQ-train-factoid-7b-snippet-2sent.json')
    for i in range(20):
        index = i * 10
        paragraph, question, actual_answer = data_set.get_data(index)
        predicted_answer = qa.reply(paragraph, question)
        print('context: ', paragraph)
        print('question: ', question)
        print({'guessed_answer': predicted_answer, 'actual_answer': actual_answer})
        score+=f1_score(predicted_answer,actual_answer)
    score/=20
    print("f1_score "+str(score))



if __name__ == '__main__':
    main()
