from deepproblog.utils.confusion_matrix import ConfusionMatrix
from deepproblog.model import Model
from deepproblog.dataset import Dataset
from typing import Optional
import re

def get_confusion_matrix_and_errors(
    model: Model, dataset: Dataset, verbose: int = 0, eps: Optional[float] = None
) -> ConfusionMatrix:
    """

    :param model: The model to evaluate.
    :param dataset: The dataset to evaluate the model on.
    :param verbose: Set the verbosity. If verbose > 0, then print confusion matrix and accuracy.
    If verbose > 1, then print all wrong answers.
    :param eps: If set, then the answer will be treated as a float, and will be considered correct if
    the difference between the predicted and ground truth value is smaller than eps.
    :return: The confusion matrix when evaluating model on dataset.
    """
    confusion_matrix = ConfusionMatrix()
    errors_indices = []
    model.eval()
    for i, gt_query in enumerate(dataset.to_queries()):
        test_query = gt_query.variable_output()
        answer = model.solve([test_query])[0]
        actual = str(gt_query.output_values()[0])
        if len(answer.result) == 0:
            predicted = "no_answer"
            if verbose > 1:
                print("no answer for query {}".format(gt_query))
        else:
            max_ans = max(answer.result, key=lambda x: answer.result[x])
            p = answer.result[max_ans]
            if eps is None:
                predicted = str(max_ans.args[gt_query.output_ind[0]])
            else:
                predicted = float(max_ans.args[gt_query.output_ind[0]])
                actual = float(gt_query.output_values()[0])
                if abs(actual - predicted) < eps:
                    predicted = actual
            if verbose > 1 and actual != predicted:
                print(
                    "{} {} vs {}::{} for query {}".format(
                        i, actual, p, predicted, test_query
                    )
                )
            if actual != predicted:
                s = str(test_query)
                match = re.search(r'dataset\((\d+)\)', s)
                number = int(match.group(1))
                errors_indices.append(number)
        confusion_matrix.add_item(predicted, actual)

    if verbose > 0:
        print(confusion_matrix)
        print("Accuracy", confusion_matrix.accuracy())

    return confusion_matrix, errors_indices