import pandas as pd
from collections import defaultdict
from deepproblog.utils.confusion_matrix import ConfusionMatrix
from deepproblog.model import Model
from deepproblog.dataset import Dataset
from typing import Optional
import re

def get_confusion_matrix_and_write_files(
    model: Model, dataset: Dataset, actual_path: str, predicted_path: str, verbose: int = 0, eps: Optional[float] = None
) -> ConfusionMatrix:
    """
    Evaluates the model and prints the Average Confidence Score 
    strictly for correctly predicted samples per class.
    """
    confusion_matrix = ConfusionMatrix()
    errors_indices = []

    correct_scores_tracker = defaultdict(list)

    open(actual_path, "w", encoding="utf-8").close()
    open(predicted_path, "w", encoding="utf-8").close()

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

            if actual == predicted:
                correct_scores_tracker[str(actual)].append(float(p))

            if verbose > 1 and actual != predicted:
                print(
                    "{} {} vs {}::{} for query {}".format(
                        i, actual, p, predicted, test_query
                    )
                )
            if actual != predicted:
                s = str(test_query)
                try:
                    match = re.search(r'dataset\((\d+)\)', s)
                    if match:
                        number = int(match.group(1))
                        errors_indices.append(number)
                except:
                    pass
                    
        confusion_matrix.add_item(predicted, actual)

        with open(actual_path, "a", encoding="utf-8") as f:
            f.write(f"{actual}\n")

        with open(predicted_path, "a", encoding="utf-8") as f:
            f.write(f"{predicted}\n")
        

    if verbose > 0:
        print(confusion_matrix)
        print("Accuracy", confusion_matrix.accuracy())

        print("\n" + "="*50)
        print("AVERAGE CONFIDENCE (CORRECT PREDICTIONS ONLY)")
        print("="*50)
        
        results = []
        sorted_labels = sorted(correct_scores_tracker.keys())
        
        for label in sorted_labels:
            scores = correct_scores_tracker[label]
            avg_score = sum(scores) / len(scores)
            count = len(scores)
            results.append({
                "Class": label,
                "Avg Confidence": round(avg_score, 4),
                "Correct Count": count
            })
            
        if results:
            df_correct = pd.DataFrame(results)
            print(df_correct.to_string(index=False))
        else:
            print("No correct predictions found.")
        print("="*50 + "\n")

    return confusion_matrix, errors_indices