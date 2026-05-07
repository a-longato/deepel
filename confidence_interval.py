import numpy as np

def confidence_interval(actual_path: str, predicted_path: str):
    with open(actual_path, 'r') as f:
        actual = [line.strip() for line in f]

    with open(predicted_path, 'r') as f:
        predicted = [line.strip() for line in f]

    y_true = np.array(actual) # True labels
    y_pred = np.array(predicted) # Predicted labels

    rng = np.random.RandomState(seed=12345)
    idx = np.arange(y_true.shape[0])

    test_accuracies = []

    for _ in range(1000):

        pred_idx = rng.choice(idx, size=idx.shape[0], replace=True)
        acc_test_boot = np.mean(y_pred[pred_idx] == y_true[pred_idx])
        test_accuracies.append(acc_test_boot)

    ci_lower = np.percentile(test_accuracies, 2.5)
    ci_upper = np.percentile(test_accuracies, 97.5)

    print(ci_lower, ci_upper)
