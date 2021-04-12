from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import common_utils as utils


if __name__ == '__main__':
    df_training_tfidf = utils.store_data('./project_data/data/tfidf/tfidf.train.libsvm', 10001)
    df_testing_tfidf = utils.store_data('./project_data/data/tfidf/tfidf.test.libsvm', 10001)
    df_eval_tfidf = utils.store_data('./project_data/data/tfidf/tfidf.eval.anon.libsvm', 10001)

    only_training_data = df_training_tfidf.iloc[:, 1:]
    only_training_label = df_training_tfidf.iloc[:, 0]
    testing_instances = df_testing_tfidf.iloc[:, 1:]
    testing_label = df_testing_tfidf.iloc[:, 0]

    eval_instances = df_eval_tfidf.iloc[:, 1:]

    clf = RandomForestClassifier(n_estimators=10)
    clf = clf.fit(only_training_data, only_training_label)

    test_predictions = clf.predict(testing_instances)
    score = accuracy_score(testing_label, test_predictions)
    print("Testing accuracy for random forests using sklearn: " + str(score))

    eval_predictions = clf.predict(eval_instances)

    f = open("variation_random_forests_sklearn.csv", "w")
    f.write("example_id,label")
    f.write("\n")
    for prediction_index in range(0, len(eval_predictions)):
        if eval_predictions[prediction_index] == -1:
            eval_predictions[prediction_index] = 0
        f.write(str(prediction_index) + "," + str(int(eval_predictions[prediction_index])) + "\n")
    f.close()
