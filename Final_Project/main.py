# Paarth Lakhani
# u0936913

import pandas as pd
import numpy as np
import random
import common_utils as utils
import os

random.seed(42)
cross_validation_epochs = 10
epochs_to_train_on = 10


def decision_tree_processing(processing_dataset_name):
    processing_dataset = open(processing_dataset_name, 'r')
    lines = processing_dataset.readlines()
    feature_freq_dict = {}

    for line in lines:
        line.split()

    #with open(processing_dataset_name, newline='') as csv_file:
    #    rows = csv.reader(csv_file)
    #    for row in rows:
    #        print(row)

    #label_dataset = open(label_dataset_name, 'r')
    #lines = label_dataset.readlines()
    #print(lines)


'''def store_data(path):
    file1 = open(path, 'r')
    lines = file1.readlines()
    feature_list = []

    for line in lines:
        feature_vector = np.zeros([10001])
        # feature_vector = np.zeros([301]) # for glove
        one_data_set = line.split()
        if int(one_data_set[0]) == 0:
            feature_vector[0] = int(-1)
        else:
            feature_vector[0] = int(one_data_set[0])
        one_data_set = one_data_set[1:]
        for attribute_value in one_data_set:
            split_attribute_value = attribute_value.split(":")
            feature_vector[int(split_attribute_value[0])] = float(split_attribute_value[1])
        feature_list.append(feature_vector)
    df = pd.DataFrame(feature_list)
    return df'''


def predict_label(weights, bias, example):
    prediction = 0
    prediction = np.dot(np.transpose(weights), example) + bias
    # prediction = prediction + model[1]  # Adding bias
    if prediction <= 0:
        return -1  # negative label
    else:
        return 1  # positive label


def predict_label_for_evaluation(weights, bias, example):
    prediction = 0
    # for weight in range(0, len(model) - 1):
    #    prediction = prediction + model[weight] * example[weight + 1]
    # prediction = prediction + model[len(model) - 1]  # Adding bias
    prediction = np.dot(np.transpose(weights), example) + bias
    if prediction <= 0:
        return 0  # negative label This dataset deals with 0 instead of -1
    else:
        return 1  # positive label


def evaluate_accuracy(weights, bias, dataset):
    accuracy = 0
    for instance in range(0, dataset.shape[0]):
        predicted_label = predict_label(weights, bias, dataset.iloc[instance][1:])
        if predicted_label == dataset.iloc[instance][0]:
            accuracy += 1
    accuracy_percent = accuracy / dataset.shape[0] * 100
    return accuracy_percent


# 0 - Simple perceptron
# 1 - Simple perceptron with decaying the learning rate
# 2 - averaged perceptron
# 3 - Majority baseline perceptron
def simple_perceptron_n_cross_validation(list_of_folds, perceptron_variant, **kwargs):
    max_accuracy_learning = -1
    learning_rate = -1
    for hyper_parameter, combinations in kwargs.items():
        for combination in combinations:
            testing_accuracy_learning = 0
            for fold_no in range(len(list_of_folds)):
                training_fold_list = []
                testing_fold_df = list_of_folds[fold_no]
                for train_fold_no in range(len(list_of_folds)):
                    if train_fold_no != fold_no:
                        training_fold_list.append(list_of_folds[train_fold_no])
                training_fold_df = pd.concat(training_fold_list, ignore_index=True)
                # weights_from_fold = np.zeros([1, training_fold_df.shape[1] - 1])
                # print("Shape is: ")
                # print(str(training_fold_df.shape[1]))
                weights = np.zeros([training_fold_df.shape[1] - 1])
                if perceptron_variant == 0:
                    weights, bias, updates, dict_accuracies = simple_perceptron(training_fold_df.to_numpy(),
                                                                                cross_validation_epochs, combination)
                elif perceptron_variant == 1:
                    pass  # weights_from_fold = simple_perceptron_decaying_learning_rate(training_fold_df.to_numpy(), cross_validation_epochs, combination)
                elif perceptron_variant == 2:
                    weights, bias, updates, dict_accuracies = averaged_perceptron(training_fold_df.to_numpy(),
                                                                                  cross_validation_epochs, combination)
                elif perceptron_variant == 3:
                    pass  # weights_from_fold = simple_perceptron_majority_baseline(training_fold_df, cross_validation_epochs, combination)
                testing_accuracy_learning += evaluate_accuracy(weights, bias, testing_fold_df)
            testing_accuracy_learning = testing_accuracy_learning / len(list_of_folds)
            if max_accuracy_learning < testing_accuracy_learning:
                max_accuracy_learning = testing_accuracy_learning
                learning_rate = combination
    print("Best learning rate is: " + str(learning_rate))
    # print("Cross validation accuracy for the best hyper parameters: " + str(max_accuracy_learning))
    return learning_rate


def evaluate_dataset(weights, bias, dataset, filename):
    f = open(filename, "w")
    f.write("example_id,label")
    f.write("\n")
    for instance in range(0, dataset.shape[0]):
        predicted_label = predict_label_for_evaluation(weights, bias, dataset.iloc[instance][1:])
        dataset.iloc[instance][0] = predicted_label
        f.write(str(instance) + "," + str(predicted_label) + "\n")
    f.close()
    return dataset.iloc[:, 0]


def simple_perceptron(dataset, epochs_to_train_on, learning_rate):
    # print("In simple perceptron: ")
    weights = np.random.uniform(-0.01, 0.01, size=np.shape(dataset)[1] - 1)
    bias = np.random.uniform(-0.01, 0.01)
    updates = 0
    accuracy = 0
    dict_accuracies = {}

    for epoch_no in range(epochs_to_train_on):
        np.random.shuffle(dataset)
        for example_index in range(len(dataset)):
            label = dataset[example_index, 0]
            training_set = dataset[example_index, 1:]
            prediction = np.dot(np.transpose(weights), training_set) + bias
            # for weight_index in range(len(weights)):
            #    prediction = prediction + weights[weight_index] * training_set[weight_index]
            # prediction = prediction + bias
            prediction = label * prediction
            if prediction <= 0:
                updates = updates + 1
                # for weight_update_index in range(len(weights)):
                #    weights[weight_update_index] = weights[weight_update_index] + learning_rate * label * training_set[weight_update_index]
                weights = weights + learning_rate * label * training_set
                bias = bias + learning_rate * label
            else:
                accuracy = accuracy + 1
        accuracy = accuracy / len(dataset)
        dict_accuracies[epoch_no] = accuracy
    # print("End of simple perceptron: ")
    return weights, bias, updates, dict_accuracies


def averaged_perceptron(dataset, epochs_to_train_on, learning_rate):
    weights = np.random.uniform(-0.01, 0.01, size=np.shape(dataset)[1] - 1)
    average_weight_vector = np.zeros([dataset.shape[1] - 1])
    averaged_bias = 0
    bias = np.random.uniform(-0.01, 0.01)
    updates = 0
    accuracy = 0
    dict_accuracies = {}

    for epoch_no in range(epochs_to_train_on):
        np.random.shuffle(dataset)
        for example_index in range(len(dataset)):
            label = dataset[example_index, 0]
            training_set = dataset[example_index, 1:]
            prediction = np.dot(np.transpose(weights), training_set) + bias
            # for weight_index in range(len(weights)):
            #    prediction = prediction + weights[weight_index] * training_set[weight_index]
            # prediction = prediction + bias
            prediction = label * prediction
            if prediction < 0:
                updates = updates + 1
                # for weight_update_index in range(len(weights)):
                #    weights[weight_update_index] = weights[weight_update_index] + learning_rate * label * training_set[weight_update_index]
                weights = weights + learning_rate * label * training_set
                bias = bias + learning_rate * label
            else:
                accuracy = accuracy + 1
            average_weight_vector = average_weight_vector + weights
            # for average_weight_index in range(len(average_weight_vector)):
            #    average_weight_vector[average_weight_index] = average_weight_vector[average_weight_index] + weights[average_weight_index]
            averaged_bias = averaged_bias + bias
        accuracy = accuracy / len(dataset)
        dict_accuracies[epoch_no] = accuracy
    return average_weight_vector, averaged_bias, updates, dict_accuracies


'''def write_to_file(df_to_write, file_name):
    f = open("./preprocessed_data/"+file_name, "w")
    f.write(str(df_to_write.to_csv()))
    f.write("\n")
    f.close()'''


if __name__ == '__main__':
    # df_training_bow = store_data('./project_data/data/bag-of-words/bow.train.libsvm')
    # df_testing_bow = store_data('./project_data/data/bag-of-words/bow.test.libsvm')
    # df_eval_bow = store_data('./project_data/data/bag-of-words/bow.eval.anon.libsvm')
    if os.path.exists("./preprocessed_data/training_tfidf.csv") and os.path.exists("./preprocessed_data/testing_tfidf.csv"):
        training_numpy = np.genfromtxt("./preprocessed_data/training_tfidf.csv", delimiter=',')
        training_numpy = np.delete(training_numpy, 0, 0)  # delete first row
        training_numpy = np.delete(training_numpy, 0, 1)  # delete first column
        df_training_tfidf = pd.DataFrame(data=training_numpy)

        testing_numpy = np.genfromtxt("./preprocessed_data/testing_tfidf.csv", delimiter=',')
        testing_numpy = np.delete(testing_numpy, 0, 0)  # delete first row
        testing_numpy = np.delete(testing_numpy, 0, 1)  # delete first column
        df_testing_tfidf = pd.DataFrame(data=testing_numpy)
    else:
        df_training_tfidf = utils.store_data('./project_data/data/tfidf/tfidf.train.libsvm', 10001)
        utils.write_to_file(df_training_tfidf, "training_tfidf.csv")
        df_testing_tfidf = utils.store_data('./project_data/data/tfidf/tfidf.test.libsvm', 10001)
        utils.write_to_file(df_testing_tfidf, "testing_tfidf.csv")
    df_eval_tfidf = utils.store_data('./project_data/data/tfidf/tfidf.eval.anon.libsvm', 10001)

    df_training_numpy = df_training_tfidf.to_numpy()
    folds = np.split(df_training_numpy, 5)

    learning_rate = [1, 0.1, 0.01]

    # First submission
    print("_______________Variation 1: Simple Perceptron_______________")
    # Cross validation below gives a learning rate of 1. Uncomment it if you want to see it
    '''best_simple_perceptron_hyper_parameter = simple_perceptron_n_cross_validation([pd.DataFrame(folds[0]),
                                                                                   pd.DataFrame(folds[1]),
                                                                                   pd.DataFrame(folds[2]),
                                                                                   pd.DataFrame(folds[3]),
                                                                                   pd.DataFrame(folds[4])],
                                                                                  0, learning_rate=learning_rate)'''
    # print("Learning rate is: ")
    # print(best_simple_perceptron_hyper_parameter)
    best_simple_perceptron_hyper_parameter = 1
    weights, bias, updates, dict_accuracies = simple_perceptron(df_training_tfidf.to_numpy(), epochs_to_train_on,
                                                                best_simple_perceptron_hyper_parameter)
    simple_perceptron_training_accuracy = evaluate_accuracy(weights, bias, df_training_tfidf)
    simple_perceptron_testing_accuracy = evaluate_accuracy(weights, bias, df_testing_tfidf)
    print("Training accuracy is: ")
    print(simple_perceptron_training_accuracy)
    print("Testing accuracy is: ")
    print(simple_perceptron_testing_accuracy)
    eval_results = evaluate_dataset(weights, bias, df_eval_tfidf, "variation_simple_perceptron.csv")
    print("_______________End of Simple Perceptron_______________")

    # Second submission
    print("_______________Variation 2: Averaged Perceptron_______________")
    # Cross validation below gives a learning rate of 0.1. Uncomment it if you want to see it
    '''best_averaged_perceptron_learning_rate_hyper_parameter = simple_perceptron_n_cross_validation(
                                                                                        [pd.DataFrame(folds[0]),
                                                                                         pd.DataFrame(folds[1]),
                                                                                         pd.DataFrame(folds[2]),
                                                                                         pd.DataFrame(folds[3]),
                                                                                         pd.DataFrame(folds[4])],
                                                                                        2, learning_rate=learning_rate)'''
    best_averaged_perceptron_learning_rate_hyper_parameter = 0.1
    average_weight_vector, averaged_bias, updates, dict_accuracies = averaged_perceptron(df_training_tfidf.to_numpy(),
                                                                                         epochs_to_train_on,
                                                                                         best_averaged_perceptron_learning_rate_hyper_parameter)

    averaged_perceptron_learning_training_accuracy = evaluate_accuracy(average_weight_vector, averaged_bias,
                                                                       df_training_tfidf)
    averaged_perceptron_learning_testing_accuracy = evaluate_accuracy(average_weight_vector, averaged_bias, df_testing_tfidf)
    print("Training accuracy is: ")
    print(averaged_perceptron_learning_training_accuracy)
    print("Testing accuracy is: ")
    print(averaged_perceptron_learning_testing_accuracy)
    # plot_learning_graph(averaged_perceptron_learning_rate_weights[2], "Averaged Perceptron")
    eval_results = evaluate_dataset(average_weight_vector, averaged_bias, df_eval_tfidf, "variation_averaged_perceptron.csv")
    print("_______________End of Averaged Perceptron_______________")

