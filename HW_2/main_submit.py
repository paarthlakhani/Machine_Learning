# Name: Paarth Lakhani
# u0936913

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

random.seed(42)
cross_validation_epochs = 10
epochs_to_train_on = 20


def store_data(path):
    df = pd.DataFrame()
    file1 = open(path, 'r')
    lines = file1.readlines()

    for line in lines:
        feature_vector = np.zeros([207])
        one_data_set = line.split()
        feature_vector[0] = one_data_set[0]
        one_data_set = one_data_set[1:]
        for attribute_value in one_data_set:
            split_attribute_value = attribute_value.split(":")
            feature_vector[int(split_attribute_value[0])] = split_attribute_value[1]
        series = pd.Series(feature_vector)
        df = df.append(series, ignore_index=True)
    return df


def margin_perceptron(dataset, epochs_to_train_on, learning_rate, margin):
    weights = np.random.uniform(-0.01, 0.01, size=np.shape(dataset)[1] - 1)
    bias = np.random.uniform(-0.01, 0.01)
    updates = 0
    accuracy = 0
    dict_accuuracies = {}

    for epoch_no in range(epochs_to_train_on):
        np.random.shuffle(dataset)
        for example_index in range(len(dataset)):
            label = dataset[example_index, 0]
            training_set = dataset[example_index, 1:]
            prediction = 0
            for weight_index in range(len(weights)):
                prediction = prediction + weights[weight_index] * training_set[weight_index]
            prediction = prediction + bias
            prediction = label * prediction
            if prediction < margin:
                updates = updates + 1
                for weight_update_index in range(len(weights)):
                    weights[weight_update_index] = weights[weight_update_index] + learning_rate * label * training_set[weight_update_index]
                bias = bias + learning_rate * label
            else:
                accuracy = accuracy + 1
        accuracy = accuracy / len(dataset)
        dict_accuuracies[epoch_no] = accuracy
    return np.append(weights, bias), updates, dict_accuuracies


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
            prediction = 0
            for weight_index in range(len(weights)):
                prediction = prediction + weights[weight_index] * training_set[weight_index]
            prediction = prediction + bias
            prediction = label * prediction
            if prediction < 0:
                updates = updates + 1
                for weight_update_index in range(len(weights)):
                    weights[weight_update_index] = weights[weight_update_index] + learning_rate * label * training_set[weight_update_index]
                bias = bias + learning_rate * label
            else:
                accuracy = accuracy + 1
            for average_weight_index in range(len(average_weight_vector)):
                average_weight_vector[average_weight_index] = average_weight_vector[average_weight_index] + weights[average_weight_index]
            averaged_bias = averaged_bias + bias
        accuracy = accuracy / len(dataset)
        dict_accuracies[epoch_no] = accuracy
    return np.append(average_weight_vector, averaged_bias), updates, dict_accuracies


def simple_perceptron_decaying_learning_rate(dataset, epochs_to_train_on, learning_rate):
    weights = np.random.uniform(-0.01, 0.01, size=np.shape(dataset)[1] - 1)
    bias = np.random.uniform(-0.01, 0.01)
    t = -1
    updates = 0
    accuracy = 0
    dict_accuracies = {}

    for epoch_no in range(epochs_to_train_on):
        t = t + 1
        learning_rate = learning_rate / (1 + t)
        np.random.shuffle(dataset)
        for example_index in range(len(dataset)):
            label = dataset[example_index, 0]
            training_set = dataset[example_index, 1:]
            prediction = 0
            for weight_index in range(len(weights)):
                prediction = prediction + weights[weight_index] * training_set[weight_index]
            prediction = prediction + bias
            prediction = label * prediction
            if prediction < 0:
                updates = updates + 1
                for weight_update_index in range(len(weights)):
                    weights[weight_update_index] = weights[weight_update_index] + learning_rate * label * training_set[weight_update_index]
                bias = bias + learning_rate * label
            else:
                accuracy = accuracy + 1
        accuracy = accuracy / len(dataset)
        dict_accuracies[epoch_no] = accuracy
    return np.append(weights, bias), updates, dict_accuracies


def simple_perceptron(dataset, epochs_to_train_on, learning_rate):
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
            prediction = 0
            for weight_index in range(len(weights)):
                prediction = prediction + weights[weight_index] * training_set[weight_index]
            prediction = prediction + bias
            prediction = label * prediction
            if prediction < 0:
                updates = updates + 1
                for weight_update_index in range(len(weights)):
                    weights[weight_update_index] = weights[weight_update_index] + learning_rate * label * training_set[weight_update_index]
                bias = bias + learning_rate * label
            else:
                accuracy = accuracy + 1
        accuracy = accuracy / len(dataset)
        dict_accuracies[epoch_no] = accuracy
    return np.append(weights, bias), updates, dict_accuracies


def simple_perceptron_majority_baseline(dataset, epochs_to_train_on, learning_rate):
    weights = np.random.uniform(-0.01, 0.01, size=dataset.shape[1] - 1)
    bias = np.random.uniform(-0.01, 0.01)
    if dataset[0].value_counts().iloc[0] > dataset[0].value_counts().iloc[1]:
        prediction = 1
    else:
        prediction = -1
    updates = 0
    accuracy = 0
    dict_accuracies = {}
    dataset = dataset.to_numpy()
    for epoch_no in range(epochs_to_train_on):
        np.random.shuffle(dataset)
        for example_index in range(len(dataset)):
            label = dataset[example_index, 0]
            training_set = dataset[example_index, 1:]
            if prediction != label:
                updates = updates + 1
                for weight_update_index in range(len(weights)):
                    weights[weight_update_index] = weights[weight_update_index] + learning_rate * label * training_set[weight_update_index]
                bias = bias + learning_rate * label
            else:
                accuracy = accuracy + 1
        accuracy = accuracy / len(dataset)
        dict_accuracies[epoch_no] = accuracy
    return np.append(weights, bias), updates, dict_accuracies


def predict_label(model, example):
    prediction = 0
    for weight in range(0, len(model) - 1):  # model length 207; 0...206 (0-205); example length: 207; 1-206
        prediction = prediction + model[weight] * example[weight + 1]
    prediction = prediction + model[len(model) - 1]  # Adding bias
    if prediction <= 0:
        return -1  # negative label
    else:
        return 1  # positive label


def evaluate_accuracy(model, dataset):
    accuracy = 0
    for instance in range(0, dataset.shape[0]):
        predicted_label = predict_label(model, dataset.iloc[instance])
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
                weights_from_fold = np.zeros([1, training_fold_df.shape[1] - 1])
                if perceptron_variant == 0:
                    weights_from_fold = simple_perceptron(training_fold_df.to_numpy(), cross_validation_epochs, combination)
                elif perceptron_variant == 1:
                    weights_from_fold = simple_perceptron_decaying_learning_rate(training_fold_df.to_numpy(), cross_validation_epochs, combination)
                elif perceptron_variant == 2:
                    weights_from_fold = averaged_perceptron(training_fold_df.to_numpy(), cross_validation_epochs, combination)
                elif perceptron_variant == 3:
                    weights_from_fold = simple_perceptron_majority_baseline(training_fold_df, cross_validation_epochs, combination)
                testing_accuracy_learning += evaluate_accuracy(weights_from_fold[0], testing_fold_df)
            testing_accuracy_learning = testing_accuracy_learning/len(list_of_folds)
            if max_accuracy_learning < testing_accuracy_learning:
                max_accuracy_learning = testing_accuracy_learning
                learning_rate = combination
    print("Best learning rate is: " + str(learning_rate))
    print("Cross validation accuracy for the best hyper parameters: " + str(max_accuracy_learning))
    return learning_rate


def margin_perceptron_n_cross_validation(list_of_folds, learning_rates, margins):
    max_accuracy_learning = -1
    max_margin = -1
    max_learning_rate = -1
    for learning_rate in learning_rates:
        for margin in margins:
            testing_accuracy_learning = 0
            for fold_no in range(len(list_of_folds)):
                training_fold_list = []
                testing_fold_df = list_of_folds[fold_no]
                for train_fold_no in range(len(list_of_folds)):
                    if train_fold_no != fold_no:
                        training_fold_list.append(list_of_folds[train_fold_no])
                training_fold_df = pd.concat(training_fold_list, ignore_index=True)
                margin_weights_from_fold = margin_perceptron(training_fold_df.to_numpy(), cross_validation_epochs, learning_rate, margin)
                testing_accuracy_learning += evaluate_accuracy(margin_weights_from_fold[0], testing_fold_df)
            testing_accuracy_learning = testing_accuracy_learning/len(list_of_folds)
            if max_accuracy_learning < testing_accuracy_learning:
                max_accuracy_learning = testing_accuracy_learning
                max_learning_rate = learning_rate
                max_margin = margin
    print("Best learning rate: " + str(max_learning_rate) + "; Best margin: " + str(max_margin))
    print("Cross validation accuracy for the best hyper parameters: " + str(max_accuracy_learning))
    return max_learning_rate, max_margin


def plot_learning_graph(epoch_accuracies, graph_name):
    epoches = []
    accuracies = []
    for key, value in epoch_accuracies.items():
        epoches.append(key)
        accuracies.append(value)
    plt.plot(epoches, accuracies)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(graph_name)
    plt.show()


if __name__ == '__main__':
    df_training = store_data('./data/libSVM-format/train')
    df_testing = store_data('./data/libSVM-format/test')
    df_fold1 = store_data('./data/libSVM-format/CVfolds/fold1')
    df_fold2 = store_data('./data/libSVM-format/CVfolds/fold2')
    df_fold3 = store_data('./data/libSVM-format/CVfolds/fold3')
    df_fold4 = store_data('./data/libSVM-format/CVfolds/fold4')
    df_fold5 = store_data('./data/libSVM-format/CVfolds/fold5')

    print("_______________Majority baseline Perceptron_______________")
    best_majority_perceptron_hyper_parameter = simple_perceptron_n_cross_validation([df_fold1, df_fold2, df_fold3, df_fold4, df_fold5], 3, learning_rate=[1, 0.1, 0.01])
    majority_baseline_perception_weights = simple_perceptron_majority_baseline(df_training, epochs_to_train_on, best_majority_perceptron_hyper_parameter)
    majority_baseline_training_updates = majority_baseline_perception_weights[1]
    majority_baseline_training_accuracy = evaluate_accuracy(majority_baseline_perception_weights[0], df_training)
    print("Training accuracy is: " + str(majority_baseline_training_accuracy))
    majority_baseline_testing_accuracy = evaluate_accuracy(majority_baseline_perception_weights[0], df_testing)
    print("Testing accuracy is: " + str(majority_baseline_testing_accuracy))
    print("_______________End of Majority baseline Perceptron_______________")

    print("_______________Simple Perceptron_______________")
    best_simple_perceptron_hyper_parameter = simple_perceptron_n_cross_validation([df_fold1, df_fold2, df_fold3, df_fold4, df_fold5], 0, learning_rate=[1, 0.1, 0.01])
    simple_perceptron_weights = simple_perceptron(df_training.to_numpy(), epochs_to_train_on, best_simple_perceptron_hyper_parameter)
    simple_perceptron_updates = simple_perceptron_weights[1]
    simple_perceptron_training_accuracy = evaluate_accuracy(simple_perceptron_weights[0], df_training)
    simple_perceptron_testing_accuracy = evaluate_accuracy(simple_perceptron_weights[0], df_testing)
    # plot_learning_graph(simple_perception_weights[2], "Simple Perceptron")
    print("_______________End of Simple Perceptron_______________")

    print("_______________Simple Perceptron decaying the learning rate_______________")
    best_simple_perceptron_decaying_learning_rate_hyper_parameter = simple_perceptron_n_cross_validation([df_fold1, df_fold2, df_fold3, df_fold4, df_fold5], 1, learning_rate=[1, 0.1, 0.01])
    simple_perception_decaying_learning_rate_weights = simple_perceptron_decaying_learning_rate(df_training.to_numpy(), epochs_to_train_on, best_simple_perceptron_decaying_learning_rate_hyper_parameter)
    simple_perceptron_decaying_learning_rate_updates = simple_perception_decaying_learning_rate_weights[1]
    simple_perceptron_decaying_learning_rate_training_accuracy = evaluate_accuracy(simple_perception_decaying_learning_rate_weights[0], df_training)
    simple_perceptron_decaying_learning_rate_testing_accuracy = evaluate_accuracy(simple_perception_decaying_learning_rate_weights[0], df_testing)
    #plot_learning_graph(simple_perception_decaying_learning_rate_weights[2], "Simple Perceptron decaying the learning rate")
    print("_______________End of Simple Perceptron decaying the learning rate_______________")

    print("_______________Averaged Perceptron_______________")
    best_averaged_perceptron_learning_rate_hyper_parameter = simple_perceptron_n_cross_validation([df_fold1, df_fold2, df_fold3, df_fold4, df_fold5], 2, learning_rate=[1, 0.1, 0.01])
    averaged_perceptron_learning_rate_weights = averaged_perceptron(df_training.to_numpy(), epochs_to_train_on, best_averaged_perceptron_learning_rate_hyper_parameter)
    averaged_perceptron_learning_rate_updates = averaged_perceptron_learning_rate_weights[1]
    averaged_perceptron_learning_training_accuracy = evaluate_accuracy(averaged_perceptron_learning_rate_weights[0], df_training)
    averaged_perceptron_learning_testing_accuracy = evaluate_accuracy(averaged_perceptron_learning_rate_weights[0], df_testing)
    # plot_learning_graph(averaged_perceptron_learning_rate_weights[2], "Averaged Perceptron")
    print("_______________End of Averaged Perceptron_______________")

    print("_______________Margin Perceptron_______________")
    margin_perceptron_hyper_parameters = margin_perceptron_n_cross_validation([df_fold1, df_fold2, df_fold3, df_fold4, df_fold5], [1, 0.1, 0.01], [1, 0.1, 0.01])
    margin_perceptron_weights = margin_perceptron(df_training.to_numpy(), epochs_to_train_on, margin_perceptron_hyper_parameters[0], margin_perceptron_hyper_parameters[1])
    margin_perceptron_updates = margin_perceptron_weights[1]
    margin_perceptron_training_accuracy = evaluate_accuracy(margin_perceptron_weights[0], df_training)
    margin_perceptron_testing_accuracy = evaluate_accuracy(margin_perceptron_weights[0], df_testing)
    # plot_learning_graph(margin_perceptron_weights[2], "Margin Perceptron")
    print("_______________End of Margin Perceptron_______________")

    print("_______________Total updates_______________")
    print("Simple Perceptron:" + str(simple_perceptron_updates))
    print("Simple Perceptron decaying the learning rate: " + str(simple_perceptron_decaying_learning_rate_updates))
    print("Averaged Perceptron: " + str(averaged_perceptron_learning_rate_updates))
    print("Margin Perceptron: " + str(margin_perceptron_updates))
    print("_______________End of Total updates_______________")

    print("_______________Accuracy on training set_______________")
    print("Simple Perceptron:" + str(simple_perceptron_training_accuracy))
    print("Simple Perceptron decaying the learning rate: " + str(simple_perceptron_decaying_learning_rate_training_accuracy))
    print("Averaged Perceptron: " + str(averaged_perceptron_learning_training_accuracy))
    print("Margin Perceptron: " + str(margin_perceptron_training_accuracy))
    print("_______________End of Accuracy on training set_______________")

    print("_______________Accuracy on testing set_______________")
    print("Simple Perceptron:" + str(simple_perceptron_testing_accuracy))
    print("Simple Perceptron decaying the learning rate: " + str(simple_perceptron_decaying_learning_rate_testing_accuracy))
    print("Averaged Perceptron: " + str(averaged_perceptron_learning_testing_accuracy))
    print("Margin Perceptron: " + str(margin_perceptron_testing_accuracy))
    print("_______________End of Accuracy on testing set_______________")
