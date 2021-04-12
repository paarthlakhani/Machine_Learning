# Name: Paarth Lakhani
# u0936913

import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import common_utils as utils
import os

random.seed(42)

def find_max_accuracy_weights(epoch_weight_accuracy_dict):
    final_weights = []
    accuracy = -1
    loss = -1
    epochs = -1
    for epoch, epoch_stat in epoch_weight_accuracy_dict.items():
        if epoch_stat[1] > accuracy:
            final_weights = epoch_stat[0]
            accuracy = epoch_stat[1]
            loss = epoch_stat[2]
            epochs = epoch_stat[3]
    return final_weights, accuracy, loss, epochs


def stochastic_sgd_svm(dataset, initial_learning_rate, loss_tradeoff):
    epochs = 0
    max_epochs = 100
    weights = np.zeros(dataset.shape[1] - 1)
    #prev_loss = float(np.Infinity)
    prev_loss = float('-inf')
    epsilon = 100
    epoch_weight_accuracy_dict = {}

    for epoch_number in range(max_epochs):
        epochs += 1
        learning_rate = float(initial_learning_rate / (1 + epochs))
        np.random.shuffle(dataset)
        for example_index in range(len(dataset)):
            label = dataset[example_index, 0]
            training_set = dataset[example_index, 1:]
            prediction = label * np.dot(np.transpose(weights), training_set)
            weights = (1 - learning_rate) * weights
            if prediction <= 1:
                weights = weights + learning_rate * loss_tradeoff * label * training_set
        per_epoch_loss = 0.5 * np.dot(np.transpose(weights), weights)
        for example_index in range(len(dataset)):
            label = dataset[example_index, 0]
            possible_hinge_loss = float(1 - label * np.dot(np.transpose(weights), dataset[example_index, 1:]))
            hinge_loss = loss_tradeoff * max([0, possible_hinge_loss])
            per_epoch_loss = per_epoch_loss + hinge_loss
        calc_epsilon = abs(prev_loss - per_epoch_loss)
        prev_loss = per_epoch_loss
        accuracy = evaluate_accuracy(weights, dataset)
        epoch_weight_accuracy_dict[epoch_number] = [weights, accuracy, prev_loss, epochs]
        if calc_epsilon < epsilon:
            break

    eff_weights, max_accuracy, loss, epochs = find_max_accuracy_weights(epoch_weight_accuracy_dict)
    return eff_weights, max_accuracy, loss, epochs, epoch_weight_accuracy_dict


def sgd_logistic_regression(dataset, initial_learning_rate, loss_tradeoff):
    #print("Learning_rate: " + str(initial_learning_rate))
    #print("loss_tradeoff: " + str(loss_tradeoff))
    # print("stochastic_sgd_svm begins...")
    epochs = 0
    max_epochs = 10
    weights = np.zeros(dataset.shape[1] - 1)
    #prev_loss = float(np.Infinity)
    prev_loss = float('-inf')
    epsilon = 100
    epoch_weight_accuracy_dict = {}

    for epoch_number in range(max_epochs):
        epochs += 1
        learning_rate = float(initial_learning_rate / (1 + epochs))
        np.random.shuffle(dataset)
        for example_index in range(len(dataset)):
            label = dataset[example_index, 0]
            training_set = dataset[example_index, 1:]
            try:
                SVM_objective_expo = math.exp(-1 * label * np.dot(np.transpose(weights), training_set))
                SVM_objective_p1 = (SVM_objective_expo * (-1 * label * training_set)) / (1 + SVM_objective_expo)
                SVM_objective_p2 = (2 * weights) / loss_tradeoff
                derivative_SVM_objective = SVM_objective_p1 + SVM_objective_p2
                weights = weights - learning_rate * derivative_SVM_objective
            except OverflowError:
                pass
        per_epoch_loss = (1 / loss_tradeoff) * np.dot(np.transpose(weights), weights)
        for example_index in range(len(dataset)):
            try:
                label = dataset[example_index, 0]
                example_loss = 1 + math.exp(-1 * label * np.dot(np.transpose(weights), dataset[example_index, 1:]))
                per_epoch_loss = per_epoch_loss + math.log(example_loss)
            except OverflowError:
                pass
        calc_epsilon = abs(prev_loss - per_epoch_loss)
        #print(prev_loss)
        # print("Calc epsilon is: " + str(calc_epsilon))
        prev_loss = per_epoch_loss
        accuracy = evaluate_accuracy(weights, dataset)
        epoch_weight_accuracy_dict[epoch_number] = [weights, accuracy, prev_loss, epochs]
        if calc_epsilon < epsilon:
            # print("I am breaking now")
            # print(epochs)
            break

    eff_weights, max_accuracy, loss, epochs = find_max_accuracy_weights(epoch_weight_accuracy_dict)
    # print("stochastic_sgd_svm ends...")
    return eff_weights, max_accuracy, loss, epochs, epoch_weight_accuracy_dict


def predict_label(weights, example):
    prediction = np.dot(np.transpose(weights), example)
    # prediction = prediction + model[1]  # Adding bias
    if prediction <= 0:
        return -1  # negative label
    else:
        return 1  # positive label


def evaluate_accuracy(weights, dataset):
    accuracy = 0

    for instance in range(0, dataset.shape[0]):
        # predicted_label = predict_label(weights, dataset.iloc[instance][1:])
        predicted_label = predict_label(weights, dataset[instance][1:])
        # if predicted_label == dataset.iloc[instance][0]:
        if predicted_label == dataset[instance][0]:
            accuracy += 1
    accuracy_percent = accuracy / dataset.shape[0] * 100
    return accuracy_percent


def stochastic_sgd_svm_n_cross_validation(list_of_folds, initial_learning_rates, loss_tradeoffs):
    print("N cross validation begins.....")
    max_accuracy = -1
    final_learning_rate = initial_learning_rates[0]
    final_loss_tradeoff = loss_tradeoffs[0]
    for learning_rate in initial_learning_rates:
        for loss_tradeoff in loss_tradeoffs:
            testing_accuracy = 0
            for test_fold_no in range(len(list_of_folds)):
                training_fold_list = []
                testing_fold = list_of_folds[test_fold_no]
                for train_fold_no in range(len(list_of_folds)):
                    if train_fold_no != test_fold_no:
                        training_fold_list.append(list_of_folds[train_fold_no])
                training_fold_dataset = np.concatenate(training_fold_list)
                weights, acccuracy, per_epoch_loss, epochs, epoch_weight_accuracy_dict = stochastic_sgd_svm(training_fold_dataset, learning_rate,
                                                                                loss_tradeoff)
                testing_accuracy += evaluate_accuracy(weights, testing_fold)
            testing_accuracy = testing_accuracy / len(list_of_folds)
            if max_accuracy < testing_accuracy:
                max_accuracy = testing_accuracy
                final_learning_rate = learning_rate
                final_loss_tradeoff = loss_tradeoff
    print("N cross validation ends.....")
    return final_learning_rate, final_loss_tradeoff, max_accuracy


def sgd_logistic_regression_n_cross_validation(list_of_folds, initial_learning_rates, loss_tradeoffs):
    # print("N cross validation begins.....")
    max_accuracy = -1
    final_learning_rate = initial_learning_rates[0]
    final_loss_tradeoff = loss_tradeoffs[0]
    for learning_rate in initial_learning_rates:
        for loss_tradeoff in loss_tradeoffs:
            testing_accuracy = 0
            for test_fold_no in range(len(list_of_folds)):
                training_fold_list = []
                testing_fold = list_of_folds[test_fold_no]
                for train_fold_no in range(len(list_of_folds)):
                    if train_fold_no != test_fold_no:
                        training_fold_list.append(list_of_folds[train_fold_no])
                training_fold_dataset = np.concatenate(training_fold_list)
                weights, acccuracy, per_epoch_loss, epochs, epoch_weight_accuracy_dict = sgd_logistic_regression(training_fold_dataset, learning_rate,
                                                                                loss_tradeoff)
                testing_accuracy += evaluate_accuracy(weights, testing_fold)
            testing_accuracy = testing_accuracy / len(list_of_folds)
            if max_accuracy < testing_accuracy:
                max_accuracy = testing_accuracy
                final_learning_rate = learning_rate
                final_loss_tradeoff = loss_tradeoff
    # print("N cross validation ends.....")
    return final_learning_rate, final_loss_tradeoff, max_accuracy


def plot_learning_graph(epoch_accuracies, graph_name):
# epoch_weight_accuracy_dict[epoch_number] = [weights, accuracy, prev_loss, epochs]
    epoches = []
    loss_function = []
    for key, value in epoch_accuracies.items():
        epoches.append(key)
        loss_function.append(value[2])
    plt.plot(epoches, loss_function)
    plt.xlabel("Epoch")
    plt.ylabel("Loss Function")
    plt.title(graph_name)
    plt.show()


if __name__ == '__main__':
    if os.path.exists("./preprocessed_data/training_tfidf.csv") and os.path.exists("./preprocessed_data/testing_tfidf.csv"):
        training_numpy = np.genfromtxt("./preprocessed_data/training_tfidf.csv", delimiter=',')
        training_numpy = np.delete(training_numpy, 0, 0)  # delete first row
        training_numpy = np.delete(training_numpy, 0, 1)  # delete first column
        training = pd.DataFrame(data=training_numpy)

        testing_numpy = np.genfromtxt("./preprocessed_data/testing_tfidf.csv", delimiter=',')
        testing_numpy = np.delete(testing_numpy, 0, 0)  # delete first row
        testing_numpy = np.delete(testing_numpy, 0, 1)  # delete first column
        test = pd.DataFrame(data=testing_numpy)
    else:
        training = utils.store_data('./project_data/data/tfidf/tfidf.train.libsvm', 10001)
        utils.write_to_file(training, "training_tfidf.csv")
        test = utils.store_data('./project_data/data/tfidf/tfidf.test.libsvm', 10001)
        utils.write_to_file(test, "testing_tfidf.csv")
    eval_dataset = utils.store_data('./project_data/data/tfidf/tfidf.eval.anon.libsvm', 10001)

    training_numpy = training.to_numpy()
    folds = np.split(training_numpy, 5)

    ###### SVM #######
    initial_learning_rates = [1, 0.1, 0.01, 0.001, 0.0001]
    loss_tradeoffs = [1000, 100, 10, 1, 0.1, 0.01]

    print("##### Support Vector Machine ##### ")
    '''best_learning_rate, best_loss_tradeoff, cross_validation_accuracy = stochastic_sgd_svm_n_cross_validation(
                                                                                    [folds[0],
                                                                                     folds[1],
                                                                                     folds[2],
                                                                                     folds[3],
                                                                                     folds[4]],
                                                                            initial_learning_rates, loss_tradeoffs)

    print("Best learning rate: " + str(best_learning_rate))
    print("Best loss trade off: " + str(best_loss_tradeoff))
    print("Cross validation accuracy: " + str(cross_validation_accuracy))'''

    print("Starting SVM SGD...")
    # Cross validation above gives a best learning rate of 0.01 and best loss trade off of 1000
    # Uncomment above if you want to see the cross validation
    best_learning_rate = 0.01
    best_loss_tradeoff = 1000
    weights, train_accuracy, loss, epochs, epoch_weight_accuracy_dict = stochastic_sgd_svm(training_numpy, best_learning_rate, best_loss_tradeoff)
    print("SVM SGD complete...")
    #plot_learning_graph(epoch_weight_accuracy_dict, "Support Vector Machine")
    test_accuracy = evaluate_accuracy(weights, test.to_numpy())

    print("Training accuracy: " + str(train_accuracy))
    print("Testing accuracy: " + str(test_accuracy))
    eval_results_svm = utils.evaluate_dataset(weights, 0, eval_dataset, "variation_svm.csv")
    print("##### End of Support Vector Machine ##### ")

    # Logistic Regression
    lg_initial_learning_rates = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
    # lg_loss_tradeoffs = [0.1, 1, 10, 100, 1000, 10000]
    lg_loss_tradeoffs = [1, 10, 100, 1000, 10000]

    print("##### Logistic Regression ##### ")

    '''best_learning_rate, best_loss_tradeoff, cross_validation_accuracy = sgd_logistic_regression_n_cross_validation(
                                                                                        [folds[0],
                                                                                         folds[1],
                                                                                         folds[2],
                                                                                         folds[3],
                                                                                         folds[4]],
                                                                            lg_initial_learning_rates, lg_loss_tradeoffs)

    print("Best learning rate: " + str(best_learning_rate))
    print("Best loss trade off: " + str(best_loss_tradeoff))
    print("Cross validation accuracy: " + str(cross_validation_accuracy))'''

    print("Starting logistic Regression...")
    # Cross validation above gives a best learning rate of 1 and best loss trade off of 10000
    # Uncomment above if you want to see the cross validation
    best_learning_rate = 1
    best_loss_tradeoff = 10000
    weights, train_accuracy, loss, epochs, epoch_weight_accuracy_dict = sgd_logistic_regression(training_numpy, best_learning_rate, best_loss_tradeoff)
    print("Logistic Regression complete...")
    #plot_learning_graph(epoch_weight_accuracy_dict, "Logistic Regression")
    test_accuracy = evaluate_accuracy(weights, test.to_numpy())

    print("Training accuracy: " + str(train_accuracy))
    print("Testing accuracy: " + str(test_accuracy))
    eval_results_lg = utils.evaluate_dataset(weights, 0, eval_dataset, "variation_logistic_regression.csv")
    print("##### End of Logistic Regression ##### ")
