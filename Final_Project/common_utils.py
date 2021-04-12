import pandas as pd
import numpy as np


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


def write_to_file(df_to_write, file_name):
    f = open("./preprocessed_data/"+file_name, "w")
    f.write(str(df_to_write.to_csv()))
    f.write("\n")
    f.close()


def store_data(path, num_features):
    file1 = open(path, 'r')
    lines = file1.readlines()
    feature_list = []

    for line in lines:
        feature_vector = np.zeros([num_features])  # 10001 for tfidf
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
    return df
