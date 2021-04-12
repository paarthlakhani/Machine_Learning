# Name: Paarth Lakhani
# u0936913

import pandas as pd
import numpy as np
import math
import re
from TreeNode import TreeNode
import common_utils as utils
import os

global_counter = 0
victim_genders_list = []
offence_category_list = []
offence_subcategory_list = []
units = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
                 "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]

tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
hundreds_place = ["hundred", "thousand", "million", "billion", "trillion"]

num_words = {}
num_words["and"] = (1, 0)
for idx, word in enumerate(units):
    num_words[word] = (1, idx)
for idx, word in enumerate(tens):
    num_words[word] = (1, idx * 10)
for idx, word in enumerate(hundreds_place):
    num_words[word] = (10 ** (idx * 3 or 2), 0)


def max_info_gain_entropy(df, attributes):
    max_info_gain_value = -1
    feature = 0
    for i in attributes:
        if i != 0:
            info_gain_attribute = information_gain_entropy(i, df)
            if info_gain_attribute > max_info_gain_value:
                max_info_gain_value = info_gain_attribute
                feature = i

    return feature, max_info_gain_value


def information_gain_entropy(attribute_no, df):
    feature_columns_groups = df.groupby(attribute_no)
    expected_entropy = 0
    for possible_value, possible_value_freq in df[attribute_no].value_counts().items():
        possible_value_entropy = 0
        for sample_label, sample_label_freq in feature_columns_groups.get_group(possible_value)[0].value_counts().items():
            proportion = sample_label_freq / possible_value_freq
            possible_value_entropy = possible_value_entropy + -1 * proportion * math.log2(proportion)
        expected_entropy = expected_entropy + (possible_value_freq / df.shape[0]) * possible_value_entropy

    dataset_entropy = cal_dataset_entropy(df)
    info_gain = dataset_entropy - expected_entropy
    return info_gain


def max_info_gain_gini(df, attributes):
    max_info_gain_value = -1
    feature = 0
    for i in attributes:
        if i != 0:
            info_gain_attribute = information_gain_gini(i, df)
            if info_gain_attribute > max_info_gain_value:
                max_info_gain_value = info_gain_attribute
                feature = i

    return feature, max_info_gain_value


def information_gain_gini(attribute_no, df):
    feature_columns_group = df.groupby(attribute_no)
    expected_gini = 0
    for key in feature_columns_group.groups.keys():
        total_number_of_examples_for_key = feature_columns_group.get_group(key).shape[0]
        part_of_gini = 0
        for index, value in feature_columns_group.get_group(key)[0].value_counts().items():
            proportion_of_labels = value / total_number_of_examples_for_key
            part_of_gini += proportion_of_labels * (1 - proportion_of_labels)
            # index is -1 and 1
        expected_gini_proportion = total_number_of_examples_for_key / df.shape[0]
        expected_gini += expected_gini_proportion * part_of_gini
    dataset_gini = cal_dataset_gini_index(df)
    info_gain = dataset_gini - expected_gini
    return info_gain


def cal_dataset_entropy(df):
    df_column = df[0]
    column_counts = df_column.value_counts()
    # print(column_counts.iloc[0])
    # print(column_counts.iloc[1])
    minus1_count = column_counts.iloc[0]
    plus1_count = column_counts.iloc[1]
    minus1_p = minus1_count / (minus1_count + plus1_count)
    plus1_p = plus1_count / (minus1_count + plus1_count)
    return -1 * plus1_p * math.log2(plus1_p) - minus1_p * math.log2(minus1_p)


def cal_dataset_gini_index(df):
    gini_index = 0
    for index, value in df[0].value_counts().items():
        proportion = value / df.shape[0]
        gini_index += proportion * (1 - proportion)
    return gini_index


def feature_selection(path, selected_feature_set):
    print("Inside feature selection:\n")
    file1 = open(path, 'r')
    lines = file1.readlines()
    feature_freq_dict = {}
    feature_occurances_dataset = 4
    # selected_feature_set = set()
    feature_list = []

    if len(selected_feature_set) == 0:
        for line in lines:
            one_data_set = line.split()
            one_data_set = one_data_set[1:]
            for attribute_value in one_data_set:
                split_attribute_value = attribute_value.split(":")
                feature_freq_dict[int(split_attribute_value[0])] = int(split_attribute_value[1])

        for feature, feature_freq in feature_freq_dict.items():
            if feature_freq > feature_occurances_dataset:
                selected_feature_set.add(feature)

    selected_feature_list = list(sorted(selected_feature_set))
    # print(selected_feature_list)
    # print(len(selected_feature_list))
    # print(list.index(selected_feature_list, 3))

    for line in lines:
        feature_vector = np.zeros(len(selected_feature_list) + 1)
        one_data_set = line.split()
        feature_vector[0] = int(one_data_set[0])
        one_data_set = one_data_set[1:]
        for attribute_value in one_data_set:
            split_attribute_value = attribute_value.split(":")  # 3:4; feature = 3; freq = 4
            if int(split_attribute_value[0]) in selected_feature_list:
                mapped_index = list.index(selected_feature_list, int(split_attribute_value[0]))
                feature_vector[mapped_index] = int(split_attribute_value[1])
        feature_list.append(feature_vector)
    df = pd.DataFrame(feature_list)
    print(df.shape)
    print("Feature selection complete:\n")
    return df, selected_feature_set


def store_data(path):
    file1 = open(path, 'r')
    lines = file1.readlines()
    feature_list = []

    for line in lines:
        feature_vector = np.zeros([10001])
        # feature_vector = np.zeros([301]) # for glove
        one_data_set = line.split()
        feature_vector[0] = int(one_data_set[0])
        one_data_set = one_data_set[1:]
        for attribute_value in one_data_set:
            split_attribute_value = attribute_value.split(":")
            feature_vector[int(split_attribute_value[0])] = int(split_attribute_value[1])
        feature_list.append(feature_vector)
    df = pd.DataFrame(feature_list)
    return df


def id3(df, attributes, entropy_gini_index=0):
    #print(df.groupby(0))
    #print(df.groupby(0).groups)
    #print(df.groupby(0).groups.keys())
    labels_grouping_data_frame = pd.DataFrame(df.groupby(0).groups.keys())
    if labels_grouping_data_frame.shape[0] == 1:
        return TreeNode(labels_grouping_data_frame.iloc[0][0])
    else:
        if entropy_gini_index == 0:
            feature_node = max_info_gain_entropy(df, attributes)
        else:
            feature_node = max_info_gain_gini(df, attributes)
        feature_number = feature_node[0]

        tree_node = TreeNode(feature_number)
        list_of_children = []
        feature_grouping = df.groupby(feature_number)
        for possible_value in feature_grouping.groups.keys():
            subset_examples = feature_grouping.get_group(possible_value)
            #print(subset_examples.shape)
            #print(subset_examples.shape[0])
            #print(subset_examples.shape[1])
            if subset_examples.shape[0] == 0:
                minus1_count = df[0].value_counts().iloc[0]
                plus1_count = df[0].value_counts().iloc[1]
                if minus1_count >= plus1_count:
                    return TreeNode(0)  # -1
                else:
                    return TreeNode(1)
            else:
                attributes_modify = attributes.copy()
                attributes_modify.remove(feature_number)
                child_node = id3(subset_examples, attributes_modify, entropy_gini_index)
                child_dict = {"input": possible_value, "child": child_node}
                list_of_children.append(child_dict)
        tree_node.list_of_children = list_of_children
        return tree_node


def predict_label(model, testing_instance):
    if model.list_of_children is None:
        return model.feature_number
    next_node = None
    feature_number = model.feature_number
    feature_value = testing_instance[feature_number]
    for child_dict in model.list_of_children:
        if child_dict.get('input') == feature_value:
            next_node = child_dict.get('child')
    if next_node is None:
        next_node = model.list_of_children[0].get('child')
    return predict_label(next_node, testing_instance)


def evaluate_accuracy(model, dataset):
    print("Evaluating accuracy:\n")
    accuracy = 0
    for instance in range(0, dataset.shape[0]):
        # print(instance)
        predicted_label = predict_label(model, dataset.iloc[instance])
        if predicted_label == dataset.iloc[instance][0]:
            accuracy += 1
    accuracy_percent = accuracy / dataset.shape[0] * 100
    print("Accuracy evaluated:\n")
    return accuracy_percent


def evaluate_dataset(model, dataset, filename):
    print("Evaluating blind dataset")
    f = open(filename, "w")
    f.write("example_id,label")
    f.write("\n")
    for instance in range(0, dataset.shape[0]):
        predicted_label = predict_label(model, dataset.iloc[instance])
        if predicted_label == -1:
            predicted_label = 0
        dataset.iloc[instance][0] = predicted_label
        f.write(str(instance) + "," + str(int(predicted_label)) + "\n")
    f.close()
    print("Blind dataset evaluated")
    return dataset.iloc[:, 0]


def tree_depth(model):
    if model.list_of_children is None:
        return 0
    depth_subtree = -1
    for child_dict in model.list_of_children:
        child_depth = tree_depth(child_dict.get('child'))
        if child_depth > depth_subtree:
            depth_subtree = child_depth
    return 1 + depth_subtree


def n_fold_cross_validation(list_of_folds, depths):
    max_depth = [-1, -1]
    for depth in depths:
        testing_accuracy_depth = 0
        for fold_no in range(len(list_of_folds)):
            training_fold_list = []
            testing_fold_df = list_of_folds[fold_no]
            for train_fold_no in range(len(list_of_folds)):
                if train_fold_no != fold_no:
                    training_fold_list.append(list_of_folds[train_fold_no])
            training_fold_df = pd.concat(training_fold_list, ignore_index=True)
            root_node_fold = id3_depth_hyper_parameter(training_fold_df, training_fold_df.columns.values.tolist(),
                                                       depth)
            testing_accuracy_depth += evaluate_accuracy(root_node_fold, testing_fold_df)
        testing_accuracy_depth = testing_accuracy_depth / len(list_of_folds)
        print("Depth: " + str(depth) + " Accuracy: " + str(testing_accuracy_depth))
        if max_depth[1] < testing_accuracy_depth:
            max_depth[0] = depth
            max_depth[1] = testing_accuracy_depth
    print("Max Depth is: ")
    print(max_depth)
    return max_depth


def id3_depth_hyper_parameter(df, attributes, depth):
    labels_grouping_data_frame = pd.DataFrame(df.groupby(0).groups.keys())
    if labels_grouping_data_frame.shape[0] == 1:
        return TreeNode(labels_grouping_data_frame.iloc[0][0])
    elif depth == 0:
        minus1_count = df[0].value_counts().iloc[0]
        plus1_count = df[0].value_counts().iloc[1]
        if minus1_count >= plus1_count:
            return TreeNode(-1)
        else:
            return TreeNode(+1)
    else:
        feature_node = max_info_gain_entropy(df, attributes)
        feature_number = feature_node[0]

        tree_node = TreeNode(feature_number)
        list_of_children = []
        feature_grouping = df.groupby(feature_number)
        for possible_value in feature_grouping.groups.keys():
            subset_examples = feature_grouping.get_group(possible_value)
            if subset_examples.shape[0] == 0:
                minus1_count = df[0].value_counts().iloc[0]
                plus1_count = df[0].value_counts().iloc[1]
                if minus1_count >= plus1_count:
                    return TreeNode(-1)
                else:
                    return TreeNode(+1)
            else:
                attributes_modify = attributes.copy()
                attributes_modify.remove(feature_number)
                child_node = id3_depth_hyper_parameter(subset_examples, attributes_modify, depth - 1)
                child_dict = {"input": possible_value, "child": child_node}
                list_of_children.append(child_dict)
        tree_node.list_of_children = list_of_children
        return tree_node


def find_number_in_text(value):
    if value == "not known":
        return 0
    elif value == "fourscore":
        return 80
    else:
        current = result = 0
        for text_number in value.split():
            if text_number not in num_words:
                try:
                    result = int(text_number)
                except:
                    pass
            else:
                scale, increment = num_words[text_number]
                current = current * scale + increment
                if scale > 100:
                    result += current
                    current = 0

        return result + current


def remove_non_numeric(value):
    non_numeric_strings = ["", "years", "year", "months", "month", "days", "day", "about",
                           "of", "age", "old", "his", "or", "and"]
    value_words = re.split('\(|\)|-| ', value)
    #print(value_words)

    sentence_without_non_numeric = " ".join([word for word in value_words if word not in non_numeric_strings])
    return sentence_without_non_numeric


def scale_defendant_age(actual_age):
    global scaled_age
    if actual_age == 0:
        scaled_age = 0
    elif 1 <= actual_age <= 20:
        scaled_age = 1
    elif 21 <= actual_age <= 40:
        scaled_age = 2
    elif 41 <= actual_age <= 60:
        scaled_age = 3
    elif 61 <= actual_age <= 80:
        scaled_age = 4
    elif 81 <= actual_age <= 100:
        scaled_age = 5
    return scaled_age


def discretize_defendant_gender(gender):
    if gender == "male":
        return 1
    elif gender == "female":
        return 2
    return 0


def get_discretized_age(actual_age):
    actual_age = actual_age.strip()
    actual_age = actual_age.lower()
    actual_age = remove_non_numeric(actual_age)
    try:
        age_number = int(actual_age)
        age_number = scale_defendant_age(age_number)
        return age_number
    except ValueError:
        discretized_age = find_number_in_text(actual_age)
        discretized_age = scale_defendant_age(discretized_age)
        return discretized_age


def discretize_victim_gender(victim_gender):
    try:
        return victim_genders_list.index(victim_gender) + 1
    except ValueError:
        return 0


def discretize_offence_category(offence_category):
    try:
        return offence_category_list.index(offence_category) + 1
    except ValueError:
        return 0


def discretize_offence_subcategory(offence_subcategory):
    try:
        return offence_subcategory_list.index(offence_subcategory) + 1
    except ValueError:
        return 0


def pre_process_data(path_to_data):
    misc_data = pd.read_csv(path_to_data, ',', header=0)

    discretized_dataset = []
    for index, example in misc_data.iterrows():
        discretized_row = np.zeros([6])
        #discretized_row = np.zeros([5])
        actual_age = example["defendant_age"]
        discretized_row[0] = get_discretized_age(actual_age)

        discretized_row[1] = discretize_defendant_gender(example["defendant_gender"])
        discretized_row[2] = example["num_victims"]
        discretized_row[3] = discretize_victim_gender(example["victim_genders"])
        discretized_row[4] = discretize_offence_category(example["offence_category"])
        #discretized_row[3] = discretize_offence_category(example["offence_category"])
        discretized_row[5] = discretize_offence_subcategory(example["offence_subcategory"])
        #discretized_row[4] = discretize_offence_subcategory(example["offence_subcategory"])

        discretized_dataset.append(discretized_row)
    df = pd.DataFrame(discretized_dataset)
    return df.to_numpy()


def create_mapping_for_features(training_data_path):
    training_data = pd.read_csv(training_data_path, ',', header=0)
    column_names = training_data.columns.values
    for column_name in column_names:
        grouping = training_data.groupby(column_name)
        possible_feature_values = grouping.groups.keys()

        if column_name == "victim_genders":
            global victim_genders_list
            victim_genders_list = list(possible_feature_values)

        if column_name == "offence_category":
            global offence_category_list
            offence_category_list = list(possible_feature_values)

        if column_name == "offence_subcategory":
            global offence_subcategory_list
            offence_subcategory_list = list(possible_feature_values)


def extract_labels(misc_numpy, label_dataset_path):
    #extract_labels(training_misc_numpy, './preprocessed_data/training_tfidf.csv')
    #extract_labels(testing_misc_numpy, './preprocessed_data/testing_tfidf.csv')
    if os.path.exists(label_dataset_path):
        label_dataset = np.genfromtxt(label_dataset_path, delimiter=',')
    else:
        if "training_tfidf.csv" in label_dataset_path:
            training_tfidf = utils.store_data('./project_data/data/tfidf/tfidf.train.libsvm', 10001)
            utils.write_to_file(training_tfidf, "training_tfidf.csv")
        elif "testing_tfidf.csv" in label_dataset_path:
            testing_tfidf = utils.store_data('./project_data/data/tfidf/tfidf.test.libsvm', 10001)
            utils.write_to_file(testing_tfidf, "testing_tfidf.csv")
        label_dataset = np.genfromtxt(label_dataset_path, delimiter=',')
    labels = label_dataset[1:, 1]
    misc_numpy = np.column_stack((labels, misc_numpy))
    return misc_numpy


def add_empty_col_beginning(eval_misc_numpy):
    ones_column = np.ones([eval_misc_numpy.shape[0], 1])
    eval_misc_numpy = np.column_stack((ones_column, eval_misc_numpy))
    return eval_misc_numpy


if __name__ == '__main__':
    if os.path.exists("./preprocessed_data/training_misc.csv") and os.path.exists("./preprocessed_data/testing_misc.csv") and os.path.exists("./preprocessed_data/eval_misc.csv"):
        training_misc_numpy_id3 = np.genfromtxt("./preprocessed_data/training_misc.csv", delimiter=',')
        training_misc_numpy_id3 = np.delete(training_misc_numpy_id3, 0, 0)  # delete first row
        training_misc_numpy_id3 = np.delete(training_misc_numpy_id3, 0, 1)  # delete first column
        training_misc_numpy_id3_pd = pd.DataFrame(data=training_misc_numpy_id3)
        # training_misc_numpy_id3_pd = pd.read_csv("./preprocessed_data/training_misc.csv", header=None)
        #training_misc_numpy_id3_pd.drop(training_misc_numpy_id3_pd.columns[[0]], axis=1, inplace=True)
        #training_misc_numpy_id3_pd.drop([0], axis=0, inplace=True)
        #training_misc_numpy_id3_pd = training_misc_numpy_id3_pd.reset_index(drop=True)
        testing_misc_numpy_id3 = np.genfromtxt("./preprocessed_data/testing_misc.csv", delimiter=',')
        testing_misc_numpy_id3 = np.delete(testing_misc_numpy_id3, 0, 0)  # delete first row
        testing_misc_numpy_id3 = np.delete(testing_misc_numpy_id3, 0, 1)  # delete first column
        testing_misc_numpy_id3_pd = pd.DataFrame(data=testing_misc_numpy_id3)

        eval_misc_numpy_id3 = np.genfromtxt("./preprocessed_data/eval_misc.csv", delimiter=',')
        eval_misc_numpy_id3 = np.delete(eval_misc_numpy_id3, 0, 0)  # delete first row
        eval_misc_numpy_id3 = np.delete(eval_misc_numpy_id3, 0, 1)  # delete first column
        eval_misc_numpy_id3_pd = pd.DataFrame(data=eval_misc_numpy_id3)
    else:
        create_mapping_for_features('./project_data/data/misc-attributes/misc-attributes-train.csv')
        training_misc_numpy = pre_process_data('./project_data/data/misc-attributes/misc-attributes-train.csv')
        training_misc_numpy = extract_labels(training_misc_numpy, './preprocessed_data/training_tfidf.csv')
        testing_misc_numpy = pre_process_data('./project_data/data/misc-attributes/misc-attributes-test.csv')
        testing_misc_numpy = extract_labels(testing_misc_numpy, './preprocessed_data/testing_tfidf.csv')
        eval_misc_numpy = pre_process_data('./project_data/data/misc-attributes/misc-attributes-eval.csv')
        eval_misc_numpy = add_empty_col_beginning(eval_misc_numpy)

        training_misc_numpy_id3_pd = pd.DataFrame(data=training_misc_numpy)
        testing_misc_numpy_id3_pd = pd.DataFrame(data=testing_misc_numpy)
        eval_misc_numpy_id3_pd = pd.DataFrame(data=eval_misc_numpy)
        utils.write_to_file(training_misc_numpy_id3_pd, "training_misc.csv")
        utils.write_to_file(testing_misc_numpy_id3_pd, "testing_misc.csv")
        utils.write_to_file(eval_misc_numpy_id3_pd, "eval_misc.csv")

    print(".........Using entropy........")
    print("Starting id3 algorithm\n")
    root_node_entropy = id3(training_misc_numpy_id3_pd, training_misc_numpy_id3_pd.columns.values.tolist(), 0)
    print("id3 algorithm finished executing\n")
    training_accuracy_entropy = evaluate_accuracy(root_node_entropy, training_misc_numpy_id3_pd)
    print("Training data accuracy: " + str(training_accuracy_entropy))
    testing_accuracy_entropy = evaluate_accuracy(root_node_entropy, testing_misc_numpy_id3_pd)
    print("Testing data accuracy: " + str(testing_accuracy_entropy))
    eval_results = evaluate_dataset(root_node_entropy, eval_misc_numpy_id3_pd, "variation_decision_trees.csv")
    print(".........End of using entropy........")

    #  df_testing_bow, selected_feature_set = feature_selection('./project_data/data/misc-attributes/misc-attributes-test.csv')
    #  df_eval_bow, selected_feature_set = feature_selection('./project_data/data/misc-attributes/misc-attributes-eval.csv')

    '''df_training_bow, selected_feature_set = feature_selection('./project_data/data/bag-of-words/bow.train.libsvm', set())
    df_testing_bow, selected_feature_set = feature_selection('./project_data/data/bag-of-words/bow.test.libsvm', selected_feature_set)
    df_eval_bow, selected_feature_set = feature_selection('./project_data/data/bag-of-words/bow.eval.anon.libsvm', selected_feature_set)
    '''
