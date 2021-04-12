# Name: Paarth Lakhani
# u0936913

import pandas as pd
import numpy as np
import math
from TreeNode import TreeNode

global_counter = 0


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
        expected_entropy = expected_entropy + (possible_value_freq/df.shape[0]) * possible_value_entropy

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
            part_of_gini += proportion_of_labels*(1-proportion_of_labels)
            # index is -1 and 1
        expected_gini_proportion = total_number_of_examples_for_key/df.shape[0]
        expected_gini += expected_gini_proportion*part_of_gini
    dataset_gini = cal_dataset_gini_index(df)
    info_gain = dataset_gini - expected_gini
    return info_gain


def cal_dataset_entropy(df):
    df_column = df[0]
    column_counts = df_column.value_counts()
    #print(column_counts.iloc[0])
    #print(column_counts.iloc[1])
    minus1_count = column_counts.iloc[0]
    plus1_count = column_counts.iloc[1]
    minus1_p = minus1_count / (minus1_count + plus1_count)
    plus1_p = plus1_count / (minus1_count + plus1_count)
    return -1 * plus1_p * math.log2(plus1_p) - minus1_p * math.log2(minus1_p)


def cal_dataset_gini_index(df):
    gini_index = 0
    for index, value in df[0].value_counts().items():
        proportion = value/df.shape[0]
        gini_index += proportion*(1-proportion)
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
    #print(selected_feature_list)
    #print(len(selected_feature_list))
    #print(list.index(selected_feature_list, 3))

    for line in lines:
        feature_vector = np.zeros(len(selected_feature_list) + 1)
        one_data_set = line.split()
        feature_vector[0] = int(one_data_set[0])
        one_data_set = one_data_set[1:]
        for attribute_value in one_data_set:
            split_attribute_value = attribute_value.split(":") # 3:4; feature = 3; freq = 4
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
            print(subset_examples.shape)
            print(subset_examples.shape[0])
            print(subset_examples.shape[1])
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
        predicted_label = predict_label(model, dataset.iloc[instance])
        if predicted_label == dataset.iloc[instance][0]:
            accuracy += 1
    accuracy_percent = accuracy/dataset.shape[0]*100
    print("Accuracy evaluated:\n")
    return accuracy_percent


def evaluate_dataset(model, dataset, filename):
    print("Evaluating blind dataset")
    f = open(filename, "w")
    f.write("example_id,label")
    f.write("\n")
    for instance in range(0, dataset.shape[0]):
        predicted_label = predict_label(model, dataset.iloc[instance])
        dataset.iloc[instance][0] = predicted_label
        f.write(str(instance) + "," + str(predicted_label) + "\n")
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
            root_node_fold = id3_depth_hyper_parameter(training_fold_df, training_fold_df.columns.values.tolist(), depth)
            testing_accuracy_depth += evaluate_accuracy(root_node_fold, testing_fold_df)
        testing_accuracy_depth = testing_accuracy_depth/len(list_of_folds)
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
