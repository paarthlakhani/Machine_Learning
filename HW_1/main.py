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
    #print("##################")
    #print("In information gain")
    #print("Feature number")
    #print(attribute_no)
    # feature_columns = df.iloc[0:, [0, attribute_no]]
    feature_one_count = 0
    if 1 in df[attribute_no].value_counts():
        feature_one_count = df[attribute_no].value_counts()[1]

    feature_zero_count = 0
    if 0 in df[attribute_no].value_counts():
        feature_zero_count = df[attribute_no].value_counts()[0]

    feature_columns_group = df.groupby(attribute_no)

    feature_one_entropy = 0
    if 1 in feature_columns_group.groups.keys():
        feature_one_label_minus1_calc = 0
        feature_one_label_plus1_calc = 0

        for index, value in feature_columns_group.get_group(1)[0].value_counts().items():
            if index == -1:
                feature_one_label_minus1 = value
                feature_one_label_minus1_proportion = feature_one_label_minus1 / feature_one_count
                feature_one_label_minus1_calc = -1 * feature_one_label_minus1_proportion * math.log2(
                    feature_one_label_minus1_proportion)
            elif index == 1:
                feature_one_label_plus1 = value
                feature_one_label_plus1_proportion = feature_one_label_plus1 / feature_one_count
                feature_one_label_plus1_calc = -1 * feature_one_label_plus1_proportion * math.log2(
                    feature_one_label_plus1_proportion)

        feature_one_entropy = feature_one_label_minus1_calc + feature_one_label_plus1_calc
        '''if -1 in feature_columns_group.get_group(1)[0].value_counts():
            feature_one_label_minus1 = feature_columns_group.get_group(1)[0].value_counts().iloc[0]
            feature_one_label_minus1_proportion = feature_one_label_minus1 / feature_one_count
            feature_one_label_minus1_calc = -1 * feature_one_label_minus1_proportion * math.log2(feature_one_label_minus1_proportion)
        if 1 in feature_columns_group.get_group(1)[0].value_counts():
            feature_one_label_plus1 = feature_columns_group.get_group(1)[0].value_counts().iloc[1]
            feature_one_label_plus1_proportion = feature_one_label_plus1 / feature_one_count
            feature_one_label_plus1_calc = -1 * feature_one_label_plus1_proportion * math.log2(feature_one_label_plus1_proportion)'''

    feature_zero_entropy = 0
    if 0 in feature_columns_group.groups.keys():
        feature_zero_label_minus1_calc = 0
        feature_zero_label_plus1_calc = 0

        for index, value in feature_columns_group.get_group(0)[0].value_counts().items():
            if index == -1:
                feature_zero_label_minus1 = value
                feature_zero_label_minus1_proportion = feature_zero_label_minus1 / feature_zero_count
                feature_zero_label_minus1_calc = -1 * feature_zero_label_minus1_proportion * math.log2(
                    feature_zero_label_minus1_proportion)
            elif index == 1:
                feature_zero_label_plus1 = value
                feature_zero_label_plus1_proportion = feature_zero_label_plus1 / feature_zero_count
                feature_zero_label_plus1_calc = -1 * feature_zero_label_plus1_proportion * math.log2(
                    feature_zero_label_plus1_proportion)
        feature_zero_entropy = feature_zero_label_minus1_calc + feature_zero_label_plus1_calc

        '''if -1 in feature_columns_group.get_group(0)[0].value_counts():
            feature_zero_label_minus1 = feature_columns_group.get_group(0)[0].value_counts().iloc[0]
            feature_zero_label_minus1_proportion = feature_zero_label_minus1 / feature_zero_count
            feature_zero_label_minus1_calc = -1 * feature_zero_label_minus1_proportion * math.log2(feature_zero_label_minus1_proportion)
        if 1 in feature_columns_group.get_group(0)[0].value_counts():
            feature_zero_label_plus1 = feature_columns_group.get_group(0)[0].value_counts().iloc[1]
            feature_zero_label_plus1_proportion = feature_zero_label_plus1 / feature_zero_count
            feature_zero_label_plus1_calc = -1 * feature_zero_label_plus1_proportion * math.log2(feature_zero_label_plus1_proportion)
            '''
    expected_entropy = (feature_one_count / (feature_one_count + feature_zero_count)) * feature_one_entropy + (
            feature_zero_count / (feature_one_count + feature_zero_count)) * feature_zero_entropy

    dataset_entropy = cal_dataset_entropy(df)
    info_gain = dataset_entropy - expected_entropy
    #print("Information gain")
    #print(info_gain)
    #print("End of info gain")
    #print("######################")
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


def store_data(path):
    df = pd.DataFrame()
    file1 = open(path, 'r')
    # file1 = open('./data/data/restaurant.train', 'r')
    lines = file1.readlines()

    for line in lines:
        feature_vector = np.zeros([124])
        #feature_vector = np.zeros([10])
        one_data_set = line.split()
        feature_vector[0] = one_data_set[0]
        one_data_set = one_data_set[1:]
        for attribute_value in one_data_set:
            split_attribute_value = attribute_value.split(":")
            feature_vector[int(split_attribute_value[0])] = split_attribute_value[1]
        series = pd.Series(feature_vector)
        df = df.append(series, ignore_index=True)
    return df


def id3(df, attributes, entropy_gini_index=0):
    labels_grouping_data_frame = pd.DataFrame(df.groupby(0).groups.keys())
    if labels_grouping_data_frame.shape[0] == 1:
        # global global_counter
        # global_counter += 1
        # print(global_counter)
        # print(labels_grouping_data_frame.iloc[0][0])
        return TreeNode(labels_grouping_data_frame.iloc[0][0])
    else:
        if entropy_gini_index == 0:
            feature_node = max_info_gain_entropy(df, attributes)
        else:
            feature_node = max_info_gain_gini(df, attributes)
        feature_number = feature_node[0]
        # print("##### In id3: #####")
        # print("feature number: ")
        # print(feature_number)
        # global global_counter
        # global_counter += 1
        # print(global_counter)

        tree_node = TreeNode(feature_number)
        list_of_children = []
        feature_grouping = df.groupby(feature_number)
        # print("feature grouping is: ")
        # print(feature_grouping)
        # print("###################")
        for possible_value in feature_grouping.groups.keys():
            subset_examples = feature_grouping.get_group(possible_value)
            if subset_examples.shape[0] == 0:
                # global_counter += 1
                # print(global_counter)
                minus1_count = df[0].value_counts().iloc[0]
                plus1_count = df[0].value_counts().iloc[1]
                if minus1_count >= plus1_count:
                    return TreeNode(-1)
                else:
                    return TreeNode(+1)
            else:
                # np.delete(attribute_array, 7)
                # attributes = np.delete(attributes, feature_number)
                attributes_modify = attributes.copy()
                attributes_modify.remove(feature_number)
                child_node = id3(subset_examples, attributes_modify, entropy_gini_index)
                # child_node = id3(subset_examples, attributes.remove(feature_number))
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
    return predict_label(next_node, testing_instance)


def evaluate_accuracy(model, dataset):
    accuracy = 0
    for instance in range(0, dataset.shape[0]):
        predicted_label = predict_label(model, dataset.iloc[instance])
        if predicted_label == dataset.iloc[instance][0]:
            accuracy += 1
    accuracy_percent = accuracy/dataset.shape[0]*100
    return accuracy_percent


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


if __name__ == '__main__':
    #df = store_data('./data/data/a1a.train')
    df = store_data('./data/data/a1a.train')
    df_testing_data = store_data('./data/data/a1a.test')
    '''df_fold1 = store_data('./data/data/CVfolds/fold1')
    df_fold2 = store_data('./data/data/CVfolds/fold2')
    df_fold3 = store_data('./data/data/CVfolds/fold3')
    df_fold4 = store_data('./data/data/CVfolds/fold4')
    df_fold5 = store_data('./data/data/CVfolds/fold5')'''
    print(".........Using entropy........")
    root_node_entropy = id3(df, df.columns.values.tolist(), 0)
    # depth_entropy = tree_depth(root_node_entropy)
    # print("Depth: " + str(depth_entropy))
    testing_accuracy_entropy = evaluate_accuracy(root_node_entropy, df_testing_data)
    print("Testing data accuracy: " + str(testing_accuracy_entropy))
    training_accuracy_entropy = evaluate_accuracy(root_node_entropy, df)
    print("Training data accuracy: " + str(training_accuracy_entropy))
    print(".........End of using entropy........")

    '''print(".........Using gini index........")
    root_node_gini = id3(df, df.columns.values.tolist(), 1)
    depth_gini = tree_depth(root_node_gini)
    print("Depth: " + str(depth_gini))
    testing_accuracy_gini = evaluate_accuracy(root_node_gini, df_testing_data)
    print("Testing data accuracy: " + str(testing_accuracy_gini))
    training_accuracy_gini = evaluate_accuracy(root_node_gini, df)
    print("Training data accuracy: " + str(training_accuracy_gini))
    print(".........End of using gini........")

    print("...........Cross Validation.............")
    max_depth_final = n_fold_cross_validation([df_fold1, df_fold2, df_fold3, df_fold4, df_fold5], [1, 2, 3, 4, 5])
    print("Depth to be used: ")
    print(max_depth_final)
    root_node = id3_depth_hyper_parameter(df, df.columns.values.tolist(), max_depth_final[0])
    testing_accuracy = evaluate_accuracy(root_node, df_testing_data)
    training_accuracy = evaluate_accuracy(root_node, df)
    print("Testing accuracy with hyper parameter: " + str(testing_accuracy))
    print("Training accuracy with hyper parameter: " + str(training_accuracy))
    print("...........End of Cross Validation.............")'''
