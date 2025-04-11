"""Imports necessary modules"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.tree._tree import TREE_UNDEFINED

from rules import extract_rules
from utils import aux_folders_rules


def train_tree(rep, reps, seed, knowledge_base_file, tree_log_path):
    """Trains a decision tree classifier using only the GNG parameters and class."""
    clf = DecisionTreeClassifier(random_state=seed, min_samples_leaf=3, criterion="gini")

    df = pd.read_csv(knowledge_base_file, delimiter=",", skiprows=2)

    metrics_to_exclude = [
        "clusters_number",
        "silhouette_avg",
        "davies_bouldin_index",
        "calinski_harabasz_index",
        "adjusted_rand_index",
        "global_error",
        "combined_score",
        "execution_time",
        "class",
    ]

    X = df.drop(metrics_to_exclude, axis=1)
    y = df["class"]

    clf.fit(X, y)

    names = list(X.columns)
    classes = ["1", "0"]

    show_tree(clf, names, classes, rep, tree_log_path)

    text_representation = export_text(clf, feature_names=names)
    print(text_representation)
    print("-" * 50)

    return get_rules(clf, names, classes, rep, reps, seed)


def get_rules(tree, feature_names, class_names, rep, reps, seed):
    """Extract rules from the tree"""
    tree_ = tree.tree_
    feature_name = [feature_names[i] if i != TREE_UNDEFINED else "undefined!" for i in tree_.feature]

    paths = []
    path = []

    def recurse(node, path, paths):
        """Rule extraction recursion"""
        if tree_.feature[node] != TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"({name} <= {np.round(threshold, 3)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"({name} > {np.round(threshold, 3)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]

    recurse(0, path, paths)

    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]

    rules = []

    file_name = aux_folders_rules(seed, rep, reps)
    with open(file_name, "w") as f:
        for path in paths:
            rule = "if "
            for p in path[:-1]:
                if rule != "if ":
                    rule += " and "
                rule += str(p)
            rule += " then "
            if class_names is None:
                rule += "response: " + str(np.round(path[-1][0][0][0], 3))
            else:
                classes = path[-1][0][0]
                l = np.argmax(classes)
                rule += f"class: {class_names[l]} (proba: {np.round(100.0*classes[l]/np.sum(classes),2)}%)"
            rule += f" | based on {path[-1][1]:,} samples"
            f.write(rule + "\n")
            rules += [rule]

    extract_rules(file_name)

    return rules


def show_tree(clf, names, classes, rep, tree_log_path):
    """Tree plot"""
    plt.figure(figsize=(20, 10))
    plot_tree(clf, filled=True, feature_names=names, class_names=classes)
    plt.savefig(f"{tree_log_path}/decision_tree_rep{rep}.png")
    plt.close()
