import json, joblib
from sklearn.tree import _tree

model = joblib.load("model.pkl")
feature_names = joblib.load("train_columns.pkl")
classes = list(getattr(model, "classes_", [0,1]))

tree_ = model.tree_

def node_to_dict(node_id):
    if tree_.feature[node_id] != _tree.TREE_UNDEFINED:
        feature = feature_names[tree_.feature[node_id]]
        threshold = float(tree_.threshold[node_id])
        return {
            "type": "node",
            "feature": feature,
            "threshold": threshold,
            "left": node_to_dict(tree_.children_left[node_id]),
            "right": node_to_dict(tree_.children_right[node_id]),
        }
    else:
        # leaf: value = class counts
        value = tree_.value[node_id][0].tolist()
        return {
            "type": "leaf",
            "value": value
        }

export = {
    "classes": classes,
    "tree": node_to_dict(0)
}

with open("tree_model.json", "w") as f:
    json.dump(export, f)

print("✅ tree_model.json dibuat")
