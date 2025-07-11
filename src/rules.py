"""Imports necessary modules"""

import re

from params import list_params


def extract_limits(rule):
    "Extrai os valores dos limites das regras"
    matches = re.finditer(r"(?P<feature>\w+) (?P<op>\<=|<>|>=|>) (?P<value>\d+\.\d+)", rule)

    limits = {}
    for match in matches:
        feature = match.group("feature")
        op = match.group("op")
        value = float(match.group("value"))

        if op == "<=":
            limits[feature] = value
        elif op == ">=":
            limits[feature] = -value

    return limits


def extract_rules(rule_log_file):
    """Function to extract the rules from the tree"""

    with open(f"{rule_log_file}", "r") as f:
        rules = f.readlines()

    all_limits = []
    for rule in rules:
        if "then class: 1" in rule:
            all_limits.append(extract_limits(rule))


def get_positive_rules(rules):
    """
    Extracts parameter conditions from rules that lead to class 1.
    Returns a list of tuples: (param_name, operator, value)
    where operator is either '<=' or '>'
    Skips rules with no conditions (e.g., 'if  then ...').
    """
    params = list_params()
    positive_conditions = []

    for rule in rules:
        try:
            if "class: 1" in rule:
                # Split the rule into conditions and result
                if "then" not in rule or "if" not in rule:
                    continue
                conditions_part = rule.split("then")[0].strip()
                # Extract the part after 'if '
                if not conditions_part.startswith("if "):
                    continue
                conditions_str = conditions_part[3:].strip()
                if not conditions_str:
                    # No conditions present
                    print(f"No positive conditions found in rule: {rule.strip()}")
                    continue
                conditions = conditions_str.split(" and ")
                for condition in conditions:
                    condition = condition.strip("()")
                    for param in params:
                        if param in condition:
                            if "<=" in condition:
                                value = float(condition.split("<=")[1].strip())
                                positive_conditions.append((param, "<=", value))
                            elif ">" in condition:
                                value = float(condition.split(">", 1)[1].strip())
                                positive_conditions.append((param, ">", value))
        except Exception as e:
            print(f"Error processing rule: {rule.strip()} | Exception: {e}")

    return positive_conditions  # Always a list, even if empty


def adjust_parameters_based_on_rule(positive_conditions, limits):
    """Adjusts parameter limits based on conditions that lead to class 1"""
    new_limits = limits.copy()
    param_indices = {param: i for i, param in enumerate(list_params())}

    for param, operator, value in positive_conditions:
        idx = param_indices[param]
        lower_limit, upper_limit = new_limits[idx]

        if operator == "<=":
            new_upper_limit = min(upper_limit, value)
            new_limits[idx] = (lower_limit, new_upper_limit)
        elif operator == ">":
            new_lower_limit = max(lower_limit, value)
            new_limits[idx] = (new_lower_limit, upper_limit)

        # Ensure lower_limit <= upper_limit
        if new_limits[idx][0] > new_limits[idx][1]:
            new_limits[idx] = (new_limits[idx][1], new_limits[idx][0])

    return new_limits
