def list_params():
    """Returns a list of parameters used in the GNG algorithm"""
    params = ["e_b", "e_n", "a_max", "l", "a", "d", "passes"]
    return params


def list_limits():
    """Returns a list of parameter limits used in the GNG algorithm"""
    limits = [
        (0.0, 1.0),
        (0.0, 1.0),
        (0, 10),
        (1, 30),
        (0.0, 1.0),
        (0.0, 1.0),
        (0, 10),
    ]
    return limits


def show_descriptions():
    """Returns a list of descriptions of the parameters used in the GNG algorithm"""
    descriptions = [
        {
            "param": "e_b",
            "description": "Controls the movement of the winning neuron at each iteration",
            "min_value": 0.0,
            "max_value": 1.0,
        },
        {
            "param": "e_n",
            "description": "Controls the movement of the topological neighbors of the winning neuron at each iteration",
            "min_value": 0.0,
            "max_value": 1.0,
        },
        {
            "param": "a_max",
            "description": "Sets the maximum age that an edge can have within the network",
            "min_value": 0.0,
            "max_value": 10,
        },
        {
            "param": "l",
            "description": "Parameter for inserting neurons into the network",
            "min_value": 1,
            "max_value": 30,
        },
        {
            "param": "a",
            "description": "Controls the learning rate during network training",
            "min_value": 0.0,
            "max_value": 1.0,
        },
        {
            "param": "d",
            "description": "Controls the reduction of the influence of the oldest neurons in the network",
            "min_value": 0.0,
            "max_value": 1.0,
        },
        {
            "param": "passes",
            "description": "Number of iterations within the algorithm",
            "min_value": 0,
            "max_value": 10,
        },
    ]
    return descriptions
