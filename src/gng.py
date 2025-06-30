"""Imports necessary modules"""

import time
from datetime import timedelta

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy import spatial
from scipy.stats import pearsonr
from sklearn import decomposition

"""
Simple implementation of the Growing Neural Gas algorithm, based on:
A Growing Neural Gas Network Learns Topologies. B. Fritzke, Advances in Neural
Information Processing Systems 7, 1995.
"""


class GrowingNeuralGas:
    """GrowingNeuralGas class"""

    def __init__(self, base_path, input_data, seed, rep, reps, i, distance_metric="euclidean", max_nodes=None):
        """Initializes the GrowingNeuralGas class"""
        if not isinstance(input_data, np.ndarray):
            self.original_data = np.array(input_data, dtype=float)
            self.data = np.array(input_data, dtype=float)
        else:
            self.original_data = np.array(input_data, dtype=float)
            self.data = input_data.astype(float)
        self.network = None
        self.units_created = 0
        self.seed = seed
        self.rep = rep
        self.reps = reps
        self.i = i
        self.base_path = base_path
        self.max_nodes = max_nodes
        valid_metrics = ["euclidean", "cityblock", "cosine", "pearson"]
        if distance_metric not in valid_metrics:
            raise ValueError(f"distance_metric must be one of {valid_metrics}")
        self.distance_metric = distance_metric
        plt.style.use("ggplot")

    def compute_distance(self, vector1, vector2):
        """Computes distance between two vectors based on the specified metric"""
        vector1 = np.array(vector1, dtype=float) if not isinstance(vector1, np.ndarray) else vector1
        vector2 = np.array(vector2, dtype=float) if not isinstance(vector2, np.ndarray) else vector2
        try:
            if self.distance_metric == "pearson":
                corr, _ = pearsonr(vector1, vector2)
                return 1 - corr
            else:
                return spatial.distance.cdist(vector1.reshape(1, -1), vector2.reshape(1, -1), metric=self.distance_metric)[0][0]
        except ValueError:
            return 1.0

    def find_nearest_units(self, observation):
        """Finds the nearest unit to a given observation"""
        distance = []
        for u, attributes in self.network.nodes(data=True):
            vector = attributes["vector"]
            dist = self.compute_distance(vector, observation)
            distance.append((u, dist))
        distance.sort(key=lambda x: x[1])
        ranking = [u for u, dist in distance]
        return ranking

    def prune_connections(self, a_max):
        """Removes edges from the network that have an age greater than a_max"""
        nodes_to_remove = []
        for u, v, attributes in self.network.edges(data=True):
            if attributes["age"] > a_max:
                nodes_to_remove.append((u, v))
        for u, v in nodes_to_remove:
            self.network.remove_edge(u, v)
        nodes_to_remove = []
        for u in self.network.nodes():
            if self.network.degree(u) == 0:
                nodes_to_remove.append(u)
        for u in nodes_to_remove:
            self.network.remove_node(u)

    def fit_network(self, e_b, e_n, a_max, l, a, d, passes, plot_evolution=False):
        base_dir = self.base_path
        """Trains the GNG algorithm on a given dataset"""
        accumulated_local_error = []
        clustering_error = []
        network_order = []
        network_size = []
        total_units = []
        self.units_created = 0
        np.random.seed(self.seed)
        w_a = np.random.uniform(-2, 2, size=np.shape(self.data)[1])
        w_b = np.random.uniform(-2, 2, size=np.shape(self.data)[1])
        self.network = nx.Graph()
        self.e_b = e_b
        self.e_n = e_n
        self.network.add_node(self.units_created, vector=w_a, error=0)
        self.units_created += 1
        self.network.add_node(self.units_created, vector=w_b, error=0)
        self.units_created += 1
        sequence = 0
        start_time = time.time()
        max_vector_value = 1e6
        max_error_value = 1e10

        for p in range(passes):
            start = time.time()
            print("   Pass #%d" % (p + 1))
            np.random.shuffle(self.data)
            steps = 0
            for observation in self.data:
                nearest_units = self.find_nearest_units(observation)
                s_1 = nearest_units[0]
                s_2 = nearest_units[1]
                for u, v, attributes in self.network.edges(data=True, nbunch=[s_1]):
                    self.network.add_edge(u, v, age=attributes["age"] + 1)
                error_increment = self.compute_distance(observation, self.network.nodes[s_1]["vector"]) ** 2
                self.network.nodes[s_1]["error"] += error_increment
                self.network.nodes[s_1]["error"] = min(self.network.nodes[s_1]["error"], max_error_value)
                update_w_s_1 = self.e_b * (np.subtract(observation, self.network.nodes[s_1]["vector"]))
                self.network.nodes[s_1]["vector"] = np.add(self.network.nodes[s_1]["vector"], update_w_s_1)
                self.network.nodes[s_1]["vector"] = np.clip(
                    self.network.nodes[s_1]["vector"], -max_vector_value, max_vector_value
                )
                for neighbor in self.network.neighbors(s_1):
                    update_w_s_n = self.e_n * (np.subtract(observation, self.network.nodes[neighbor]["vector"]))
                    self.network.nodes[neighbor]["vector"] = np.add(self.network.nodes[neighbor]["vector"], update_w_s_n)
                    self.network.nodes[neighbor]["vector"] = np.clip(
                        self.network.nodes[neighbor]["vector"], -max_vector_value, max_vector_value
                    )
                self.network.add_edge(s_1, s_2, age=0)
                self.prune_connections(a_max)
                steps += 1
                if steps % l == 0:
                    if plot_evolution:
                        self.plot_network(f"{base_dir}/sequence/" + str(sequence) + ".png")
                    sequence += 1
                    q = 0
                    error_max = 0
                    for u in self.network.nodes():
                        if self.network.nodes[u]["error"] > error_max:
                            error_max = self.network.nodes[u]["error"]
                            q = u
                    f = -1
                    largest_error = -1
                    for u in self.network.neighbors(q):
                        if self.network.nodes[u]["error"] > largest_error:
                            largest_error = self.network.nodes[u]["error"]
                            f = u
                    w_r = 0.5 * (np.add(self.network.nodes[q]["vector"], self.network.nodes[f]["vector"]))
                    w_r = np.clip(w_r, -max_vector_value, max_vector_value)
                    r = self.units_created
                    self.units_created += 1
                    self.network.add_node(r, vector=w_r, error=0)
                    self.network.add_edge(r, q, age=0)
                    self.network.add_edge(r, f, age=0)
                    self.network.remove_edge(q, f)
                    self.network.nodes[q]["error"] *= a
                    self.network.nodes[f]["error"] *= a
                    self.network.nodes[r]["error"] = self.network.nodes[q]["error"]
                    self.network.nodes[q]["error"] = min(self.network.nodes[q]["error"], max_error_value)
                    self.network.nodes[f]["error"] = min(self.network.nodes[f]["error"], max_error_value)
                    self.network.nodes[r]["error"] = min(self.network.nodes[r]["error"], max_error_value)
                error = 0
                for u in self.network.nodes():
                    error += self.network.nodes[u]["error"]
                accumulated_local_error.append(error)
                network_order.append(self.network.order())
                network_size.append(self.network.size())
                total_units.append(self.units_created)
                for u in self.network.nodes():
                    self.network.nodes[u]["error"] *= d
                    self.network.nodes[u]["error"] = min(self.network.nodes[u]["error"], max_error_value)
            clustering_error.append(self.compute_clustering_error())

            end = time.time()
            execution = end - start
            total_seconds = int(execution)
            milliseconds = int((execution % 1) * 1000)
            hours, remainder = divmod(total_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            print(f"   Pass #{p + 1} - DONE in ~{hours}h:{minutes:02d}m:{seconds:02d}s.{milliseconds:03d}ms.\n")

        end_time = time.time()
        total_time = end_time - start_time
        total_seconds = int(total_time)
        milliseconds = int((total_time % 1) * 1000)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Total execution time in ~{hours}h:{minutes:02d}m:{seconds:02d}s.{milliseconds:03d}ms.\n")

        accumulated_local_error = np.clip(accumulated_local_error, 0, max_error_value)
        clustering_error = np.clip(clustering_error, 0, max_error_value)

        plt.clf()
        plt.title("Accumulated local error")
        plt.xlabel("iterations")
        plt.plot(range(len(accumulated_local_error)), accumulated_local_error)
        plt.savefig(f"{base_dir}/accumulated_local_error.png")
        plt.close()

        plt.clf()
        plt.title("Clustering error (NQECP)")
        plt.xlabel("passes")
        plt.plot(range(len(clustering_error)), clustering_error)
        plt.savefig(f"{base_dir}/clustering_error.png")
        plt.close()

        plt.clf()
        plt.title("Neural network properties")
        plt.plot(range(len(network_order)), network_order, label="Network order")
        plt.plot(range(len(network_size)), network_size, label="Network size")
        plt.legend()
        plt.savefig(f"{base_dir}/network_properties.png")
        plt.close()

        return start_time, end_time

    def plot_network(self, file_path):
        """Plots the GNG network to a file"""
        plt.clf()
        clipped_data = np.clip(self.data, -1e6, 1e6)
        plt.scatter(clipped_data[:, 0], clipped_data[:, 1])
        node_pos = {}
        for u in self.network.nodes():
            vector = np.clip(self.network.nodes[u]["vector"], -1e6, 1e6)
            node_pos[u] = (vector[0], vector[1])
        nx.draw(self.network, pos=node_pos)
        plt.draw()
        plt.savefig(file_path)
        plt.close()

    def number_of_clusters(self):
        """Returns the number of clusters in the network"""
        return nx.number_connected_components(self.network)

    def cluster_data(self):
        unit_to_cluster = np.zeros(self.units_created)
        cluster = 0
        for c in nx.connected_components(self.network):
            for unit in c:
                unit_to_cluster[unit] = cluster
            cluster += 1
        clustered_data = []
        for observation in self.original_data:
            nearest_units = self.find_nearest_units(observation)
            s = nearest_units[0]
            clustered_data.append((observation, unit_to_cluster[s]))
        return clustered_data

    def reduce_dimension(self, clustered_data):
        """Reduces the dimensionality of the clustered data using PCA"""
        transformed_clustered_data = []
        svd = decomposition.PCA(n_components=2)
        transformed_observations = svd.fit_transform(self.data)
        for i in range(len(clustered_data)):
            transformed_clustered_data.append((transformed_observations[i], clustered_data[i][1]))
        return transformed_clustered_data

    def plot_clusters(self, clustered_data):
        """Plots the clustered data with a large variety of colors"""
        number_of_clusters = nx.number_connected_components(self.network)
        plt.clf()
        plt.title("Cluster affectation")
        cmap = plt.cm.get_cmap("tab20")
        colors = [cmap(i / 20) for i in range(20)]
        cmap2 = plt.cm.get_cmap("Set3")
        colors.extend([cmap2(i / 12) for i in range(12)])
        cmap3 = plt.cm.get_cmap("Paired")
        colors.extend([cmap3(i / 12) for i in range(12)])
        cmap4 = plt.cm.get_cmap("viridis")
        colors.extend([cmap4(i / 20) for i in range(20)])
        if number_of_clusters > len(colors):
            cmap_fallback = plt.cm.get_cmap("rainbow")
            colors = [cmap_fallback(i / number_of_clusters) for i in range(number_of_clusters)]
        else:
            colors = colors[:number_of_clusters]
        for i in range(number_of_clusters):
            observations = [observation for observation, s in clustered_data if s == i]
            if len(observations) > 0:
                observations = np.clip(np.array(observations), -1e6, 1e6)
                plt.scatter(observations[:, 0], observations[:, 1], color=colors[i], label=f"cluster #{i}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
        plt.tight_layout()
        plt.savefig(f"{self.base_path}/clusters.png")
        plt.close()

    def compute_clustering_error(self):
        """Calculates clustering error as normalized quantization error with connectivity penalty"""
        quantization_error = 0
        for observation in self.data:
            nearest_units = self.find_nearest_units(observation)
            s_1 = nearest_units[0]
            dist = self.compute_distance(observation, self.network.nodes[s_1]["vector"])
            quantization_error += dist**2

        data_variance = np.sum(np.var(self.data, axis=0)) if np.sum(np.var(self.data, axis=0)) > 0 else 1.0
        normalized_quantization_error = quantization_error / data_variance

        local_errors = [self.network.nodes[u]["error"] for u in self.network.nodes()]
        if len(local_errors) > 1:
            error_std = np.std(local_errors)
            error_range = max(local_errors) - min(local_errors)
            k_opt = max(2, min(int(error_range / error_std) if error_std > 0 else 2, int(np.sqrt(len(self.data)))))
        else:
            k_opt = 2

        k = self.number_of_clusters()
        connectivity_penalty = abs(k - k_opt) / max(k, k_opt) if max(k, k_opt) > 0 else 1.0

        lambda_weight = 0.5
        clustering_error = normalized_quantization_error + lambda_weight * connectivity_penalty
        return clustering_error
