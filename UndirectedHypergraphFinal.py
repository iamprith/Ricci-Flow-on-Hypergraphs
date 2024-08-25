from collections import deque
from itertools import permutations, combinations
from gurobipy import Model, GRB, quicksum
import time
import requests
import json
import pandas as pd
import ast
import random
import csv
import numpy as np

class UndirectedHypergraph:
    '''Initializing the hypergraph'''
    def __init__(self):
        self.nodes = set()
        self.hyperedges = {}
        self.weights = {}
        self.ricci_curvature = {}

    '''Function to add a node to the hypergraph'''
    def add_node(self,node):
        self.nodes.add(node)
        print(f"Node added: {node}")

    """Add a hyperedge to the hypergraph. Automatically adds missing nodes."""
    def add_hyperedge(self, hyperedge_id, nodes):
        # Ensure nodes is a list
        if not isinstance(nodes, list):
            raise ValueError("Nodes must be provided as a list")
        
        # Check if hyperedge already exists
        if hyperedge_id in self.hyperedges:
            print(f"Hyperedge {hyperedge_id} already exists with nodes {self.hyperedges[hyperedge_id]}")
            return
        # Add missing nodes to the node set
        for node in nodes:
            if node not in self.nodes:
                self.add_node(node)

        # Add the hyperedge
        self.hyperedges[hyperedge_id] = nodes
        self.weights[hyperedge_id] = [1]
        print(f"Hyperedge {hyperedge_id} added with nodes {nodes}")

    '''Function to add ollivier ricci curvature for all hyperedges for every iteration'''
    def add_ricci_curvature(self, hyperedge_id, orc):
        if hyperedge_id not in self.ricci_curvature:
            self.ricci_curvature[hyperedge_id] = []  # Initialize with an empty list if key doesn't exist

        self.ricci_curvature[hyperedge_id].append(orc)

    '''Function to add weights for all hyperedges for every iteration'''
    def add_weights(self, hyperedge_id, weights):
        if weights is not None:
            self.weights[hyperedge_id].append(weights)


    '''Build hypergraph from a DataFrame'''
    def build_from_dataframe(self, df):
        for _, row in df.iterrows():
            paper_id = row['paper_id']
            author_ids = ast.literal_eval(row['author_ids'])
            self.add_hyperedge(paper_id, author_ids)

    '''Retrieve authors for a given paper_id'''
    def get_authors_by_paper_id(self, paper_id):
        if paper_id in self.hyperedges:
            return self.hyperedges[paper_id]
        else:
            return "No hyperedge found for the given paper_id."
        

    def node_degree(self, node):
        """Calculate the degree of a node. Degree is the number of hyperedges containing this node."""
        if node not in self.nodes:
            raise ValueError("Node does not exist in the graph.")
        return sum(node in hyperedge for hyperedge in self.hyperedges.values())

    def neighbours(self, node):
        """
        Find all nodes that share at least one hyperedge with the specified node.

        :param node: The node for which to find neighbors.
        :return: A set of neighboring nodes.
        """
        if node not in self.nodes:
            return set()  # Return an empty set if the node does not exist

        neighbours = set()
        # Iterate through all hyperedges
        for hyperedge in self.hyperedges.values():
            if node in hyperedge:
                neighbours.update(hyperedge)  # Add all nodes in the hyperedge

        neighbours.discard(node)  # Remove the node itself from the set of neighbours
        return neighbours

    def floyd_warshall(self):
        node_list = list(self.nodes)
        index = {node: idx for idx, node in enumerate(node_list)}
        n = len(node_list)
        dist = [[float('inf') for _ in range(n)] for _ in range(n)]
        
        for i in range(n):
            dist[i][i] = 0
        
        for hyperedge_id, nodes in self.hyperedges.items():
            weight = self.weights[hyperedge_id][-1]
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    idx_i = index[nodes[i]]
                    idx_j = index[nodes[j]]
                    if dist[idx_i][idx_j] > weight:
                        dist[idx_i][idx_j] = weight
                        dist[idx_j][idx_i] = weight  # Graph is undirected

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
                        dist[j][i] = dist[i][j]

         # Replace 'inf' with 0 for pairs of nodes that have no path between them
        for i in range(n):
            for j in range(n):
                if dist[i][j] == float('inf'):
                    dist[i][j] = 0 

        return dist

    def find_shortest_distance(self, start, end):
        """
        Find the shortest distance (number of edges) between two nodes in an undirected hypergraph,
        with a maximum allowed distance of 3. Returns 3 if no path is found within this limit.

        :param start: The starting node.
        :param end: The ending node.
        :return: The shortest distance as an integer, or 3 if no path exists within the limit.
        """
        if start not in self.nodes or end not in self.nodes:
            return 0  # Return 0 if either node does not exist, assuming '0' as a 'not found' indicator

        if start == end:
            return 0  # The distance to itself is 0

        max_distance = 3  # Maximum distance allowed
        queue = deque([(start, 0)])
        visited = set([start])  # Set to keep track of visited nodes

        while queue:
            current_node, current_distance = queue.popleft()

            if current_distance == max_distance:
                if current_node == end:
                    return current_distance
                continue  # Do not expand this node further if max distance reached

            # Traverse each hyperedge in the graph
            for hyperedge in self.hyperedges.values():
                if current_node in hyperedge:
                    # Check each node in the current hyperedge
                    for node in hyperedge:
                        if node == end:
                            return current_distance + 1  # Found the end node, return the distance
                        if node not in visited:
                            visited.add(node)
                            queue.append((node, current_distance + 1))

        return 0  # Return 3 if no path is found within the limit
    
    def calculate_distance_matrix(self):
        n = len(self.nodes)
        distance_matrix = [[0 for _ in range(n)] for _ in range(n)]
        for i, node1 in enumerate(self.nodes):
            for j, node2 in enumerate(self.nodes):
                distance_matrix[i][j] = self.find_shortest_distance(node1, node2)
        return distance_matrix
    
    def diameter(self,hyperedge_id):
        hyperedge = self.hyperedges[hyperedge_id]
        diameter = 0
        for node1 in hyperedge:
            for node2 in hyperedge:
                if node1 != node2:
                    diam = self.find_shortest_distance(node1,node2)
                    if diam > diameter:
                        diameter = diam
        
        return diameter

    def node_probability(self, node):
        alpha = 0.1  # Self-transition probability factor
        probability_distribution = {n: 0.0 for n in self.nodes}  # Initialize probabilities

        if node not in self.nodes:
            raise ValueError("Node does not exist in the hypergraph.")

        # Calculate the denominator: sum of (|f| - 1) for all f containing node
        denominator = 0
        hyperedges_containing_node = self.find_hyperedges_containing_nodes(node)
        for hyperedge_id in hyperedges_containing_node:
            hyperedge = self.hyperedges[hyperedge_id]
            denominator += (len(hyperedge) - 1)

        if denominator == 0:
            # If the denominator is zero, we should handle this edge case gracefully.
            probability_distribution[node] = 1.0
            return probability_distribution

        # Calculate the numerator for each neighbor node j and update their probabilities
        for neighbour in self.neighbours(node):
            hyperedges_containing_both = self.find_hyperedges_containing_nodes(node, neighbour)
            numerator = len(hyperedges_containing_both)
            '''
            for hyperedge_id in hyperedges_containing_both:
                hyperedge = self.hyperedges[hyperedge_id]
                numerator += (len(hyperedge))
            '''
            # Update the probability of transitioning to the neighbor
            probability_distribution[neighbour] = (1 - alpha) * numerator / denominator

        # Assign the self-loop probability
        probability_distribution[node] = alpha

        # Normalization step
        
        total_probability = sum(probability_distribution.values())
        for n in probability_distribution:
            probability_distribution[n] /= total_probability
        
        return probability_distribution
    

    '''Find hyperedges that contain any of the specified nodes.'''
    def find_hyperedges_containing_nodes(self, *nodes):
        '''
        # Ensure input is treated as a list even if a single node is passed
        if isinstance(nodes, str):
            nodes = [nodes]  # Convert single string node to a list
        '''
        nodes_set = set(nodes)  # Convert list to set for efficient intersection checks
        #print(nodes_set)
        # Handle different types of inputs
        for node in nodes:
            if isinstance(node, (list, set, tuple)):  # If the input is any kind of collection
                nodes_set.update(node)  # Add all elements to the set
            else:
                nodes_set.add(node)  # Add the single element to the set

        found_hyperedges = []
        # Ensure all nodes in the set are in our nodes list
        if not nodes_set.issubset(self.nodes):
            print("Some nodes are not in the hypergraph.")
        
        # Iterate through all hyperedges
        for hyperedge_id, hyperedge_nodes in self.hyperedges.items():
            if nodes_set.intersection(hyperedge_nodes):  # Check if intersection is not empty
                found_hyperedges.append(hyperedge_id)
        
        return found_hyperedges
    
    def earthmover_distance_gurobi_distance_matrix(self, node_A, node_B, distance_matrix):
        if node_A not in self.nodes or node_B not in self.nodes:
            print(f"Node {node_A} or {node_B} does not exist in the hypergraph.")
            return None  # Return None if either node does not exist
        
        # Get the probability distributions for the two specified nodes.
        mu_A = self.node_probability(node_A)
        mu_B = self.node_probability(node_B)

        
        # Convert distributions from dictionary to list format and print for debugging
        nodes_A = sorted(mu_A.keys())
        nodes_B = sorted(mu_B.keys())
        distribution1 = [mu_A[node] for node in nodes_A]
        distribution2 = [mu_B[node] for node in nodes_B]
    
        # Print the distributions to verify correctness
        print("Nodes in mu_A:", nodes_A)
        print("Nodes in mu_B:", nodes_B)
        print("Distribution mu_A:", distribution1)
        print("Distribution mu_B:", distribution2)

        # Check if distributions sum to the same value
        total_mass_A = sum(distribution1)
        total_mass_B = sum(distribution2)
        print("Total mass in mu_A:", total_mass_A)
        print("Total mass in mu_B:", total_mass_B)
    
        if abs(total_mass_A - total_mass_B) > 1e-6:
            raise ValueError('The total mass of the distributions mu_A and mu_B are not equal.')
        

        # Create a mapping of nodes to their indices in the distance matrix.
        node_to_index = {node: idx for idx, node in enumerate(list(self.nodes))}

        try:
            # Create a new model in Gurobi.
            model = Model("EarthMoverDistance")

            # Set up the log file
            #log_filename = f"gurobi_log_{hyperedge_id}.log"
            # Set up the log file
            '''
            log_filename = f"gurobi_log_{hyperedge_id}.log"
            model.setParam('LogFile', log_filename)
            '''
            #model.setParam('OutputFlag', 1)
            # Create variables for the linear program.
            variables = model.addVars(mu_A.keys(), mu_B.keys(), name="z", lb=0)

            # Set the objective of the linear program to minimize the total cost.
            model.setObjective(quicksum(distance_matrix[node_to_index[x]][node_to_index[y]] * variables[x, y]
                                for x in mu_A for y in mu_B), GRB.MINIMIZE)

            # Add constraints to ensure the conservation of mass.
            for x in mu_A:
                model.addConstr(quicksum(variables[x, y] for y in mu_B) == mu_A[x], f"dirt_leaving_{x}")

            for y in mu_B:
                model.addConstr(quicksum(variables[x, y] for x in mu_A) == mu_B[y], f"dirt_filling_{y}")

            # Start the timer, solve the model, and calculate the time taken.
            start_time = time.time()
            model.optimize()
            end_time = time.time()

            time_taken = end_time - start_time

            # Check the model status and process the results.
            if model.status == GRB.OPTIMAL:
                total_cost = model.getObjective().getValue()
                print("Total EMD Cost:", total_cost)
                print("Time taken to find the optimal solution: {:.4f} seconds".format(time_taken))
                for x in mu_A:
                    for y in mu_B:
                        amount_moved = variables[x, y].X
                        if amount_moved > 0:
                            print(f"Move {amount_moved} from {x} to {y}")
                return total_cost
            else:
                print("No optimal solution found.")
                return None

        except Exception as e:
            print(f"Gurobi Error: {e}")
            return None

    def earthmover_distance_hyperedge_combinations(self, hyperedge_id, distance_matrix):
        """

        :param hyperedge_id: The identifier for the hyperedge.
        :return: The average EMD for all permutations of node pairs, or None if the hyperedge does not exist or has errors.
        """
        if hyperedge_id not in self.hyperedges:
            print("Hyperedge does not exist.")
            return None
        
        nodes = self.hyperedges[hyperedge_id]
        print(nodes)
        if len(nodes) < 2:
            return 1
        
        sum_emd = 0
        pair_count = 0
        # Generate all combinations of pairs of nodes
        for node_A, node_B in combinations(nodes, 2):
            emd = self.earthmover_distance_gurobi_distance_matrix(node_A, node_B, distance_matrix)
            if emd is not None:
                sum_emd += emd
                pair_count += 1

        if pair_count > 0:
            # Compute the average EMD
            average_emd = sum_emd /pair_count
            print(f"Calculated EMD for hyperedge {hyperedge_id} considering all combinations: {average_emd}")
            weight = self.weights[hyperedge_id][-1]
            if weight == 0:
                return 1 - average_emd
            else:
                return 1 - average_emd/weight
        else:
            print("No valid EMD computations were possible.")
            return None


    def check_weak_connectivity(self):
        """
        Check if the hypergraph is weakly connected.
        A hypergraph is weakly connected if there is a path between any pair of nodes.

        :return: True if the hypergraph is weakly connected, False otherwise.
        """
        if not self.nodes:
            return True  # An empty hypergraph or one with no nodes is trivially connected.

        start_node = next(iter(self.nodes))  # Get an arbitrary starting node
        visited = set()
        queue = deque([start_node])

        # Perform a breadth-first search (BFS) to find all reachable nodes
        while queue:
            current = queue.popleft()
            if current not in visited:
                visited.add(current)
                for neighbor in self.neighbours(current):
                    if neighbor not in visited:
                        queue.append(neighbor)

        # If the size of the visited set is the same as the number of nodes, all nodes are connected
        return len(visited) == len(self.nodes)
    
    def connected_components(self):
      """
      Find all connected components of the hypergraph.
      Each component is a set of nodes that are connected.

      :return: A list of sets, each set being a connected component.
      """
      if not self.nodes:
          return []

      visited = set()
      components = []

      for node in self.nodes:
          if node not in visited:
              # Start a new component
              current_component = set()
              queue = deque([node])

              # Perform BFS to find all nodes in this component
              while queue:
                  current = queue.popleft()
                  if current not in visited:
                      visited.add(current)
                      current_component.add(current)
                      # Enqueue all non-visited neighbors
                      queue.extend(self.neighbours(current) - visited)

              components.append(current_component)

      return components
    
    def remove_hyperedge(self, hyperedge_id):
        """
        Remove a hyperedge and any nodes that are not part of any other hyperedge.
        :param hyperedge_id: ID of the hyperedge to remove.
        """
        if hyperedge_id in self.hyperedges:
            # Retrieve the nodes in the hyperedge to be removed
            nodes_to_check = self.hyperedges.pop(hyperedge_id)
            self.weights.pop(hyperedge_id, None)  # Safely remove weight entry if exists

            # Determine if the nodes are part of any other hyperedges
            nodes_to_remove = set(nodes_to_check)
            for other_hyperedge_nodes in self.hyperedges.values():
                nodes_to_remove.difference_update(other_hyperedge_nodes)
                if not nodes_to_remove:
                    break  # Exit early if all nodes are found in other hyperedges

            # Remove nodes that are not in any other hyperedge
            self.nodes.difference_update(nodes_to_remove)
            print(f"Removed hyperedge {hyperedge_id} and isolated nodes {nodes_to_remove}")
        else:
            print(f"Hyperedge ID {hyperedge_id} not found.")
    
    
    def max_degree(self):
        max_degree = 0
        for node in self.nodes:
                degree = self.node_degree(node)

                if degree > max_degree:
                    max_degree = degree
                
        return max_degree
    
    def min_degree(self):
        min_degree = 0
        for node in self.nodes:
            degree = self.node_degree(node)

            if degree < min_degree:
                min_degree = degree

        return min_degree
    
    def avg_degree(self):
        total_degree = 0
        for node in self.nodes:
            degree =self.node_degree(node)
            total_degree+=degree
        
        average_degree = total_degree / len(self.nodes)
        return average_degree
            
    
    
df = pd.read_csv('/Users/Undirected Hypergraph/dataset_turingpapers_clean.csv')  #Add the data file here
hypergraph = UndirectedHypergraph()
hypergraph.build_from_dataframe(df)


def save_matrix_csv(matrix, filename):
        pd.DataFrame(matrix).to_csv(filename, index=False, header=False)

def load_matrix_csv(filename):
    return pd.read_csv(filename, header=None).values


print(len(hypergraph.nodes))
print(len(hypergraph.hyperedges))

print(hypergraph.check_weak_connectivity())

def calculate_degrees(hypergraph):
    degrees = {}
    
    # Iterate over each node in the hypergraph
    for node in hypergraph.nodes:
        degrees[node] = hypergraph.node_degree(node)

    # If there are no nodes or degrees calculated, handle the case gracefully
    if not degrees:
        max_degree = 0
        min_degree = 0
        avg_degree = 0.0
    else:
        max_degree = max(degrees.values())
        min_degree = min(degrees.values())
        avg_degree = sum(degrees.values()) / len(degrees)
    
    return max_degree, min_degree, avg_degree

# Example usage:
# Assuming 'hypergraph' is an instance of UndirectedHypergraph
max_degree, min_degree, avg_degree = calculate_degrees(hypergraph)
print(f"Max Degree: {max_degree}")
print(f"Min Degree: {min_degree}")
print(f"Average Degree: {avg_degree:.2f}")


def adjusted_sigmoid_0_to_1(x):
    # Clip x to a range that prevents overflow in exp.
    # The range of -709 to 709 is chosen based on the practical limits of np.exp()
        x_clipped = np.clip(x, -709, 709)
        a, b = 0, 1  # Define the target range
        return a + (b - a) / (1 + np.exp(-x_clipped))

def update_orc_and_weights_iter(distance_matrix, iteration, file_format='csv'):
        file_name = f'dataset_networkscience_normalized_weights_data_iteration_{iteration}.{file_format}'

        with open(file_name, 'a', newline='') as file:
            if file_format == 'csv':
                writer = csv.writer(file)
                # Check if the file is empty to write headers
                if file.tell() == 0:
                    writer.writerow(['Hyperedge ID', 'ORC', 'Weight'])

                for hyperedge_id in hypergraph.hyperedges:
                    orc = hypergraph.earthmover_distance_hyperedge_combinations(hyperedge_id, distance_matrix)
                    hypergraph.add_ricci_curvature(hyperedge_id, orc)
                    weight = hypergraph.weights[hyperedge_id][-1]
                    if weight != 0:
                        
                        weight = weight * (1 - orc)
                        normalized_weight = adjusted_sigmoid_0_to_1(weight)
                    else:
                        normalized_weight = 0


                    hypergraph.add_weights(hyperedge_id, normalized_weight)

                    writer.writerow([hyperedge_id, orc, normalized_weight])

def find_top_n_weighted_hyperedges(file_path, n):

        # Load the CSV file
        df = pd.read_csv(file_path)

        # Sort the DataFrame based on the 'Weight' column in descending order
        df_sorted = df.sort_values(by='Weight', ascending=False)

        # Select the top n rows and only the 'Hyperedge ID' column
        top_n_hyperedges_ids = df_sorted.head(n)['Hyperedge ID'].tolist()

        # Select the top n rows
        top_n_hyperedges = df_sorted.head(n)

        return top_n_hyperedges_ids


def save_and_update(distance_matrix, iteration):
        filename = f'distance_matrix_normalized_weights_{iteration}.csv'
        save_matrix_csv(distance_matrix, filename)
        update_orc_and_weights_iter(distance_matrix, iteration)


def delete_hyperedges(file_path, percentage=0.08):
        total_hyperedges = len(hypergraph.hyperedges)
        del_hyperedges = int(percentage * total_hyperedges)
        hyperedges_to_remove = find_top_n_weighted_hyperedges(file_path, del_hyperedges)
        for he in hyperedges_to_remove:
            hypergraph.remove_hyperedge(he)

def write_hypergraph_stats(file_path, iteration):
        with open(file_path, 'w') as file:
            file.write(f"Number of reactions or hyperedges: {len(hypergraph.hyperedges)}\n")
            file.write(f"Number of nodes or metabolites: {len(hypergraph.nodes)}\n")
            connected = hypergraph.check_weak_connectivity()
            file.write("The hypergraph is weakly connected:\n" if connected else "The hypergraph is not weakly connected.\n")
            components = hypergraph.connected_components()
            file.write(f"Connected Components: {components}\n")
            file.write(f"No. of modules: {len(components)}\n")
            # # Listing all hyperedges
            file.write("\nList of all hyperedges:\n")
            for hyperedge_id, edge_data in hypergraph.hyperedges.items():
                file.write(f"Hyperedge ID: {hyperedge_id}, Hyperedge: {edge_data}\n")

def update_orc_and_weights_iter0(distance_matrix, iteration, file_format='csv'):
        file_name = f'dataset_networkscience_ORC_weights_iteration_{iteration}.{file_format}'
        
        with open(file_name, 'a', newline='') as file:
            if file_format == 'csv':
                writer = csv.writer(file)
                # Check if the file is empty to write headers
                if file.tell() == 0:
                    writer.writerow(['Hyperedge ID', 'ORC', 'Weight'])
                
                for hyperedge_id in hypergraph.hyperedges:
                    orc = hypergraph.earthmover_distance_hyperedge_combinations(hyperedge_id, distance_matrix)
                    hypergraph.add_ricci_curvature(hyperedge_id, orc)
                    weight = hypergraph.weights[hyperedge_id][-1]
                    if weight != 0:
                        weight = weight * (1 - orc)
                        normalized_weight = adjusted_sigmoid_0_to_1(weight)
                    else:
                        normalized_weight == 0

                    hypergraph.add_weights(hyperedge_id, normalized_weight)
                    
                    writer.writerow([hyperedge_id, orc, normalized_weight])

distance_matrix = hypergraph.calculate_distance_matrix()

update_orc_and_weights_iter0(distance_matrix,iteration=0)

total_iterations = 40
for i in range(1, total_iterations + 1):
    distance_matrix_i = hypergraph.floyd_warshall()
    save_and_update(distance_matrix_i, i)

    if i % 2 == 0:
        file_path = f'dataset_networkscience_normalized_weights_data_iteration_{i}.csv'
        delete_hyperedges(file_path)
        stats_file_path = f'/Users//networkscience_8percentsurgery_RF_normalized{i // 2}.txt'
        write_hypergraph_stats(stats_file_path, i)



