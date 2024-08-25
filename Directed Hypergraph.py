from ortools.linear_solver import pywraplp
from collections import defaultdict, deque
import random
import time
import numpy as np
import ot
import json
import re
import numpy as np
from gurobipy import Model, GRB, quicksum
import pandas as pd
import csv


class DirectedHypergraph:
    '''Initializing the hypergraph'''
    def __init__(self):
        self.nodes = set()
        self.hyperedges = {}
        self.enzymes = {}  # Dictionary to store enzymes associated with hyperedges
        self.weights = {}
        self.ricci_curvature = {}

    '''Function to add a node to the hypergraph'''
    def add_node(self, node):
        self.nodes.add(node)

    '''Function to add a hyperedge to the hypergraph'''
    def add_hyperedge(self, hyperedge_id, tail_set, head_set, enzymes=None):
        self.hyperedges[hyperedge_id] = (tail_set, head_set)
        self.weights[hyperedge_id]=[1]
        if enzymes is not None:
            self.enzymes[hyperedge_id] = enzymes
        else:
            self.enzymes[hyperedge_id] = []  # Initialize with an empty list if no enzymes are provided

    '''Function to add ollivier ricci curvature for all hyperedges for every iteration'''
    def add_ricci_curvature(self, hyperedge_id, orc):
        if hyperedge_id not in self.ricci_curvature:
            self.ricci_curvature[hyperedge_id] = []  # Initialize with an empty list if key doesn't exist
        
        self.ricci_curvature[hyperedge_id].append(orc)


    '''Function to add weights for all hyperedges for every iteration'''
    def add_weights(self, hyperedge_id, weights):
        if weights is not None:
            self.weights[hyperedge_id].append(weights)
            


    '''Function to get the edges from the hyperedges'''
    def get_underlying_edges(self):
        '''Extract all edges from the hyperedges'''
        edges = set()
        for tail_set, head_set in self.hyperedges.values():
            for tail in tail_set:
                for head in head_set:
                    edge = frozenset([tail, head])
                    edges.add(edge)
        return edges
    

    '''Function to remove nodes from the hypergraph'''
    def remove_node(self, node):
        if node in self.nodes:
            self.nodes.remove(node)
            for hyperedge_id, hyperedge in list(self.hyperedges.items()):
                if node in hyperedge[0] or node in hyperedge[1]:
                    self.remove_hyperedge(hyperedge_id)


    '''Function to remove hyperedges from the hypergraph'''
    def remove_hyperedge(self, hyperedge_id):
        if hyperedge_id in self.hyperedges:
            # Retrieve tail and head sets of the hyperedge
            tail_set, head_set = self.hyperedges[hyperedge_id]

            # Remove the hyperedge from the hyperedges dictionary
            del self.hyperedges[hyperedge_id]

            # Remove associated enzymes if any
            if hyperedge_id in self.enzymes:
                del self.enzymes[hyperedge_id]

            # Function to check if a node is in any other hyperedge
            def is_node_in_other_hyperedges(node):
                for he in self.hyperedges.values():
                    if node in he[0] or node in he[1]:  # Check in both tail and head sets
                        return True
                return False

            # Remove nodes that are not in any other hyperedge
            all_nodes_to_remove = tail_set.union(head_set)
            for node in all_nodes_to_remove:
                if not is_node_in_other_hyperedges(node):
                    self.nodes.remove(node)

        else:
            print(f"No hyperedge found with ID: {hyperedge_id}")
   

    '''Function to calculate the in-degree of a node in the hypergraph, i.e., number of times a node appears in the head set of a hyperedge'''
    def calculate_d_in_x(self, node):
        d_in_x = 0
        for _, (_, head_set) in self.hyperedges.items():
            if node in head_set:
                d_in_x += 1
        return d_in_x

    '''Function to calculate the out-degree of a node in the hypergraph, i.e., number of times a node appears in the tail set of a hyperedge'''
    def calculate_d_out_x(self, node):
        d_out_x = 0
        for _, (tail_set, _) in self.hyperedges.items():
            if node in tail_set:
                d_out_x += 1
        return d_out_x

    
    '''Function that outputs the distance matrix for all pair shortest path'''
    def floyd_warshall_with_weights(self):
        # Initialize the distance matrix with "infinite" distances
        # Assume self.nodes is a list or set of nodes
        node_list = list(self.nodes)  # Convert to list to ensure consistent ordering
        node_count = len(node_list)

        # Create a mapping of node to index
        node_index = {node: idx for idx, node in enumerate(node_list)}

        # Initialize a 2D list (matrix) with "infinite" distances
        dist = [[float('inf') for _ in range(node_count)] for _ in range(node_count)]
        #dist = [[100 for _ in range(node_count)] for _ in range(node_count)]
        # Set the diagonal to 0 (distance from each node to itself)
        for i in range(node_count):
            dist[i][i] = 0
                # Set the distance from each node to itself to 0
        
    
        # Set the distance for directly connected nodes based on edge weights
        for edge_id, (tail_set, head_set) in self.hyperedges.items():
            # Set distances within tail set and head set to 0
            for tail in tail_set:
                for another_tail in tail_set:
                    if tail != another_tail:
                        dist[node_index[tail]][node_index[another_tail]] = 0
            for head in head_set:
                for another_head in head_set:
                    if head != another_head:
                        dist[node_index[head]][node_index[another_head]] = 0

            for tail in tail_set:
                for head in head_set:
                    # Update the distance with the weight of the edge
                    # Assuming edge_id is used to access weights; adjust accordingly
                    dist[node_index[tail]][node_index[head]] = min(dist[node_index[tail]][node_index[head]],self.weights[edge_id][-1])  # Using the last weight in the list
        
        # Floyd-Warshall algorithm to update distances
        for k in self.nodes:
            for i in self.nodes:
                for j in self.nodes:
                    if dist[node_index[i]][node_index[k]] + dist[node_index[k]][node_index[j]] < dist[node_index[i]][node_index[j]]:
                        dist[node_index[i]][node_index[j]] = dist[node_index[i]][node_index[k]] + dist[node_index[k]][node_index[j]]

         # Replace 'inf' with 0 for pairs of nodes that have no path between them
        for i in range(node_count):
            for j in range(node_count):
                if dist[i][j] == float('inf'):
                    dist[i][j] = 0               

        return dist

    
    def find_shortest_distance(self, start, end):
        """
        Find the shortest distance (number of hops) from start to end, but only up to a maximum number of hops.

        :param start: The starting node.
        :param end: The ending node.
        :param max_hops: Maximum number of hops allowed (default is 3).
        :return: The shortest distance as an integer, or None if no path exists within the hop limit.
        """
        max_distance = 3
        if start not in self.nodes or end not in self.nodes:
            return 0

        # Queue for BFS, each element is a tuple (current_node, hops)
        queue = deque([(start, 0)])
        visited = set()  # Set to keep track of visited nodes

        while queue:
            current, distance = queue.popleft()

            if distance > max_distance:
                break  # Stop if the number of hops exceeds the limit

            if current == end:
                return distance

            if current in visited:
                continue

            visited.add(current)

            for edge, (tail_set, head_set) in self.hyperedges.items():
                if current in tail_set and all(node not in visited for node in head_set):
                    for next_node in head_set:
                        queue.append((next_node, distance + 1))

        # Return 0 if no path is found within the hop limit. Note that these distances won't be used anyway in the LP so returning 0 works fine.
        return 0
    
    
    
    '''Function to calculate the probability distributions over all nodes based on the hyperedge'''
    def calculate_probability_distributions(self, hyperedge_id):
        tail_set, head_set = self.hyperedges[hyperedge_id]

        # Initialize mu_A and mu_B only for nodes in the tail set and head set respectively
       
        mu_A_in = {node: 0 for node in self.nodes}
        for node in tail_set:
            d_x_in = self.calculate_d_in_x(node)
            if d_x_in != 0:
                mu_A_in[node] = 0
            else:
                mu_A_in[node] = 1 / len(tail_set)

       
        mu_B_out = {node: 0 for node in self.nodes}
        for node in head_set:
            d_x_out = self.calculate_d_out_x(node)
            if d_x_out != 0:
                mu_B_out[node] = 0
            else:
                mu_B_out[node] = 1 / len(head_set)
        
        # Third Case
        for edge in self.hyperedges:
            if edge != hyperedge_id:
                tail_set_prime, head_set_prime = self.hyperedges[edge]
                common_tail_nodes = set(tail_set) & set(head_set_prime)
                if common_tail_nodes:
                    for node in common_tail_nodes:
                        deg_x_in = self.calculate_d_in_x(node)
                        for nodes in tail_set_prime:
                            if deg_x_in != 0:  
                                mu_A_in[nodes] += 1 / (len(tail_set) * len(tail_set_prime) * deg_x_in)

                common_head_nodes = set(head_set) & set(tail_set_prime)
                if common_head_nodes:
                    for node in common_head_nodes:
                        deg_x_out = self.calculate_d_out_x(node)
                        for nodes in head_set_prime:
                            if deg_x_out != 0:    
                                mu_B_out[nodes] += 1 / (len(head_set) * len(head_set_prime) * deg_x_out)
              
        total_mass_A = sum(mu_A_in.values())
        total_mass_B = sum(mu_B_out.values())
        
        # Normalize the probability distributions
        if total_mass_A == 0:
            mu_A_in ={node: mass for node, mass in mu_A_in.items()}
        else:
            mu_A_in = {node: mass / total_mass_A for node, mass in mu_A_in.items()}
        
        if total_mass_B == 0:
            mu_B_out = {node: mass for node, mass in mu_B_out.items()}
        else:
            mu_B_out = {node: mass / total_mass_B for node, mass in mu_B_out.items()}
        
        return mu_A_in, mu_B_out

        

    '''Function to calculate EMD using the distance matrix (Optimized)'''
    def earthmover_distance_gurobi_distance_matrix(self, hyperedge_id, distance_matrix):
        # Get the probability distributions for the specified hyperedge.
        mu_A, mu_B = self.calculate_probability_distributions(hyperedge_id)

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
        node_to_index = {node: idx for idx, node in enumerate(self.nodes)}

        
        try:
            model = Model("EarthMoverDistance")

            # Set up the log file
            log_filename = f"gurobi_log_{hyperedge_id}.log"
            model.setParam('LogFile', log_filename)

            variables = model.addVars(mu_A.keys(), mu_B.keys(), name="z", lb=0)

            # Update the objective function to use the distance matrix.
            model.setObjective(quicksum(distance_matrix[node_to_index[x]][node_to_index[y]] * variables[x, y]
                                for x in mu_A for y in mu_B), GRB.MINIMIZE)

            # Add constraints
            for x in mu_A:
                model.addConstr(quicksum(variables[x, y] for y in mu_B) == mu_A[x], f"dirt_leaving_{x}")

            for y in mu_B:
                model.addConstr(quicksum(variables[x, y] for x in mu_A) == mu_B[y], f"dirt_filling_{y}")

            start_time = time.time()
            model.optimize()
            end_time = time.time()

            time_taken = end_time - start_time

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
    
        
    def import_reactions(self, json_file_path):
        with open(json_file_path, 'r') as file:
            data = json.load(file)

    # Regular expression pattern to match enzyme IDs
        enzyme_pattern = re.compile(r'b\d+')

        for reaction in data['reactions']:
            reaction_id = reaction['id']
            metabolites = reaction['metabolites']
            gene_reaction_rule = reaction.get('gene_reaction_rule', '')

            # Extract enzymes using regular expression
            enzymes = enzyme_pattern.findall(gene_reaction_rule)

            # Check if the reaction is reversible
            is_reversible = reaction.get('lower_bound', 0.0) < 0.0
            

            # Separating reactants and products
            reactants = [metabolite for metabolite, coefficient in metabolites.items() if coefficient < 0]
            products = [metabolite for metabolite, coefficient in metabolites.items() if coefficient > 0]

            # Add sink_node if no products are present
            if not products:
                products = ['sink_node']

            # Add nodes to the hypergraph
            for node in reactants + products:
                self.add_node(node)

            # Add hyperedge for the forward reaction
            self.add_hyperedge(reaction_id + "_forward", set(reactants), set(products), enzymes)

            
            # If the reaction is reversible, add another hyperedge for the reverse reaction
            if is_reversible:
               self.add_hyperedge(reaction_id + "_reverse", set(products), set(reactants), enzymes)
    

    '''Check if the underlying graph is weakly connected'''
    def is_weakly_connected(self):
        if not self.nodes:
            return True

        edges = self.get_underlying_edges()
        visited = set()

        def dfs(node):
            '''Depth First Search'''
            if node in visited:
                return
            visited.add(node)
            for edge in edges:
                if node in edge:
                    for next_node in edge:
                        if next_node != node:
                            dfs(next_node)

        # Start DFS from any node
        start_node = next(iter(self.nodes))
        dfs(start_node)

        return visited == self.nodes
    
    '''Function to check if the Hypergraph is strongly connected'''
    def is_strongly_connected(self):
        for node1 in self.nodes:
            for node2 in self.nodes:
                if node1 != node2:
                    if self.find_shortest_distance(node1, node2) == 0:
                        return False
        return True
    
    def average_degree(self):
        total_in_degree, total_out_degree = 0, 0
        for node in self.nodes:
            total_in_degree += self.calculate_d_in_x(node)
            total_out_degree += self.calculate_d_out_x(node)
    
        average_in_degree = total_in_degree / len(self.nodes)
        average_out_degree = total_out_degree / len(self.nodes)

        return average_in_degree, average_out_degree
    
    def lowest_degree(self):
        min_in_degree = float('inf')
        min_out_degree = float('inf')
        node_min_in_degree = []
        node_min_out_degree = []

        for node in self.nodes:
            in_degree = self.calculate_d_in_x(node)
            out_degree = self.calculate_d_out_x(node)

            if in_degree < min_in_degree:
                min_in_degree = in_degree
                node_min_in_degree = [node]
            elif in_degree == min_in_degree:
                node_min_in_degree.append(node)

            if out_degree < min_out_degree:
                min_out_degree = out_degree
                node_min_out_degree = [node]
            elif out_degree == min_out_degree:
                node_min_out_degree.append(node)

        return (min_in_degree, node_min_in_degree), (min_out_degree, node_min_out_degree)
    
    def highest_degree(self):
        max_in_degree = 0
        max_out_degree = 0
        node_max_in_degree = []
        node_max_out_degree = []

        for node in self.nodes:
            in_degree = self.calculate_d_in_x(node)
            out_degree = self.calculate_d_out_x(node)

            if in_degree > max_in_degree:
                max_in_degree = in_degree
                node_max_in_degree = [node]
            elif in_degree == max_in_degree:
                node_max_in_degree.append(node)

            if out_degree > max_out_degree:
                max_out_degree = out_degree
                node_max_out_degree = [node]
            elif out_degree == max_out_degree:
                node_max_out_degree.append(node)

        return (max_in_degree, node_max_in_degree), (max_out_degree, node_max_out_degree)

    

    def calculate_distance_matrix(self):
        n = len(self.nodes)
        distance_matrix = [[0 for _ in range(n)] for _ in range(n)]
        for i, node1 in enumerate(self.nodes):
            for j, node2 in enumerate(self.nodes):
                distance_matrix[i][j] = self.find_shortest_distance(node1, node2)
        return distance_matrix

    

    def get_connected_components(self):
        '''Find all connected components in the graph'''
        if not self.nodes:
            return []

        edges = self.get_underlying_edges()
        visited = set()
        components = []

        def dfs(node, component):
            '''Depth First Search to find a connected component'''
            if node in visited:
                return
            visited.add(node)
            component.add(node)
            for edge in edges:
                if node in edge:
                    for next_node in edge:
                        if next_node != node:
                            dfs(next_node, component)

        for node in self.nodes:
            if node not in visited:
                component = set()
                dfs(node, component)
                components.append(component)

        return components

    
    

# Sample usage
if __name__ == "__main__":
    hypergraph = DirectedHypergraph()
    hypergraph.import_reactions('/Users/iJN678.json')  # Replace with the actual file path. We get these json files from BiGG Models.
   
    
    # Print the hyperedges and associated enzymes for verification
    for hyperedge_id, enzymes in hypergraph.enzymes.items():
        print(f"Hyperedge {hyperedge_id} has enzymes: {enzymes}")
        

    for hyperedge_id, hyperedge in hypergraph.hyperedges.items():
        print(f"Hyperedge {hyperedge_id} has these tail set and head set: {hyperedge}")
        


    print("Number of reactions or hypergedges:",len(hypergraph.hyperedges)) #Printing the number of hyperedges or reactions in our network.
    print("Number of nodes or metabolites",len(hypergraph.nodes)) #Printing the number of nodes or 'metabolites' in the network.
    connected = hypergraph.is_weakly_connected()
    print("The hypergraph is weakly connected:" if connected else "The hypergraph is not weakly connected.")
    strongly_connected = hypergraph.is_strongly_connected()
    print("The hypergraph is strongly connected." if strongly_connected else "The hypergraph is not strongly connected.")

    min_enzymes = hypergraph.greedy_enzyme_set_cover()
    print("Minimum set of enzymes to cover all hyperedges:", min_enzymes)

    # Call the functions
    avg_degree = hypergraph.average_degree()
    lowest_degree = hypergraph.lowest_degree()
    highest_degree = hypergraph.highest_degree()

    # Print the results
    print("Average Degree (In, Out):", avg_degree)
    print("Lowest Degree (In, Out):", lowest_degree)
    print("Highest Degree (In, Out):", highest_degree)
    
    
    def adjusted_sigmoid_0_to_1(x):
    # Clip x to a range that prevents overflow in exp.
    # The range of -709 to 709 is chosen based on the practical limits of np.exp()
        x_clipped = np.clip(x, -709, 709)
        a, b = 0, 1  # Define the target range
        return a + (b - a) / (1 + np.exp(-x_clipped))
    
    



    def update_orc_and_weights_iter(distance_matrix, iteration, file_format='csv'):
        file_name = f'methanosarcina_normalized_weights_data_iteration_{iteration}.{file_format}' #Name this file corresponding to the metabolic network
        
        with open(file_name, 'a', newline='') as file:
            if file_format == 'csv':
                writer = csv.writer(file)
                # Check if the file is empty to write headers
                if file.tell() == 0:
                    writer.writerow(['Hyperedge ID', 'ORC', 'Weight'])
                
                for hyperedge_id in hypergraph.hyperedges:
                    emd = hypergraph.earthmover_distance_gurobi_distance_matrix(hyperedge_id, distance_matrix)
                    weight = hypergraph.weights[hyperedge_id][-1] 
                    if weight != 0:
                        orc = 1 - (emd/weight)
                        hypergraph.add_ricci_curvature(hyperedge_id, orc)
                        weight = weight * (1 - orc)
                        normalized_weight = adjusted_sigmoid_0_to_1(weight)
                    else:
                        orc = 1 - (emd)
                        hypergraph.add_ricci_curvature(hyperedge_id, orc)
                        normalized_weight = 0
                    

                    hypergraph.add_weights(hyperedge_id, normalized_weight)
                    
                    writer.writerow([hyperedge_id, orc, normalized_weight])
                    
    
    

    # Function to save the matrix as a CSV file
    def save_matrix_csv(matrix, filename):
        pd.DataFrame(matrix).to_csv(filename, index=False, header=False)

    def load_matrix_csv(filename):
        return pd.read_csv(filename, header=None).values
    
    #Function to calculate ORC for the 0th iteration
    def update_orc_and_weights_iter0(distance_matrix, iteration, file_format='csv'):
        file_name = f'methanosarcina_ORC_weights_iteration_{iteration}.{file_format}'
        
        with open(file_name, 'a', newline='') as file:
            if file_format == 'csv':
                writer = csv.writer(file)
                # Check if the file is empty to write headers
                if file.tell() == 0:
                    writer.writerow(['Hyperedge ID', 'ORC', 'Weight'])
                
                for hyperedge_id in hypergraph.hyperedges:
                    emd = hypergraph.earthmover_distance_gurobi_distance_matrix(hyperedge_id, distance_matrix)
                    orc = 1 - emd
                    hypergraph.add_ricci_curvature(hyperedge_id, orc)
                    
                    if hypergraph.weights[hyperedge_id][-1] == 0:
                        normalized_weight = 0
                    else:
                        normalized_weight = hypergraph.weights[hyperedge_id][-1] * (1 - orc)
                        #normalized_weight = 1

                    hypergraph.add_weights(hyperedge_id, normalized_weight)
                    
                    writer.writerow([hyperedge_id, orc, normalized_weight])

    

    #Ricci Flow helper functions
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
        filename = f'distance_matrix_methanosarcina_normalized_weights_{iteration}.csv'
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
            connected = hypergraph.is_weakly_connected()
            file.write("The hypergraph is weakly connected:\n" if connected else "The hypergraph is not weakly connected.\n")
            components = hypergraph.get_connected_components()
            file.write(f"Connected Components: {components}\n")
            file.write(f"No. of modules: {len(components)}\n")
            # # Listing all hyperedges
            file.write("\nList of all hyperedges:\n")
            for hyperedge_id, edge_data in hypergraph.hyperedges.items():
                tail_set = edge_data[0]
                head_set = edge_data[1]
                file.write(f"Hyperedge ID: {hyperedge_id}, Tail Set: {tail_set}, Head Set: {head_set}\n")



    #Ricci Flow with Surgery script
    
    
    distance_matrix = hypergraph.calculate_distance_matrix()
    #save_matrix_csv(distance_matrix,'synechocystis_iter_0_distance_matrix.csv')
    #loaded_matrix = load_matrix_csv('methanosarcina_iter_0_distance_matrix.csv')
    update_orc_and_weights_iter0(distance_matrix,iteration=0)

    total_iterations = 40
    for i in range(1, total_iterations + 1):
        distance_matrix_i = hypergraph.floyd_warshall_with_weights()
        save_and_update(distance_matrix_i, i)

        if i % 2 == 0:  #Surgery step
            file_path = f'methanosarcina_normalized_weights_data_iteration_{i}.csv' 
            delete_hyperedges(file_path)
            stats_file_path = f'/Users/Methanosarcina RF/methanosarcina_8percentsurgery_RF_normalized{i // 2}.txt' #This file would contain the details about the largest module in the network after the corresponding surgery
            write_hypergraph_stats(stats_file_path, i)

    