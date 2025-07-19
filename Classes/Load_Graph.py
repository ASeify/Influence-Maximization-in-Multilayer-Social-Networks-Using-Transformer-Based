import networkx as nx
import os
import sys
from termcolor import colored


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if not (CURRENT_DIR in sys.path):
    sys.path.append(CURRENT_DIR)

from Bcolors_Class import Bcolors
bcolors = Bcolors()

class Load_Graph:

    color_list = ["light_red", "light_green", "light_yellow",
                    "light_blue","light_magenta", "light_cyan",
                    "blue", "red", "white", "green", "yellow",
                        "magenta", "cyan", ]

    @staticmethod
    def load_monoplex_graph(file_path:str, file_info:dict, convert_node_labels:bool=False,
               nodes_first_label:int=None, convert_undirected:bool=True)->nx.Graph:
    
        if not (os.path.exists(file_path)):
            print(f"Error: File {file_path} not found.")
            sys.exit(1)
        else:
            # print(file_info)
            if (file_info['type'] == ".edgelist" or file_info['type'] == ".edges" 
                or file_info['type'] == ".txt"):
                graph = nx.read_edgelist(file_path, comments="#")
            elif (file_info['type'] == ".gml"):
                graph = nx.read_gml(file_path)
            elif (file_info['type'] == ".mtx"):
                graph = nx.read_edgelist(file_path, comments="#")
            elif (file_info['type'] == ".csv"):
                graph = nx.read_edgelist(file_path, delimiter=' ', nodetype=str)
                if graph.number_of_nodes() == 0:
                    graph = nx.read_edgelist(file_path, delimiter='\t', nodetype=str)
                if graph.number_of_nodes() == 0:
                    graph = nx.read_edgelist(file_path, delimiter=',', nodetype=str)
                if 'source' in graph:
                    graph.remove_node('source')
                if 'target' in graph:
                    graph.remove_node('target')
            
            if graph.number_of_nodes() > 0:
                print(f"\t{bcolors.green_fg}Graph with {bcolors.end_color}{bcolors.yellow_fg}{bcolors.bold}{graph.number_of_nodes()}{bcolors.end_color}" +
                    f"{bcolors.green_fg} nodes and {bcolors.end_color}{bcolors.yellow_fg}{bcolors.bold}{graph.number_of_edges()}{bcolors.end_color}" +
                    f"{bcolors.green_fg} edges loaded successfully.{bcolors.end_color}")
                if convert_node_labels:
                    graph = nx.convert_node_labels_to_integers(graph, first_label=nodes_first_label, ordering='default', label_attribute='orginal_label')#, label_attribute='old_label')
                    print(f"\tConvert node labels to integers. Nodes first label is {bcolors.yellow_fg}{bcolors.bold}{nodes_first_label}{bcolors.end_color}" +
                        f" and Nodes last label is {bcolors.yellow_fg}{bcolors.bold}{nodes_first_label + graph.number_of_nodes()-1}{bcolors.end_color}")
                if convert_undirected:
                    if nx.is_directed(graph):
                        graph = graph.to_undirected()
            else:
                print(f"\t{bcolors.red_fg}No nodes found in the graph.{bcolors.end_color}")
                graph = nx.Graph()
        return graph
    
    @staticmethod
    def load_multilayer_graph(file_path:str)->nx.Graph:
        entra_layer_edges = {}
        intra_layer_edges = {}
        layer_id = 1
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    if line != None and line.strip() != "" and line.strip()[0] != "#" and line.strip()[0] != "%":
                        content_parts = line.strip().split(" ")
                        if len(content_parts) == 4:
                            source_node_id = content_parts[0].strip()
                            source_layer_id = content_parts[1].strip()
                            destination_node_id = content_parts[2].strip()
                            destination_layer_id = content_parts[3].strip()
                            if source_layer_id == destination_layer_id:
                                layer_id = source_layer_id
                                if not(layer_id in entra_layer_edges.keys()):
                                    entra_layer_edges[layer_id] = []
                                entra_layer_edges[layer_id].append((source_node_id, destination_node_id))
                            else:
                                if not(source_layer_id in intra_layer_edges.keys()):
                                    intra_layer_edges[source_layer_id] = {}
                                if not(destination_layer_id in intra_layer_edges[source_layer_id].keys()):
                                    intra_layer_edges[source_layer_id][destination_layer_id] = []
                                intra_layer_edges[source_layer_id][destination_layer_id].append((source_node_id, destination_node_id))

                                if not(destination_layer_id in intra_layer_edges.keys()):
                                    intra_layer_edges[destination_layer_id] = {}
                                if not(source_layer_id in intra_layer_edges[destination_layer_id].keys()):
                                    intra_layer_edges[destination_layer_id][source_layer_id] = []
                                intra_layer_edges[destination_layer_id][source_layer_id].append((destination_node_id, source_node_id))
            file.close()
        except Exception as e:
            print(e)
        
        graphs_of_network = []
        i, j = 0, 0
        for k, v in entra_layer_edges.items():
            if len(v) > 0:
                graphs_of_network.append(nx.Graph())
                graphs_of_network[-1].add_edges_from(v)
                graphs_of_network[-1].graph["id"]  = k

                print(colored("Layer "  + str(i+1) + ": " + str(graphs_of_network[-1].number_of_nodes()) + " Node And " +
                            str(graphs_of_network[-1].number_of_edges()) + " Edge", Load_Graph.color_list[j]))
                i += 1
                j += 1
                if j >= len(Load_Graph.color_list):
                    j = 0
        network_entier_nodes_list = []
        for graph in graphs_of_network:
            if graph.number_of_nodes() > 0:
                network_entier_nodes_list.extend(list(graph.nodes()))
            else:
                graphs_of_network.remove(graph)
        network_entier_nodes_list = list(set(network_entier_nodes_list))
        network_entier_nodes_count = len(network_entier_nodes_list)
        print(f"\n{bcolors.green_fg}Network with {bcolors.end_color}{bcolors.yellow_fg}{bcolors.bold}{network_entier_nodes_count}{bcolors.end_color}" +
            f"{bcolors.green_fg} nodes and {bcolors.end_color}{bcolors.yellow_fg}{bcolors.bold}{len(entra_layer_edges)}{bcolors.end_color}" +
            f"{bcolors.green_fg} layers loaded successfully.{bcolors.end_color}")
        
        return graphs_of_network, intra_layer_edges, network_entier_nodes_list, network_entier_nodes_count