import networkx as nx
import math
from tqdm import tqdm
import random
import os
import sys
import lzma
import json
import h5py
from typing import Hashable, TypeAlias

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if not (CURRENT_DIR in sys.path):
    sys.path.append(CURRENT_DIR)

from Bcolors_Class import Bcolors
bcolors = Bcolors()

Node: TypeAlias = Hashable  # Alias for readability

class Random_Walk:

    @staticmethod
    def random_walk(graph:nx.Graph, start_node:Node, walk_length:int, attribute:str='Label'):
        walk = [start_node]
        if nx.is_isolate(graph, start_node):
            walk [start_node] * walk_length
        else:
            i = 1
            last_node = walk[-1]
            while i <= walk_length:
                neighbors = list(graph.neighbors(last_node))
                if neighbors:
                    walk.append(random.choice(neighbors))
                    last_node = walk[-1]
                    i += 1
                else:
                    last_node = walk[-2]
                    if not(list(graph.neighbors(last_node))):
                        last_node = walk[-2]     
        if attribute == 'Degree':
            temp_walk = [graph.degree(node) for node in walk]
            walk = temp_walk

        return walk
    
    @staticmethod
    def biased_random_walk(graph:nx.Graph, start_node:Node, walk_length:int=64, walk_depth:int=3, attribute:str='Label'):
        walk = []
        if nx.is_isolate(graph, start_node):
            walk [start_node] * walk_length
        else:
            walk = [start_node]
            i = 1
            mini_walk_count = math.ceil((walk_length - 1) / walk_depth)
            while i <= mini_walk_count:
                current_node = start_node
                temp_walk_ = []
                j = 0
                while j < walk_depth:
                    neighbors = list(graph.neighbors(current_node))
                    if neighbors:
                        next_node = random.choice(neighbors)
                        temp_walk_.append(next_node)
                        current_node = next_node
                        j += 1
                if len(temp_walk_) == walk_depth:
                    walk_new_length = len(walk) + len(temp_walk_)
                    if walk_new_length > walk_length:
                        extra_item_count = walk_new_length - walk_length
                        walk.extend(temp_walk_[:len(temp_walk_)-extra_item_count])
                    else:    
                        walk.extend(temp_walk_)
                i += 1
            if len(walk) > walk_length:
                walk = walk[:walk_length]
        if attribute == 'Degree':
            temp_walk = [graph.degree(node) for node in walk]
            walk = temp_walk
            
        return walk

    @staticmethod
    def node_random_walk_matrix(graph:nx.Graph, start_node:Node, walk_length:int=64, num_walks:int=64, walk_depth:int=None, attribute:str='Label'):
        temp_walks = []
        walk_valid = True 
        if graph.degree[start_node] > 0:
            walks_counter = 0
            while walks_counter < num_walks:
                if walk_depth is None:
                    temp_walk = Random_Walk.random_walk(graph, start_node, walk_length, attribute)
                else:
                    temp_walk = Random_Walk.biased_random_walk(graph, start_node, walk_length, walk_depth, attribute)
                if len(temp_walk) == walk_length:
                    temp_walks.append(temp_walk)
                    walks_counter += 1
                else:
                    walk_valid = False
                    # break
        else:
            for i in range(num_walks):
                temp_walks.append([start_node]*walk_length)
            walk_valid = True
        if walk_valid:
            return temp_walks
        else:
            return False

    @staticmethod
    def node_random_walk_vector(graph:nx.Graph, start_node:Node, walk_length:int=64, walk_depth:int=None, attribute:str='Label'):
        walk_valid = False 
        if graph.degree[start_node] > 0:
            while not walk_valid:
                if walk_depth is None:
                    temp_walk = Random_Walk.random_walk(graph, start_node, walk_length, attribute)
                else:
                    temp_walk = Random_Walk.biased_random_walk(graph, start_node, walk_length, walk_depth, attribute)
                if len(temp_walk) == walk_length:
                    walk_valid = True
        if walk_valid:
            return temp_walk
        else:
            temp_walk = [start_node] * walk_length
            return temp_walk

    @staticmethod
    def graph_random_walk_matrix(graph:nx.Graph, walk_length:int=64, num_walks:int=64, walk_depth:int=None, attribute:str='Label'):
        nodes_random_walks = {}
        print(f"\tCreate nodes random walk matrix.\n\t\twalk_length: {bcolors.yellow_fg}{bcolors.bold}{walk_length}{bcolors.end_color}," +
            f" walk_depth: {bcolors.yellow_fg}{bcolors.bold}{walk_depth}{bcolors.end_color}," +
            f" num_walks: {bcolors.yellow_fg}{bcolors.bold}{num_walks}{bcolors.end_color}")
        pbar = tqdm(total=len(list(graph.nodes())))
        pbar.unit = ' Node'
        pbar.colour = 'Green'
        for node in graph:
            pbar.desc = '\t\tNode ' + '{0: <16}'.format(str(node))
            nodes_random_walks[node] = Random_Walk.node_random_walk_matrix(graph, node, walk_length, num_walks, walk_depth, attribute)
            pbar.update(1)
        pbar.close()
        return nodes_random_walks

    @staticmethod
    def graph_random_walk_vector(graph:nx.Graph, walk_length:int=64, walk_depth:int=None, attribute:str='Label'):
        nodes_random_walks = {}
        print(f'\tCreate nodes random walk vector:\n\t\t' + 
            f'walk_length: {bcolors.yellow_fg}{bcolors.bold}{walk_length}{bcolors.end_color},' + 
            f' walk_depth: {bcolors.yellow_fg}{bcolors.bold}{walk_depth}{bcolors.end_color}')
        pbar = tqdm(total=len(list(graph.nodes())))
        pbar.unit = ' Node'
        pbar.colour = 'Green'
        for node in graph:
            pbar.desc = '\t\tNode ' + '{0: <16}'.format(str(node))
            nodes_random_walks[node] = Random_Walk.node_random_walk_vector(graph, node, walk_length, walk_depth, attribute)
            pbar.update(1)
        pbar.close()
        return nodes_random_walks

    @staticmethod
    def write_graph_nodes_random_walk(graph:nx.Graph, embedding_attribute:str, embedding_type:str,
                                    walk_length:int, walk_depth:int, num_walks:int,
                                    compression_method:str=None):
        result_file_name = f"{graph.graph['name']} nodes {embedding_attribute} random walk {embedding_type}"
        result_file_name += f" walk_length={walk_length} walk_depth={walk_depth} num_walks={num_walks}"

        print(f"\tWrite {graph.graph['name']} nodes random walk in file using " + 
            f"{bcolors.black_fg}{bcolors.yellow_bg_l}{bcolors.bold}{compression_method}{bcolors.end_color} compression method.")
        if compression_method == 'lzma':
            with lzma.open(graph.graph['result_path'] + result_file_name + ".xz", "wt", encoding="utf-8") as f:
                    json.dump(nx.get_node_attributes(graph, 'x'), f)
                    result_file_name += ".xz"
        elif compression_method == 'h5py':
            with h5py.File(graph.graph['result_path'] + result_file_name + ".h5", "w") as f:
                for key, data in nx.get_node_attributes(graph, 'x').items():
                    f.create_dataset(str(key), data=data)
                result_file_name += ".h5"
        else:
            with open(graph.graph['result_path'] + result_file_name + '.json', 'w') as f:
                json.dump(nx.get_node_attributes(graph, 'x'), f)
                result_file_name += ".json"
        
        print(f"\t\t{bcolors.black_bg_l}{bcolors.green_fg}{bcolors.bold}Write don.{bcolors.end_color}\n"
            f"\t\tFile path: {bcolors.cyan_fg}{bcolors.underline}{graph.graph['result_path']}{bcolors.end_color}\n"
            f"\t\tFile name: {bcolors.bold}{bcolors.italic}{result_file_name}{bcolors.end_color}")
        pass
    
    @staticmethod
    def load_graph_nodes_random_walk(graph:nx.Graph, embedding_attribute:str, embedding_type:str,
                                    walk_length:int, walk_depth:int, num_walks:int,
                                    compression_method:str=None):
        graph_nodes_random_walk = None

        result_file_name_info = f"{graph.graph['name']} nodes {embedding_attribute} random walk {embedding_type}"
        result_file_name_info += f" walk_length={walk_length} walk_depth={walk_depth} num_walks={num_walks}"

        result_file = None
        if compression_method == 'lzma':
            result_file = graph.graph['result_path'] + result_file_name_info + ".xz"
            if os.path.isfile(result_file):
                print(f"\t{bcolors.black_bg}{bcolors.yellow_fg}Nodes random walk loading...{bcolors.end_color}")
                with lzma.open(result_file, "rt", encoding="utf-8") as f:
                    graph_nodes_random_walk = json.load(f)
                print(f"\t{bcolors.black_bg}{bcolors.green_fg}Load done.{bcolors.end_color}")
        elif compression_method == 'h5py':
            result_file = graph.graph['result_path'] + result_file_name_info + ".h5"
            if os.path.isfile(result_file):
                print(f"\t{bcolors.black_bg}{bcolors.yellow_fg}Nodes random walk loading...{bcolors.end_color}")
                with h5py.File(result_file, "r") as f:
                    graph_nodes_random_walk = {key: f[key][()] for key in f.keys()}
                print(f"\t{bcolors.black_bg}{bcolors.green_fg}Load done.{bcolors.end_color}")
        else:
            result_file = graph.graph['result_path'] + result_file_name_info + ".json"
            if os.path.isfile(result_file):
                print(f"\t{bcolors.black_bg}{bcolors.yellow_fg}Nodes random walk loading...{bcolors.end_color}")
                with open(result_file, "rt", encoding="utf-8") as f:
                    graph_nodes_random_walk = json.load(f)
                print(f"\t{bcolors.black_bg}{bcolors.green_fg}Load done.{bcolors.end_color}")
            
        return graph_nodes_random_walk

    @staticmethod
    def write_multilayer_network_nodes_random_walk(random_walks:dict, network_infos:dict, embedding_attribute:str, embedding_type:str,
                                    walk_length:int, walk_depth:int, num_walks:int,
                                    compression_method:str=None, tabs:str=''):
        result_file_name = f"{network_infos['name']} nodes {embedding_attribute} random walk {embedding_type}"
        result_file_name += f" walk_length={walk_length} walk_depth={walk_depth} num_walks={num_walks}"

        print(f"{tabs}Write {network_infos['name']} nodes random walk in file using " + 
            f"{bcolors.black_fg}{bcolors.yellow_bg_l}{bcolors.bold}{compression_method}{bcolors.end_color} compression method.")
        
        if compression_method == 'lzma':
            with lzma.open(network_infos['results_path'] + result_file_name + ".xz", "wt", encoding="utf-8") as f:
                    json.dump(random_walks, f)
                    result_file_name += ".xz"
        elif compression_method == 'h5py':
            with h5py.File(network_infos['results_path'] + result_file_name + ".h5", "w") as f:
                for layer , walks_data in random_walks:
                    for key, x in walks_data.items():
                        f.create_dataset(str(f"{layer},{key}"), data=x)
                result_file_name += ".h5"
        else:
            with open(network_infos['results_path'] + result_file_name + '.json', 'w') as f:
                json.dump(random_walks, f)
                result_file_name += ".json"
        
        print(f"{tabs}\t{bcolors.black_bg_l}{bcolors.green_fg}{bcolors.bold}Write don.{bcolors.end_color}\n"
            f"{tabs}\tFile path: {bcolors.cyan_fg}{bcolors.underline}{network_infos['results_path']}{bcolors.end_color}\n"
            f"{tabs}\tFile name: {bcolors.bold}{bcolors.italic}{result_file_name}{bcolors.end_color}")
        pass
    
    @staticmethod
    def load_multilayer_network_nodes_random_walk(network_infos:dict, embedding_attribute:str,
                                                embedding_type:str, walk_length:int, walk_depth:int, num_walks:int,
                                                compression_method:str=None, tabs:str=''):
        network_nodes_random_walk = None

        result_file_name_info = f"{network_infos['name']} nodes {embedding_attribute} random walk {embedding_type}"
        result_file_name_info += f" walk_length={walk_length} walk_depth={walk_depth} num_walks={num_walks}"

        result_file = None
        if compression_method == 'lzma':
            result_file = network_infos['results_path'] + result_file_name_info + ".xz"
            if os.path.isfile(result_file):
                print(f"{tabs}{bcolors.black_bg}{bcolors.yellow_fg}Nodes random walk loading...{bcolors.end_color}")
                with lzma.open(result_file, "rt", encoding="utf-8") as f:
                    network_nodes_random_walk = json.load(f)
                print(f"{tabs}{bcolors.black_bg}{bcolors.green_fg}Load done.{bcolors.end_color}")
            else:
                print(f"{bcolors.bold}{bcolors.red_fg}File not found.{bcolors.ENDC}")
        elif compression_method == 'h5py':
            result_file = network_infos['results_path'] + result_file_name_info + ".h5"
            if os.path.isfile(result_file):
                print(f"{tabs}{bcolors.black_bg}{bcolors.yellow_fg}Nodes random walk loading...{bcolors.end_color}")
                with h5py.File(result_file, "r") as f:
                    temp_network_nodes_random_walk = {key: f[key][()] for key in f.keys()}
                for k, v in temp_network_nodes_random_walk.items():
                    layer_node = k.split(',')
                    layer = layer_node[0]
                    node = layer_node[1]
                    if layer in network_nodes_random_walk.keys():
                        network_nodes_random_walk[layer][node] = v
                    else:
                        network_nodes_random_walk[layer] = {}
                        network_nodes_random_walk[layer][node] = v

                print(f"{tabs}{bcolors.black_bg}{bcolors.green_fg}Load done.{bcolors.end_color}")
            else:
                print(f"{bcolors.bold}{bcolors.red_fg}File not found.{bcolors.ENDC}")
        else:
            result_file = network_infos['results_path'] + result_file_name_info + ".json"
            if os.path.isfile(result_file):
                print(f"{tabs}{bcolors.black_bg}{bcolors.yellow_fg}Nodes random walk loading...{bcolors.end_color}")
                with open(result_file, "rt", encoding="utf-8") as f:
                    network_nodes_random_walk = json.load(f)
                print(f"{tabs}{bcolors.black_bg}{bcolors.green_fg}Load done.{bcolors.end_color}")
            else:
                print(f"{bcolors.bold}{bcolors.red_fg}File not found.{bcolors.ENDC}")
        return network_nodes_random_walk
