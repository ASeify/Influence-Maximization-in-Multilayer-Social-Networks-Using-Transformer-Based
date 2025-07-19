import networkx as nx
import numpy as np
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.feature_extraction import FeatureHasher
from tqdm import tqdm
import os
import sys
import lzma
import json
import h5py

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if not (CURRENT_DIR in sys.path):
    sys.path.append(CURRENT_DIR)

from Bcolors_Class import Bcolors
bcolors = Bcolors()


class Generate_Embedings:
    @staticmethod
    def get_Word2Vec_sentence_vector(sentence, model):
        vector = []
        for word in sentence:
            temp_embedding = model.wv[word]
            vector.append(float(sum(temp_embedding)/len(temp_embedding)))
        # print(len(vector[0]))
        return vector

    @staticmethod
    def get_scale_of_list(list_of_numbers:list, min_value:int=0, max_value:int=10000000):
        scaled = []
        for item in list_of_numbers:
            if item < min_value or item > max_value:
                raise ValueError(f"Item {item} is out of range [{min_value}, {max_value}]")
            else:
                scaled.append((item - min_value) / (max_value - min_value))
        return scaled

    @staticmethod
    def crate_embedding_matrix(graph:nx.Graph, embedding_model, embedding_method:str='Word2Vec'): 
        graph = graph
        if embedding_method == 'Word2Vec':
            nodes_random_walks = nx.get_node_attributes(graph, 'x')
            pbar = tqdm(total=len(nodes_random_walks.keys()))
            pbar.unit = ' Node'
            pbar.colour = 'Blue'
            for k, v in nodes_random_walks.items():
                pbar.desc = '\t\tNode ' + '{0: <16}'.format(str(k))
                temp_embedding = []
                for row in v:
                    temp_embedding.append(list(Generate_Embedings.get_Word2Vec_sentence_vector(row, embedding_model)))
                graph.nodes[k]['x'] = temp_embedding
                pbar.update(1)
            pbar.close()
        
        elif embedding_method == 'Doc2Vec':
            nodes_list = list(graph.nodes())
            nodes_random_walks = nx.get_node_attributes(graph, 'x')
            pbar = tqdm(total=len(nodes_list))
            pbar.unit = ' Node'
            pbar.colour = 'Blue'
            for node in nodes_list:
                pbar.desc = '\t\tNode ' + '{0: <16}'.format(str(node))
                node_embedding_mat = []
                for i in range(len(nodes_random_walks[node])):
                    nodes_random_walk_vector = [str(item) for item in nodes_random_walks[node][i]]
                    node_embedding_mat.append(list(np.float64(embedding_model.infer_vector(nodes_random_walk_vector))))
                graph.nodes[node]['x'] = node_embedding_mat
                pbar.update(1)
            pbar.close()
        
        elif embedding_method ==  'FeatureHasher':
            nodes_list = list(graph.nodes())
            nodes_random_walks = nx.get_node_attributes(graph, 'x')
            for key, values in nodes_random_walks.items():
                graph.nodes[key]['x'] = [embedding_model.transform([{str(num): 1 for num in row}]).toarray().tolist()[0] for row in values]

        elif embedding_method == 'Scale':
            nodes_random_walks = nx.get_node_attributes(graph, 'x')
            pbar = tqdm(total=len(nodes_random_walks.keys()))
            pbar.unit = ' Node'
            pbar.colour = 'Blue'
            for k, v in nodes_random_walks.items():
                pbar.desc = '\t\tNode ' + '{0: <16}'.format(str(k))
                temp_embedding = []
                for row in v:
                    temp_embedding.append(Generate_Embedings.get_scale_of_list(row))
                graph.nodes[k]['x'] = temp_embedding
                pbar.update(1)
            pbar.close()
        else:
            raise ValueError(f"Unsupported embedding method: {embedding_method}")
        return graph

    @staticmethod
    def crate_embedding_matrix_return_dict(graph:nx.Graph, embedding_model, embedding_method:str='Word2Vec'): 
        graph = graph
        embeddings = {}
        if embedding_method == 'Word2Vec':
            nodes_random_walks = nx.get_node_attributes(graph, 'x')
            pbar = tqdm(total=len(nodes_random_walks.keys()))
            pbar.unit = ' Node'
            pbar.colour = 'Blue'
            for k, v in nodes_random_walks.items():
                pbar.desc = '\t\tNode ' + '{0: <16}'.format(str(k))
                temp_embedding = []
                for row in v:
                    temp_embedding.append(list(Generate_Embedings.get_Word2Vec_sentence_vector(row, embedding_model)))
                embeddings[k] = temp_embedding
                pbar.update(1)
            pbar.close()
        
        elif embedding_method == 'Doc2Vec':
            nodes_list = list(graph.nodes())
            nodes_random_walks = nx.get_node_attributes(graph, 'x')
            pbar = tqdm(total=len(nodes_list))
            pbar.unit = ' Node'
            pbar.colour = 'Blue'
            for node in nodes_list:
                pbar.desc = '\t\tNode ' + '{0: <16}'.format(str(node))
                node_embedding_mat = []
                for i in range(len(nodes_random_walks[node])):
                    nodes_random_walk_vector = [str(item) for item in nodes_random_walks[node][i]]
                    node_embedding_mat.append(list(np.float64(embedding_model.infer_vector(nodes_random_walk_vector))))
                embeddings[node] = node_embedding_mat
                pbar.update(1)
            pbar.close()
        
        elif embedding_method ==  'FeatureHasher':
            nodes_list = list(graph.nodes())
            nodes_random_walks = nx.get_node_attributes(graph, 'x')
            for key, values in nodes_random_walks.items():
                embeddings[key] = [embedding_model.transform([{str(num): 1 for num in row}]).toarray().tolist()[0] for row in values]

        elif embedding_method == 'Scale':
            nodes_random_walks = nx.get_node_attributes(graph, 'x')
            pbar = tqdm(total=len(nodes_random_walks.keys()))
            pbar.unit = ' Node'
            pbar.colour = 'Blue'
            for k, v in nodes_random_walks.items():
                pbar.desc = '\t\tNode ' + '{0: <16}'.format(str(k))
                temp_embedding = []
                for row in v:
                    temp_embedding.append(Generate_Embedings.get_scale_of_list(row))
                embeddings[k] = temp_embedding
                pbar.update(1)
            pbar.close()
        else:
            raise ValueError(f"Unsupported embedding method: {embedding_method}")
        return graph

    @staticmethod
    def create_embedding_vector(graph:nx.Graph, embedding_model, embedding_method: str = 'Word2Vec'):
        
        graph = graph
        
        if embedding_method == 'Word2Vec':
            nodes_random_walks = nx.get_node_attributes(graph, 'x')
            pbar = tqdm(total=len(graph.nodes()), desc="\tProcessing Nodes", colour='Blue')
            for node in graph.nodes():
                graph.nodes[node]['x'] = list(Generate_Embedings.get_Word2Vec_sentence_vector(nodes_random_walks[node], embedding_model))
                pbar.update(1)
            pbar.close()
        
        elif embedding_method == 'Doc2Vec':
            nodes_random_walks = nx.get_node_attributes(graph, 'x')
            pbar = tqdm(total=len(graph.nodes()), desc="\tProcessing Nodes", colour='Blue')
            for node in graph.nodes():
                nodes_random_walk_vector = [str(item) for item in nodes_random_walks[node]]
                graph.nodes[node]['x'] = list(np.float64(embedding_model.infer_vector(nodes_random_walk_vector)))
                pbar.update(1)
            pbar.close()
        
        elif embedding_method == 'FeatureHasher':
            nodes_random_walks = nx.get_node_attributes(graph, 'x')
            pbar = tqdm(total=len(graph.nodes()), desc="\tProcessing Nodes", colour='Blue')
            for node in graph.nodes():
                graph.nodes[node]['x'] = embedding_model.transform([{str(num): 1 for num in nodes_random_walks[node]}]).toarray().tolist()[0]
                pbar.update(1)
            pbar.close()
        
        elif embedding_method == 'Scale':
            nodes_random_walks = nx.get_node_attributes(graph, 'x')
            pbar = tqdm(total=len(graph.nodes()), desc="\tProcessing Nodes", colour='Blue')
            for node in graph.nodes():
                graph.nodes[node]['x'] = Generate_Embedings.get_scale_of_list(nodes_random_walks[node])
                pbar.update(1)
            pbar.close()
        
        else:
            raise ValueError(f"Unsupported embedding method: {embedding_method}")
        
        return graph

    @staticmethod
    def create_embedding_vector_return_dict(graph:nx.Graph, embedding_model, embedding_method: str = 'Word2Vec'):
        
        graph = graph
        embeddings = {}
        
        if embedding_method == 'Word2Vec':
            nodes_random_walks = nx.get_node_attributes(graph, 'x')
            pbar = tqdm(total=len(graph.nodes()), desc="\tProcessing Nodes", colour='Blue')
            for node in graph.nodes():
                embeddings[node] = list(Generate_Embedings.get_Word2Vec_sentence_vector(nodes_random_walks[node], embedding_model))
                pbar.update(1)
            pbar.close()
        
        elif embedding_method == 'Doc2Vec':
            nodes_random_walks = nx.get_node_attributes(graph, 'x')
            pbar = tqdm(total=len(graph.nodes()), desc="\tProcessing Nodes", colour='Blue')
            for node in graph.nodes():
                nodes_random_walk_vector = [str(item) for item in nodes_random_walks[node]]
                embeddings[node] = list(np.float64(embedding_model.infer_vector(nodes_random_walk_vector)))
                pbar.update(1)
            pbar.close()
        
        elif embedding_method == 'FeatureHasher':
            nodes_random_walks = nx.get_node_attributes(graph, 'x')
            pbar = tqdm(total=len(graph.nodes()), desc="\tProcessing Nodes", colour='Blue')
            for node in graph.nodes():
                embeddings[node] = embedding_model.transform([{str(num): 1 for num in nodes_random_walks[node]}]).toarray().tolist()[0]
                pbar.update(1)
            pbar.close()
        
        elif embedding_method == 'Scale':
            nodes_random_walks = nx.get_node_attributes(graph, 'x')
            pbar = tqdm(total=len(graph.nodes()), desc="\tProcessing Nodes", colour='Blue')
            for node in graph.nodes():
                embeddings[node] = Generate_Embedings.get_scale_of_list(nodes_random_walks[node])
                pbar.update(1)
            pbar.close()
        
        else:
            raise ValueError(f"Unsupported embedding method: {embedding_method}")
        
        return embeddings
    
    @staticmethod  
    def create_embedding_vector_mdoel(graphs_of_networks:list[nx.Graph], embedding_method: str,
                        dimensions: int = 64, window: int = 3, min_count: int = 1,
                        workers: int = 4, epochs: int = 100, batch_words: int = 4,
                        p: int = 1, q: float = 0.5):
        embedding_model = None
        nodes_random_walks = {}
        print(f"Generate networks random walk vector...\n")
        for i, graph in enumerate(graphs_of_networks):
            print(f"{i+1}- {graph.graph['name']}:  with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
            nodes_random_walks.update(nx.get_node_attributes(graph, 'x'))
        print(f'\nPreparing {embedding_method} model.\n\t' +
                f'vector_size: {bcolors.black_fg}{bcolors.yellow_bg_l}{bcolors.bold}{bcolors.bold}{dimensions}{bcolors.end_color},'+
                f' window: {bcolors.black_fg}{bcolors.yellow_bg_l}{bcolors.bold}{bcolors.bold}{window}{bcolors.end_color},'
                f' min_count: {bcolors.black_fg}{bcolors.yellow_bg_l}{bcolors.bold}{bcolors.bold}{min_count}{bcolors.end_color},'
                f' workers: {bcolors.black_fg}{bcolors.yellow_bg_l}{bcolors.bold}{bcolors.bold}{workers}{bcolors.end_color}')
        if embedding_method == 'Word2Vec':
            print(f"\tVocabulary size: {bcolors.yellow_fg}{bcolors.bold}{len(nodes_random_walks)}{bcolors.end_color}")
            embedding_model = Word2Vec(sentences=nodes_random_walks.values(), vector_size=3, window=window, min_count=min_count, workers=workers)
        elif embedding_method == 'Doc2Vec':
            documents = []
            for k, v in nodes_random_walks.items():
                documents.append(TaggedDocument(words=v, tags=[str(k)]))
            print(f"\tDocuments count: {bcolors.yellow_fg}{bcolors.bold}{len(documents)}{bcolors.end_color}")
            embedding_model = Doc2Vec(documents, vector_size=dimensions, window=window, min_count=min_count, workers=workers, epochs=epochs, dm=1)
        elif embedding_method == 'FeatureHasher':
            embedding_model = FeatureHasher(n_features=dimensions, input_type="dict")
        print(f"\t{bcolors.black_bg}{bcolors.green_fg}Mode {embedding_method} prepared successfully.{bcolors.end_color}")
        return embedding_model

    @staticmethod   
    def create_embedding_matrix_mdoel(graphs_of_networks:list[nx.Graph], embedding_method: str,
                        dimensions: int = 64, window: int = 3, min_count: int = 1,
                        workers: int = 4, epochs: int = 100, batch_words: int = 4,
                        p: int = 1, q: float = 0.5):
        embedding_model = None
        nodes_random_walks = {}
        print(f"Generate networks random walk matrix...\n")
        for i, graph in enumerate(graphs_of_networks):
            print(f"{i+1}- {graph.graph['name']}: with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
            nodes_random_walks.update(nx.get_node_attributes(graph, 'x'))
        print(f'\nPreparing {embedding_method} model.\n\t' +
                f'vector_size: {bcolors.black_fg}{bcolors.yellow_bg_l}{bcolors.bold}{bcolors.bold}{dimensions}{bcolors.end_color},'+
                f' window: {bcolors.black_fg}{bcolors.yellow_bg_l}{bcolors.bold}{bcolors.bold}{window}{bcolors.end_color},'
                f' min_count: {bcolors.black_fg}{bcolors.yellow_bg_l}{bcolors.bold}{bcolors.bold}{min_count}{bcolors.end_color},'
                f' workers: {bcolors.black_fg}{bcolors.yellow_bg_l}{bcolors.bold}{bcolors.bold}{workers}{bcolors.end_color}')
        if embedding_method == 'Word2Vec':
            sentences = [row for values in nodes_random_walks.values() for row in values]
            print(f"\tVocabulary size: {bcolors.yellow_fg}{bcolors.bold}{len(sentences)}{bcolors.end_color}")
            embedding_model = Word2Vec(sentences=sentences, vector_size=2, window=window, min_count=min_count, workers=workers)
        elif embedding_method == 'Doc2Vec':
            documents = []
            counter = 0
            for k, v in nodes_random_walks.items():
                for item in v:
                    documents.append(TaggedDocument(words=item, tags=[str(counter)]))
                counter += 1
            print(f"\tDocuments count: {bcolors.yellow_fg}{bcolors.bold}{len(documents)}{bcolors.end_color}")
            embedding_model = Doc2Vec(documents, vector_size=dimensions, window=window, min_count=min_count, workers=workers, epochs=epochs, dm=1)
        elif embedding_method == 'FeatureHasher':
            embedding_model = FeatureHasher(n_features=dimensions, input_type="dict")
        print(f"\t{bcolors.black_bg}{bcolors.green_fg}Mode {embedding_method} prepared successfully.{bcolors.end_color}")
        return embedding_model


    def write_graph_nodes_embedding(graph:nx.Graph, embedding_method:str, embedding_type:str,
                                    embedding_attribute:str, walk_length:int, walk_depth:int, num_walks:int,
                                    compression_method:str=None):
        result_file_name = f"{graph.graph['name']} nodes {embedding_attribute} {embedding_method} {embedding_type}"
        result_file_name += f" wl={walk_length}, wd={walk_depth} nw={num_walks}"
        print(f"\t\tWrite {graph.graph['name']} nodes embedding in file using " + 
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
    def load_graph_nodes_embedding(graph:nx.Graph, embedding_method:str, embedding_type:str,
                                embedding_attribute:str, walk_length:int, walk_depth:int, num_walks:int,
                                compression_method:str=None, load_in_RAM:bool=False):
        graph_nodes_embedding = None

        result_file_name_info = f"{graph.graph['name']} nodes {embedding_attribute} {embedding_method} {embedding_type}"
        result_file_name_info += f" wl={walk_length}, wd={walk_depth} nw={num_walks}"
        result_file = None
        if compression_method == 'lzma':
            result_file = graph.graph['result_path'] + result_file_name_info + ".xz"
            if os.path.isfile(result_file):
                if load_in_RAM:
                    print(f"\t{bcolors.yellow_fg}Nodes embedding loading...{bcolors.end_color}")
                    with lzma.open(result_file, "rt", encoding="utf-8") as f:
                        graph_nodes_embedding = json.load(f)
                    print(f"\t{bcolors.green_fg}Load embedding done.{bcolors.end_color}")
                else:
                    graph_nodes_embedding = "Embedding file exist and dont need to load on RAM."
                    print(f"\t{bcolors.green_fg}Embedding file exist.\n{bcolors.end_color}"
                        f"\t\tFile path: {bcolors.cyan_fg}{bcolors.underline}{graph.graph['result_path']}{bcolors.end_color}\n"
                        f"\t\tFile name: {bcolors.bold}{bcolors.italic}{result_file_name_info}.xz{bcolors.end_color}")
        elif compression_method == 'h5py':
            result_file = graph.graph['result_path'] + result_file_name_info +  ".h5"
            if os.path.isfile(result_file):
                if load_in_RAM:
                    print(f"\t{bcolors.yellow_fg}Nodes embedding loading...{bcolors.end_color}")
                    with h5py.File(result_file, "rt") as f:
                        graph_nodes_embedding = json.load(f)
                    print(f"\t{bcolors.green_fg}Load embedding done.{bcolors.end_color}")
                else:
                    graph_nodes_embedding = "Embedding file exist and dont need to load on RAM."
                    print(f"\t{bcolors.green_fg}Embedding file exist.\n{bcolors.end_color}"
                        f"\t\tFile path: {bcolors.cyan_fg}{bcolors.underline}{graph.graph['result_path']}{bcolors.end_color}\n"
                        f"\t\tFile name: {bcolors.bold}{bcolors.italic}{result_file_name_info}.h5{bcolors.end_color}")
        else:
            result_file = graph.graph['result_path'] + result_file_name_info +  ".json"
            if os.path.isfile(result_file):
                if load_in_RAM:
                    print(f"\t{bcolors.yellow_fg}Nodes embedding loading...{bcolors.end_color}")
                    with open(result_file, "rt") as f:
                        graph_nodes_embedding= json.load(f)
                    print(f"\t{bcolors.green_fg}Load embedding done.{bcolors.end_color}")
                else:
                    graph_nodes_embedding = "Embedding file exist and dont need to load on RAM."
                    print(f"\t{bcolors.green_fg}Embedding file exist.\n{bcolors.end_color}"
                        f"\t\tFile path: {bcolors.cyan_fg}{bcolors.underline}{graph.graph['result_path']}{bcolors.end_color}\n"
                        f"\t\tFile name: {bcolors.bold}{bcolors.italic}{result_file_name_info}.json{bcolors.end_color}")
            
        return graph_nodes_embedding

    @staticmethod
    def load_nodes_embedding(result_path:str, network_name:str, embedding_method:str, embedding_type:str,
                                embedding_attribute:str, walk_length:int, walk_depth:int, num_walks:int,
                                compression_method:str=None, load_in_RAM:bool=True, tabs:str=''):
        graph_nodes_embedding = None
        result_file_name_info = f"{network_name} nodes {embedding_attribute} {embedding_method} {embedding_type}"
        result_file_name_info += f" wl={walk_length}, wd={walk_depth} nw={num_walks}"
        result_file = None
        
        if compression_method == 'lzma':
            result_file = result_path + result_file_name_info + ".xz"
            if os.path.isfile(result_file):
                if load_in_RAM:
                    print(f"{tabs}\t{bcolors.yellow_fg}Nodes embedding loading...{bcolors.end_color}")
                    with lzma.open(result_file, "rt", encoding="utf-8") as f:
                        graph_nodes_embedding = json.load(f)
                    print(f"{tabs}\t{bcolors.green_fg}Load embedding done.{bcolors.end_color}")
                else:
                    graph_nodes_embedding = "Embedding file exist and dont need to load on RAM."
                    print(f"{tabs}\t{bcolors.green_fg}Embedding file exist.\n{bcolors.end_color}"
                        f"{tabs}\t\tFile path: {bcolors.cyan_fg}{bcolors.underline}{result_path}{bcolors.end_color}\n"
                        f"{tabs}\t\tFile name: {bcolors.bold}{bcolors.italic}{result_file_name_info}.xz{bcolors.end_color}")
            else:
                print(f"{tabs}\t{bcolors.red_fg}File not found.{bcolors.end_color}")
                graph_nodes_embedding = "File not found."
        elif compression_method == 'h5py':
            result_file = result_path + result_file_name_info +  ".h5"
            if os.path.isfile(result_file):
                if load_in_RAM:
                    print(f"{tabs}\t{bcolors.yellow_fg}Nodes embedding loading...{bcolors.end_color}")
                    with h5py.File(result_file, "rt") as f:
                        graph_nodes_embedding = json.load(f)
                    print(f"{tabs}\t{bcolors.green_fg}Load embedding done.{bcolors.end_color}")
                else:
                    graph_nodes_embedding = "Embedding file exist and dont need to load on RAM."
                    print(f"{tabs}\t{bcolors.green_fg}Embedding file exist.\n{bcolors.end_color}"
                        f"{tabs}\t\tFile path: {bcolors.cyan_fg}{bcolors.underline}{result_path}{bcolors.end_color}\n"
                        f"{tabs}\t\tFile name: {bcolors.bold}{bcolors.italic}{result_file_name_info}.h5{bcolors.end_color}")
            else:
                print(f"{tabs}\t{bcolors.red_fg}File not found.{bcolors.end_color}")
                graph_nodes_embedding = "File not found."
        else:
            result_file = result_path + result_file_name_info +  ".json"
            if os.path.isfile(result_file):
                if load_in_RAM:
                    print(f"{tabs}\t{bcolors.yellow_fg}Nodes embedding loading...{bcolors.end_color}")
                    with open(result_file, "rt") as f:
                        graph_nodes_embedding= json.load(f)
                    print(f"{tabs}\t{bcolors.green_fg}Load embedding done.{bcolors.end_color}")
                else:
                    graph_nodes_embedding = "Embedding file exist and dont need to load on RAM."
                    print(f"{tabs}\t{bcolors.green_fg}Embedding file exist.\n{bcolors.end_color}"
                        f"{tabs}\t\tFile path: {bcolors.cyan_fg}{bcolors.underline}{result_path}{bcolors.end_color}\n"
                        f"{tabs}\t\tFile name: {bcolors.bold}{bcolors.italic}{result_file_name_info}.json{bcolors.end_color}")
            else:
                print(f"{tabs}\t{bcolors.red_fg}File not found.{bcolors.end_color}")
                graph_nodes_embedding = "File not found."
            
        return graph_nodes_embedding

    @staticmethod
    def load_embedding_model(networks_root_dir,
                            embedding_method, embedding_attribute, embedding_type,
                            dimensions, window, min_count, tabs:str=''):
        print(f"{tabs}{bcolors.yellow_fg}{bcolors.bold}Loadin embedding model...{bcolors.end_color}")
        embedding_model = None
        embedding_model_dir = '/'.join(networks_root_dir.split('/')[:-2]) + '/Embedding Models/'
        embedding_model_name = f"{embedding_method} {embedding_attribute} {embedding_type}"
        embedding_model_name += f" dimensions={dimensions} window={window} min_count={min_count}.embModel"
        if embedding_method == 'Word2Vec':
            embedding_model = Word2Vec.load(embedding_model_dir + embedding_model_name)
        elif embedding_method == 'Doc2Vec':
            embedding_model= Doc2Vec.load(embedding_model_dir + embedding_model_name)
            
        print(f"{tabs}\tfile path: {bcolors.cyan_fg}{bcolors.underline}{embedding_model_dir}{bcolors.end_color}")
        print(f"{tabs}\tfile name: {bcolors.italic}{embedding_model_name}{bcolors.end_color}")
        print(f"{tabs}{bcolors.green_fg}{bcolors.bold}Loadin embedding model successfully.{bcolors.end_color}")
        return embedding_model

    @staticmethod
    def write_multilayer_network_nodes_embedding(embeddins:dict, network_infos:dict,
                                                embedding_method:str, embedding_type:str, embedding_attribute:str,
                                                walk_length:int, walk_depth:int, num_walks:int,
                                                compression_method:str=None, tabs:str=''):
        result_file_name = f"{network_infos['name']} nodes {embedding_attribute} {embedding_method} {embedding_type}"
        result_file_name += f" wl={walk_length}, wd={walk_depth} nw={num_walks}"
        print(f"{tabs}Write {network_infos['name']} nodes embedding in file using " + 
            f"{bcolors.black_fg}{bcolors.yellow_bg_l}{bcolors.bold}{compression_method}{bcolors.end_color} compression method.")
        if compression_method == 'lzma':
            with lzma.open(network_infos['results_path'] + result_file_name + ".xz", "wt", encoding="utf-8") as f:
                    json.dump(embeddins, f)
                    result_file_name += ".xz"
        elif compression_method == 'h5py':
            with h5py.File(network_infos['results_path'] + result_file_name + ".h5", "w") as f:
                for layer , walks_data in embeddins.items():
                    for key, x in walks_data.items():
                        f.create_dataset(str(f"{layer},{key}"), data=x)
                result_file_name += ".h5"
        else:
            with open(network_infos['results_path'] + result_file_name + '.json', 'w') as f:
                json.dump(embeddins, f)
                result_file_name += ".json"
        
        print(f"{tabs}{bcolors.black_bg_l}{bcolors.green_fg}{bcolors.bold}Write don.{bcolors.end_color}\n"
            f"{tabs}File path: {bcolors.cyan_fg}{bcolors.underline}{network_infos['results_path']}{bcolors.end_color}\n"
            f"{tabs}File name: {bcolors.bold}{bcolors.italic}{result_file_name}{bcolors.end_color}")
        pass

    @staticmethod
    def load_multilayer_network_nodes_embedding(network_infos:dict,
                                                embedding_method:str, embedding_type:str, embedding_attribute:str,
                                                walk_length:int, walk_depth:int, num_walks:int,
                                                compression_method:str=None, load_in_RAM:bool=False, tabs:str=''):
        
        network_nodes_embedding = None

        result_file_name_info = f"{network_infos['name']} nodes {embedding_attribute} {embedding_method} {embedding_type}"
        result_file_name_info += f" wl={walk_length}, wd={walk_depth} nw={num_walks}"
        
        result_file = None
        if compression_method == 'lzma':
            result_file = network_infos['results_path'] + result_file_name_info + ".xz"
            if os.path.isfile(result_file):
                if load_in_RAM:
                    print(f"{tabs}{bcolors.black_bg}{bcolors.yellow_fg}Nodes embedding loading...{bcolors.end_color}")
                    with lzma.open(result_file, "rt", encoding="utf-8") as f:
                        network_nodes_embedding = json.load(f)
                    print(f"{tabs}{bcolors.black_bg}{bcolors.green_fg}Load embedding done.{bcolors.end_color}")
                else:
                    network_nodes_embedding = "Embedding file exist and dont need to load on RAM."
                    print(f"{tabs}{bcolors.black_bg}{bcolors.green_fg}Embedding file exist.\n{bcolors.end_color}"
                        f"{tabs}\tFile path: {bcolors.cyan_fg}{bcolors.underline}{network_infos['results_path']}{bcolors.end_color}\n"
                        f"{tabs}\tFile name: {bcolors.bold}{bcolors.italic}{result_file_name_info}.xz{bcolors.end_color}")
            else:
                print(f"{bcolors.bold}{bcolors.red_fg}File not found.{bcolors.ENDC}")
        elif compression_method == 'h5py':
            result_file = network_infos['results_path'] + result_file_name_info +  ".h5"
            if os.path.isfile(result_file):
                if load_in_RAM:
                    print(f"{tabs}{bcolors.black_bg}{bcolors.yellow_fg}Nodes embedding loading...{bcolors.end_color}")
                    with h5py.File(result_file, "rt") as f:
                        temp_network_nodes_embedding = json.load(f)
                    for k, v in temp_network_nodes_embedding.items():
                        layer_node = k.split(',')
                        layer = layer_node[0]
                        node = layer_node[1]
                        if layer in network_nodes_embedding.keys():
                            network_nodes_embedding[layer][node] = v
                        else:
                            network_nodes_embedding[layer] = {}
                            network_nodes_embedding[layer][node] = v
                    print(f"{tabs}{bcolors.black_bg}{bcolors.green_fg}Load embedding done.{bcolors.end_color}")
                else:
                    network_nodes_embedding = "Embedding file exist and dont need to load on RAM."
                    print(f"{tabs}{bcolors.black_bg}{bcolors.green_fg}Embedding file exist.\n{bcolors.end_color}"
                        f"{tabs}\tFile path: {bcolors.cyan_fg}{bcolors.underline}{network_infos['results_path']}{bcolors.end_color}\n"
                        f"{tabs}\tFile name: {bcolors.bold}{bcolors.italic}{result_file_name_info}.h5{bcolors.end_color}")
            else:
                print(f"{bcolors.bold}{bcolors.red_fg}File not found.{bcolors.ENDC}")
        else:
            result_file = network_infos['results_path'] + result_file_name_info +  ".json"
            if os.path.isfile(result_file):
                if load_in_RAM:
                    print(f"{tabs}{bcolors.black_bg}{bcolors.yellow_fg}Nodes embedding loading...{bcolors.end_color}")
                    with open(result_file, "rt") as f:
                        network_nodes_embedding= json.load(f)
                    print(f"{tabs}{bcolors.black_bg}{bcolors.green_fg}Load embedding done.{bcolors.end_color}")
                else:
                    network_nodes_embedding = "Embedding file exist and dont need to load on RAM."
                    print(f"{tabs}{bcolors.black_bg}{bcolors.green_fg}Embedding file exist.\n{bcolors.end_color}"
                        f"{tabs}\tFile path: {bcolors.cyan_fg}{bcolors.underline}{network_infos['results_path']}{bcolors.end_color}\n"
                        f"{tabs}\tFile name: {bcolors.bold}{bcolors.italic}{result_file_name_info}.json{bcolors.end_color}")
            else:
                print(f"{bcolors.bold}{bcolors.red_fg}File not found.{bcolors.ENDC}")
        return network_nodes_embedding
