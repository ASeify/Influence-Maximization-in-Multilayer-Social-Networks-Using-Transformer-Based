# version 1.0.1
import os
import numpy as np


class Resd_Network_Infos:

    @staticmethod
    def read_nodeFrom_layerFrom_nodeTo_layerTo(
        network_path: str,
        network_name: str,
        network_type: str,
        directed: bool = False,
        weighted: bool = False,
    ):
        network_layers_info = []
        entra_layer_edges = []
        inter_layer_edge = {}
        network_layers_nodes = []
        entra_layer_edges_features = {}
        layer_id = 0
        try:
            file_opened = open(network_path + network_name + network_type)
            content = file_opened.readlines()
            file_opened.close()
            i = 0
            for item in content:
                if item != None and item.strip() != "" and item.strip()[0] != "#":
                    content_parts = item.strip().split(" ")
                    if len(content_parts) > 3:
                        source_node_id = content_parts[0].strip()
                        source_node_layer_id = content_parts[1].strip()
                        destination_node_id = content_parts[2].strip()
                        destination_node_layer_id = content_parts[3].strip()

                        layer_id = int(source_node_layer_id)
                        while len(network_layers_nodes) < layer_id + 1:
                            network_layers_info.append([])
                            entra_layer_edges.append([])
                            network_layers_nodes.append([])

                        layer_id = int(destination_node_layer_id)
                        while len(network_layers_nodes) < layer_id + 1:
                            network_layers_info.append([])
                            entra_layer_edges.append([])
                            network_layers_nodes.append([])

                        edge_info = list()
                        edge_info.append(source_node_id)
                        edge_info.append(destination_node_id)
                        edge_features = []
                        if len(content_parts) > 4:
                            j = 4
                            while j < len(content_parts):
                                edge_features.append(content_parts[j])
                                j += 1

                        if source_node_layer_id == destination_node_layer_id:
                            layer_id = int(source_node_layer_id)
                            layer_info = list()
                            layer_info.append(layer_id)
                            layer_info.append(str(layer_id))

                            network_layers_info[layer_id] = layer_info
                            network_layers_nodes[layer_id].append(source_node_id)
                            network_layers_nodes[layer_id].append(destination_node_id)
                            entra_layer_edges[layer_id].append(edge_info)
                            if weighted:
                                edge_key = str(
                                    source_node_layer_id + " " + source_node_id + " " + destination_node_id
                                )
                                entra_layer_edges_features[edge_key] = edge_features

                        else:
                            if not inter_layer_edge.get(source_node_layer_id):
                                inter_layer_edge[source_node_layer_id] = {}
                            if not inter_layer_edge[source_node_layer_id].get(source_node_id):
                                inter_layer_edge[source_node_layer_id][source_node_id] = {}
                            if not inter_layer_edge[source_node_layer_id][source_node_id].get(destination_node_layer_id):
                                if weighted:
                                    inter_layer_edge[source_node_layer_id][source_node_id][destination_node_layer_id] = {}
                                    if not inter_layer_edge[source_node_layer_id][source_node_id][destination_node_layer_id].get(destination_node_id):
                                        inter_layer_edge[source_node_layer_id][source_node_id][destination_node_layer_id][destination_node_id] = {}
                                    inter_layer_edge[source_node_layer_id][source_node_id][destination_node_layer_id][destination_node_id] = edge_features
                                else:
                                    inter_layer_edge[source_node_layer_id][source_node_id][destination_node_layer_id] = []
                                    inter_layer_edge[source_node_layer_id][source_node_id][destination_node_layer_id].append(destination_node_id)
                            if not directed:
                                if not inter_layer_edge.get(destination_node_layer_id):
                                    inter_layer_edge[destination_node_layer_id] = {}
                                if not inter_layer_edge[destination_node_layer_id].get(destination_node_id):
                                    inter_layer_edge[destination_node_layer_id][destination_node_id] = {}
                                if not inter_layer_edge[destination_node_layer_id][
                                    destination_node_id].get(source_node_layer_id):
                                    if weighted:
                                        inter_layer_edge[destination_node_layer_id][destination_node_id][source_node_layer_id] = {}
                                        if not inter_layer_edge[destination_node_layer_id][destination_node_id][source_node_layer_id].get(source_node_id):
                                            inter_layer_edge[destination_node_layer_id][destination_node_id][source_node_layer_id][source_node_id] = {}
                                        inter_layer_edge[destination_node_layer_id][destination_node_id][source_node_layer_id][source_node_id] = edge_features
                                    else:
                                        inter_layer_edge[destination_node_layer_id][destination_node_id][source_node_layer_id] = []
                                        inter_layer_edge[destination_node_layer_id][destination_node_id][source_node_layer_id].append(source_node_id)

            # file_opened.close()
        except Exception as e:
            print(e)

        return (
            network_layers_info,
            network_layers_nodes,
            entra_layer_edges,
            entra_layer_edges_features,
            inter_layer_edge,
        )

    @staticmethod
    def read_layer_node_node_weight_dataset(file_path: os.path) -> dict:
        inputDataset = np.loadtxt(file_path)
        inputDataset = np.delete(inputDataset, [-1], axis=1)
        edgeList = {}
        for i in inputDataset:
            edgeList.setdefault(i[0], []).append((int(i[1]), int(i[2])))
        edgeList = edgeList.values()
        return edgeList
    
    @staticmethod
    def read_layer_node_node_dataset(file_path: os.path) -> dict:
        inputDataset = np.loadtxt(file_path)
        edgeList = {}
        for i in inputDataset:
            edgeList.setdefault(i[0], []).append((int(i[1]), int(i[2])))
        edgeList = edgeList.values()
        return edgeList
    
    @staticmethod
    def read_node_node_layer_dataset(file: os.path) -> dict:
        file_opened = open(file)
        content = file_opened.readlines()
        file_opened.close()
        layers_id = {}
        edgeList = {}
        layer_id = None
        layer_counter = 0
        for line in content:
            edge = line.strip().split(' ')
            if len(edge) > 3:
                print(edge)
            if edge[2] in layers_id:
                layer_id = layers_id[edge[2]]
            else:
                layer_counter += 1
                # print(edge[2], layer_counter)
                layers_id[edge[2]] = layer_counter
                layer_id = layers_id[edge[2]]
            if layer_id in edgeList:
                edgeList[layer_id].append((edge[0], edge[1]))
            else:
                edgeList[layer_id] = []
                edgeList[layer_id].append((edge[0], edge[1]))
            
        return edgeList
    
    @staticmethod
    def read_node_node_weight_layer_dataset(file: os.path) -> dict:
        file_opened = open(file)
        content = file_opened.readlines()
        file_opened.close()
        layers_id = {}
        edgeList = {}
        layer_id = None
        layer_counter = 0
        for line in content:
            edge = line.strip().split(' ')
            if len(edge) > 4:
                print(edge)
            if edge[3] in layers_id:
                layer_id = layers_id[edge[3]]
            else:
                layer_counter += 1
                layers_id[edge[3]] = layer_counter
                layer_id = layers_id[edge[3]]
            if layer_id in edgeList:
                edgeList[layer_id].append((edge[0], edge[1]))
            else:
                edgeList[layer_id] = []
                edgeList[layer_id].append((edge[0], edge[1]))
            
        return edgeList

    @staticmethod
    def read_network_from_seperated_files(root_parh:os.path, valid_files_type:list[str]=['txt', 'edgelist'], convert_lables:bool=False):
        network_layers_info = []
        network_layers_nodes = []
        entra_layer_edges = []
        nodes_labels = {}
        node_label_counter = 1
        files_list = next(os.walk(root_parh), (None, None, []))[2]
        layer_ocunter = 0
        for layer in files_list:
            file_type = layer.strip().split('.')[-1]
            if file_type in valid_files_type:
                layer_name = layer.split('.')[0]
                layer_id = layer_ocunter + 1
                network_layers_info.append((layer_id, layer_name))
                file_opened = open(root_parh + layer)
                content = file_opened.readlines()
                file_opened.close()
                network_layers_nodes.append([])
                entra_layer_edges.append([])
                for i, item in enumerate(content):
                    edge = item.strip().split(' ')
                    if len(edge) < 2:
                        print(i, edge)
                    source_node = edge[0]
                    dest_node = edge[1]
                    if convert_lables:
                        if edge[0] in nodes_labels:
                            source_node = nodes_labels[edge[0]]
                        else:
                            nodes_labels[edge[0]] = node_label_counter
                            source_node = nodes_labels[edge[0]]
                            node_label_counter += 1
                        if edge[1] in nodes_labels:
                            dest_node = nodes_labels[edge[1]]
                        else:
                            nodes_labels[edge[1]] = node_label_counter
                            dest_node = nodes_labels[edge[1]]
                            node_label_counter += 1
                    entra_layer_edges[-1].append((source_node, dest_node))
                    network_layers_nodes[-1].append(edge[0])
                    network_layers_nodes[-1].append(edge[1])
                layer_ocunter += 1

        return (
            network_layers_info,
            network_layers_nodes,
            entra_layer_edges,
        )
    pass