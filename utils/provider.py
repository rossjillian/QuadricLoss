# Copyright (c) 2019 Nitin Agarwal (agarwal@uci.edu)

from __future__ import print_function
import numpy as np
import os
import sys
import scipy.sparse
import pandas as pd

import torch
import torch.utils.data as data

sys.path.append('./utils')
from pc_utils import *


class getDataset(data.Dataset):
    def __init__(self, root, train=True, data_augment=True, small=False, category='abc_2.5k', color=False):
        
        self.root = root
        self.train = train
        self.data_augment = data_augment    
        self.small = small         # test on a small dataset

        shape_paths = []           # path of all mesh files
        if color and os.path.isfile(os.path.join(root, category, 'color_data.csv')):     # mesh color feature
            self.color = pd.read_csv(os.path.join(root, category, 'color_data.csv'))
            self.color.set_index('mesh_file')
        elif color:
            self.color = generate_color_data(os.path.join(root, category))
            self.color.set_index('mesh_file')
        else:
            self.color = None
    
        if self.train:
            if self.small:
                self.file = os.path.join(root, category, 'train.txt')
            else:
                self.file = os.path.join(root, category, 'train_full.txt')
        else:
            if self.small:
                self.file = os.path.join(root, category, 'test.txt')
            else:
                self.file = os.path.join(root, category, 'test_full.txt')
    
        with open(self.file) as f:
            for line in f:
                shape_paths.append(os.path.join(root, category, line.strip()))

        self.datapath=[]
        if self.data_augment:
            """ Data augment by scaling and rotation """
            for line in shape_paths:
                mesh_path = line
                mesh={}
                mesh["rotate"] = False
                mesh["scale"] = True
                mesh["path"] = mesh_path
                self.datapath.append(mesh)
                
                mesh={}
                mesh["rotate"] = True
                mesh["scale"] = False
                mesh["path"] = mesh_path
                self.datapath.append(mesh)

                mesh={}
                mesh["rotate"] = True
                mesh["scale"] = True
                mesh["path"] = mesh_path
                self.datapath.append(mesh)

        else:
            for line in shape_paths:
                mesh={}
                mesh_path = line
                mesh["rotate"] = False
                mesh["scale"] = False
                mesh["path"] = mesh_path
                self.datapath.append(mesh)

    def __getitem__(self, index):

        fn = self.datapath[index]

        if fn["path"].endswith('obj'):
            vertices, faces = load_obj_data(fn["path"])
        else:
            vertices, faces = load_ply_data(fn["path"])

        # vertices = uniform_sampling(vertices, faces, 2500)

        if fn["scale"]:
            vertices = scale_vertices(vertices)
        if fn["rotate"]:
            vertices = rotate_vertices(vertices)
        
        vertices = normalize_shape(vertices)
        Q = compute_Q_matrix(vertices, faces)

        adj = get_adjacency_matrix(vertices, faces, K_max=271)
        face_coords = get_face_coordinates(vertices, faces, K_max=271)
        normal = compute_vertex_normals(vertices, faces)
        # vertices = farthest_point_sample(vertices, 2500)
        file_name = os.path.basename(os.path.normpath(fn["path"]))
        if self.color is not None:
            color = get_color(self.color, file_name, len(vertices))

        vertices = self.convert_to_tensor(vertices)
        Q = self.convert_to_tensor(Q)
        Q = Q.view(vertices.size()[0], -1)
        adj = self.convert_to_tensor(adj)
        normal = self.convert_to_tensor(normal)
        face_coords = self.convert_to_tensor(face_coords)
        if self.color is not None:
            color = self.convert_to_tensor(color)
        else:
            color = torch.zeros(3)

        return vertices, Q, adj, normal, face_coords, color

    def convert_to_tensor(self, x):
        x = torch.from_numpy(x.astype(np.float32))
        return x

    def __len__(self):
        return len(self.datapath)


if __name__ == "__main__":
    
    path = '../data'

    obj = getDataset(root=path, train=False, data_augment=False, small=False, category='mujoco_data', color=True)
    
    testdataloader = torch.utils.data.DataLoader(obj, batch_size=1, shuffle=False, num_workers=4)

    for i, data in enumerate(testdataloader, 0):
        v, q, adj, normal, f, c = data
        print(v)
        print(c)
        print(v.size(), q.size(), adj.size(), normal.size(), f.size(), c.size())
        inputs = torch.cat((v, c), 2)
        print(inputs)
        print(inputs.size())
