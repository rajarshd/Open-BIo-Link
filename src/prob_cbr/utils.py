import os
import logging
from tqdm import tqdm
import pickle
import numpy as np
from typing import *
from prob_cbr.data.data_utils import read_graph


def get_adj_mat(kg_file, entity_vocab, rel_vocab):
    adj_mat = read_graph(kg_file, entity_vocab, rel_vocab)
    l2norm = np.linalg.norm(adj_mat, axis=-1)
    adj_mat = adj_mat / l2norm.reshape(l2norm.shape[0], 1)
    return adj_mat


def get_programs(e: str, ans: str, all_paths_around_e: List[List[str]]):
    """
    Given an entity and answer, get all paths which end at that ans node in the subgraph surrounding e
    """
    all_programs = []
    for path in all_paths_around_e:
        for l, (r, e_dash) in enumerate(path):
            if e_dash == ans:
                # get the path till this point
                all_programs.append([x for (x, _) in path[:l + 1]])  # we only need to keep the relations
    return all_programs


def execute_one_program(train_map, e: str, path: List[str], depth: int, max_branch: int):
    """
    starts from an entity and executes the path by doing depth first search. If there are multiple edges with the same label, we consider
    max_branch number.
    """
    if depth == len(path):
        # reached end, return node
        return [e]
    next_rel = path[depth]
    next_entities = train_map[(e, path[depth])]
    # next_entities = list(set(self.train_map[(e, path[depth])] + self.args.rotate_edges[(e, path[depth])][:5]))
    if len(next_entities) == 0:
        # edge not present
        return []
    if len(next_entities) > max_branch:
        # select max_branch random entities
        next_entities = np.random.choice(next_entities, max_branch, replace=False).tolist()
    answers = []
    for e_next in next_entities:
        answers += execute_one_program(train_map, e_next, path, depth + 1, max_branch)
    return answers
