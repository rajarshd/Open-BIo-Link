import argparse
import numpy as np
import os
from tqdm import tqdm
from collections import defaultdict
import pickle
import torch
import uuid
from typing import *
import logging
import json
import sys
import wandb

from prob_cbr.data.data_utils import create_vocab, load_data, get_unique_entities, \
    read_graph, get_entities_group_by_relation, get_inv_relation, load_data_all_triples, create_adj_list
from prob_cbr.utils import execute_one_program, get_programs, get_adj_mat

logger = logging.getLogger()
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s \t %(message)s]",
                              "%Y-%m-%d %H:%M:%S")
ch.setFormatter(formatter)
logger.addHandler(ch)


def combine_maps(dir_name, output_file_name="precision.pkl"):
    """
    Combines all the individual maps
    :param dir_name:
    :return:
    """
    all_numerator_maps, all_denominator_maps = [], []
    combined_numerator_map, combined_denominator_map = {}, {}
    logger.info("Combining precision map located in {}".format(dir_name))
    for filename in os.listdir(dir_name):
        if filename.endswith("_precision_map.pkl"):
            with open(os.path.join(data_dir, filename), "rb") as fin:
                count_maps = pickle.load(fin)
                all_numerator_maps.append(count_maps["numerator_map"])
                all_denominator_maps.append(count_maps["denominator_map"])
    assert len(all_numerator_maps) == len(all_denominator_maps)
    for numerator_map, denominator_map in zip(all_numerator_maps, all_denominator_maps):
        for c, _ in numerator_map.items():
            for r, _ in numerator_map[c].items():
                for path, s_c in numerator_map[c][r].items():
                    if c not in combined_numerator_map:
                        combined_numerator_map[c] = {}
                    if r not in combined_numerator_map[c][r]:
                        combined_numerator_map[c][r] = {}
                    if path not in combined_numerator_map[c][r]:
                        combined_numerator_map[c][r][path] = 0
                    combined_numerator_map[c][r][path] += numerator_map[c][r][path]

        for c, _ in denominator_map.items():
            for r, _ in denominator_map[c].items():
                for path, s_c in denominator_map[c][r].items():
                    if c not in combined_denominator_map:
                        combined_denominator_map[c] = {}
                    if r not in combined_denominator_map[c][r]:
                        combined_denominator_map[c][r] = {}
                    if path not in combined_denominator_map[c][r]:
                        combined_denominator_map[c][r][path] = 0
                    combined_denominator_map[c][r][path] += combined_denominator_map[c][r][path]
    # now calculate precision
    ratio_map = {}
    for c, _ in combined_numerator_map.items():
        for r, _ in combined_numerator_map[c].items():
            if c not in ratio_map:
                ratio_map[c] = {}
            if r not in ratio_map[c]:
                ratio_map[c][r] = {}
            for path, s_c in combined_numerator_map[c][r].items():
                ratio_map[c][r][path] = s_c / combined_denominator_map[c][r][path]

    output_filenm = os.path.join(dir_name, output_file_name)
    logger.info("Dumping ratio map at {}".format(output_filenm))
    with open(output_filenm, "wb") as fout:
        pickle.dump(ratio_map, fout)
    logger.info("Done...")


def calc_precision_map_parallel(args, dir_name, job_id=0, total_jobs=1):
    """
    Calculates precision of each path wrt a query relation, i.e. ratio of how many times, a path was successful when executed
    to how many times the path was executed.
    Note: In the current implementation, we compute precisions for the paths stored in the path_prior_map
    :return:
    """
    logger.info("Calculating precision map")
    success_map, total_map = {}, {}  # map from query r to a dict of path and ratio of success
    # not sure why I am getting RuntimeError: dictionary changed size during iteration.
    train_map = [((e1, r), e2_list) for ((e1, r), e2_list) in args.train_map.items()]
    # sort this list so that every job gets the same list for processing
    train_map = [((e1, r), e2_list) for ((e1, r), e2_list) in sorted(train_map, key=lambda item: item[0])]
    job_size = len(train_map) / total_jobs
    st = job_id * job_size
    en = min((job_id + 1) * job_size, len(train_map))
    logger.info("Start of partition: {}, End of partition: {}".format(st, en))
    for e_ctr, ((e1, r), e2_list) in tqdm(enumerate(train_map)):
        if e_ctr < st or e_ctr >= en:
            # not this partition
            continue
        c = args.cluster_assignments[args.entity_vocab[e1]]
        if c not in success_map:
            success_map[c] = {}
        if c not in total_map:
            total_map[c] = {}
        if r not in success_map[c]:
            success_map[c][r] = {}
        if r not in total_map[c]:
            total_map[c][r] = {}
        if r in args.path_prior_map_per_relation[c]:
            paths_for_this_relation = args.path_prior_map_per_relation[c][r]
        else:
            continue  # if a relation is missing from prior map, then no need to calculate precision for that relation.
        for p_ctr, (path, _) in enumerate(paths_for_this_relation.items()):
            ans = execute_one_program(args.train_map, e1, path, depth=0, max_branch=100)
            if len(ans) == 0:
                continue
            # execute the path get answer
            if path not in success_map[c][r]:
                success_map[c][r][path] = 0
            if path not in total_map[c][r]:
                total_map[c][r][path] = 0
            for a in ans:
                if a in e2_list:
                    success_map[c][r][path] += 1
                total_map[c][r][path] += 1
    output_filenm = os.path.join(dir_name, "{}_precision_map.pkl".format(job_id))
    logger.info("Dumping precision map at {}".format(output_filenm))
    with open(output_filenm, "wb") as fout:
        pickle.dump({"numerator_map": success_map, "denominator_map": total_map}, fout)
    logger.info("Done...")


def combine_prior_maps(dir_name, output_file_name="path_prior_map.pkl"):
    all_program_maps = []
    combined_program_maps = {}
    logger.info("Combining prior maps located in {}".format(dir_name))
    for filename in os.listdir(dir_name):
        if filename.endswith("_path_prior_map.pkl"):
            logger.info("Reading {}".format(filename))
            with open(os.path.join(dir_name, filename), "rb") as fin:
                program_maps = pickle.load(fin)
                all_program_maps.append(program_maps)

    for program_map in all_program_maps:
        for c, _ in program_map.items():
            for r, _ in program_map[c].items():
                for path, s_c in program_map[c][r].items():
                    if c not in combined_program_maps:
                        combined_program_maps[c] = {}
                    if r not in combined_program_maps[c]:
                        combined_program_maps[c][r] = {}
                    if path not in combined_program_maps[c][r]:
                        combined_program_maps[c][r][path] = 0
                    combined_program_maps[c][r][path] += program_map[c][r][path]

    for c, _ in combined_program_maps.items():
        for r, path_counts in combined_program_maps[c].items():
            sum_path_counts = 0
            for p, p_c in path_counts.items():
                sum_path_counts += p_c
            for p, p_c in path_counts.items():
                combined_program_maps[c][r][p] = p_c / sum_path_counts

    output_filenm = os.path.join(dir_name, output_file_name)
    logger.info("Dumping ratio map at {}".format(output_filenm))
    with open(output_filenm, "wb") as fout:
        pickle.dump(combined_program_maps, fout)
    logger.info("Done...")


def calc_prior_path_prob_parallel(args, output_dir_name, job_id=0, total_jobs=1):
    """
    Calculate how probable a path is given a query relation, i.e P(path|query rel)
    For each entity in the graph, count paths that exists for each relation in the
    random subgraph.
    :return:
    """
    logger.info("Calculating prior map")
    programs_map = {}
    job_size = len(args.train_map) / total_jobs
    st = job_id * job_size
    en = min((job_id + 1) * job_size, len(args.train_map))
    logger.info("Start of partition: {}, End of partition: {}".format(st, en))
    train_map = [((e1, r), e2_list) for ((e1, r), e2_list) in args.train_map.items()]
    # sort this list so that every job gets the same list for processing
    train_map = [((e1, r), e2_list) for ((e1, r), e2_list) in sorted(train_map, key=lambda item: item[0])]
    for e_ctr, ((e1, r), e2_list) in enumerate(tqdm((train_map))):
        if e_ctr < st or e_ctr >= en:
            # not this partition
            continue
        c = args.cluster_assignments[args.entity_vocab[e1]]
        if c not in programs_map:
            programs_map[c] = {}
        if r not in programs_map[c]:
            programs_map[c][r] = {}
        all_paths_around_e1 = args.all_paths[e1]
        nn_answers = e2_list
        for nn_ans in nn_answers:
            programs = get_programs(e1, nn_ans, all_paths_around_e1)
            for p in programs:
                p = tuple(p)
                if len(p) == 1:
                    if p[0] == r:  # don't store query relation
                        continue
                if p not in programs_map[c][r]:
                    programs_map[c][r][p] = 0
                programs_map[c][r][p] += 1

    output_filenm = os.path.join(output_dir_name, "{}_path_prior_map.pkl".format(job_id))
    logger.info("Dumping path prior pickle at {}".format(output_filenm))

    with open(output_filenm, "wb") as fout:
        pickle.dump(programs_map, fout)

    logger.info("Done...")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Collect subgraphs around entities")
    parser.add_argument("--dataset_name", type=str, default="nell")
    parser.add_argument("--data_dir", type=str, default="../prob_cbr_data/")
    parser.add_argument("--expt_dir", type=str, default="../prob_cbr_expts/")
    parser.add_argument("--subgraph_file_name", type=str, default="")
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--test_file_name", type=str, default='',
                        help="Useful to switch between test files for FB122")
    parser.add_argument("--only_preprocess", action="store_true",
                        help="If on, only calculate prior and precision maps")
    parser.add_argument("--calculate_precision_map_parallel", action="store_true",
                        help="If on, only calculate precision maps")
    parser.add_argument("--calculate_prior_map_parallel", action="store_true",
                        help="If on, only calculate precision maps")
    parser.add_argument("--combine_precision_map", action="store_true",
                        help="If on, only combine precision maps")
    parser.add_argument("--combine_prior_map", action="store_true",
                        help="If on, only combine prior maps")
    parser.add_argument("--total_jobs", type=int, default=50,
                        help="Total number of jobs")
    parser.add_argument("--current_job", type=int, default=0,
                        help="Current job id")
    parser.add_argument("--name_of_run", type=str, default="unset")
    # Clustering args
    parser.add_argument("--linkage", type=float, default=0.8,
                        help="Clustering threshold")
    parser.add_argument("--use_wandb", type=int, choices=[0, 1], default=1, help="Set to 1 if using W&B")
    # Path sampling args
    parser.add_argument("--num_paths_to_collect", type=int, default=1000)
    parser.add_argument("--max_path_len", type=int, default=3)

    args = parser.parse_args()
    logger.info('COMMAND: %s' % ' '.join(sys.argv))
    if args.use_wandb:
        wandb.init(project='pr-cbr')
    assert 0 <= args.current_job < args.total_jobs and args.total_jobs > 0
    if args.name_of_run == "unset":
        args.name_of_run = str(uuid.uuid4())[:8]
    args.output_dir = os.path.join(args.expt_dir, "outputs", args.dataset_name, args.name_of_run)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    logger.info(f"Output directory: {args.output_dir}")

    dataset_name = args.dataset_name
    logger.info("==========={}============".format(dataset_name))
    data_dir = os.path.join(args.data_dir, "data", dataset_name)
    subgraph_dir = os.path.join(args.data_dir, "subgraphs", dataset_name)
    kg_file = os.path.join(data_dir, "full_graph.txt") if dataset_name == "nell" else os.path.join(data_dir,
                                                                                                   "graph.txt")
    if args.small:
        args.dev_file = os.path.join(data_dir, "dev.txt.small")
        args.test_file = os.path.join(data_dir, "test.txt")
    else:
        args.dev_file = os.path.join(data_dir, "dev.txt")
        args.test_file = os.path.join(data_dir, "test.txt") if not args.test_file_name \
            else os.path.join(data_dir, args.test_file_name)

    args.train_file = os.path.join(data_dir, "graph.txt") if dataset_name == "nell" else os.path.join(data_dir,
                                                                                                      "train.txt")

    if args.combine_prior_map:
        dir_name = os.path.join(args.data_dir, "data", args.dataset_name,
                                "linkage={}".format(args.linkage), "prior_maps")
        combine_prior_maps(dir_name)
        sys.exit(0)

    # if len(args.subgraph_file_name) == 0:
    #     args.subgraph_file_name = f"paths_{args.num_paths_to_collect}_{args.max_path_len}hop"
    #     if args.prevent_loops:
    #         args.subgraph_file_name += "_no_loops"
    #     args.subgraph_file_name += ".pkl"
    # if os.path.exists(os.path.join(subgraph_dir, args.subgraph_file_name)):
    #     logger.info("Loading subgraph around entities:")
    #     with open(os.path.join(subgraph_dir, args.subgraph_file_name), "rb") as fin:
    #         all_paths = pickle.load(fin)
    #     logger.info("Done...")
    # else:
    #     logger.info("Sampling subgraph around entities:")
    #     unique_entities = get_unique_entities(kg_file)
    #     train_adj_list = create_adj_list(kg_file)
    #     all_paths = defaultdict(list)
    #     for ctr, e1 in enumerate(tqdm(unique_entities)):
    #         paths = get_paths(args, train_adj_list, e1, max_len=args.max_path_len)
    #         if paths is None:
    #             continue
    #         all_paths[e1] = paths
    #     os.makedirs(subgraph_dir, exist_ok=True)
    #     with open(os.path.join(subgraph_dir, args.subgraph_file_name), "wb") as fout:
    #         pickle.dump(all_paths, fout)
    #
    # args.all_paths = all_paths
    entity_vocab, rev_entity_vocab, rel_vocab, rev_rel_vocab = create_vocab(kg_file)
    logger.info("Loading train map")
    train_map = load_data(kg_file)
    logger.info("Loading dev map")
    dev_map = load_data(args.dev_file)
    logger.info("Loading test map")
    test_map = load_data(args.test_file)
    eval_map = dev_map
    eval_file = args.dev_file
    if args.test:
        eval_map = test_map
        eval_file = args.test_file

    # making these part of args for easier access #hack
    args.entity_vocab = entity_vocab
    args.rel_vocab = rel_vocab
    args.rev_entity_vocab = rev_entity_vocab
    args.rev_rel_vocab = rev_rel_vocab
    args.train_map = train_map
    args.dev_map = dev_map
    args.test_map = test_map

    # cluster entities
    adj_mat = get_adj_mat(kg_file, entity_vocab, rel_vocab)
    if args.linkage > 0:
        if os.path.exists(os.path.join(data_dir, "linkage={}".format(args.linkage), "cluster_assignments.pkl")):
            logger.info("Clustering with linkage {} found, loading them....".format(args.linkage))
            fin = open(os.path.join(data_dir, "linkage={}".format(args.linkage), "cluster_assignments.pkl"), "rb")
            args.cluster_assignments = pickle.load(fin)
            fin.close()
        else:
            logger.info("Clustering entities with linkage = {}...".format(args.linkage))
            args.cluster_assignments = cluster_entities(adj_mat, args.linkage)
            logger.info("There are {} unique clusters".format(np.unique(args.cluster_assignments).shape[0]))
            dir_name = os.path.join(data_dir, "linkage={}".format(args.linkage))
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            logger.info("Dumping cluster assignments of entities at {}".format(dir_name))
            fout = open(os.path.join(dir_name, "cluster_assignments.pkl"), "wb")
            pickle.dump(args.cluster_assignments, fout)
            fout.close()
    else:
        args.cluster_assignments = np.zeros(adj_mat.shape[0])

    if args.calculate_prior_map_parallel:
        logger.info(
            "Calculating prior map. Current job id: {}, Total jobs: {}".format(args.current_job, args.total_jobs))
        assert args.cluster_assignments is not None
        assert args.all_paths is not None
        assert args.train_map is not None
        dir_name = os.path.join(args.data_dir, "data", args.dataset_name,
                                "linkage={}".format(args.linkage), "prior_maps")
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        calc_prior_path_prob_parallel(args, dir_name, args.current_job, args.total_jobs)

    if args.calculate_precision_map_parallel:
        logger.info(
            "Calculating precision map. Current job id: {}, Total jobs: {}".format(args.current_job, args.total_jobs))
        assert args.train_map is not None
        logger.info("Loading prior map...")
        dir_name = os.path.join(args.data_dir, "data", args.dataset_name,
                                "linkage={}".format(args.linkage))
        with open(os.path.join(dir_name, "prior_maps", "path_prior_map.pkl"), "rb") as fin:
            args.path_prior_map_per_relation = pickle.load(fin)
        assert args.path_prior_map_per_relation is not None
        dir_name = os.path.join(dir_name, "precision_maps")
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        calc_precision_map_parallel(args, dir_name, args.current_job, args.total_jobs)

