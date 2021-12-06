import argparse
import numpy as np
from scipy.special import logsumexp
import scipy.sparse
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
from prob_cbr.preprocessing.preprocessing import combine_path_splits
from prob_cbr.utils import get_programs
from prob_cbr.data.data_utils import create_vocab, load_vocab, load_data, get_unique_entities, \
    read_graph, get_entities_group_by_relation, get_inv_relation, load_data_all_triples, create_adj_list

logger = logging.getLogger('main')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s \t %(message)s]",
                              "%Y-%m-%d %H:%M:%S")
ch.setFormatter(formatter)
logger.addHandler(ch)


class ProbCBR(object):
    def __init__(self, args, train_map, eval_map, entity_vocab, rev_entity_vocab, rel_vocab, rev_rel_vocab, eval_vocab,
                 eval_rev_vocab, all_paths, rel_ent_map):
        self.args = args
        self.eval_map = eval_map
        self.train_map = train_map
        self.all_zero_ctr = []
        self.all_num_ret_nn = []
        self.entity_vocab, self.rev_entity_vocab, self.rel_vocab, self.rev_rel_vocab = entity_vocab, rev_entity_vocab, rel_vocab, rev_rel_vocab
        self.eval_vocab, self.eval_rev_vocab = eval_vocab, eval_rev_vocab
        self.all_paths = all_paths
        self.rel_ent_map = rel_ent_map
        self.num_non_executable_programs = []
        self.nearest_neighbor_1_hop = None
        self.sparse_adj_mats = {}
        self.create_sparse_adj_mats()

    def create_sparse_adj_mats(self):
        logger.info("Building sparse adjacency matrices")
        csr_data, csr_row, csr_col = {}, {}, {}
        for (e1, r), e2_list in self.train_map.items():
            _ = csr_data.setdefault(r, [])
            _ = csr_row.setdefault(r, [])
            _ = csr_col.setdefault(r, [])
            for e2 in e2_list:
                csr_data[r].append(1)
                csr_row[r].append(self.entity_vocab[e2])
                csr_col[r].append(self.entity_vocab[e1])
        for r in self.rel_vocab:
            self.sparse_adj_mats[r] = scipy.sparse.csr_matrix((np.array(csr_data[r], dtype=np.uint32),  # data
                                                               (np.array(csr_row[r], dtype=np.int64),  # row
                                                                np.array(csr_col[r], dtype=np.int64))  # col
                                                               ),
                                                              shape=(len(self.entity_vocab), len(self.entity_vocab))
                                                              )

    def set_nearest_neighbor_1_hop(self, nearest_neighbor_1_hop):
        self.nearest_neighbor_1_hop = nearest_neighbor_1_hop

    def get_nearest_neighbor_inner_product(self, e1: str, r: str, k: Optional[int] = 5) -> Union[List[str], None]:
        try:
            nearest_entities = [self.rev_entity_vocab[e] for e in
                                self.nearest_neighbor_1_hop[self.eval_vocab[e1]].tolist()]
            # remove e1 from the set of k-nearest neighbors if it is there.
            nearest_entities = [nn for nn in nearest_entities if nn != e1]
            # making sure, that the similar entities also have the query relation
            ctr = 0
            temp = []
            for nn in nearest_entities:
                if ctr == k:
                    break
                if len(self.train_map[nn, r]) > 0:
                    temp.append(nn)
                    ctr += 1
            nearest_entities = temp
        except KeyError:
            return None
        return nearest_entities

    def get_programs_from_nearest_neighbors(self, e1: str, r: str, nn_func: Callable, num_nn: Optional[int] = 5):
        all_programs = []
        nearest_entities = nn_func(e1, r, k=num_nn)
        if nearest_entities is None:
            self.all_num_ret_nn.append(0)
            return None
        self.all_num_ret_nn.append(len(nearest_entities))
        zero_ctr = 0
        for e in nearest_entities:
            if len(self.train_map[(e, r)]) > 0:
                paths_e = self.all_paths[e]  # get the collected 3 hop paths around e
                nn_answers = self.train_map[(e, r)]
                for nn_ans in nn_answers:
                    all_programs += get_programs(e, nn_ans, paths_e)
            elif len(self.train_map[(e, r)]) == 0:
                zero_ctr += 1
        self.all_zero_ctr.append(zero_ctr)
        return all_programs

    def rank_programs(self, list_programs: List[List[str]], r: str) -> List[List[str]]:
        """
        Rank programs.
        """
        # sort it by the path score
        unique_programs = set()
        for p in list_programs:
            unique_programs.add(tuple(p))
        # now get the score of each path
        path_and_scores = []
        for p in unique_programs:
            try:
                if self.args.use_only_precision_scores:
                    path_and_scores.append((p, self.args.precision_map[self.c][r][p]))
                else:
                    path_and_scores.append((p, self.args.path_prior_map_per_relation[self.c][r][p] *
                                            self.args.precision_map[self.c][r][p]))
            except KeyError:
                # TODO: Fix key error
                if len(p) == 1 and p[0] == r:
                    continue  # ignore query relation
                else:
                    # use the fall back score
                    try:
                        c = 0
                        if self.args.use_only_precision_scores:
                            score = self.args.precision_map_fallback[c][r][p]
                        else:
                            score = self.args.path_prior_map_per_relation_fallback[c][r][p] * \
                                    self.args.precision_map_fallback[c][r][p]
                        path_and_scores.append((p, score))
                    except KeyError:
                        # still a path or rel is missing.
                        path_and_scores.append((p, 0))

        # sort wrt counts
        sorted_programs = [k for k, v in sorted(path_and_scores, key=lambda item: -item[1])]

        return sorted_programs

    def execute_one_program(self, e: str, path: List[str], depth: int, max_branch: int):
        """
        starts from an entity and executes the path by doing depth first search. If there are multiple edges with the same label, we consider
        max_branch number.
        """
        src_vec = np.zeros((len(self.entity_vocab), 1), dtype=np.uint32)
        src_vec[self.entity_vocab[e]] = 1
        ent_vec = scipy.sparse.csr_matrix(src_vec)
        for r in path:
            ent_vec = self.sparse_adj_mats[r] * ent_vec
        final_counts = ent_vec.toarray().reshape(-1)
        answers = [self.rev_entity_vocab[d_e] for d_e, d_c in enumerate(final_counts) for _ in range(d_c)]
        return answers

    def execute_programs(self, e: str, r: str, path_list: List[List[str]], max_branch: Optional[int] = 1000) \
            -> Tuple[List[Tuple[str, float, List[str]]], List[List[str]]]:

        def _fall_back(r, p):
            """
            When a cluster does not have a query relation (because it was not seen during counting)
            or if a path is not found, then fall back to no cluster statistics
            :param r:
            :param p:
            :return:
            """
            c = 0  # one cluster for all entity
            try:
                score = self.args.path_prior_map_per_relation_fallback[c][r][p] * \
                        self.args.precision_map_fallback[c][r][p]
            except KeyError:
                # either the path or relation is missing from the fall back map as well
                score = 0
            return score

        all_answers = []
        not_executed_paths = []
        execution_fail_counter = 0
        executed_path_counter = 0
        for path in path_list:
            if executed_path_counter == self.args.max_num_programs:
                break
            ans = self.execute_one_program(e, path, depth=0, max_branch=max_branch)
            temp = []
            if self.args.use_path_counts:
                try:
                    if path in self.args.path_prior_map_per_relation[self.c][r] and path in \
                            self.args.precision_map[self.c][r]:
                        path_score = self.args.path_prior_map_per_relation[self.c][r][path] * \
                                     self.args.precision_map[self.c][r][path]
                    else:
                        # logger.info("This path was not there in the cluster for the relation.")
                        path_score = _fall_back(r, path)
                except KeyError:
                    # logger.info("Looks like the relation was not found in the cluster, have to fall back")
                    # fallback to the global scores
                    path_score = _fall_back(r, path)
            else:
                path_score = 1
            for a in ans:
                path = tuple(path)
                temp.append((a, path_score, path))
            ans = temp
            if ans == []:
                not_executed_paths.append(path)
                execution_fail_counter += 1
            else:
                executed_path_counter += 1
            all_answers += ans
        self.num_non_executable_programs.append(execution_fail_counter)
        return all_answers, not_executed_paths

    @staticmethod
    def rank_answers(list_answers: List[Tuple[str, float, List[str]]], aggr_type1="none", aggr_type2="sum") -> List[
        str]:
        """
        Different ways to re-rank answers
        """

        def rank_entities_by_max_score(score_map):
            """
            sorts wrt top value. If there are same values, then it sorts wrt the second value
            :param score_map:
            :return:
            """
            # sort wrt the max value
            sorted_score_map = sorted(score_map.items(), key=lambda kv: -kv[1][0])
            sorted_score_map_second_round = []
            temp = []
            curr_val = sorted_score_map[0][1][0]  # value of the first
            for (k, v) in sorted_score_map:
                if v[0] == curr_val:
                    temp.append((k, v))
                else:
                    sorted_temp = sorted(temp, key=lambda kv: -kv[1][1] if len(
                        kv[1]) > 1 else 1)  # sort wrt second highest score
                    sorted_score_map_second_round += sorted_temp
                    temp = [(k, v)]  # clear temp and add new val
                    curr_val = v[0]  # calculate new curr_val
            # do the same for remaining elements in temp
            if len(temp) > 0:
                sorted_temp = sorted(temp,
                                     key=lambda kv: -kv[1][1] if len(kv[1]) > 1 else 1)  # sort wrt second highest score
                sorted_score_map_second_round += sorted_temp
            return sorted_score_map_second_round

        count_map = {}
        uniq_entities = set()
        for e, e_score, path in list_answers:
            if e not in count_map:
                count_map[e] = {}
            if aggr_type1 == "none":
                count_map[e][path] = e_score  # just count once for a path type.
            elif aggr_type1 == "sum":
                if path not in count_map[e]:
                    count_map[e][path] = 0
                count_map[e][path] += e_score  # aggregate for each path
            else:
                raise NotImplementedError("{} aggr_type1 is invalid".format(aggr_type1))
            uniq_entities.add(e)
        score_map = defaultdict(int)
        for e, path_scores_map in count_map.items():
            p_scores = [v for k, v in path_scores_map.items()]
            if aggr_type2 == "sum":
                score_map[e] = np.sum(p_scores)
            elif aggr_type2 == "max":
                score_map[e] = sorted(p_scores, reverse=True)
            elif aggr_type2 == "noisy_or":
                score_map[e] = 1 - np.prod(1 - np.asarray(p_scores))
            elif aggr_type2 == "logsumexp":
                score_map[e] = logsumexp(p_scores)
            else:
                raise NotImplementedError("{} aggr_type2 is invalid".format(aggr_type2))
        if aggr_type2 == "max":
            sorted_entities_by_val = rank_entities_by_max_score(score_map)
        else:
            sorted_entities_by_val = sorted(score_map.items(), key=lambda kv: -kv[1])
        return sorted_entities_by_val

    @staticmethod
    def get_rank_in_list(e, predicted_answers):
        for i, e_to_check in enumerate(predicted_answers):
            if e == e_to_check:
                return i + 1
        return -1

    def get_hits(self, list_answers: List[str], gold_answers: List[str], query: Tuple[str, str]) \
            -> Tuple[float, float, float, float, float]:
        hits_1 = 0.0
        hits_3 = 0.0
        hits_5 = 0.0
        hits_10 = 0.0
        rr = 0.0
        (e1, r) = query
        all_gold_answers = self.args.all_kg_map[(e1, r)]
        for gold_answer in gold_answers:
            # remove all other gold answers from prediction
            filtered_answers = []
            for pred in list_answers:
                if pred in all_gold_answers and pred != gold_answer:
                    continue
                else:
                    filtered_answers.append(pred)

            rank = ProbCBR.get_rank_in_list(gold_answer, filtered_answers)
            if rank > 0:
                if rank <= 10:
                    hits_10 += 1
                    if rank <= 5:
                        hits_5 += 1
                        if rank <= 3:
                            hits_3 += 1
                            if rank <= 1:
                                hits_1 += 1
                rr += 1.0 / rank
        return hits_10, hits_5, hits_3, hits_1, rr

    @staticmethod
    def get_accuracy(gold_answers: List[str], list_answers: List[str]) -> List[float]:
        all_acc = []
        for gold_ans in gold_answers:
            if gold_ans in list_answers:
                all_acc.append(1.0)
            else:
                all_acc.append(0.0)
        return all_acc

    def do_symbolic_case_based_reasoning(self):
        num_programs = []
        num_answers = []
        all_acc = []
        non_zero_ctr = 0
        hits_10, hits_5, hits_3, hits_1, mrr = 0.0, 0.0, 0.0, 0.0, 0.0
        per_relation_scores = {}  # map of performance per relation
        per_relation_query_count = {}
        total_examples = 0
        learnt_programs = defaultdict(lambda: defaultdict(int))  # for each query relation, a map of programs to count
        for ex_ctr, ((e1, r), e2_list) in enumerate(tqdm(self.eval_map.items())):
            logger.info("Executing query {}".format(ex_ctr))
            # if e2_list is in train list then remove them
            # Normally, this shouldn't happen at all, but this happens for Nell-995.
            orig_train_e2_list = self.train_map[(e1, r)]
            temp_train_e2_list = []
            for e2 in orig_train_e2_list:
                if e2 in e2_list:
                    continue
                temp_train_e2_list.append(e2)
            self.train_map[(e1, r)] = temp_train_e2_list
            # also remove (e2, r^-1, e1)
            r_inv = get_inv_relation(r, args.dataset_name)
            temp_map = {}  # map from (e2, r_inv) -> outgoing nodes
            for e2 in e2_list:
                temp_map[(e2, r_inv)] = self.train_map[e2, r_inv]
                temp_list = []
                for e1_dash in self.train_map[e2, r_inv]:
                    if e1_dash == e1:
                        continue
                    else:
                        temp_list.append(e1_dash)
                self.train_map[e2, r_inv] = temp_list

            total_examples += len(e2_list)
            if e1 not in self.entity_vocab:
                all_acc += [0.0] * len(e2_list)
                # put it back
                self.train_map[(e1, r)] = orig_train_e2_list
                for e2 in e2_list:
                    self.train_map[(e2, r_inv)] = temp_map[(e2, r_inv)]
                continue  # this entity was not seen during train; skip?
            self.c = self.args.cluster_assignments[self.entity_vocab[e1]]
            all_programs = self.get_programs_from_nearest_neighbors(e1, r, self.get_nearest_neighbor_inner_product,
                                                                    num_nn=self.args.k_adj)
            if all_programs is None or len(all_programs) == 0:
                all_acc += [0.0] * len(e2_list)
                # put it back
                self.train_map[(e1, r)] = orig_train_e2_list
                for e2 in e2_list:
                    self.train_map[(e2, r_inv)] = temp_map[(e2, r_inv)]
                continue
            for p in all_programs:
                if p[0] == r:
                    continue
                if r not in learnt_programs:
                    learnt_programs[r] = {}
                p = tuple(p)
                if p not in learnt_programs[r]:
                    learnt_programs[r][p] = 0
                learnt_programs[r][p] += 1

            # filter the program if it is equal to the query relation
            temp = []
            for p in all_programs:
                if len(p) == 1 and p[0] == r:
                    continue
                temp.append(p)
            all_programs = temp

            if len(all_programs) > 0:
                non_zero_ctr += len(e2_list)

            all_uniq_programs = self.rank_programs(all_programs, r)
            num_programs.append(len(all_uniq_programs))
            # Now execute the program
            answers, not_executed_programs = self.execute_programs(e1, r, all_uniq_programs, max_branch=args.max_branch)
            answers = self.rank_answers(answers, self.args.aggr_type1, self.args.aggr_type2)
            if len(answers) > 0:
                acc = self.get_accuracy(e2_list, [k[0] for k in answers])
                _10, _5, _3, _1, rr = self.get_hits([k[0] for k in answers], e2_list, query=(e1, r))
                hits_10 += _10
                hits_5 += _5
                hits_3 += _3
                hits_1 += _1
                mrr += rr
                if args.output_per_relation_scores:
                    if r not in per_relation_scores:
                        per_relation_scores[r] = {"hits_1": 0, "hits_3": 0, "hits_5": 0, "hits_10": 0, "mrr": 0}
                        per_relation_query_count[r] = 0
                    per_relation_scores[r]["hits_1"] += _1
                    per_relation_scores[r]["hits_3"] += _3
                    per_relation_scores[r]["hits_5"] += _5
                    per_relation_scores[r]["hits_10"] += _10
                    per_relation_scores[r]["mrr"] += rr
                    per_relation_query_count[r] += len(e2_list)
            else:
                acc = [0.0] * len(e2_list)
            all_acc += acc
            num_answers.append(len(answers))
            # put it back
            self.train_map[(e1, r)] = orig_train_e2_list
            for e2 in e2_list:
                self.train_map[(e2, r_inv)] = temp_map[(e2, r_inv)]

        if args.output_per_relation_scores:
            for r, r_scores in per_relation_scores.items():
                r_scores["hits_1"] /= per_relation_query_count[r]
                r_scores["hits_3"] /= per_relation_query_count[r]
                r_scores["hits_5"] /= per_relation_query_count[r]
                r_scores["hits_10"] /= per_relation_query_count[r]
                r_scores["mrr"] /= per_relation_query_count[r]
            out_file_name = os.path.join(args.output_dir, "per_relation_scores.json")
            fout = open(out_file_name, "w")
            logger.info("Writing per-relation scores to {}".format(out_file_name))
            fout.write(json.dumps(per_relation_scores, sort_keys=True, indent=4))
            fout.close()

        logger.info(
            "Out of {} queries, atleast one program was returned for {} queries".format(total_examples, non_zero_ctr))
        logger.info("Avg number of programs {:3.2f}".format(np.mean(num_programs)))
        logger.info("Avg number of answers after executing the programs: {}".format(np.mean(num_answers)))
        logger.info("Accuracy (Loose): {}".format(np.mean(all_acc)))
        logger.info("Hits@1 {}".format(hits_1 / total_examples))
        logger.info("Hits@3 {}".format(hits_3 / total_examples))
        logger.info("Hits@5 {}".format(hits_5 / total_examples))
        logger.info("Hits@10 {}".format(hits_10 / total_examples))
        logger.info("MRR {}".format(mrr / total_examples))
        logger.info("Avg number of nn, that do not have the query relation: {}".format(
            np.mean(self.all_zero_ctr)))
        logger.info("Avg num of returned nearest neighbors: {:2.4f}".format(np.mean(self.all_num_ret_nn)))
        logger.info("Avg number of programs that do not execute per query: {:2.4f}".format(
            np.mean(self.num_non_executable_programs)))
        if self.args.print_paths:
            for k, v in learnt_programs.items():
                logger.info("query: {}".format(k))
                logger.info("=====" * 2)
                for rel, _ in learnt_programs[k].items():
                    logger.info((rel, learnt_programs[k][rel]))
                logger.info("=====" * 2)
        if self.args.use_wandb:
            # Log all metrics
            wandb.log({'hits_1': hits_1 / total_examples, 'hits_3': hits_3 / total_examples,
                       'hits_5': hits_5 / total_examples, 'hits_10': hits_10 / total_examples,
                       'mrr': mrr / total_examples, 'total_examples': total_examples, 'non_zero_ctr': non_zero_ctr,
                       'all_zero_ctr': self.all_zero_ctr, 'avg_num_nn': np.mean(self.all_num_ret_nn),
                       'avg_num_prog': np.mean(num_programs), 'avg_num_ans': np.mean(num_answers),
                       'avg_num_failed_prog': np.mean(self.num_non_executable_programs), 'acc_loose': np.mean(all_acc)})


def main(args):
    dataset_name = args.dataset_name
    logger.info("==========={}============".format(dataset_name))
    data_dir = os.path.join(args.data_dir, "data", dataset_name)
    subgraph_dir = os.path.join(args.data_dir, "subgraphs", dataset_name,
                                "paths_{}".format(args.num_paths_around_entities))
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
    rel_ent_map = get_entities_group_by_relation(args.train_file)

    logger.info("=========Config:============")
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
    logger.info("Loading vocabs...")
    entity_vocab, rev_entity_vocab, rel_vocab, rev_rel_vocab, eval_vocab, eval_rev_vocab = load_vocab(data_dir)
    # making these part of args for easier access #hack
    args.entity_vocab = entity_vocab
    args.rel_vocab = rel_vocab
    args.rev_entity_vocab = rev_entity_vocab
    args.rev_rel_vocab = rev_rel_vocab
    args.train_map = train_map
    args.dev_map = dev_map
    args.test_map = test_map

    logger.info("Loading combined train/dev/test map for filtered eval")
    all_kg_map = load_data_all_triples(args.train_file, os.path.join(data_dir, 'dev.txt'),
                                       os.path.join(data_dir, 'test.txt'))
    args.all_kg_map = all_kg_map

    ########### Load all paths ###########
    file_prefix = "paths_{}_path_len_{}_".format(args.num_paths_around_entities, args.max_path_len)
    all_paths = combine_path_splits(subgraph_dir, file_prefix=file_prefix)

    prob_cbr_agent = ProbCBR(args, train_map, eval_map, entity_vocab, rev_entity_vocab, rel_vocab,
                             rev_rel_vocab, eval_vocab, eval_rev_vocab, all_paths, rel_ent_map)
    ########### entity sim ###########
    if os.path.exists(os.path.join(args.data_dir, "data", args.dataset_name, "ent_sim.pkl")):
        with open(os.path.join(args.data_dir, "data", args.dataset_name, "ent_sim.pkl"), "rb") as fin:
            sim_and_ind = pickle.load(fin)
            sim = sim_and_ind["sim"]
            arg_sim = sim_and_ind["arg_sim"]
    else:
        logger.info(
            "Entity similarity matrix not found at {}. Please run the preprocessing script first to generate this matrix...".format(
                os.path.join(args.data_dir, "data", args.dataset_name, "ent_sim.pkl")))
        sys.exit(1)
    assert arg_sim is not None
    prob_cbr_agent.set_nearest_neighbor_1_hop(arg_sim)

    ########### cluster entities ###########
    dir_name = os.path.join(data_dir, "data", args.dataset_name, "linkage={}".format(args.linkage))
    cluster_file_name = os.path.join(dir_name, "cluster_assignments.pkl")
    if os.path.exists(cluster_file_name):
        with open(cluster_file_name, "rb") as fin:
            args.cluster_assignments = pickle.load(fin)
    else:
        logger.info(
            "Clustering file not found at {}. Please run the preprocessing script first".format(cluster_file_name))

    ########### load prior maps ###########
    path_prior_map_filenm = os.path.join(data_dir, "linkage={}".format(args.linkage), "prior_maps",
                                         "path_{}".format(args.num_paths_around_entities), "path_prior_map.pkl")
    logger.info("Loading path prior weights")
    if os.path.exists(path_prior_map_filenm):
        with open(path_prior_map_filenm, "rb") as fin:
            args.path_prior_map_per_relation = pickle.load(fin)
    else:
        logger.info(
            "Path prior files not found at {}. Please run the preprocessing script".format(path_prior_map_filenm))

    ########### load prior maps (fall-back) ###########
    linkage_bck = args.linkage
    args.linkage = 0.0
    bck_dir_name = os.path.join(data_dir, "linkage={}".format(args.linkage), "prior_maps",
                                "path_{}".format(args.num_paths_around_entities))
    path_prior_map_filenm_fallback = os.path.join(bck_dir_name, "path_prior_map.pkl")
    if os.path.exists(bck_dir_name):
        logger.info("Loading fall-back path prior weights")
        with open(path_prior_map_filenm_fallback, "rb") as fin:
            args.path_prior_map_per_relation_fallback = pickle.load(fin)
    else:
        logger.info("Fall-back path prior weights not found at {}. Please run the preprocessing script".format(
            path_prior_map_filenm_fallback))
    args.linkage = linkage_bck

    ########### load precision maps ###########
    precision_map_filenm = os.path.join(data_dir, "linkage={}".format(args.linkage), "precision_maps",
                                        "path_{}".format(args.num_paths_around_entities), "precision_map.pkl")
    logger.info("Loading precision map")
    if os.path.exists(precision_map_filenm):
        with open(precision_map_filenm, "rb") as fin:
            args.precision_map = pickle.load(fin)
    else:
        logger.info(
            "Path precision files not found at {}. Please run the preprocessing script".format(precision_map_filenm))

    ########### load precision maps (fall-back) ###########
    linkage_bck = args.linkage
    args.linkage = 0.0
    precision_map_filenm_fallback = os.path.join(data_dir, "linkage={}".format(args.linkage), "precision_maps",
                                                 "path_{}".format(args.num_paths_around_entities), "precision_map.pkl")
    logger.info("Loading fall-back precision map")
    if os.path.exists(precision_map_filenm_fallback):
        with open(precision_map_filenm_fallback, "rb") as fin:
            args.precision_map_fallback = pickle.load(fin)
    else:
        logger.info("Path precision fall-back files not found at {}. Please run the preprocessing script".format(
            precision_map_filenm_fallback))
    args.linkage = linkage_bck

    # Finally all files are loaded, do inference!
    prob_cbr_agent.do_symbolic_case_based_reasoning()


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
    parser.add_argument("--use_path_counts", type=int, choices=[0, 1], default=1,
                        help="Set to 1 if want to weight paths during ranking")
    # Clustering args
    parser.add_argument("--linkage", type=float, default=0.8,
                        help="Clustering threshold")
    # CBR args
    parser.add_argument("--k_adj", type=int, default=5,
                        help="Number of nearest neighbors to consider based on adjacency matrix")
    parser.add_argument("--max_num_programs", type=int, default=1000)
    # Output modifier args
    parser.add_argument("--name_of_run", type=str, default="unset")
    parser.add_argument("--output_per_relation_scores", action="store_true")
    parser.add_argument("--print_paths", action="store_true")
    parser.add_argument("--use_wandb", type=int, choices=[0, 1], default=0, help="Set to 1 if using W&B")
    # Path sampling args
    parser.add_argument("--num_paths_around_entities", type=int, default=1000)
    parser.add_argument("--max_path_len", type=int, default=3)
    parser.add_argument("--prevent_loops", type=int, choices=[0, 1], default=1)
    parser.add_argument("--max_branch", type=int, default=100)
    parser.add_argument("--aggr_type1", type=str, default="none", help="none/sum")
    parser.add_argument("--aggr_type2", type=str, default="sum", help="sum/max/noisy_or/logsumexp")
    parser.add_argument("--use_only_precision_scores", action="store_true")

    args = parser.parse_args()
    if args.aggr_type2 == "noisy_or":
        if args.aggr_type1 == "sum":
            logger.info("aggr_type1 cannot be sum, when aggr_type2 is noisy_or, exiting...")
            sys.exit(0)

    logger.info('COMMAND: %s' % ' '.join(sys.argv))
    if args.use_wandb:
        wandb.init(project='pr-cbr')

    if args.name_of_run == "unset":
        args.name_of_run = str(uuid.uuid4())[:8]
    args.output_dir = os.path.join(args.expt_dir, "outputs", args.dataset_name, args.name_of_run)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    logger.info(f"Output directory: {args.output_dir}")

    args.use_path_counts = (args.use_path_counts == 1)

    main(args)
