from nltk.corpus import LazyCorpusLoader, BracketParseCorpusReader
from nltk.tree import Tree
import copy
#from utils import *
def set_tag(tree, method=1):
    '''Traverse the tree'''
    if isinstance(tree, Tree):
        tree.set_label(simplify_functional_tag(tree.label(), method))
        return tree
    else:
        return tree
    
def simply_tree_tag(tree, method=1):
    tree = set_tag(tree, method)
    for child in tree:
        if isinstance(child, Tree):
            simply_tree_tag(child, method)
    return tree

def simplify_functional_tag(tag, method=1):
    #Remove the characters after '-'
    if method == 1:
        if '-' in tag:
            tag = tag.split('-')[0]
    else:
        if '|' in tag:
            tag = tag.split('|')[0]
        if '+' in tag:
            tag = tag.split('+')[-1]
    return tag

def set_leave_lower(tree_string):
    if isinstance(tree_string, Tree):
        tree = tree_string
    else:
        tree = Tree.fromstring(tree_string)
    for idx, _ in enumerate(tree.leaves()):
        tree_location = tree.leaf_treeposition(idx)
        non_terminal = tree[tree_location[:-1]]
        non_terminal[0] = non_terminal[0].lower()
    return tree

def convert2cnf(original_tree):
    '''
    Chomsky norm form
    '''
    tree = copy.deepcopy(original_tree)
    tree = simply_tree_tag(tree)
    
    #Remove cases like NP->DT, VP->NP
    tree.collapse_unary(collapsePOS=True, collapseRoot=True)
    #Convert to Chomsky
    tree.chomsky_normal_form()
    
    tree = simply_tree_tag(tree, 2)
    tree = set_leave_lower(tree)
    return tree

def get_train_test_data():
    '''
    Load training and test set from nltk corpora
    '''
    train_num = 3800
    #Split the data into training and test set
    test_index = [0, 1, 4, 12, 16, 19, 21, 35, 37, 42, 43, 44, 45, 47, 54, 56, 62, 63, 65, 68, 71, 76, 79, 83]
    treebank = LazyCorpusLoader('treebank/combined', BracketParseCorpusReader, r'wsj_.*\.mrg')
    cnf_train = treebank.parsed_sents()[:train_num]
    cnf_test = [treebank.parsed_sents()[i+train_num] for i in test_index]
    #Convert to Chomsky norm form, remove auxiliary labels
    cnf_train = [convert2cnf(t) for t in cnf_train]
    cnf_test = [convert2cnf(t) for t in cnf_test]
    return cnf_train, cnf_test