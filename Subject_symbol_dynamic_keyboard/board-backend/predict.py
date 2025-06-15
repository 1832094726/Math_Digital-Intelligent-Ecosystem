# -*- coding: utf-8 -*-

import os
import re
import sys
pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import jieba
import jieba.analyse
import json
LATEX = False

print("###Start init global data...###")

with open('data/dict.json','r',encoding='utf-8') as file:
    f = file.read()
    data = json.loads(f)
g_l1 = data["general"]["symbols"][0].keys()
g_l2 = data["general"]["symbols-scene"][0]
g_l3 = data["general"]["other_symbols"]
g_keywords = data["discrete"]["knowledge"]
g_scenes = data["discrete"]["symbols-scene"][0]
g_dict = data["discrete"]["symbols"][0]
g_symbols = g_dict.keys()
g_df = data["discrete"]["knowledge_graph"]
g_reconmmendSymbols = []
jieba.load_userdict(data["discrete"]["knowledge"])

jieba.analyse.set_idf_path("data/idf_dict.txt")

print("###End init global data...###")

def general_l1():
    l1 = list(g_l1)
    print('l1' + str(l1))
    return l1

def general_l2():
    l2 = dict(g_l2)
    print('l2' + str(l2))
    return l2

def general_l3():
    l3 = list(g_l3)
    print('l3' + str(l3))
    return l3

def get_knowledge_points(s):
    global g_keywords
    tags = jieba.analyse.extract_tags(s, topK=10)
    tags = list(filter(lambda x:x in g_keywords, tags))
    return tags[:4]


def findRelativeConcepts(concept):
    global g_df
    map = {}
    Max = 0
    noFather = True

    for row in g_df:
        fatherConcept = row['父概念']
        childrenConcepts = list(row['子概念'].split(','))
        map.setdefault(fatherConcept,0)
        if concept in childrenConcepts or concept==fatherConcept:
            return childrenConcepts
    return []

def findFatherConcept(concept):
    global g_df
    map = {}
    Max = 0
    noFather = True
    for row in g_df:
        fatherConcept = row['父概念']
        childrenConcepts = list(row['子概念'].split(','))
        map.setdefault(fatherConcept,0)
        if concept in childrenConcepts or concept==fatherConcept:
            return fatherConcept
    return ''


def get_symbols_from_concepts(concepts):
    global g_scenes
    global g_scenes
    map = {}
    for concept in concepts:
        if concept in g_scenes:
            symbols = g_scenes[concept]
            for symbol in symbols:
                map.setdefault(symbol,0)
                map[symbol]+=1
    map = dict(sorted(map.items(), key=lambda item: item[1], reverse=True))
    return list(map.keys())


def get_symbols_from_concept(concept):
    fatherConcept = findFatherConcept(concept)
    if fatherConcept==concept:
        childrenConcepts = findRelativeConcepts(concept)
        return get_symbols_from_concepts(childrenConcepts)
    else:
        return get_symbols_from_concepts([concept])


def get_l1_symbols(question):
    global g_keywords
    global g_symbols
    global g_reconmmendSymbols
    s_from_question = list(filter(lambda x:x in g_symbols, list(question)))
    tags = jieba.analyse.extract_tags(question, topK=20)
    keywords = list(filter(lambda x:x in g_keywords, tags))
    s_from_keywords = get_symbols_from_concepts(keywords)
    # 3. 从知识图谱扩展得到符号
    symbols_from_relative_concepts = []
    relative_concepts = []
    for keyword in keywords:
        result = findRelativeConcepts(keyword)
        relative_concepts.extend(result)
    relative_concepts = list(set(relative_concepts))
    for relative_concept in relative_concepts:
        symbols_from_relative_concepts.extend(get_symbols_from_concepts(relative_concept))
    all_symbols = list(set(s_from_question+s_from_keywords+symbols_from_relative_concepts))
    all_symbols = list(re.sub('[a-zA-XZ=!]+', '', ''.join(all_symbols)).replace("−", "")) #去掉字母（除了Y）,=,-
    if LATEX:
    # latex符号转换
        latexsymbols=[]
        for j in range(len(all_symbols)):
            for i in g_dict:
                if i == all_symbols[j]:
                    latex = g_dict[i]
                    latexsymbols.append(latex)
        all_symbols=latexsymbols
    print('l1_data' + str(all_symbols))
    g_reconmmendSymbols = all_symbols
    return all_symbols

def get_l2_symbols(s):
    global g_keywords
    global g_dict
    tags = jieba.analyse.extract_tags(s, topK=10)
    tags = list(filter(lambda x:x in g_keywords, tags))
    m = {}

    for tag in tags:
        symbols = get_symbols_from_concept(tag)
        if LATEX:
            l4_latexs = []
            for symbol in symbols:
                l4_latexs.append(g_dict[symbol])
            m.setdefault(tag,l4_latexs)
        else:
            m.setdefault(tag,symbols)
    print('l2_data'+str(m))
    return m

def get_l3_symbols():
    global g_reconmmendSymbols
    global g_symbols
    global g_dict
    l3_symbols = []
    symbols = []
    if LATEX:
        for symbol in g_symbols:
            symbols.append(g_dict[symbol])
            l3_symbols = list(filter(lambda x: x not in g_reconmmendSymbols, symbols))
    else:
        l3_symbols = list(filter(lambda x:x not in g_reconmmendSymbols,g_symbols))
    print('l3_data'+str(l3_symbols))
    return l3_symbols

