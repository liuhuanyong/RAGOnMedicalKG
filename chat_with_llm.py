# ccoding = utf-8
import os
from question_classifier import *
from question_parser import *
from llm_server import *
from build_medicalgraph import *
import re

entity_parser = QuestionClassifier()

kg = MedicalGraph()
model = ModelAPI(MODEL_URL="http://10/generate")

class KGRAG():
    def __init__(self):
        self.cn_dict = {
                "name":"名称",
                "desc":"疾病简介",
                "cause":"疾病病因",
                "prevent":"预防措施",
                "cure_department":"治疗科室",
                "cure_lasttime":"治疗周期",
                "cure_way":"治疗方式",
                "cured_prob":"治愈概率",
                "easy_get":"易感人群",
                "belongs_to":"所属科室",
                "common_drug":"常用药品",
                "do_eat":"宜吃",
                "drugs_of":"生产药品",
                "need_check":"诊断检查",
                "no_eat":"忌吃",
                "recommand_drug":"好评药品",
                "recommand_eat":"推荐食谱",
                "has_symptom":"症状",
                "acompany_with":"并发症",
                "Check":"诊断检查项目",
                "Department":"医疗科目",
                "Disease":"疾病",
                "Drug":"药品",
                "Food":"食物",
                "Producer":"在售药品",
                "Symptom":"疾病症状"
        }
        self.entity_rel_dict = {
            "check":["name", 'need_check'],
            "department":["name", 'belongs_to'],
            "disease":["prevent", "cure_way", "name", "cure_lasttime", "cured_prob", "cause", "cure_department", "desc", "easy_get", 'recommand_eat', 'no_eat', 'do_eat', "common_drug", 'drugs_of', 'recommand_drug', 'need_check', 'has_symptom', 'acompany_with', 'belongs_to'],
            "drug":["name", "common_drug", 'drugs_of', 'recommand_drug'],
            "food":["name"],
            "producer":["name"],
            "symptom":["name", 'has_symptom'],
        }
        return

    def entity_linking(self, query):
        return entity_parser.check_medical(query)

    def link_entity_rel(self, query, entity, entity_type):
        cate = [self.cn_dict.get(i) for i in self.entity_rel_dict.get(entity_type)]
        prompt = "请判定问题：{query}所提及的是{entity}的哪几个信息，请从{cate}中进行选择，并以列表形式返回。".format(query=query, entity=entity, cate=cate)
        answer, history = model.chat(query=prompt, history=[])
        cls_rel = set([i for i in re.split(r"[\[。、, ;'\]]", answer)]).intersection(set(cate))
        print([prompt, answer, cls_rel])
        return cls_rel

    def recall_facts(self, cls_rel, entity_type, entity_name, depth=1):
        entity_dict = {
            "check":"Check",
            "department":"Department",
            "disease":"Disease",
            "drug":"Drug",
            "food":"Food",
            "producer":"Producer",
            "symptom":"Symptom"
        }
        # "MATCH p=(m:Disease)-[r*..2]-(n) where m.name = '耳聋' return p "
        sql = "MATCH p=(m:{entity_type})-[r*..{depth}]-(n) where m.name = '{entity_name}' return p".format(depth=depth, entity_type=entity_dict.get(entity_type), entity_name=entity_name)
        print(sql)
        ress = kg.g.run(sql).data()
        triples = set()
        for res in ress:
            p_data = res["p"]
            nodes = p_data.nodes
            rels = p_data.relationships
            for node in nodes:
                node_name = node["name"]
                for k,v in node.items():
                    # print(k)
                    if v == node_name:
                        continue
                    if self.cn_dict[k] not in cls_rel:
                        continue
                    triples.add("<" + ','.join([str(node_name), str(self.cn_dict[k]), str(v)]) + ">")
            for rel in rels:
                if rel.start_node["name"] == rel.end_node["name"]:
                    continue
                # print(rel["name"])
                if rel["name"] not in cls_rel:
                    continue
                triples.add("<" + ','.join([str(rel.start_node["name"]), str(rel["name"]), str(rel.end_node["name"])]) + ">")
        print(len(triples), list(triples)[:3])
        return list(triples)


    def format_prompt(self, query, context):
        prompt = "这是一个关于医疗领域的问题。给定以下知识三元组集合，三元组形式为<subject, relation, object>，表示subject和object之间存在relation关系" \
                 "请先从这些三元组集合中找到能够支撑问题的部分，在这里叫做证据，并基于此回答问题。如果没有找到，那么直接回答没有找到证据，回答不知道，如果找到了，请先回答证据的内容，然后在给出最终答案" \
                 "知识三元组集合为：" + str(context) + "\n问题是：" + query + "\n请回答："
        return prompt

    def chat(self, query):
        "{'耳聋': ['disease', 'symptom']}"
        print("step1: linking entity.....")
        entity_dict = self.entity_linking(query)
        depth = 1
        facts = list()
        answer = ""
        default = "抱歉，我在知识库中没有找到对应的实体，无法回答。"
        if not entity_dict:
            print("no entity founded...finished...")
            return default
        print("step2：recall kg facts....")
        for entity_name, types in entity_dict.items():
            for entity_type in types:
                rels = self.link_entity_rel(query, entity_name, entity_type)
                entity_triples = self.recall_facts(rels, entity_type, entity_name, depth)
                facts += entity_triples
        fact_prompt = self.format_prompt(query, facts)
        print("step3：generate answer...")
        answer = model.chat(query=fact_prompt, history=[])
        return answer

if __name__ == "__main__":
    chatbot = KGRAG()
    while 1:
        query = input("USER:").strip()
        answer = chatbot.chat(query)
        print("KGRAG_BOT:", answer)
