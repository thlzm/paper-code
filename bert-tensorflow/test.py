from bert_serving.client import BertClient
with BertClient(ip='39.101.138.44',port=5555,port_out=5556,timeout=10000) as bc:
    bc.encode(['你好'])