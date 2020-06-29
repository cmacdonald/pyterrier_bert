import unittest
import pandas as pd
from pyterrier_bert.axioms import AxiomEstimator
from collections import defaultdict

# Python code to find frequency of each word 
def freq(str): 
  
    # break the string into list of words  
    str = str.split()          
    str2 = [] 
  
    # loop till string values present in list str 
    for i in str:              
  
        # checking for the duplicacy 
        if i not in str2: 
  
            # insert value in str2 
            str2.append(i)  
    d = defaultdict(int)
    for i in range(0, len(str2)): 
       d[str2[i]]+=1 
  

class TestAxiom(unittest.TestCase):

    def test_TFC1A_and_3(self):
        obj = AxiomEstimator(None)
        #obj._choose = lambda self: 0
        doc = pd.DataFrame([["q1", "queryterm1 queryterm2", "d1", "queryterm1"]], columns=["qid", "query", "docno", "body"])
        qrel = pd.DataFrame([["q1", "d1", 1]], columns=["qid", "docno", "label"])
        new_doc, new_qrel = obj._changeDF(doc, qrel)
        self.assertEqual(2, len(new_doc))
        self.assertEqual(2, len(new_qrel))
        
        seenNew = False
        for i in [0,1]:
            row = new_doc.iloc[i]
            print(row["docno"])
            if "%r" in row["docno"]:
                seenNew = True
                self.assertTrue("queryterm2" in row["body"] or freq(row["body"])["queryterm1"] == 2)



