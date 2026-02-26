'''
Docstring for metadata_normalizer.data.preprocessing
'''
import pandas as pd
import re

class TagEncoder:
   
    def __init__(self):
        self.label2id = {}
        self.id2label = {}
    

    def buid_vocab_from_df(self, df: pd.DataFrame):

        labels_per_head = {
              "modality":set(), 
              "vendor": set(),
              "series_type": set(), 
              "plane": set(), 
              "acquisition":set(), 
              "body": set(), 
              "contrast": set()}
        
        for harmonized in df["harmonized"]:
            slots = self._parse_harmonized(harmonized)
            for head, value in slots.items():
                labels_per_head[head]. add(value)

            for head, values in labels_per_head.items():
                labels_per_head[head].add("none")
            
            
            for head , values in labels_per_head.items():
                sorted_values = sorted(values)
                self.label2id[head] = {v: i for i, v in enumerate(sorted_values)}
                self.id2label[head] = {i: v  for i, v in self.label2id[head].items()}

    
    def _parse_harmonized(self, harmonized: str) -> dict:

        tokens = re.findall(r"<([^>]+)>", harmonized or "")
        expected_slots = 7
        if len(tokens) < expected_slots:
            tokens = tokens + ["none"] * (expected_slots - len(tokens))

        heads = {
            "modality":   tokens[0],
            "vendor":     tokens[1],
            "series_type":tokens[2],
            "plane":      tokens[3],
            "acquisition":tokens[4],   
            "body":       tokens[5],
            "contrast":   tokens[6],
        }
        return heads
    
    def encode_row(self, row) -> dict:
        slots = self._parse_harmonized(row["harmonized"])
        return {
            head: self.label2id[head][slots.get(head, "none")]
            for head in self.label2id.keys()
        }
    






