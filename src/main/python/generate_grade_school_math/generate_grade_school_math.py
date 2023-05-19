import os
import json
import random
import training_pb2 as trn
import numpy as np

def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


def get_examples(path, split):
    path = os.path.join(path, f"{split}.jsonl")
    examples = read_jsonl(path)
    for ex in examples:
        ex.update(question=ex["question"] + "\n")
        ex.update(answer=ex["answer"] + "<|endoftexl|>")

    print(f"{len(examples)} {split} examples")
    return examples

def main():

    examples = get_examples("../grade-school-math/grade_school_math/data/", "train")

    # Print all
    #for example in examples:
    #    print(example)
    #    print(f"########")
    #    print()

    # Print a random example 
    #randomnum = random.randint(0, len(examples))
    #print ("example " , randomnum)
    #print (examples[randomnum])

    # Get maximum length of features and labels
    features_len = len(max(examples, key=lambda x:x["question"])["question"])

    # Get standard deviation of the questions
    listr = []
    for example in examples:
        listr.append(len(example["question"]))
    print("mean: ", np.mean(listr), "; stddev:", np.std(listr))
    print("min: ", min(listr), "; max:", max(listr))
    
    #print("longest answer length: ", features_len)
    #labels_len = max()


    result = trn.DataSetPackage()


if __name__ == "__main__":
    main()