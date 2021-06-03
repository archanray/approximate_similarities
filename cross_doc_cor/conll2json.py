
import json
import sys

def line2list(line):
    splt = line.split("\t")
    doc_id = splt[2]
    data = splt[3:7]
    data[0] = int(data[0])
    data[1] = int(data[1])
    data[-1] = bool(data[-1])
    return doc_id,data

def process(filename,outfile):
    from collections import defaultdict
    result = defaultdict(list)
    with open(filename, 'r') as fin:
        for line in fin:
            if not line.startswith('#'):
                doc_id,data = line2list(line)
                result[doc_id].append(data)
    with open(outfile,'w') as fout:
        json.dump(result,fout,indent=4)

if __name__ == "__main__":
    process(sys.argv[1] + '/train.conll',sys.argv[1] + '/train.json')
    process(sys.argv[1] + '/dev.conll', sys.argv[1] + '/dev.json')
    process(sys.argv[1] + '/test.conll', sys.argv[1] + '/test.json')
