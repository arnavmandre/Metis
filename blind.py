"""Three explicit stages: prepare, fit/freeze, evaluate once."""
import argparse,json
from pathlib import Path
from src.blind_research import prepare,fit,evaluate
from src.historical import download_french
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','fit','evaluate']);p.add_argument('--archive');p.add_argument('--work',default='data/blind');p.add_argument('--out',default='outputs/blind');p.add_argument('--protocol',default='blind_protocol.json');a=p.parse_args()
    c=json.loads(Path(a.protocol).read_text())
    if a.stage=='prepare':prepare(a.archive or download_french(Path(a.work)/'french.zip'),a.work,c)
    elif a.stage=='fit':fit(a.work,a.out,c)
    else:evaluate(a.work,a.out)
