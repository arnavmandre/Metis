import argparse
import yaml
from src.data_loader import load_csv,download,synthetic
from src.pipeline import run

if __name__=='__main__':
    p=argparse.ArgumentParser(description='Metis market regime research')
    p.add_argument('--csv'); p.add_argument('--download',action='store_true'); p.add_argument('--demo',action='store_true'); p.add_argument('--config',default='config.yaml'); p.add_argument('--out',default='outputs')
    a=p.parse_args()
    if sum([bool(a.csv),a.download,a.demo])!=1: p.error('Choose exactly one of --csv, --download, --demo')
    with open(a.config) as h: cfg=yaml.safe_load(h)
    prices=synthetic() if a.demo else download('data/raw/market.csv') if a.download else load_csv(a.csv)
    print(run(prices,cfg,a.out,a.demo).to_string())
