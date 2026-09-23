"""Traceable adapter for Kenneth French's daily market and T-bill returns."""
import hashlib
import io
import re
import urllib.request
import zipfile
from pathlib import Path
import pandas as pd
URL='https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip'
DETAILS='https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/f-f_factors.html'
def french_prices(archive,start='1999-01-01',end='2025-12-31'):
    with zipfile.ZipFile(archive) as z:
        members=[n for n in z.namelist() if n.lower().endswith('.csv')]
        if len(members)!=1: raise ValueError('Expected exactly one CSV')
        text=z.read(members[0]).decode('utf-8-sig')
    lines=[line for line in text.splitlines() if re.match(r'^\d{8},',line)]
    frame=pd.read_csv(io.StringIO('\n'.join(lines)),header=None,names=['date','market_excess','smb','hml','rf'])
    frame['date']=pd.to_datetime(frame.date.astype(str),format='%Y%m%d')
    frame=frame.set_index('date').sort_index()
    if frame.index.has_duplicates or frame.isna().any().any(): raise ValueError('Invalid source records')
    if (frame[['market_excess','rf']]<=-99).any().any(): raise ValueError('Source has missing-value sentinels')
    frame=frame.loc[start:end]/100.
    returns=pd.DataFrame({'equity':frame.market_excess+frame.rf,'defensive':frame.rf})
    if (returns<=-1).any().any(): raise ValueError('Invalid total return')
    prices=100*(1+returns).cumprod(); prices.index.name='date'
    provenance={'source_url':URL,'description_url':DETAILS,'archive_sha256':hashlib.sha256(Path(archive).read_bytes()).hexdigest(),'source_header':text.splitlines()[0],'start':str(prices.index.min().date()),'end':str(prices.index.max().date()),'rows':len(prices),'equity':'US value-weight CRSP market total-return research proxy; not SPY','defensive':'Daily one-month Treasury-bill return research proxy; not a bond ETF','point_in_time':False,'publication_delay':'French data are released after observation dates; not a contemporaneous live feed.'}
    return prices,provenance

def download_french(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with urllib.request.urlopen(URL,timeout=30) as response:data=response.read()
    if not data.startswith(b'PK'):raise ValueError('Provider did not return a ZIP')
    path.write_bytes(data);return path
