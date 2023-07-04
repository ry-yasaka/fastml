import os
import httpx, requests
import streamlit as st
import pandas as pd
from typing import List


def format_results(result_files: List[str]) -> pd.DataFrame:
    job_indices, filenames = [], []
    for _, job_id, filename in map(lambda s: s.split('/'), result_files):
        job_indices.append(job_id)
        filenames.append(filename)
    df = pd.DataFrame({'job_id': job_indices, 'filename': filenames})
    return df


BACKEND_HOST = os.environ.get('BACKEND_HOST', '127.0.0.1:8080')


image_files = st.file_uploader('Target image file',
                               type=['png', 'jpg'],
                               accept_multiple_files=True)

if len(image_files) > 0 and st.button('Submit'):
    files = [('files', file) for file in image_files]
    st.text(files[0])

    # r = requests.post(f'http://{BACKEND_HOST}/api/predict', files=files)
    # r = requests.post(f'http://{BACKEND_HOST}/api/predict')
    r = requests.post(f'http://{BACKEND_HOST}/api/predict', files=files)
    

    st.success(r.json())
    st.success('success!')



if st.button('Refresh'):
    st.success('Refreshed')
    
# r = httpx.get(f'http://{BACKEND_HOST}/results')
# df_results = format_results(r.json())
# st.write(df_results)

