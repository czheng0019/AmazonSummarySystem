#!/bin/bash
python3.12 -m venv project_venv
source project_venv/bin/activate
pip install -r requirements.txt
python3 preprocessing.py
python3 clustering.py
python3 retriever.py