# Requirments
Requires Python version >= 3.12.0 and pip version >= 23.2.1

Need to Install Packages first:</br>
Create virtual environment with name project_venv (follow standard steps to create virtual environment on your machine): </br>
Below is just a reference </br>
For Linux/Mac: </br>
`python3.12 -m venv project_venv` </br>
For Windows: </br>
`python -m venv project_venv` </br>

Activate virtual environment: </br>
Please follow your standard steps to activate virtual environment on your machine. </br>
Below is just a reference: </br>
For Max/Linux bash shell: </br>
`source project_venv/bin/activate` </br>

For Windows command prompt: </br>
`project_venv\Scripts\activate`

# To Run:
Can try following command script (if on Linux/MacOS) (note that the script will attempt to create a virtual environment and install packages):</br>
`./script_runner.sh`
**Note: Above script may fail (was tested on Ubuntu 20.04 only)**

Alternatively run the script_runner.ipynb file

# Note: Running preprocessing.py may take ~30mins-1hr and clustering.py will at least take ~75mins
**processed_nouns.csv and clusterer_state.pkl are the outputs of preprocessing.py and clustering.py respectively. You can just run the retriever.py block in the IPYNB (or comment out the lines to run the preprocessing.py and clustering.py files in shell script) to try out the model without needing to wait**