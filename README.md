# DeepEL: Combining Neural Perception and Description Logic Reasoning
Code for the neuro-symbolic framework DeepEL, that aims at combining DeepProbLog and the desscription logic EL.

## How to run
1. Clone the repository and navigate to the project directory:
```sh
git clone https://github.com/a-longato/deepel.git
cd deepel
```
2. Create and activate a virtual environment (recommended):

Linux/MacOS:
```sh
python3 -m venv venv
source venv/bin/activate
```
Windows:
```sh
python -m venv venv
venv\Scripts\activate
```
3. Install the dependencies:
```sh
pip install -r requirements.txt
```
4. The notebook **`main.ipynb`** contains the code for the experiments. A demo UI is also provided. To run it:

```sh
streamlit run demo.py
```