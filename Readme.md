1. create a environment variable :
python -m venv env

2. activate environment variable:
env/Scripts/activate

3. Install Lib:
pip install fastapi uvicorn tensorflow pillow python-multipart numpy

4. Run the server
run the model :  uvicorn main:app --reload