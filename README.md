# streamlit_app
Public Repository for ML Genre Classifier

The files in this repository are for the creation of a streamlit app to share Michael Renken's Multi Label Genre Classifier. Below is a quick overview of all of the files
The file for the application is hosted on google cloud, as it is too big for a free github repository.

-.gitignore
Files on my laptop not shared with git

-Procfile
specifies to streamlit to first run the shell file, then the app file

-TFDIF.sav
The code the model uses to vectorize an input

-app.py
The main code for the streamlit app. This specifies the infrustructure of the web app, including page elements, and how to take and process user inputs.

bash.exe.stackdump
-crash file log. Can ignore

genrelist.sav
- turns the outputs of the model into readable genres

mlknn_streamlit.sav
-mlknn version of this model. Not used in the final site. The final model is hosted on google cloud, as it is too large for a free github repository.

requirements.txt
-file that specifies the python packages streamlit needs to load for the application

setup.sh
- boilerplate setup file for the application on startup.
