# Personal Research Assistant


## Local Usage
0. Install streamlit using `pip install streamlit`
1. Navigate to the root directory
2. Open `.streamlit/secrets.toml` and input your api keys.
3. In terminal write ```streamlit run App.py``` (along with correct venv)



## Note:
If you have to clear agent chat memory and start fresh, click the `Reset` button on the page.

# Changelog:

## 0.1.3
- Added `Clear Files` functionality to give more control in which files are being considered for a query.

## 0.1.2
- Added `Download Response` button for downloading the current response as a markdown file.

## 0.1.1
- Added control over splitting size and overlap size to suit any query task. Helps if the uploaded file has to be analyzed as one chunk.