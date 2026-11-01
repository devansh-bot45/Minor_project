import urllib.request
import zipfile
import os

url = "https://archive.ics.uci.edu/static/public/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset.zip"
zip_path = "wisdm.zip"

print(f"Downloading from {url}...")
urllib.request.urlretrieve(url, zip_path)
print("Download complete.")

print("Extracting...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("wisdm_data")
print("Extraction complete.")

# The desired files are under wisdm_data/wisdm-dataset/raw/phone/accel and gyro
# We will just verify they exist
import glob
print(f"Accel files found: {len(glob.glob('wisdm_data/wisdm-dataset/raw/phone/accel/*.txt'))}")
