import zipfile
import subprocess
import os

def extract_and_delete(zip_path, extract_to):
    # List all files in the zip archive using Python's zipfile module
    with zipfile.ZipFile(zip_path, 'r') as zf:
        file_list = zf.namelist()

    print(f"Found {len(file_list)} files in the archive.")

    # Process each file individually
    for file in file_list:
        print(f"Extracting: {file}")
        try:
            # Open the zip file again and extract the current file
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extract(file, path=extract_to)
        except Exception as e:
            print(f"Error extracting {file}: {e}")
            continue

        # After extraction, delete the file from the archive using zip -d
        print(f"Deleting {file} from the archive.")
        # The command is: zip -d <zipfile> <file>
        cmd = ['zip', '-d', zip_path, file]
        try:
            subprocess.run(cmd, check=True)
            print(f"Deleted {file} from the archive.")
        except subprocess.CalledProcessError as e:
            print(f"Error deleting {file} from the archive: {e}")
            # You can choose to break here or continue with the next file
            continue

if __name__ == "__main__":
    zip_file_path = "yfcc100m_phone.zip"  # Your large zip file
    extract_destination = "./images"   # Where to extract files

    # Create the extraction destination folder if it doesn't exist
    if not os.path.exists(extract_destination):
        os.makedirs(extract_destination)
    
    extract_and_delete(zip_file_path, extract_destination)
