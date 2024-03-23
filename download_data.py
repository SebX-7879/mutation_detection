import gdown
import os
import zipfile

## Download the challenge folder from Google Drive 

def download_data(url):
    print("Data downloading...")
    gdown.download_folder(url, quiet=False)
    print("Data successfully downloaded!")
    return None

def rename_folder(old_name='owkin_challenge', new_name='data'):
    try :
        os.rename(old_name, new_name)
        print("Folder {} successfully renamed into {} !".format(old_name, new_name))
    except FileNotFoundError:
        print("Folder \"{}\" not found !!!".format(old_name))
    
    return None

def unzip_file(file_path, output_path):
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(output_path)
    return None

## Unzip the content of a zipped folder, skipping a specified folder (here 'images')
## due to the limited number of files per user allowed on school's servers.

def unzip_file_skip_folder(file_path, output_path, skip_folder):
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        for file in zip_ref.namelist():
            if not file.startswith(file_path.split('/')[-1].split('.')[0] + '/' + skip_folder):
                zip_ref.extract(file, output_path)
    return None

def rename_file(old_name, new_name):
    try:
        os.rename(old_name, new_name)
        print("File {} successfully renamed into {} !".format(old_name, new_name))
    except FileNotFoundError:
        print("File \"{}\" not found !!!".format(old_name))
    return None


if __name__ == "__main__":
    ## Download the data
    url = 'https://drive.google.com/drive/folders/1W1rkukMXesdROzK1u-RZ_Nioh7Mm9BW1'
    download_data(url)
    
    ## Rename the folder to 'data'
    rename_folder()
    
    ## Unzip all the zip files in the data folder
    for file in os.listdir('data'):
        if file.endswith('.zip'):
            print("Unzipping file {}...".format(file))
            file_path = os.path.join('data', file)
            unzip_file_skip_folder(file_path, 'data', 'images')
            os.remove(file_path)
            print("File {} successfully unzipped and removed!".format(file))
            
    # Rename the file train_output
    rename_file('data/train_output_76GDcgx.csv', 'data/train_output.csv')
            
    

    
    
    