import os
import urllib.request
import tarfile
from tqdm import tqdm
import shutil

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    """Download a file from URL with progress bar"""
    print(f"Downloading {url}...")
    with DownloadProgressBar(unit='B', unit_scale=True,
                           miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def extract_tar(filepath, extract_path):
    """Extract a tar.gz file with progress bar"""
    print(f"Extracting {filepath}...")
    with tarfile.open(filepath, 'r:gz') as tar:
        total = len(tar.getmembers())
        with tqdm(total=total) as pbar:
            for member in tar.getmembers():
                tar.extract(member, path=extract_path)
                pbar.update(1)

def prepare_oxford_pets_dataset():
    """Download and prepare the Oxford-IIIT Pet Dataset"""
    
    # Create directories
    dataset_dir = "Oxford-IIIT Pet Dataset"
    download_dir = "downloads"
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(download_dir, exist_ok=True)
    
    # URLs for the dataset
    urls = {
        'images': 'https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz',
        'annotations': 'https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz'
    }
    
    try:
        # Download files
        for name, url in urls.items():
            output_path = os.path.join(download_dir, f"{name}.tar.gz")
            if not os.path.exists(output_path):
                download_url(url, output_path)
            else:
                print(f"Found existing {name}.tar.gz, skipping download")
        
        # Extract files
        for name in urls.keys():
            tar_path = os.path.join(download_dir, f"{name}.tar.gz")
            extract_tar(tar_path, dataset_dir)
        
        print("\nVerifying dataset structure...")
        # Verify the dataset structure
        required_dirs = [
            os.path.join(dataset_dir, "images"),
            os.path.join(dataset_dir, "annotations", "trimaps")
        ]
        
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                raise ValueError(f"Required directory not found: {dir_path}")
            
            # Count files in directory
            files = os.listdir(dir_path)
            print(f"Found {len(files)} files in {dir_path}")
        
        print("\nDataset preparation completed successfully!")
        print(f"Dataset is ready at: {os.path.abspath(dataset_dir)}")
        
        # Cleanup
        if os.path.exists(download_dir):
            print("\nCleaning up downloaded files...")
            shutil.rmtree(download_dir)
            
    except Exception as e:
        print(f"Error preparing dataset: {str(e)}")
        # Cleanup on failure
        if os.path.exists(download_dir):
            shutil.rmtree(download_dir)
        raise

if __name__ == "__main__":
    prepare_oxford_pets_dataset() 