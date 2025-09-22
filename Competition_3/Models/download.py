import kagglehub
import shutil

def test():
    # Download latest version
    path = kagglehub.dataset_download("cdeotte/gemma2-9b-it-cv945")
    print(f"Downloaded to: {path}")

    # Move to your desired location
    desired_path = "/home/server2/Desktop/My_Github/Kaggle/Kaggle_Project/Competition_3/Models"
    shutil.move(path, desired_path)
    print(f"Moved to: {desired_path}")


def download_gemma2():
    # Download latest version
    path = kagglehub.dataset_download("cdeotte/gemma2-9b-it-bf16")
    print("Path to dataset files:", path)
    desired_path = "/home/server2/Desktop/My_Github/Kaggle/Kaggle_Project/Competition_3/Models"
    shutil.move(path, desired_path)
    print(f"Moved to: {desired_path}")

if __name__ == '__main__':
    download_gemma2()