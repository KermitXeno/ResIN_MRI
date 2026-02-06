import os
from huggingface_hub import HfApi, hf_hub_download

UTILSp = os.path.dirname(os.path.abspath(__file__))
MRIp   = os.path.dirname(UTILSp)
MODELp = os.path.join(MRIp, "Models")
MODELp = os.path.join(MODELp, "weights")

def download_keras_files(repoid: str, localdir: str = ".", revision: str = "main"):

    api = HfApi()
    os.makedirs(localdir, exist_ok = True)

    files = api.list_repo_files(repo_id = repoid, revision = revision)

    kerasfiles = [f for f in files if f.endswith(".weights.h5")]

    if not kerasfiles:
        print("No weights files found in the repository.")
        return

    for file_path in kerasfiles:
        filename = os.path.basename(file_path)
        destination_path = os.path.join(localdir, filename)

        if os.path.exists(destination_path):
            os.remove(destination_path)

        hf_hub_download(
            repo_id = repoid,
            filename = file_path,
            revision = revision,
            local_dir = localdir,
            local_dir_use_symlinks = False,
            force_download = True
        )

        print(f"Downloaded and replaced: {filename}")

    print("Download complete.")

if __name__ == "__main__":
    download_keras_files(
        repoid = "KermitXeno/MRIBLandRESIN",
        localdir = MODELp
    )
