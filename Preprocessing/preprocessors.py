"""This module contains all the Preprocessing classes."""

import os
from typing import Any

import pandas as pd

from storage.storage import StorageClient


class PreprocessorBlueprint:
    """An abstract class that other Preprocessors inherit from."""

    def __init__(self, local_path: str, gcs_path: str, storage_client: StorageClient):
        """Initialize a concrete Preprocessor."""
        self._storage_client = storage_client
        self._processed_data = None
        self._raw_data_local_path = os.path.join(local_path, "raw")
        self._processed_data_local_path = os.path.join(local_path, "preprocessed.csv")
        self._raw_data_gcs_path = os.path.join(gcs_path, "raw")
        self._processed_data_gcs_path = os.path.join(gcs_path, "preprocessed.csv")

    def _download_raw_data(self, force_download: bool):
        """Download the raw data from GCS.

        :param force_download: Download the raw data even if it already exists locally.
        """
        if not os.path.exists(self._raw_data_local_path) or force_download:
            self._storage_client.download_from_gcs_to_local_directory_or_file(
                local_path="", gcs_path=self._raw_data_gcs_path
            )

    def _load_raw_data_from_local(self):
        """Load the raw data from local directory. This needs to be implemented."""
        raise NotImplementedError

    def _preprocess(self, raw_data: Any) -> pd.DataFrame:
        """Preprocess the raw data. This needs to be implemented."""
        raise NotImplementedError

    def _save_data_to_local(self, processed_data: Any):
        """Save the processed data locally. This method needs to be overwritten in case we have multiple files to save."""
        processed_data.to_csv(self._processed_data_local_path, index=False)

    def process(
        self,
        force_download: bool = False,
        force_upload: bool = False,
        force_preprocessing: bool = False,
        write_to_local: bool = False,
    ):
        """Process a dataset.

        This method is the entrypoint of every concrete Dataset preprocessor. It downloads the raw data if necessary,
        stores it locally, loads it into memory and preprocesses it. After preprocessing, it saves it locally and
        uploads it to GCS.
        :param write_to_local: Write preprocessed file to local.
        :param force_preprocessing: Preprocess the dataset even if it already exists preprocessed in GCS.
        :param force_upload: Upload the preprocessed dataset even if it already exists preprocessed in GCS.
        :param force_download: Download the raw data even if it already exists locally.
        """
        exists_preprocessed_gcs = self._storage_client.blob_exists(gcs_path=self._processed_data_gcs_path)

        if exists_preprocessed_gcs and not force_preprocessing:
            # If the dataset is already processed, and we don't force preprocessing
            print("Not processing as the dataset is already processed in GCS and no force_preprocessing flag is set.")
            return

        self._download_raw_data(force_download=force_download)
        raw_data = self._load_raw_data_from_local()
        processed_data = self._preprocess(raw_data=raw_data)
        if write_to_local:
            self._save_data_to_local(processed_data=processed_data)

        if (not exists_preprocessed_gcs) or force_upload:
            self._storage_client.upload_local_directory_or_file_to_gcs(
                local_path=self._processed_data_local_path, gcs_path=self._processed_data_gcs_path
            )
