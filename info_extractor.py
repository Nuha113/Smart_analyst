"""Production-safe compatibility facade for the original Colab extractor.

The Llama model is loaded lazily by the backend service from LLAMA_MODEL_PATH.
No downloads, notebook commands, retrieval over the transaction itself, or demo
execution occur at import time.
"""

import pandas as pd

from backend.app.services.info_extractor_service import InfoExtractorService, check_prince_meeting


class Info_Extractor2:
    """Keep the original public name while delegating to the backend service."""

    column_order = [
        "name", "gender", "nationality", "extracted_ids", "extracted_phones",
        "problem", "keywords", "main_category", "requests_prince_meeting",
    ]

    def __init__(self, retriever=None):
        self.service = InfoExtractorService()

    def _check_prince_meeting(self, case_text):
        return check_prince_meeting(case_text)

    def extract(self, input_data):
        texts = [input_data] if isinstance(input_data, str) else list(input_data)
        rows = [self.service.extract(text) for text in texts]
        return pd.DataFrame(rows).reindex(columns=self.column_order)
