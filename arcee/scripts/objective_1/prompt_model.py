from __future__ import annotations

from arcee.ee_models.ee_models import EEOllamaModel

ee_model = EEOllamaModel("llama3.2:1b")
ee_model.predict("", "")
