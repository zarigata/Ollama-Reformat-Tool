from __future__ import annotations

import customtkinter as ctk


class DownloadTab(ctk.CTkFrame):
    def __init__(self, master: ctk.CTkBaseClass | None = None, **kwargs) -> None:
        super().__init__(master, **kwargs)
        self._build_ui()

    def _build_ui(self) -> None:
        title = ctk.CTkLabel(self, text="Model Downloads", font=ctk.CTkFont(size=20, weight="bold"))
        title.pack(anchor="w", pady=(0, 12))

        description = ctk.CTkLabel(
            self,
            text="Manage Ollama model downloads. Configure sources and monitor progress here.",
            justify="left",
        )
        description.pack(anchor="w")

        download_area = ctk.CTkFrame(self)
        download_area.pack(expand=True, fill="both", pady=(20, 10))
        download_placeholder = ctk.CTkLabel(
            download_area,
            text="Download queue and controls coming soon",
        )
        download_placeholder.pack(expand=True)
