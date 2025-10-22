from __future__ import annotations

import customtkinter as ctk


class LibraryTab(ctk.CTkFrame):
    def __init__(self, master: ctk.CTkBaseClass | None = None, **kwargs) -> None:
        super().__init__(master, **kwargs)
        self._build_ui()

    def _build_ui(self) -> None:
        title = ctk.CTkLabel(self, text="Model Library", font=ctk.CTkFont(size=20, weight="bold"))
        title.pack(anchor="w", pady=(0, 12))

        description = ctk.CTkLabel(
            self,
            text="Browse and manage your converted models. Filtering and actions coming soon.",
            justify="left",
        )
        description.pack(anchor="w")

        library_area = ctk.CTkFrame(self)
        library_area.pack(expand=True, fill="both", pady=(20, 10))
        library_placeholder = ctk.CTkLabel(
            library_area,
            text="Library table and controls coming soon",
        )
        library_placeholder.pack(expand=True)
