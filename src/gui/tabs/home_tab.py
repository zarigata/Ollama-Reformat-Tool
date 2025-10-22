from __future__ import annotations

import customtkinter as ctk


class HomeTab(ctk.CTkFrame):
    def __init__(self, master: ctk.CTkBaseClass | None = None, **kwargs) -> None:
        super().__init__(master, **kwargs)
        self._build_ui()

    def _build_ui(self) -> None:
        title = ctk.CTkLabel(self, text="Welcome", font=ctk.CTkFont(size=20, weight="bold"))
        title.pack(anchor="w", pady=(0, 12))

        intro = ctk.CTkLabel(
            self,
            text="Use the tabs to manage downloads, conversions, your library, and settings.",
            justify="left",
        )
        intro.pack(anchor="w")

        placeholder = ctk.CTkFrame(self)
        placeholder.pack(expand=True, fill="both", pady=(20, 0))
        placeholder_label = ctk.CTkLabel(placeholder, text="Dashboard overview coming soon")
        placeholder_label.pack(expand=True)
