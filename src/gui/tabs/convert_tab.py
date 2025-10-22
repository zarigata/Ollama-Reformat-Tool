from __future__ import annotations

import customtkinter as ctk


class ConvertTab(ctk.CTkFrame):
    def __init__(self, master: ctk.CTkBaseClass | None = None, **kwargs) -> None:
        super().__init__(master, **kwargs)
        self._build_ui()

    def _build_ui(self) -> None:
        title = ctk.CTkLabel(self, text="Convert Models", font=ctk.CTkFont(size=20, weight="bold"))
        title.pack(anchor="w", pady=(0, 12))

        description = ctk.CTkLabel(
            self,
            text="Convert downloaded models into supported formats. Configure options below.",
            justify="left",
        )
        description.pack(anchor="w")

        convert_area = ctk.CTkFrame(self)
        convert_area.pack(expand=True, fill="both", pady=(20, 10))
        convert_placeholder = ctk.CTkLabel(
            convert_area,
            text="Conversion controls coming soon",
        )
        convert_placeholder.pack(expand=True)
