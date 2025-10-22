from __future__ import annotations

import customtkinter as ctk

from src.core.config_manager import ConfigManager
from src.gui.tabs.convert_tab import ConvertTab
from src.gui.tabs.download_tab import DownloadTab
from src.gui.tabs.home_tab import HomeTab
from src.gui.tabs.library_tab import LibraryTab
from src.gui.tabs.settings_tab import SettingsTab
from src.utils.logger import setup_logger


class App(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.config_manager = ConfigManager()
        self.logger = setup_logger(__name__, self.config_manager.get("log_level", "INFO"))
        self._configure_styles()
        self._configure_window()
        self._build_header()
        self._build_tabs()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _configure_styles(self) -> None:
        theme = self.config_manager.get("theme", "System")
        ctk.set_appearance_mode(theme)
        ctk.set_default_color_theme("blue")

    def _configure_window(self) -> None:
        self.title("Ollama Reformat Tool")
        width, height = 1100, 720
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x_position = int((screen_width / 2) - (width / 2))
        y_position = int((screen_height / 2) - (height / 2))
        self.geometry(f"{width}x{height}+{x_position}+{y_position}")
        self.minsize(900, 600)

    def _build_header(self) -> None:
        header_frame = ctk.CTkFrame(self)
        header_frame.pack(fill="x", padx=20, pady=20)
        title_label = ctk.CTkLabel(header_frame, text="Ollama Reformat Tool", font=ctk.CTkFont(size=24, weight="bold"))
        title_label.pack(side="left")

    def _build_tabs(self) -> None:
        tab_view = ctk.CTkTabview(self)
        tab_view.pack(expand=True, fill="both", padx=20, pady=(0, 20))

        home_tab_frame = tab_view.add("Home")
        home_tab = HomeTab(home_tab_frame)
        home_tab.pack(expand=True, fill="both", padx=10, pady=10)

        download_tab_frame = tab_view.add("Download")
        download_tab = DownloadTab(download_tab_frame)
        download_tab.pack(expand=True, fill="both", padx=10, pady=10)

        convert_tab_frame = tab_view.add("Convert")
        convert_tab = ConvertTab(convert_tab_frame)
        convert_tab.pack(expand=True, fill="both", padx=10, pady=10)

        library_tab_frame = tab_view.add("Library")
        library_tab = LibraryTab(library_tab_frame)
        library_tab.pack(expand=True, fill="both", padx=10, pady=10)

        settings_tab_frame = tab_view.add("Settings")
        settings_tab = SettingsTab(settings_tab_frame, self.config_manager)
        settings_tab.pack(expand=True, fill="both", padx=10, pady=10)

        self.tabs = {
            "home": home_tab,
            "download": download_tab,
            "convert": convert_tab,
            "library": library_tab,
            "settings": settings_tab,
        }

    def on_closing(self) -> None:
        self.config_manager.save_config()
        self.destroy()
*** End***
