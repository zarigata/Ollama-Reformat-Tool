from __future__ import annotations

import os
import shutil
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

from src.core.config_manager import ConfigManager
from src.utils.logger import LOG_FILE, get_logger, setup_logger


class SettingsTab(ctk.CTkFrame):
    def __init__(
        self,
        master: ctk.CTkBaseClass | None = None,
        config_manager: ConfigManager | None = None,
        **kwargs,
    ) -> None:
        super().__init__(master, **kwargs)
        self.config_manager = config_manager or ConfigManager()
        self.logger = get_logger(__name__)
        self._download_dir_var = ctk.StringVar(value="")
        self._ollama_path_var = ctk.StringVar(value="")
        self._theme_var = ctk.StringVar(value="System")
        self._log_level_var = ctk.StringVar(value="INFO")
        self._build_ui()
        self._load_values()

    def _build_ui(self) -> None:
        form_frame = ctk.CTkFrame(self)
        form_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Download directory
        download_label = ctk.CTkLabel(form_frame, text="Download Directory")
        download_label.grid(row=0, column=0, sticky="w", pady=(10, 5))
        download_entry = ctk.CTkEntry(form_frame, textvariable=self._download_dir_var, width=400)
        download_entry.grid(row=1, column=0, sticky="we", padx=(0, 10))
        browse_download_btn = ctk.CTkButton(
            form_frame,
            text="Browse",
            command=self._browse_download_directory,
            width=120,
        )
        browse_download_btn.grid(row=1, column=1, pady=(0, 10))

        # Ollama path
        ollama_label = ctk.CTkLabel(form_frame, text="Ollama Executable Path")
        ollama_label.grid(row=2, column=0, sticky="w", pady=(10, 5))
        ollama_entry = ctk.CTkEntry(form_frame, textvariable=self._ollama_path_var, width=400)
        ollama_entry.grid(row=3, column=0, sticky="we", padx=(0, 10))
        browse_ollama_btn = ctk.CTkButton(
            form_frame,
            text="Browse",
            command=self._browse_ollama_path,
            width=120,
        )
        browse_ollama_btn.grid(row=3, column=1, pady=(0, 10))

        # Theme selector
        theme_label = ctk.CTkLabel(form_frame, text="Appearance Theme")
        theme_label.grid(row=4, column=0, sticky="w", pady=(10, 5))
        theme_menu = ctk.CTkOptionMenu(
            form_frame,
            values=["Dark", "Light", "System"],
            variable=self._theme_var,
            command=self._on_theme_change,
        )
        theme_menu.grid(row=5, column=0, sticky="w")

        # Log level selector
        log_label = ctk.CTkLabel(form_frame, text="Log Level")
        log_label.grid(row=6, column=0, sticky="w", pady=(10, 5))
        log_menu = ctk.CTkOptionMenu(
            form_frame,
            values=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            variable=self._log_level_var,
        )
        log_menu.grid(row=7, column=0, sticky="w")

        # Clear logs button
        clear_logs_btn = ctk.CTkButton(
            form_frame,
            text="Clear Logs",
            command=self._clear_logs,
            width=120,
        )
        clear_logs_btn.grid(row=7, column=1, padx=(10, 0))

        # Action buttons
        button_frame = ctk.CTkFrame(self)
        button_frame.pack(fill="x", padx=10, pady=(0, 10))
        save_btn = ctk.CTkButton(button_frame, text="Save Settings", command=self._save_settings)
        save_btn.pack(side="left", padx=(0, 10))
        reset_btn = ctk.CTkButton(button_frame, text="Reset to Defaults", command=self._reset_defaults)
        reset_btn.pack(side="left")

        form_frame.columnconfigure(0, weight=1)

    def _load_values(self) -> None:
        self._download_dir_var.set(self.config_manager.get("download_directory", ""))
        self._ollama_path_var.set(self.config_manager.get("ollama_path", ""))
        theme = self.config_manager.get("theme", "System")
        self._theme_var.set(theme)
        ctk.set_appearance_mode(theme)
        log_level = self.config_manager.get("log_level", "INFO")
        self._log_level_var.set(log_level)
        setup_logger(__name__, log_level)

    def _browse_download_directory(self) -> None:
        selected = filedialog.askdirectory()
        if selected:
            self._download_dir_var.set(selected)

    def _browse_ollama_path(self) -> None:
        initial_dir = self._ollama_path_var.get() or os.getcwd()
        selected = filedialog.askdirectory(initialdir=initial_dir)
        if selected:
            self._ollama_path_var.set(selected)

    def _on_theme_change(self, value: str) -> None:
        ctk.set_appearance_mode(value)

    def _save_settings(self) -> None:
        download_dir = Path(self._download_dir_var.get()).expanduser()
        ollama_path = Path(self._ollama_path_var.get()).expanduser() if self._ollama_path_var.get() else Path()

        if download_dir and not download_dir.exists():
            try:
                download_dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                messagebox.showerror("Invalid Directory", f"Unable to create download directory: {exc}")
                return

        if ollama_path and ollama_path != Path() and not ollama_path.exists():
            messagebox.showerror("Invalid Ollama Path", "The selected Ollama path does not exist.")
            return

        self.config_manager.set("download_directory", str(download_dir))
        self.config_manager.set("ollama_path", str(ollama_path) if ollama_path else "")
        self.config_manager.set("theme", self._theme_var.get())
        self.config_manager.set("log_level", self._log_level_var.get())
        self.config_manager.save_config()
        setup_logger(__name__, self._log_level_var.get())
        messagebox.showinfo("Settings Saved", "Configuration settings have been saved successfully.")

    def _reset_defaults(self) -> None:
        confirm = messagebox.askyesno("Reset to Defaults", "Are you sure you want to reset to default settings?")
        if not confirm:
            return
        self.config_manager.reset_to_defaults()
        self._load_values()
        messagebox.showinfo("Settings Reset", "Settings have been reset to defaults.")

    def _clear_logs(self) -> None:
        try:
            if LOG_FILE.exists():
                with LOG_FILE.open("w", encoding="utf-8"):
                    pass
            messagebox.showinfo("Logs Cleared", "Log file has been cleared.")
        except OSError as exc:
            messagebox.showerror("Error", f"Failed to clear log file: {exc}")
        self.logger.info("Log file cleared by user")
        try:
            shutil.copy(LOG_FILE, LOG_FILE)
        except Exception:
            pass
*** End***
