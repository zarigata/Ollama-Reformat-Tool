"""
Web interface for the One Ring AI Fine-Tuning Platform.

This module provides a Gradio-based web interface for interacting with the platform.
"""

import gradio as gr
from loguru import logger

from one_ring import __version__
from one_ring.core.config import settings
from one_ring.core.hardware import hardware_manager


def create_header():
    """Create the header section of the web interface."""
    with gr.Blocks() as header:
        gr.Markdown(
            f"""
            # 🧙‍♂️ One Ring to Tune Them All
            ### AI Model Fine-Tuning Platform v{__version__}
            """
        )
    return header


def create_model_selection_tab():
    """Create the model selection tab."""
    with gr.Blocks() as tab:
        with gr.Row():
            model_dropdown = gr.Dropdown(
                label="Base Model",
                choices=["meta-llama/Llama-2-7b-chat-hf", "mistralai/Mistral-7B-v0.1", "tiiuae/falcon-7b"],
                value="meta-llama/Llama-2-7b-chat-hf",
                interactive=True
            )
            
            model_status = gr.Textbox(
                label="Model Status",
                value="Not loaded",
                interactive=False
            )
        
        with gr.Row():
            load_btn = gr.Button("Load Model", variant="primary")
            unload_btn = gr.Button("Unload Model")
        
        # Model info section
        with gr.Accordion("Model Information", open=False):
            model_info = gr.JSON(
                label="Model Details",
                value={"status": "No model loaded"}
            )
        
        # Event handlers
        def load_model(model_name):
            logger.info(f"Loading model: {model_name}")
            # TODO: Implement model loading logic
            return {
                model_status: "Model loaded successfully",
                model_info: {
                    "name": model_name,
                    "status": "loaded",
                    "device": str(hardware_manager.get_default_device())
                }
            }
        
        def unload_model():
            logger.info("Unloading model")
            # TODO: Implement model unloading logic
            return {
                model_status: "Model unloaded",
                model_info: {"status": "No model loaded"}
            }
        
        load_btn.click(
            fn=load_model,
            inputs=[model_dropdown],
            outputs=[model_status, model_info]
        )
        
        unload_btn.click(
            fn=unload_model,
            inputs=[],
            outputs=[model_status, model_info]
        )
    
    return tab


def create_training_tab():
    """Create the model training tab."""
    with gr.Blocks() as tab:
        with gr.Row():
            with gr.Column(scale=2):
                # Dataset selection
                dataset_dropdown = gr.Dropdown(
                    label="Dataset",
                    choices=["Custom Dataset"],  # Will be populated dynamically
                    interactive=True
                )
                
                # Training parameters
                with gr.Accordion("Training Parameters", open=True):
                    epochs = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=3,
                        step=1,
                        label="Epochs"
                    )
                    
                    batch_size = gr.Slider(
                        minimum=1,
                        maximum=32,
                        value=4,
                        step=1,
                        label="Batch Size"
                    )
                    
                    learning_rate = gr.Number(
                        value=2e-5,
                        label="Learning Rate"
                    )
                
                # Advanced options
                with gr.Accordion("Advanced Options", open=False):
                    use_peft = gr.Checkbox(
                        label="Use PEFT (Parameter-Efficient Fine-Tuning)",
                        value=True
                    )
                    
                    peft_method = gr.Dropdown(
                        label="PEFT Method",
                        choices=["LoRA", "QLoRA", "AdaLoRA"],
                        value="LoRA",
                        interactive=True
                    )
                    
                    max_seq_length = gr.Number(
                        value=2048,
                        label="Max Sequence Length"
                    )
                
                # Start training button
                train_btn = gr.Button("Start Training", variant="primary")
            
            # Training output
            with gr.Column(scale=3):
                training_output = gr.Textbox(
                    label="Training Log",
                    lines=20,
                    max_lines=20,
                    interactive=False,
                    show_copy_button=True
                )
        
        # Training progress
        progress = gr.Slider(
            minimum=0,
            maximum=100,
            value=0,
            step=1,
            label="Training Progress",
            interactive=False
        )
        
        # Event handlers
        def start_training(
            dataset, epochs_val, batch_size_val, lr, 
            use_peft_flag, peft_method_val, max_seq_len
        ):
            logger.info("Starting training...")
            # TODO: Implement training logic
            for i in range(1, 101):
                yield {
                    training_output: f"Epoch {i//10 + 1}/{epochs_val} - Loss: {1.0 - i/100:.4f}\n",
                    progress: i
                }
        
        train_btn.click(
            fn=start_training,
            inputs=[
                dataset_dropdown, epochs, batch_size, learning_rate,
                use_peft, peft_method, max_seq_length
            ],
            outputs=[training_output, progress]
        )
    
    return tab


def create_inference_tab():
    """Create the model inference tab."""
    with gr.Blocks() as tab:
        with gr.Row():
            with gr.Column(scale=1):
                # Prompt input
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your prompt here...",
                    lines=5,
                    max_lines=10
                )
                
                # Generation parameters
                with gr.Accordion("Generation Parameters", open=False):
                    max_length = gr.Slider(
                        minimum=10,
                        maximum=2048,
                        value=512,
                        step=10,
                        label="Max Length"
                    )
                    
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=0.7,
                        step=0.1,
                        label="Temperature"
                    )
                    
                    top_p = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.9,
                        step=0.05,
                        label="Top-p (nucleus) sampling"
                    )
                
                # Generate button
                generate_btn = gr.Button("Generate", variant="primary")
                stop_btn = gr.Button("Stop")
            
            # Output
            with gr.Column(scale=1):
                output = gr.Textbox(
                    label="Generated Text",
                    lines=15,
                    interactive=False,
                    show_copy_button=True
                )
        
        # Event handlers
        def generate_text(prompt_text, max_len, temp, top_p_val):
            logger.info("Generating text...")
            # TODO: Implement text generation logic
            return f"Generated response for: {prompt_text[:50]}... (max_length={max_len}, temperature={temp}, top_p={top_p_val})"
        
        generate_btn.click(
            fn=generate_text,
            inputs=[prompt, max_length, temperature, top_p],
            outputs=output
        )
        
        stop_btn.click(
            fn=lambda: "Generation stopped by user",
            inputs=[],
            outputs=output
        )
    
    return tab


def create_web_interface():
    """Create the main web interface."""
    with gr.Blocks(
        title=f"One Ring v{__version__}",
        theme=gr.themes.Soft()
    ) as demo:
        # Header
        header = create_header()
        
        # Main tabs
        with gr.Tabs() as tabs:
            with gr.TabItem("Model"):
                model_tab = create_model_selection_tab()
            
            with gr.TabItem("Training"):
                training_tab = create_training_tab()
            
            with gr.TabItem("Inference"):
                inference_tab = create_inference_tab()
            
            with gr.TabItem("Settings"):
                with gr.Blocks():
                    gr.Markdown("## Settings")
                    # TODO: Add settings UI
                    gr.Markdown("Application settings will appear here.")
        
        # Footer
        with gr.Row():
            gr.Markdown(
                f"""
                ---
                *Running on {hardware_manager.info.device_type.name} with {hardware_manager.info.device_count} devices*  
                *Python {'.'.join(map(str, sys.version_info[:3]))} | PyTorch {torch.__version__} | Gradio {gr.__version__}*
                """
            )
    
    return demo


def serve_web_interface(host: str = "0.0.0.0", port: int = 7860, share: bool = False):
    """Start the Gradio web interface.
    
    Args:
        host: Host to bind the server to.
        port: Port to run the server on.
        share: Whether to create a public link for the interface.
    """
    logger.info(f"Starting web interface on {host}:{port}")
    
    # Create the interface
    demo = create_web_interface()
    
    # Launch the interface
    demo.launch(
        server_name=host,
        server_port=port,
        share=share,
        show_error=True
    )


if __name__ == "__main__":
    serve_web_interface()
