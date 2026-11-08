"""Start a browser UI at http://127.0.0.1:7860 ."""
from voice_ranker.ui import make_app

if __name__ == "__main__":
    app = make_app()
    app.queue(default_concurrency_limit=1)
    app.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True,
               share=False, show_error=True)
