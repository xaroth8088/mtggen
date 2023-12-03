import http.server
import socketserver

import numpy as np
from tensorflow.keras.models import load_model

from sampling_module import generate_text
from vectorizer import build_vectorizer


class MyHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        global g_model
        global g_vectorizer
        global g_temperature

        if self.path == '/card-data.js':
            self.send_response(200)
            self.send_header('Content-type', 'text/javascript')
            self.end_headers()

            generated_text = generate_text(
                200,
                g_model,
                g_vectorizer,
                temperature=g_temperature,
                show_token_breaks=False
            )

            self.wfile.write(f'export default\n{generated_text}'.encode())
        elif self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()

            with open('./html/card.html', 'r') as file:
                self.wfile.write(file)
        else:
            # Call the parent class method to handle other requests
            super().do_GET()


def start_server(
        model_path=None,
        data_path=None,
        temperature=None,
        port=8000
):
    global g_model
    global g_vectorizer
    global g_temperature

    # See note on vectorizer.py::custom_splitter() for more info on why we need this here
    with open(data_path, 'r', encoding='utf-8') as file:
        raw_text = np.array(file.readlines())

    g_vectorizer = build_vectorizer(raw_text)

    g_model = load_model(model_path)

    g_temperature=temperature

    # Specify the custom handler
    handler = MyHandler

    # Create the server
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"Serving on port {port}")
        httpd.serve_forever()
