import http.server
import socketserver

from tensorflow.keras.models import load_model

from sampling_module import generate_text
from vectorizer import load_vectorizer


class MyHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        global g_model
        global g_vectorizer
        global g_temperature
        global g_max_output_tokens

        if self.path == '/card-data.js':
            self.send_response(200)
            self.send_header('Content-type', 'text/javascript')
            self.end_headers()

            generated_text = generate_text(
                g_max_output_tokens,
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
        temperature=None,
        vectorizer_path=None,
        max_output_tokens=None,
        listen_address=None,
        port=None
):
    global g_model
    global g_vectorizer
    global g_temperature
    global g_max_output_tokens

    g_vectorizer = load_vectorizer(vectorizer_path)
    g_model = load_model(model_path)

    g_temperature = temperature
    g_max_output_tokens = max_output_tokens

    # Specify the custom handler
    handler = MyHandler

    # Create the server
    with socketserver.TCPServer((listen_address, port), handler) as httpd:
        print(f"Serving on {listen_address}:{port}")
        httpd.serve_forever()
