import http.server
import socketserver
from io import BytesIO
from urllib.parse import urlparse, parse_qs
import torch
from diffusers import AutoPipelineForText2Image
from tensorflow.keras.models import load_model

from sampling_module import generate_text
from vectorizer import load_vectorizer


class MyHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        global g_model
        global g_vectorizer
        global g_temperature
        global g_max_output_tokens
        global g_pipe

        parsed_path = urlparse(self.path)
        params = parse_qs(parsed_path.query)
        print(parsed_path)
        print(params)

        if parsed_path.path == '/card-data.js':
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
        elif parsed_path.path == '/card-image.png':
            self.send_response(200)
            self.send_header('Content-type', 'image')
            self.end_headers()

            # TODO: fiddle with the prompt and style some more
            # Repeat the name, to give it greater weight in the prompt
            prompt_items = [f'"{params["name"][0]}"'] * 8
            prompt_items.extend([
                "art by Alayna Lemmer",
                "art by Alex Horley"
            ])
            if "type[]" in params:
                prompt_items.extend(params["type[]"])
            if "subtype[]" in params:
                prompt_items.extend(params["subtype[]"] * 4)
            if "supertype[]" in params:
                prompt_items.extend(params["supertype[]"])

            prompt = ", ".join(prompt_items)
            print(prompt)

            image = g_pipe(
                prompt=prompt,
                num_inference_steps=2,
                guidance_scale=0.0,
                width=768,
                height=512
            ).images[0]

            with BytesIO() as buffer:
                image.save(buffer, format="PNG")
                image_bytes = buffer.getvalue()

            self.wfile.write(image_bytes)

        elif parsed_path.path == '/':
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
    global g_pipe

    g_vectorizer = load_vectorizer(vectorizer_path)
    g_model = load_model(model_path)

    g_temperature = temperature
    g_max_output_tokens = max_output_tokens

    # Diffusers
    g_pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16"
    )
    g_pipe.to("cuda")

    # Specify the custom handler
    handler = MyHandler

    # Create the server
    with socketserver.TCPServer((listen_address, port), handler) as httpd:
        print(f"Serving on {listen_address}:{port}")
        httpd.serve_forever()
