from flask import Flask, render_template_string, request
import speedtest

app = Flask(__name__)

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Internet Speed Test</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            font-family: Arial, sans-serif;
            height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .card {
            border-radius: 20px;
            box-shadow: 0px 8px 20px rgba(0,0,0,0.3);
            background-color: #1e1e2f;
            padding: 30px;
            width: 500px;
            text-align: center;
        }
        .btn-custom {
            background-color: #667eea;
            border: none;
            padding: 10px 20px;
            font-size: 18px;
            border-radius: 12px;
            color: white;
            transition: 0.3s;
        }
        .btn-custom:hover {
            background-color: #5765d3;
        }
        .result-box {
            background: #2d2d44;
            padding: 15px;
            border-radius: 12px;
            margin-top: 20px;
        }
        .download { color: #4ade80; font-weight: bold; }
        .upload { color: #38bdf8; font-weight: bold; }
        .ping { color: #facc15; font-weight: bold; }
        h2 { margin-bottom: 20px; color: white; }
        table { width: 100%; margin-top: 15px; }
        th, td { padding: 8px; text-align: center; }
        th { color: #facc15; }

        /* Spinner */
        .spinner {
            display: none;
            margin-top: 20px;
        }
        .spinner-border {
            width: 3rem;
            height: 3rem;
        }
    </style>
    <script>
        function showSpinner() {
            document.getElementById("spinner").style.display = "block";
        }
    </script>
</head>
<body>
    <div class="card">
        <h2>🚀 Internet Speed Test</h2>
        <form action="/test" method="post" onsubmit="showSpinner()">
            <button type="submit" class="btn-custom">Run Test</button>
        </form>

        <!-- Spinner -->
        <div id="spinner" class="spinner">
            <div class="spinner-border text-warning" role="status">
                <span class="visually-hidden">Testing...</span>
            </div>
            <p style="margin-top:10px;">Running test... Please wait</p>
        </div>

        {% if result %}
            <div class="result-box mt-4">
                <p><b style="color:white;">Download:</b> <span class="download">{{ result['download'] }} Mbps</span></p>
                <p><b style="color:white;">Upload:</b> <span class="upload">{{ result['upload'] }} Mbps</span></p>
                <p><b style="color:white;">Ping:</b> <span class="ping">{{ result['ping'] }} ms</span></p>

                <h5 style="margin-top:20px; color:white;">📂 File Transfer Estimates</h5>
                <table border="1" class="table table-dark table-striped">
                    <tr>
                        <th>File Size</th>
                        <th>Download Time</th>
                        <th>Upload Time</th>
                    </tr>
                    {% for size, times in result['file_times'].items() %}
                    <tr>
                        <td>{{ size }}</td>
                        <td>{{ times['download'] }} sec</td>
                        <td>{{ times['upload'] }} sec</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_PAGE)

@app.route("/test", methods=["GET", "POST"])
def run_test():
    if request.method == "POST":
        st = speedtest.Speedtest()
        st.get_best_server()
        download = round(st.download() / 1_000_000, 2)  
        upload = round(st.upload() / 1_000_000, 2)      
        ping = round(st.results.ping, 2)

        file_sizes = [1, 10, 100]
        file_times = {}
        for size in file_sizes:
            download_time = round((size * 8) / download, 2) if download > 0 else "∞"
            upload_time = round((size * 8) / upload, 2) if upload > 0 else "∞"
            file_times[f"{size} MB"] = {
                "download": download_time,
                "upload": upload_time
            }

        result = {
            "download": download,
            "upload": upload,
            "ping": ping,
            "file_times": file_times
        }
        return render_template_string(HTML_PAGE, result=result)
    else:
        return render_template_string(HTML_PAGE)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
