from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from vector_reporting import main_f as m

with open("config.json", "r") as file:
    config = json.load(file)

host = config.get("host")
port = config.get("port")

app = Flask(__name__)
CORS(app)

# Helper function to parse UNIX timestamps or ISO strings
def parse_time_input(value):
    try:
        return int(value)
    except ValueError:
        raise ValueError("Invalid time format. Use UNIX timestamp (int).")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Interface Report API is up and running!"}), 200

def normalize_to_seconds(timestamp):
    """
    Takes a UNIX timestamp in seconds or milliseconds.
    Returns timestamp in integer seconds.
    """
    # If it's too big, assume milliseconds
    if timestamp > 1e12:
        return int(timestamp / 1000)
    # Otherwise, assume seconds
    return int(timestamp)


@app.route("/generate-report", methods=["POST"])
def generate_interface_report():

    data = request.data
    json_str = data.decode("utf-8")
    data = json.loads(json_str)

    met = []
    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    report_type = data.get("report_type")
    hostname = data.get("hostname")
    from_time_raw = data.get("from_time")
    to_time_raw = data.get("to_time")

    # Get metric name based on report type
    if report_type in ['interface', 'filesystem']:
        metric_name = data.get("metric_name")
        met.append(metric_name)
    elif report_type == "Resource":
        metric_name = data.get("metric")
        met.append(metric_name)
    elif report_type == "alert":
        metric_name = None  # Not needed for alert

    # Validation
    if report_type == "alert":
        if not all([hostname, from_time_raw, to_time_raw]):
            return jsonify({"error": "Missing required fields for alert report"}), 400
    else:
        if not all([hostname, metric_name, from_time_raw, to_time_raw]):
            return jsonify({"error": "Missing required fields"}), 400

    result, status = m(
        hostname=str(hostname),
        metric_name=met,
        time_from=normalize_to_seconds(int(from_time_raw)),
        time_to=normalize_to_seconds(int(to_time_raw)),
        report_type=str(report_type)
    )

    return jsonify(result), status

if __name__ == "__main__":
    app.run(debug=True, host=host, port=port)