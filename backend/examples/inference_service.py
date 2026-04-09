import autopilot_telemetry as at


def handle_request() -> None:
    at.init(job_name="inference-example")


if __name__ == "__main__":
    handle_request()
