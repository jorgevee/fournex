import autopilot_telemetry as at


def main() -> None:
    at.init(job_name="resnet-example")

    batch = {
        "input_ids": FakeTensor((32, 224, 224, 3)),
        "labels": FakeTensor((32,)),
    }

    with at.step_context(step=0, batch=batch) as step:
        with at.phase("forward", step=step["step"], parent_span_id=step["span_id"]):
            pass


class FakeTensor:
    def __init__(self, shape):
        self.shape = shape


if __name__ == "__main__":
    main()
