import autopilot_telemetry as at


class FakeDataLoader:
    batch_size = 4
    num_workers = 2
    prefetch_factor = 2
    pin_memory = True

    def __iter__(self):
        for _ in range(2):
            yield {
                "input_ids": FakeTensor((4, 128)),
                "attention_mask": FakeTensor((4, 128)),
            }


class FakeTensor:
    def __init__(self, shape):
        self.shape = shape


def main() -> None:
    at.init(job_name="transformer-example")
    at.configure_sampled_profiler(wait=1, warmup=1, record=1, repeat=1)

    loader = at.instrument_dataloader(FakeDataLoader(), loader_name="fake-transformer-loader")
    for step, batch in enumerate(loader):
        with at.step_context(step=step, batch=batch) as ctx:
            with at.time_memcpy(
                copy_kind="h2d",
                step=ctx["step"],
                parent_span_id=ctx["span_id"],
                src_device="cpu",
                dst_device="cuda:0",
                num_bytes=4 * 128 * 8,
                non_blocking=True,
                device="cuda:0",
            ):
                pass
            with at.phase("forward", step=ctx["step"], parent_span_id=ctx["span_id"], device="cuda:0"):
                pass
            with at.phase("backward", step=ctx["step"], parent_span_id=ctx["span_id"], device="cuda:0"):
                pass


if __name__ == "__main__":
    main()
