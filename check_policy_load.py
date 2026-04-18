from eval_libero import Qwen35VLALiberoPolicy


def main() -> None:
    policy = Qwen35VLALiberoPolicy(
        "checkpoints/qwen35_2b_vla_libero_layerwise_v2_bs128_ga1_deepspeed_train_30k_bs128fresh/checkpoint-30000",
        chunk_size=1,
        num_inference_steps=10,
        deterministic_seed=0,
    )
    print("policy_load_ok", policy._chunk_size)


if __name__ == "__main__":
    main()
