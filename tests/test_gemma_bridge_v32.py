import math
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.qwen_backbone_adapter import QwenBackboneAdapter
from model.qwen_gemma_bridge_action_head import (
    QwenGemmaBridgeActionConfig,
    QwenGemmaBridgeActionHead,
)


class QwenBackboneAdapterTapTests(unittest.TestCase):
    def _make_adapter(self):
        adapter = object.__new__(QwenBackboneAdapter)
        adapter._cached_full_attention_indices = [1, 3]
        return adapter

    def test_last_all_concat_and_scalar_mix_memory_layout(self):
        adapter = self._make_adapter()
        batch_size = 2
        seq_len = 5
        hidden_dim = 4
        hidden_states = [
            torch.zeros(batch_size, seq_len, hidden_dim),
            torch.full((batch_size, seq_len, hidden_dim), 1.0),
            torch.full((batch_size, seq_len, hidden_dim), 2.0),
            torch.full((batch_size, seq_len, hidden_dim), 3.0),
            torch.full((batch_size, seq_len, hidden_dim), 4.0),
        ]
        attention_mask = torch.tensor(
            [[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]],
            dtype=torch.long,
        )
        token_type_ids = torch.tensor(
            [[1, 1, 0, 0, 0], [1, 0, 0, 0, 0]],
            dtype=torch.long,
        )
        encoded_prefix = {
            "hidden_states": hidden_states,
            "attention_mask": attention_mask,
            "vlm_inputs": {"mm_token_type_ids": token_type_ids},
        }

        last = adapter.get_prefix_memory(encoded_prefix, tap_strategy="last")
        self.assertEqual(tuple(last["memory"].shape), (batch_size, seq_len, hidden_dim))
        self.assertTrue(torch.equal(last["memory_mask"], attention_mask.bool()))
        self.assertTrue(torch.equal(last["tap_ids"], torch.full_like(attention_mask, 3)))
        self.assertTrue(torch.equal(last["token_type_ids"], token_type_ids))

        for strategy in ("all_concat", "scalar_mix"):
            memory = adapter.get_prefix_memory(encoded_prefix, tap_strategy=strategy)
            self.assertEqual(tuple(memory["memory"].shape), (batch_size, seq_len * 2, hidden_dim))
            self.assertEqual(tuple(memory["memory_mask"].shape), (batch_size, seq_len * 2))
            self.assertEqual(memory["tap_indices"], [1, 3])
            self.assertTrue(torch.equal(memory["memory_mask"][:, :seq_len], attention_mask.bool()))
            self.assertTrue(torch.equal(memory["memory_mask"][:, seq_len:], attention_mask.bool()))
            self.assertTrue(torch.equal(memory["tap_ids"][:, :seq_len], torch.full_like(attention_mask, 1)))
            self.assertTrue(torch.equal(memory["tap_ids"][:, seq_len:], torch.full_like(attention_mask, 3)))
            self.assertTrue(torch.equal(memory["token_type_ids"][:, :seq_len], token_type_ids))
            self.assertTrue(torch.equal(memory["token_type_ids"][:, seq_len:], token_type_ids))


class GemmaBridgeV32ActionHeadTests(unittest.TestCase):
    def _make_config(
        self,
        tap_strategy: str = "last",
        stop_gradient_backbone: bool = True,
        bridge_num_queries: int = 3,
        bridge_layers: int = 1,
    ):
        return QwenGemmaBridgeActionConfig(
            hidden_dim=64,
            num_heads=8,
            num_kv_heads=1,
            num_layers=2,
            mlp_dim=128,
            action_dim=7,
            chunk_size=4,
            action_horizon=4,
            n_action_steps=4,
            max_action_dim=8,
            max_state_dim=8,
            state_dim=4,
            vlm_hidden_dim=32,
            bridge_policy_dim=64,
            bridge_num_queries=bridge_num_queries,
            bridge_layers=bridge_layers,
            tap_strategy=tap_strategy,
            stop_gradient_backbone=stop_gradient_backbone,
            bridge_dropout=0.0,
            dropout=0.0,
            bridge_scalar_mix=(tap_strategy == "scalar_mix"),
        )

    def _toy_inputs(self, seq_len: int = 5):
        prefix_memory = torch.randn(2, seq_len, 32)
        prefix_mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]], dtype=torch.bool)
        tap_ids = torch.zeros(2, seq_len, dtype=torch.long)
        token_type_ids = torch.tensor([[1, 1, 0, 0, 0], [1, 0, 0, 0, 0]], dtype=torch.long)
        actions = torch.randn(2, 4, 7)
        state = torch.randn(2, 4)
        return prefix_memory, prefix_mask, tap_ids, token_type_ids, actions, state

    def _concat_inputs(self):
        prefix_memory = torch.randn(2, 10, 32)
        prefix_mask = torch.tensor(
            [[1, 1, 1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 1, 1, 0, 0, 0]],
            dtype=torch.bool,
        )
        tap_ids = torch.tensor(
            [[1, 1, 1, 1, 1, 3, 3, 3, 3, 3], [1, 1, 1, 1, 1, 3, 3, 3, 3, 3]],
            dtype=torch.long,
        )
        token_type_ids = torch.tensor(
            [[1, 1, 0, 0, 0, 1, 1, 0, 0, 0], [1, 0, 0, 0, 0, 1, 0, 0, 0, 0]],
            dtype=torch.long,
        )
        actions = torch.randn(2, 4, 7)
        state = torch.randn(2, 4)
        return prefix_memory, prefix_mask, tap_ids, token_type_ids, actions, state

    def test_forward_outputs_are_finite(self):
        torch.manual_seed(0)
        head = QwenGemmaBridgeActionHead(self._make_config())
        prefix_memory, prefix_mask, tap_ids, token_type_ids, actions, state = self._toy_inputs()

        outputs = head(
            prefix_memory=prefix_memory,
            prefix_attention_mask=prefix_mask,
            tap_ids=tap_ids,
            token_type_ids=token_type_ids,
            actions=actions,
            state=state,
        )
        self.assertTrue(torch.isfinite(outputs["loss"]))
        self.assertEqual(tuple(outputs["predicted_velocity"].shape), (2, 4, 8))
        for key in (
            "raw_qwen_token_norm",
            "bridge_token_norm",
            "query_token_norm",
            "bridge_memory_norm",
            "expert_token_norm",
            "cross_attn_entropy",
            "memory_norm_ratio",
            "mean_norm_saturation_fraction",
        ):
            self.assertIn(key, outputs)
            self.assertTrue(torch.isfinite(outputs[key]).all())

    def test_predict_action_runs_with_all_concat_tokens(self):
        torch.manual_seed(0)
        head = QwenGemmaBridgeActionHead(self._make_config(tap_strategy="all_concat"))
        prefix_memory, prefix_mask, tap_ids, token_type_ids, _, state = self._concat_inputs()

        pred = head.predict_action(
            prefix_memory=prefix_memory,
            prefix_attention_mask=prefix_mask,
            tap_ids=tap_ids,
            token_type_ids=token_type_ids,
            state=state,
            num_steps=3,
            deterministic_seed=7,
        )
        self.assertEqual(tuple(pred.shape), (2, 4, 7))
        self.assertTrue(torch.isfinite(pred).all())

    def test_scalar_mix_forward_reports_normalized_weights(self):
        torch.manual_seed(0)
        head = QwenGemmaBridgeActionHead(self._make_config(tap_strategy="scalar_mix"))
        prefix_memory, prefix_mask, tap_ids, token_type_ids, actions, state = self._concat_inputs()

        outputs = head(
            prefix_memory=prefix_memory,
            prefix_attention_mask=prefix_mask,
            tap_ids=tap_ids,
            token_type_ids=token_type_ids,
            actions=actions,
            state=state,
        )
        self.assertIn("tap_weight_entropy", outputs)
        self.assertIn("scalar_mix_gamma", outputs)
        self.assertTrue(torch.isfinite(outputs["tap_weight_entropy"]).all())
        self.assertTrue(torch.isfinite(outputs["scalar_mix_gamma"]).all())

        stats = head.last_bridge_stats
        self.assertIn("tap_weight_distribution", stats)
        weights = stats["tap_weight_distribution"]
        self.assertEqual(len(weights), 2)
        self.assertTrue(math.isclose(sum(weights), 1.0, rel_tol=1e-5, abs_tol=1e-5))
        self.assertGreaterEqual(stats["tap_weight_entropy"], 0.0)
        self.assertGreater(stats["scalar_mix_gamma"], 0.0)

        pred = head.predict_action(
            prefix_memory=prefix_memory,
            prefix_attention_mask=prefix_mask,
            tap_ids=tap_ids,
            token_type_ids=token_type_ids,
            state=state,
            num_steps=3,
            deterministic_seed=11,
        )
        self.assertEqual(tuple(pred.shape), (2, 4, 7))
        self.assertTrue(torch.isfinite(pred).all())

    def test_stop_gradient_backbone_blocks_prefix_grads(self):
        torch.manual_seed(0)
        head = QwenGemmaBridgeActionHead(self._make_config())
        prefix_memory = torch.randn(2, 5, 32, requires_grad=True)
        prefix_mask = torch.ones(2, 5, dtype=torch.bool)
        tap_ids = torch.zeros(2, 5, dtype=torch.long)
        token_type_ids = torch.zeros(2, 5, dtype=torch.long)
        actions = torch.randn(2, 4, 7)
        state = torch.randn(2, 4)

        outputs = head(
            prefix_memory=prefix_memory,
            prefix_attention_mask=prefix_mask,
            tap_ids=tap_ids,
            token_type_ids=token_type_ids,
            actions=actions,
            state=state,
        )
        outputs["loss"].backward()

        self.assertIsNone(prefix_memory.grad)
        self.assertIsNotNone(head.bridge.query_tokens.grad)
        self.assertIsNotNone(head.action_out_proj.weight.grad)

    def test_stop_gradient_backbone_false_allows_prefix_grads(self):
        torch.manual_seed(0)
        head = QwenGemmaBridgeActionHead(
            self._make_config(tap_strategy="scalar_mix", stop_gradient_backbone=False)
        )
        prefix_memory = torch.randn(2, 10, 32, requires_grad=True)
        prefix_mask = torch.ones(2, 10, dtype=torch.bool)
        tap_ids = torch.tensor(
            [[1, 1, 1, 1, 1, 3, 3, 3, 3, 3], [1, 1, 1, 1, 1, 3, 3, 3, 3, 3]],
            dtype=torch.long,
        )
        token_type_ids = torch.zeros(2, 10, dtype=torch.long)
        actions = torch.randn(2, 4, 7)
        state = torch.randn(2, 4)

        outputs = head(
            prefix_memory=prefix_memory,
            prefix_attention_mask=prefix_mask,
            tap_ids=tap_ids,
            token_type_ids=token_type_ids,
            actions=actions,
            state=state,
        )
        outputs["loss"].backward()

        self.assertIsNotNone(prefix_memory.grad)
        self.assertGreater(prefix_memory.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(head.bridge.scalar_mixer.tap_logits.grad)


if __name__ == "__main__":
    unittest.main()
