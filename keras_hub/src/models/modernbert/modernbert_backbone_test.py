import pytest
from keras import ops

from keras_hub.src.models.modernbert.modernbert_backbone import (
    ModernBertBackbone,
)
from keras_hub.src.tests.test_case import TestCase


class ModernBertBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 10,
            "num_layers": 2,
            "num_query_heads": 4,
            "num_key_value_heads": 2,
            "hidden_dim": 8,
            "intermediate_dim": 8,
        }
        self.input_data = {
            "token_ids": ops.ones((2, 5), dtype="int32"),
            "padding_mask": ops.ones((2, 5), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=ModernBertBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 8),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=ModernBertBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=ModernBertBackbone,
            preset="bert_tiny_en_uncased",
            input_data={
                "token_ids": ops.array([[101, 1996, 4248, 102]], dtype="int32"),
                "segment_ids": ops.zeros((1, 4), dtype="int32"),
                "padding_mask": ops.ones((1, 4), dtype="int32"),
            },
            expected_output_shape={
                "sequence_output": (1, 4, 128),
                "pooled_output": (1, 128),
            },
            # The forward pass from a preset should be stable!
            expected_partial_output={
                "sequence_output": (
                    ops.array([-1.38173, 0.16598, -2.92788, -2.66958, -0.61556])
                ),
                "pooled_output": (
                    ops.array([-0.99999, 0.07777, -0.99955, -0.00982, -0.99967])
                ),
            },
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in ModernBertBackbone.preset:
            self.run_preset_test(
                cls=ModernBertBackbone,
                preset=preset,
                input_data=self.input_data,
            )
