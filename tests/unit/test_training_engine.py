import tempfile
import unittest
from pathlib import Path

import torch

from mmctr.core import Batch, ModelOutput
from mmctr.data import DatasetManifest
from mmctr.training import (
    AlternatingPhase,
    CheckpointManager,
    TrainingEngine,
    build_optimizer,
    build_phased_adam,
)


class TinyLoader:
    dataset_name = "fixture"
    manifest = DatasetManifest(
        name="fixture",
        version="v1",
        storage_format="memory",
        sequence_length=2,
        padding_id=0,
        feature_dimensions={"id": 1},
    )

    def iter_batches(self, split):
        labels = torch.tensor([0.0, 1.0])
        yield Batch(
            user_features={"id": torch.tensor([[1], [2]])},
            item_features={"score": torch.tensor([[0.0], [1.0]])},
            history_features={"id": torch.tensor([[0, 0], [1, 0]])},
            history_mask=torch.tensor([[False, False], [True, False]]),
            labels=labels,
            metadata={"split": split},
        )


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, batch):
        return ModelOutput(self.linear(batch.item_features["score"]))


class TinyAlternatingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.main = torch.nn.Parameter(torch.tensor([0.25]))
        self.discriminator = torch.nn.Parameter(torch.tensor([0.2]))
        self.generator = torch.nn.Parameter(torch.tensor([0.75]))
        self.phase_calls = []

    def forward(self, batch):
        return ModelOutput(self.main.expand(batch.batch_size))

    def discriminator_loss(self, batch):
        self.phase_calls.append("discriminator")
        return (self.discriminator - batch.labels.mean()).pow(2).sum()

    def generator_loss(self, batch):
        self.phase_calls.append("generator")
        return (self.generator - batch.labels.mean()).pow(2).sum()


class TrainingEngineTests(unittest.TestCase):
    def test_alternating_phases_start_on_schedule_and_run_in_declared_order(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            model = TinyAlternatingModel()
            optimizer = build_phased_adam(
                {
                    "main": (model.main,),
                    "discriminator": (model.discriminator,),
                    "generator": (model.generator,),
                },
                {"main": 0.1, "discriminator": 0.1, "generator": 0.1},
            )
            engine = TrainingEngine(
                model,
                optimizer,
                torch.device("cpu"),
                CheckpointManager(Path(directory) / "checkpoints"),
                alternating_phases=(
                    AlternatingPhase("discriminator", 1, model.discriminator_loss),
                    AlternatingPhase("generator", 1, model.generator_loss),
                ),
            )
            batch = next(TinyLoader().iter_batches("train"))
            discriminator_before = model.discriminator.detach().clone()
            generator_before = model.generator.detach().clone()

            engine.train_epoch((batch,), epoch=0)
            self.assertEqual([], model.phase_calls)
            self.assertTrue(torch.equal(discriminator_before, model.discriminator))
            self.assertTrue(torch.equal(generator_before, model.generator))

            engine.train_epoch((batch,), epoch=1)
            self.assertEqual(["discriminator", "generator"], model.phase_calls)
            self.assertFalse(torch.equal(discriminator_before, model.discriminator))
            self.assertFalse(torch.equal(generator_before, model.generator))

    def test_phased_adam_updates_only_the_selected_parameter_group(self):
        parameters = {
            "main": torch.nn.Parameter(torch.tensor([1.0])),
            "discriminator": torch.nn.Parameter(torch.tensor([2.0])),
            "generator": torch.nn.Parameter(torch.tensor([3.0])),
        }
        optimizer = build_phased_adam(
            {name: (parameter,) for name, parameter in parameters.items()},
            {name: 0.1 for name in parameters},
        )
        before = {name: parameter.detach().clone() for name, parameter in parameters.items()}
        for parameter in parameters.values():
            parameter.grad = torch.ones_like(parameter)

        optimizer.step_phase("discriminator")

        self.assertTrue(torch.equal(before["main"], parameters["main"]))
        self.assertFalse(torch.equal(before["discriminator"], parameters["discriminator"]))
        self.assertTrue(torch.equal(before["generator"], parameters["generator"]))

    def test_phased_adam_closure_cannot_bypass_parameter_group_isolation(self):
        parameters = {
            "main": torch.nn.Parameter(torch.tensor([1.0])),
            "discriminator": torch.nn.Parameter(torch.tensor([2.0])),
            "generator": torch.nn.Parameter(torch.tensor([3.0])),
        }
        optimizer = build_phased_adam(
            {name: (parameter,) for name, parameter in parameters.items()},
            {name: 0.1 for name in parameters},
        )
        before = {name: parameter.detach().clone() for name, parameter in parameters.items()}

        def closure():
            optimizer.zero_grad()
            loss = sum(parameter.pow(2).sum() for parameter in parameters.values())
            loss.backward()
            return loss

        loss = optimizer.step_phase("discriminator", closure=closure)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(torch.equal(before["main"], parameters["main"]))
        self.assertFalse(torch.equal(before["discriminator"], parameters["discriminator"]))
        self.assertTrue(torch.equal(before["generator"], parameters["generator"]))

    def test_phased_adam_state_round_trips_with_the_run_checkpoint(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            model = TinyAlternatingModel()
            optimizer = build_phased_adam(
                {
                    "main": (model.main,),
                    "discriminator": (model.discriminator,),
                    "generator": (model.generator,),
                },
                {"main": 0.01, "discriminator": 0.02, "generator": 0.03},
            )
            for phase, parameter in (
                ("main", model.main),
                ("discriminator", model.discriminator),
                ("generator", model.generator),
            ):
                optimizer.zero_grad()
                parameter.grad = torch.ones_like(parameter)
                optimizer.step_phase(phase)
            manager = CheckpointManager(Path(directory) / "checkpoints")
            manager.save("last", model, optimizer, epoch=3)

            restored_model = TinyAlternatingModel()
            restored_optimizer = build_phased_adam(
                {
                    "main": (restored_model.main,),
                    "discriminator": (restored_model.discriminator,),
                    "generator": (restored_model.generator,),
                },
                {"main": 0.01, "discriminator": 0.02, "generator": 0.03},
            )
            state = manager.restore("last", restored_model, restored_optimizer)

            self.assertEqual(3, state.epoch)
            self.assertEqual(3, len(restored_optimizer.state))
            self.assertEqual(
                ["main", "discriminator", "generator"],
                [group["phase"] for group in restored_optimizer.param_groups],
            )

    def test_train_save_load_and_resume(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            model = TinyModel()
            optimizer = build_optimizer(model, "adam", 0.01)
            checkpoints = CheckpointManager(Path(directory) / "checkpoints")
            engine = TrainingEngine(model, optimizer, torch.device("cpu"), checkpoints)
            result = engine.fit(TinyLoader(), 2, 1, "fixture-run", Path(directory))
            self.assertTrue(result.succeeded)
            self.assertTrue(checkpoints.best_path.is_file())
            self.assertTrue(checkpoints.last_path.is_file())
            next_epoch = engine.resume()
            self.assertGreaterEqual(next_epoch, 1)
            resumed = engine.fit(
                TinyLoader(),
                next_epoch + 1,
                1,
                "fixture-run",
                Path(directory),
                start_epoch=next_epoch,
            )
            self.assertTrue(resumed.succeeded)

    def test_test_split_is_not_read_during_fit(self):
        class GuardedLoader(TinyLoader):
            def iter_batches(self, split):
                if split == "test":
                    raise AssertionError("fit must not read test")
                return super().iter_batches(split)

        with tempfile.TemporaryDirectory(dir=".") as directory:
            model = TinyModel()
            optimizer = build_optimizer(model, "sgd", 0.01)
            engine = TrainingEngine(
                model,
                optimizer,
                torch.device("cpu"),
                CheckpointManager(Path(directory) / "checkpoints"),
            )
            engine.fit(GuardedLoader(), 1, 1, "fixture-run", Path(directory))


if __name__ == "__main__":
    unittest.main()
