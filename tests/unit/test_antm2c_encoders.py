import io
import tarfile
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from mmctr.data.datasets.antm2c.encoders import fingerprint_checkpoint, load_tar_images


class AntM2CEncoderTests(unittest.TestCase):
    def test_checkpoint_fingerprint_anchors_every_local_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text('{"hidden_size": 2}\n', encoding="utf-8")
            (root / "weights.bin").write_bytes(b"weights-v1")

            identity = fingerprint_checkpoint(root)
            repeated = fingerprint_checkpoint(root)
            (root / "weights.bin").write_bytes(b"weights-v2")
            changed = fingerprint_checkpoint(root)

        self.assertEqual(identity, repeated)
        self.assertEqual(("config.json", "weights.bin"), tuple(identity.files))
        self.assertEqual(64, len(identity.sha256))
        self.assertNotEqual(identity.sha256, changed.sha256)

    def test_tar_images_are_streamed_without_extracting_the_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "images.tar.gz"
            buffer = io.BytesIO()
            Image.new("RGB", (3, 2), color=(10, 20, 30)).save(buffer, format="PNG")
            payload = buffer.getvalue()
            with tarfile.open(archive, "w:gz") as stream:
                member = tarfile.TarInfo("AntM2C_image/item-1.png")
                member.size = len(payload)
                stream.addfile(member, io.BytesIO(payload))

            images = load_tar_images(archive, ["AntM2C_image/item-1.png"])

        self.assertEqual((3, 2), images["AntM2C_image/item-1.png"].size)
        self.assertEqual("RGB", images["AntM2C_image/item-1.png"].mode)


if __name__ == "__main__":
    unittest.main()
