import hashlib, unittest

from tinygrad.runtime.support.tt.chip import TTChip
from tinygrad.runtime.support.tt.firmware.consts import Firmware, TensixL1
from tinygrad.runtime.support.tt.firmware.core import build_brisc, build_ncrisc, build_trisc
from tinygrad.runtime.support.tt.firmware.cq import build_dispatch, build_prefetch
from tinygrad.runtime.support.tt.firmware.dram_cq import build_dram_brisc, build_dram_ncrisc


class TestTTFirmware(unittest.TestCase):
  def test_golden_images(self):
    # These match blackhole-py at 8f303b4. The fixed pcie_mid makes the CQ images deterministic.
    images = {
      "worker_brisc": build_brisc().lower(),
      "worker_ncrisc": build_ncrisc().lower(),
      **{f"worker_trisc{i}": build_trisc(i).lower() for i in range(3)},
      "prefetch": build_prefetch(0x12345678).lower(),
      "dispatch": build_dispatch(0x12345678).lower(),
      "dram_brisc": build_dram_brisc().lower(),
      "dram_ncrisc": build_dram_ncrisc().lower(),
    }
    expected = {
      "worker_brisc": "e878452ea85284edd03700b67e30681f8f86f8f250fdfe5bcf9233c634c4236b",
      "worker_ncrisc": "221cc6773fb6737af91a1f2978797fe087e02d3e65a2a9781ab9eae3fac5d08a",
      "worker_trisc0": "5ac368b93cd3770ee10d7215809ac43244803efbcbe37750440cfb6b8167ed50",
      "worker_trisc1": "5ea072c0af23c2cd2d8e8e3258f561c578c315121d0143bae6c27cd01e91b87a",
      "worker_trisc2": "7de9d93d6bccd81bd6a3e06967a07a7c42d35b8591945abdaf4e252302b09a83",
      "prefetch": "cd379bd9590d5c02d7b011c51a92a60429cc649ae4aaeb1d6023d0a7c3858ff0",
      "dispatch": "c70c0a1ac93433744edf46511822e81bfed32d9826dfe21165ad0c672309395f",
      "dram_brisc": "34bb954c4fad3baba984a3b753623c8e54cb9791583c158c170b2894286784ea",
      "dram_ncrisc": "1b7bdd8bb6e32665be5609c5d252b4ead9e5529665ea8041594804b2e536acc9",
    }
    self.assertEqual({name: hashlib.sha256(image).hexdigest() for name, image in images.items()}, expected)

  def test_images_fit_l1(self):
    workers = (build_brisc().lower(), build_ncrisc().lower(), *(build_trisc(i).lower() for i in range(3)))
    for (role, (_, capacity)), image in zip(Firmware.TEXT.items(), workers):
      with self.subTest(role=role): self.assertLessEqual(len(image), capacity)

    queues = {
      "prefetch_brisc": ("brisc", build_prefetch().lower()),
      "dispatch_brisc": ("brisc", build_dispatch().lower()),
      "dram_brisc": ("brisc", build_dram_brisc().lower()),
      "dram_ncrisc": ("ncrisc", build_dram_ncrisc().lower()),
    }
    for name, (role, image) in queues.items():
      with self.subTest(name=name): self.assertLessEqual(len(image), TensixL1.WORKER_TEXT_SIZE[role])

  def test_worker_firmware_is_partition_packed(self):
    self.assertEqual(len(TTChip._worker_firmware()), sum(size for _, size in Firmware.TEXT.values()))


if __name__ == "__main__": unittest.main()
