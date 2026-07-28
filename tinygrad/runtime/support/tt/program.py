from __future__ import annotations
import struct

from tinygrad.runtime.support.tt.firmware.consts import KERNEL_ROLES, KernelRole

PROGRAM_MAGIC, PROGRAM_VERSION = b"TTPR", 1
PROGRAM_HEADER = struct.Struct("<4sI5I")


def encode_tt_program(images:dict[KernelRole, bytes]) -> bytes:
  unknown = set(images) - set(KERNEL_ROLES)
  if unknown: raise ValueError(f"unknown TT program roles: {sorted(unknown)}")
  payloads = tuple(bytes(images.get(role, b"")) for role in KERNEL_ROLES)
  return PROGRAM_HEADER.pack(PROGRAM_MAGIC, PROGRAM_VERSION, *(len(image) for image in payloads)) + b"".join(payloads)


def decode_tt_program(lib:bytes) -> dict[KernelRole, bytes]:
  if len(lib) < PROGRAM_HEADER.size: raise ValueError("TT program container is truncated")
  magic, version, *sizes = PROGRAM_HEADER.unpack_from(lib)
  if magic != PROGRAM_MAGIC: raise ValueError("invalid TT program magic")
  if version != PROGRAM_VERSION: raise ValueError(f"unsupported TT program version {version}")
  offset, images = PROGRAM_HEADER.size, {}
  for role, size in zip(KERNEL_ROLES, sizes):
    if offset + size > len(lib): raise ValueError(f"TT {role} image is truncated")
    images[role] = lib[offset:offset+size]
    offset += size
  if offset != len(lib): raise ValueError("TT program container has trailing bytes")
  return images
