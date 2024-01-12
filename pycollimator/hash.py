import hashlib
from pycollimator.log import Log


class Hash:
    @classmethod
    def sha256sum(cls, filename: str) -> str:
        sha256_hash = hashlib.sha256()
        with open(filename, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
            value = sha256_hash.hexdigest()
            Log.trace(f"SHA256 hash of {filename} is {value}")
            return value
