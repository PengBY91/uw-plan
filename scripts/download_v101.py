"""Fetch only V1_01 members of ETH's ZIP64 archive using HTTP Range."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import struct
import time
import urllib.request
import zlib

URL = 'https://www.research-collection.ethz.ch/server/api/core/bitstreams/02ecda9a-298f-498b-970c-b7c44334d880/content'
ARCHIVE_BYTES = 6042263426


def get_range(start, size):
    end = start + size - 1
    req = urllib.request.Request(URL, headers={
        'User-Agent': 'Mozilla/5.0', 'Range': f'bytes={start}-{end}'})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                expected = f'bytes {start}-{end}/{ARCHIVE_BYTES}'
                if response.status != 206 or response.headers.get('Content-Range') != expected:
                    raise RuntimeError('Unexpected HTTP Range response')
                data = response.read(size)
                if len(data) != size:
                    raise RuntimeError('Incomplete HTTP Range payload')
                return data
        except Exception:
            if attempt == 2:
                raise
            time.sleep(3 * (attempt + 1))


def download(entry, stage):
    filename = Path(entry['name']).name
    header = get_range(entry['header_offset'], 30)
    if header[:4] != b'PK\x03\x04':
        raise RuntimeError('Invalid ZIP local header')
    name_size, extra_size = struct.unpack_from('<HH', header, 26)
    payload_start = entry['header_offset'] + 30 + name_size + extra_size
    compressed = stage / (filename + '.deflate')
    offset = compressed.stat().st_size if compressed.exists() else 0
    if offset > entry['compressed']:
        raise RuntimeError('Local compressed payload exceeds expected size')
    with compressed.open('ab') as output:
        while offset < entry['compressed']:
            count = min(16 * 1024 * 1024, entry['compressed'] - offset)
            data = get_range(payload_start + offset, count)
            output.write(data)
            output.flush()
            offset += len(data)
            if offset % (64 * 1024 * 1024) == 0 or offset == entry['compressed']:
                print(filename, offset, '/', entry['compressed'], flush=True)
    target = stage / filename
    if target.exists():
        raise RuntimeError('Refusing to overwrite final dataset file: ' + str(target))
    tmp = stage / (filename + '.partial')
    decoder = zlib.decompressobj(-15)
    crc, size, sha = 0, 0, hashlib.sha256()
    with compressed.open('rb') as source, tmp.open('wb') as output:
        while chunk := source.read(1024 * 1024):
            raw = decoder.decompress(chunk)
            output.write(raw)
            crc = zlib.crc32(raw, crc)
            sha.update(raw)
            size += len(raw)
        raw = decoder.flush()
        output.write(raw)
        crc = zlib.crc32(raw, crc)
        sha.update(raw)
        size += len(raw)
    if not decoder.eof or size != entry['size'] or f'{crc:08x}' != entry['crc32']:
        raise RuntimeError('ZIP member integrity verification failed: ' + filename)
    tmp.rename(target)
    result = {'filename': filename, 'source_archive': URL,
              'archive_member': entry['name'], 'archive_member_crc32': entry['crc32'],
              'crc32_verified': True, 'bytes': size, 'sha256': sha.hexdigest(),
              'whole_archive_md5_verified': False}
    (stage / (filename + '.receipt.json')).write_text(json.dumps(result, indent=2) + '\n')
    print('VERIFIED', json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=Path, required=True)
    args = parser.parse_args()
    entries = json.loads((args.stage / 'archive-entries.json').read_text())
    selected = [entry for entry in entries if entry['name'].endswith(
        ('/V1_01_easy.bag', '/V1_01_easy.zip'))]
    assert len(selected) == 2 and all(e['method'] == 8 for e in selected)
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(download, entry, args.stage) for entry in selected]
        for future in futures:
            future.result()


if __name__ == '__main__':
    main()
