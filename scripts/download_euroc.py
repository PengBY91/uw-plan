"""Fetch selected EuRoC sequence members of ETH's ZIP64 archives using HTTP Range.

Archive URL and size come from eth-files.json (by archive name), member offsets from
archive-entries-<archive>.json, which --list-entries builds from the ZIP64 central directory.
The vicon_room1 entry list is the historical archive-entries.json.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import struct
import time
import urllib.error
import urllib.request
import zlib

SEQUENCE_ARCHIVES = {'MH': 'machine_hall', 'V1': 'vicon_room1', 'V2': 'vicon_room2'}


class Archive:
    def __init__(self, eth_files, name):
        matches = [b for b in json.loads(eth_files.read_text())['_embedded']['bitstreams']
                   if b['name'] == name + '.zip']
        if len(matches) != 1:
            raise RuntimeError('Archive not uniquely listed in eth-files.json: ' + name)
        self.name = name
        self.url = matches[0]['_links']['content']['href']
        self.size = matches[0]['sizeBytes']
        self.md5 = matches[0]['checkSum']['value']

    def get_range(self, start, size):
        end = start + size - 1
        req = urllib.request.Request(self.url, headers={
            'User-Agent': 'Mozilla/5.0', 'Range': f'bytes={start}-{end}'})
        for attempt in range(12):
            try:
                with urllib.request.urlopen(req, timeout=600) as response:
                    expected = f'bytes {start}-{end}/{self.size}'
                    if response.status != 206 or response.headers.get('Content-Range') != expected:
                        raise RuntimeError('Unexpected HTTP Range response')
                    data = response.read(size)
                    if len(data) != size:
                        raise RuntimeError('Incomplete HTTP Range payload')
                    return data
            except Exception as error:
                if attempt == 11:
                    raise
                # The server rate-limits per client with 429 and no Retry-After; back off for minutes.
                limited = isinstance(error, urllib.error.HTTPError) and error.code == 429
                wait = min(60 * (attempt + 1), 600) if limited else 3 * (attempt + 1)
                print('retry in', wait, 's after', repr(error), flush=True)
                time.sleep(wait)


def list_entries(archive):
    """Parse the ZIP64 end-of-central-directory records and the central directory."""
    tail_size = 65536 + 22
    tail = archive.get_range(archive.size - tail_size, tail_size)
    eocd = tail.rfind(b'PK\x05\x06')
    locator = tail.rfind(b'PK\x06\x07', 0, eocd)
    if eocd < 0 or locator < 0:
        raise RuntimeError('ZIP64 end records not found')
    zip64_eocd_offset = struct.unpack_from('<Q', tail, locator + 8)[0]
    record = archive.get_range(zip64_eocd_offset, 56)
    if record[:4] != b'PK\x06\x06':
        raise RuntimeError('Invalid ZIP64 end-of-central-directory record')
    count, cd_size, cd_offset = struct.unpack_from('<QQQ', record, 32)
    directory = archive.get_range(cd_offset, cd_size)
    entries, pos = [], 0
    for _ in range(count):
        if directory[pos:pos + 4] != b'PK\x01\x02':
            raise RuntimeError('Invalid central directory header')
        method, = struct.unpack_from('<H', directory, pos + 10)
        crc, compressed, size = struct.unpack_from('<III', directory, pos + 16)
        name_len, extra_len, comment_len = struct.unpack_from('<HHH', directory, pos + 28)
        offset, = struct.unpack_from('<I', directory, pos + 42)
        name = directory[pos + 46:pos + 46 + name_len].decode('utf-8')
        extra = directory[pos + 46 + name_len:pos + 46 + name_len + extra_len]
        e = 0
        while e < len(extra):
            tag, length = struct.unpack_from('<HH', extra, e)
            if tag == 1:  # ZIP64 extended information, fields present only when saturated
                values, k = extra[e + 4:e + 4 + length], 0
                if size == 0xFFFFFFFF:
                    size, = struct.unpack_from('<Q', values, k); k += 8
                if compressed == 0xFFFFFFFF:
                    compressed, = struct.unpack_from('<Q', values, k); k += 8
                if offset == 0xFFFFFFFF:
                    offset, = struct.unpack_from('<Q', values, k); k += 8
            e += 4 + length
        entries.append({'name': name, 'size': size, 'compressed': compressed, 'method': method,
                        'crc32': f'{crc:08x}', 'header_offset': offset})
        pos += 46 + name_len + extra_len + comment_len
    if pos != cd_size:
        raise RuntimeError('Central directory size mismatch')
    return entries


def download(archive, entry, stage):
    filename = Path(entry['name']).name
    header = archive.get_range(entry['header_offset'], 30)
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
            count = min(128 * 1024 * 1024, entry['compressed'] - offset)
            data = archive.get_range(payload_start + offset, count)
            output.write(data)
            output.flush()
            offset += len(data)
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
    compressed.unlink()
    result = {'filename': filename, 'source_archive': archive.url,
              'archive_member': entry['name'], 'archive_member_crc32': entry['crc32'],
              'crc32_verified': True, 'bytes': size, 'sha256': sha.hexdigest(),
              'whole_archive_md5_verified': False}
    (stage / (filename + '.receipt.json')).write_text(json.dumps(result, indent=2) + '\n')
    print('VERIFIED', json.dumps(result), flush=True)
    return result


def entries_path(stage, archive):
    # vicon_room1 keeps its historical file name.
    return stage / ('archive-entries.json' if archive == 'vicon_room1' else f'archive-entries-{archive}.json')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--sequence', default='V1_01_easy', help='e.g. V1_01_easy, MH_01_easy')
    parser.add_argument('--list-entries', action='store_true',
                        help='Only read the archive central directory into archive-entries-<archive>.json')
    args = parser.parse_args()
    archive = Archive(args.stage / 'eth-files.json', SEQUENCE_ARCHIVES[args.sequence[:2]])
    path = entries_path(args.stage, archive.name)
    if args.list_entries:
        if path.exists():
            raise RuntimeError('Refusing to overwrite ' + str(path))
        entries = list_entries(archive)
        path.write_text(json.dumps(entries, indent=2) + '\n')
        print('LISTED', len(entries), 'entries to', path, flush=True)
        return
    entries = json.loads(path.read_text())
    selected = [entry for entry in entries if entry['name'].endswith(
        (f'/{args.sequence}.bag', f'/{args.sequence}.zip'))]
    assert len(selected) == 2 and all(e['method'] == 8 for e in selected), selected
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        futures = [pool.submit(download, archive, entry, args.stage) for entry in selected]
        for future in futures:
            future.result()


if __name__ == '__main__':
    main()
