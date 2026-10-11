"""Import verified V1_01 downloads into WSL, convert and audit."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import zipfile


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(8*1024*1024):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lab', type=Path, required=True)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--resume', action='store_true', help='Verify and reuse existing raw data; preserve prior failed audit')
    args = parser.parse_args()
    root = args.lab/'data/public/euroc'
    root.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic()+1800
    filenames = ['V1_01_easy.bag', 'V1_01_easy.zip']
    while not all((args.stage/(name+'.receipt.json')).exists() for name in filenames):
        if time.monotonic() > deadline:
            raise TimeoutError('Verified downloads did not complete')
        time.sleep(3)
    receipts = []
    for name in filenames:
        receipt = json.loads((args.stage/(name+'.receipt.json')).read_text())
        target = root/name
        if target.exists():
            if not args.resume:
                raise RuntimeError('Refusing to overwrite existing dataset: '+str(target))
        else:
            shutil.copyfile(args.stage/name, target)
        assert target.stat().st_size == receipt['bytes']
        assert sha256(target) == receipt['sha256']
        receipts.append(receipt)
        print('WSL COPY VERIFIED', name, flush=True)
    provenance = root/'provenance'
    provenance.mkdir(exist_ok=args.resume)
    for name in ['eth-item.json', 'eth-files.json', 'archive-entries.json']:
        shutil.copyfile(args.stage/name, provenance/name)
    asl = root/'V1_01_easy_asl'
    reuse_asl = asl.exists() and args.resume
    asl.mkdir(exist_ok=args.resume)
    with zipfile.ZipFile(root/'V1_01_easy.zip') as archive:
        for member in archive.infolist():
            resolved = (asl/member.filename).resolve()
            if not resolved.is_relative_to(asl.resolve()):
                raise RuntimeError('Invalid archive member path')
        if reuse_asl:
            assert archive.testzip() is None
        else:
            archive.extractall(asl)  # ZipExtFile validates member CRC.
        entries = len(archive.infolist())
    mav = asl/'mav0'
    assert (mav/'state_groundtruth_estimate0/data.csv').is_file()
    print('ASL extracted; CRC checked for', entries, 'entries', flush=True)
    destination = root/'V1_01_easy_ros2'
    command = [str(args.lab/'.venv-eval/bin/rosbags-convert'), '--src', str(root/'V1_01_easy.bag'),
               '--dst', str(destination), '--dst-version', '9', '--dst-storage', 'mcap',
               '--dst-typestore', 'ros2_jazzy']
    report = args.lab/'reports/w1'
    p = subprocess.CompletedProcess(command, 0, '', '') if args.resume and destination.exists() else subprocess.run(command, capture_output=True, text=True)
    (report/('v101-conversion-resume.log' if args.resume else 'v101-conversion.log')).write_text(repr(command)+'\n'+p.stdout+p.stderr+'\nEXIT '+str(p.returncode))
    p.check_returncode()
    print('Converted to ROS2 Jazzy MCAP', flush=True)
    audit_dir = report/'v101-data-audit'
    if args.resume and audit_dir.exists():
        audit_dir.rename(report/('v101-data-audit-prior-'+str(time.time_ns())))
    command = [str(args.lab/'.venv-eval/bin/python'), str(args.lab/'scripts/audit_euroc.py'),
               '--bag', str(root/'V1_01_easy.bag'), '--ros2', str(destination),
               '--asl', str(mav), '--report', str(audit_dir)]
    p = subprocess.run(command, capture_output=True, text=True)
    (report/'v101-audit.log').write_text(repr(command)+'\n'+p.stdout+p.stderr+'\nEXIT '+str(p.returncode))
    print(p.stdout, p.stderr, flush=True)
    p.check_returncode()
    manifest = {'schema_version': 1, 'status': 'downloaded_converted_audited', 'sequence': 'V1_01_easy',
                'source_doi': 'https://doi.org/10.3929/ethz-b-000690084',
                'raw_files': receipts, 'ros1_bag': str(root/'V1_01_easy.bag'),
                'asl_directory': str(mav), 'ros2_directory': str(destination),
                'ground_truth_original': str(mav/'state_groundtruth_estimate0/data.csv'),
                'ground_truth_corrected': str(args.lab/'ros_ws/src/open_vins/ov_data/euroc_mav/V1_01_easy.csv'),
                'conversion': {'rosbags_version': '0.11.6', 'ros2_distribution': 'jazzy',
                               'metadata_version': 9, 'storage': 'mcap'},
                'audit': str(audit_dir/'audit.json'), 'artifacts': {}}
    for path in sorted(destination.rglob('*')):
        if path.is_file():
            manifest['artifacts'][str(path.relative_to(root))] = {'bytes': path.stat().st_size, 'sha256': sha256(path)}
    for path in sorted(mav.glob('*/sensor.yaml')):
        manifest['artifacts'][str(path.relative_to(root))] = {'bytes': path.stat().st_size, 'sha256': sha256(path)}
    for name in ['ground_truth_original', 'ground_truth_corrected']:
        path = Path(manifest[name]);manifest[name+'_sha256'] = sha256(path)
    import yaml
    (args.lab/'manifests/euroc_v101.yaml').write_text(yaml.safe_dump(manifest, sort_keys=False))
    print('MANIFEST', json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
